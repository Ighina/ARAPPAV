#!/usr/bin/env python
"""Harness for **skill-tuning** self-play (the `.claude/skills/` loop).

Instead of updating model weights with GRPO/DPO, this loop updates two *skills*
— `perturb-vN` and `verify-vN` — from the episodes they generate. The agent
plays both roles; this script owns everything that must not be left to agent
judgement:

* sampling problems for a round (and not reusing problems across rounds),
* parsing/validating Perturber output exactly like the RL path
  (``parse_and_backoff``: fences, invalid LaTeX escapes, mechanical backoff),
* building the Verifier's input so it **cannot** see the ground truth,
* scoring with the repo's own ``compute_rewards`` (error units, anti-spam,
  anti-duplicate against previous rounds, graded format penalties),
* aggregating a round into the learning signal the update skills consume.

Round layout (default root ``data/skill_rollouts``)::

    round_01/
      manifest.json            round config: k, skills used, freeze, episodes
      episodes/<episode_id>/
        problem.json           {problem, solution, k, topic, level}
        perturb.json           raw Perturber output   (written by perturb-vN)
        perturb_status.json    format validity + reason + perturbed solution
        perturb_parsed.json    validated MathPerturberOutput (ground truth)
        verify_input.json      {problem, solution_to_review}  ← Verifier sees ONLY this
        verify.json            raw Verifier output    (written by verify-vN)
        score.json             RewardOutput + match details
      round_summary.json       aggregates + learning signal + deltas

Usage::

    python scripts/skill_selfplay.py init --round 1 --episodes 8 --k 3
    # agent runs perturb-vN on every episodes/*/problem.json → perturb.json
    python scripts/skill_selfplay.py prepare-verify --round 1
    # agent runs verify-vN on every episodes/*/verify_input.json → verify.json
    python scripts/skill_selfplay.py score --round 1
    python scripts/skill_selfplay.py summarize --round 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_ROOT = REPO_ROOT / "data" / "skill_rollouts"
REWARD_CONFIG = REPO_ROOT / "configs" / "reward" / "reward.yaml"

logger = logging.getLogger("skill_selfplay")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_dir(root: Path, n: int) -> Path:
    return root / f"round_{n:02d}"


def load_reward_config(path: Path = REWARD_CONFIG) -> dict:
    """Load the ``reward:`` block from reward.yaml (falls back to defaults)."""
    try:
        import yaml

        with open(path) as fh:
            return yaml.safe_load(fh)["reward"]
    except Exception as e:  # pragma: no cover - config is in-repo
        logger.warning("Could not load %s (%s) — using built-in defaults.", path, e)
        from arappav.reward.reward_fns import _default_config

        return _default_config()


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def episode_dirs(rdir: Path) -> list[Path]:
    eps = rdir / "episodes"
    if not eps.is_dir():
        return []
    return sorted(p for p in eps.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Problem pools
# ---------------------------------------------------------------------------


def _pool_from_local_rollouts() -> list[dict]:
    """Harvest (problem, solution) pairs from previously logged RL rollouts.

    Works offline — the Hendrycks MATH originals are stored verbatim in the
    perturber rollout logs under ``original_text`` / ``original_solution``.
    """
    pool: dict[str, dict] = {}
    for pattern in ("data/rollouts_math/*.jsonl", "new_rollouts/*.jsonl", "data/rollouts/*.jsonl"):
        for path in sorted(REPO_ROOT.glob(pattern)):
            if "verifier" in path.name:
                continue
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    problem = rec.get("original_text")
                    solution = rec.get("original_solution")
                    pid = rec.get("paper_id") or rec.get("chunk_id")
                    if not (problem and solution and pid) or pid in pool:
                        continue
                    topic, _, level = str(pid).partition("_")
                    pool[pid] = {
                        "source_id": pid,
                        "problem": problem,
                        "solution": solution,
                        "topic": topic,
                        "level": level.rsplit("_", 1)[0] if level else None,
                    }
    return list(pool.values())


def _pool_from_hendrycks(topics: list[str], per_topic: int, seed: int) -> list[dict]:
    from arappav.data.ingest_math import load_math_dataset

    ds = load_math_dataset(topics=topics, split="train", max_examples_per_topic=per_topic, seed=seed)
    return [
        {
            "source_id": f"{row['topic']}_{row['level']}_{i}",
            "problem": row["problem"],
            "solution": row["solution"],
            "topic": row["topic"],
            "level": row["level"],
        }
        for i, row in enumerate(ds)
    ]


def _pool_from_file(path: Path) -> list[dict]:
    pool = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pool.append(
                {
                    "source_id": rec.get("source_id") or rec.get("paper_id") or f"item_{i}",
                    "problem": rec["problem"],
                    "solution": rec["solution"],
                    "topic": rec.get("topic"),
                    "level": rec.get("level"),
                }
            )
    return pool


def _seen_source_ids(root: Path) -> set[str]:
    seen: set[str] = set()
    if not root.is_dir():
        return seen
    for rdir in sorted(root.glob("round_*")):
        for edir in episode_dirs(rdir):
            problem = edir / "problem.json"
            if problem.exists():
                seen.add(read_json(problem).get("source_id", ""))
    return seen


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    root = Path(args.root)
    rdir = round_dir(root, args.round)
    if rdir.exists() and not args.force:
        print(f"[init] {rdir} already exists — pass --force to overwrite its manifest.")
        return 1

    if args.problems_file:
        pool = _pool_from_file(Path(args.problems_file))
    elif args.source == "hendrycks":
        pool = _pool_from_hendrycks(args.topics, args.per_topic, args.seed)
    else:
        pool = _pool_from_local_rollouts()

    if not pool:
        print("[init] Empty problem pool — check --source / --problems-file.")
        return 1

    if not args.allow_repeats:
        seen = _seen_source_ids(root)
        fresh = [p for p in pool if p["source_id"] not in seen]
        if len(fresh) >= args.episodes:
            pool = fresh
        else:
            print(
                f"[init] Only {len(fresh)} unseen problems for {args.episodes} episodes — "
                "reusing some problems from earlier rounds."
            )

    rng = random.Random(args.seed + args.round)
    chosen = rng.sample(pool, min(args.episodes, len(pool)))

    episode_ids = []
    for i, item in enumerate(chosen):
        k = args.k if args.k else rng.randint(args.k_range[0], args.k_range[1])
        eid = f"ep{i:02d}"
        episode_ids.append(eid)
        write_json(
            rdir / "episodes" / eid / "problem.json",
            {
                "episode_id": eid,
                "round": args.round,
                "source_id": item["source_id"],
                "topic": item.get("topic"),
                "level": item.get("level"),
                "k": k,
                "problem": item["problem"],
                "solution": item["solution"],
            },
        )

    manifest = {
        "round": args.round,
        "mode": "math",
        "freeze": None if args.freeze == "none" else args.freeze,
        "perturb_skill": args.perturb_skill,
        "verify_skill": args.verify_skill,
        "k_fixed": args.k,
        "k_range": args.k_range,
        "seed": args.seed,
        "source": "file" if args.problems_file else args.source,
        "episodes": episode_ids,
    }
    write_json(rdir / "manifest.json", manifest)

    print(f"[init] round {args.round}: {len(episode_ids)} episodes → {rdir}")
    print(f"[init] skills: perturber={args.perturb_skill}  verifier={args.verify_skill}  freeze={manifest['freeze']}")
    print(f"[init] next: run {args.perturb_skill} on each episodes/*/problem.json → write perturb.json")
    return 0


# ---------------------------------------------------------------------------
# prepare-verify
# ---------------------------------------------------------------------------


def cmd_prepare_verify(args) -> int:
    from arappav.models.perturber import parse_and_backoff

    rdir = round_dir(Path(args.root), args.round)
    dirs = episode_dirs(rdir)
    if not dirs:
        print(f"[prepare-verify] No episodes in {rdir} — run init first.")
        return 1

    n_valid = n_invalid = n_missing = 0
    for edir in dirs:
        problem = read_json(edir / "problem.json")
        praw_path = edir / "perturb.json"
        if not praw_path.exists():
            n_missing += 1
            print(f"[prepare-verify] {edir.name}: MISSING perturb.json")
            continue

        raw = praw_path.read_text()
        parsed, err, stage = parse_and_backoff(
            raw, problem["k"], mode="math", original_text=problem["solution"],
        )

        if parsed is None:
            n_invalid += 1
            write_json(
                edir / "perturb_status.json",
                {
                    "episode_id": problem["episode_id"],
                    "format_valid": False,
                    "failure_stage": stage,
                    "reason": err,
                    "k": problem["k"],
                },
            )
            # No verifier input: a format-invalid perturbation has no text to review.
            (edir / "verify_input.json").unlink(missing_ok=True)
            (rdir / "verify_inbox" / f"{problem['episode_id']}.json").unlink(missing_ok=True)
            print(f"[prepare-verify] {edir.name}: INVALID ({stage}) — {str(err)[:110]}")
            continue

        n_valid += 1
        perturbed = parsed.perturbed_solution
        backoff_used = perturbed != json.loads(_first_json(raw)).get("perturbed_solution", perturbed)
        write_json(
            edir / "perturb_status.json",
            {
                "episode_id": problem["episode_id"],
                "format_valid": True,
                "failure_stage": None,
                "reason": None,
                "k": problem["k"],
                "num_errors": len(parsed.errors),
                "mechanical_backoff_used": backoff_used,
                "perturbed_solution": perturbed,
            },
        )
        write_json(edir / "perturb_parsed.json", parsed.model_dump(mode="json"))
        # The Verifier's whole world: the problem and the text to review.
        verify_input = {
            "episode_id": problem["episode_id"],
            "problem": problem["problem"],
            "solution_to_review": perturbed,
        }
        write_json(edir / "verify_input.json", verify_input)
        # Same payload in a flat directory that contains nothing else, so the
        # Verifier can be run against `verify_inbox/` alone and never has to
        # open an episode directory that also holds the ground truth.
        write_json(rdir / "verify_inbox" / f"{problem['episode_id']}.json", verify_input)
        print(f"[prepare-verify] {edir.name}: valid, {len(parsed.errors)} errors")

    print(
        f"[prepare-verify] round {args.round}: {n_valid} valid, {n_invalid} invalid, "
        f"{n_missing} missing ({n_valid}/{len(dirs)} format-valid)"
    )
    print(f"[prepare-verify] verifier inbox: {rdir / 'verify_inbox'}  →  answers in {rdir / 'verify_outbox'}")
    print("[prepare-verify] IMPORTANT: run the verifier with no access to the ground truth —")
    print("[prepare-verify]   it needs verify_inbox/ only, never episodes/, perturb*.json or score.json.")
    return 0


def _first_json(raw: str) -> str:
    """Best-effort extraction of the model's own JSON (for backoff detection)."""
    from arappav.utils.parsing import extract_first_json_object, strip_json_fences

    data, _ = extract_first_json_object(strip_json_fences(raw))
    return json.dumps(data if data is not None else {})


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _historical_errors(root: Path, up_to_round: int):
    """Ground-truth errors from all previous rounds (anti-duplicate history)."""
    from arappav.errors.schema_math import MathInjectedError

    history = []
    for n in range(1, up_to_round):
        for edir in episode_dirs(round_dir(root, n)):
            parsed = edir / "perturb_parsed.json"
            if not parsed.exists():
                continue
            for err in read_json(parsed).get("errors", []):
                try:
                    history.append(MathInjectedError.model_validate(err))
                except Exception:
                    continue
    return history


def cmd_score(args) -> int:
    from arappav.errors.schema_math import (
        MathPerturberOutput,
        validate_math_verifier_output,
    )
    from arappav.reward.reward_fns import compute_rewards
    from arappav.utils.parsing import extract_first_json_object, strip_json_fences

    root = Path(args.root)
    rdir = round_dir(root, args.round)
    cfg = load_reward_config()
    history = [] if args.no_history else _historical_errors(root, args.round)

    dirs = episode_dirs(rdir)
    if not dirs:
        print(f"[score] No episodes in {rdir}.")
        return 1

    for edir in dirs:
        problem = read_json(edir / "problem.json")
        k = problem["k"]
        status_path = edir / "perturb_status.json"
        if not status_path.exists():
            print(f"[score] {edir.name}: no perturb_status.json — run prepare-verify first.")
            continue
        status = read_json(status_path)

        if not status["format_valid"]:
            # Graded format penalty, mirroring grpo_trainer.make_perturber_reward_fn.
            penalty = (
                cfg.get("format_penalty", -10.0)
                if status.get("failure_stage") == "json"
                else cfg.get("format_penalty_soft", -5.0)
            )
            write_json(
                edir / "score.json",
                {
                    "episode_id": problem["episode_id"],
                    "k": k,
                    "perturber_format_valid": False,
                    "failure_stage": status.get("failure_stage"),
                    "format_violation_reason": status.get("reason"),
                    "perturber_reward": penalty,
                    "verifier_reward": None,
                    "verifier_recall": None,
                    "verifier_precision": None,
                    "verifier_f_beta": None,
                    "scored": False,
                },
            )
            print(f"[score] {edir.name}: format-invalid → r_P={penalty}")
            continue

        outbox_path = rdir / "verify_outbox" / f"{problem['episode_id']}.json"
        vraw_path = edir / "verify.json"
        if outbox_path.exists():
            vraw = outbox_path.read_text()
            vraw_path.write_text(vraw)  # keep the episode record self-contained
        elif vraw_path.exists():
            vraw = vraw_path.read_text()
        else:
            print(f"[score] {edir.name}: MISSING verifier output (verify_outbox/ or verify.json) — skipped.")
            continue

        vdata, verr = extract_first_json_object(strip_json_fences(vraw))
        if vdata is None:
            vout, verr = None, verr or "no JSON object found"
        else:
            vout, verr = validate_math_verifier_output(vdata)

        claims = vout.claims if vout is not None else []
        perturber_out = MathPerturberOutput.model_validate(read_json(edir / "perturb_parsed.json"))

        reward = compute_rewards(
            ground_truth=perturber_out.errors,
            verifier_claims=claims,
            perturbed_text=perturber_out.perturbed_solution,
            k=k,
            config=cfg,
            perturber_format_valid=True,
            historical_perturbations=history or None,
            verifier_raw_output=vraw,
        )

        record = dataclasses.asdict(reward)
        record.update(
            {
                "episode_id": problem["episode_id"],
                "source_id": problem.get("source_id"),
                "topic": problem.get("topic"),
                "level": problem.get("level"),
                "perturber_format_valid": True,
                "verifier_parse_error": verr,
                "scored": True,
                "errors": [
                    {
                        "error_id": e.error_id,
                        "error_type": e.error_type.value,
                        "original_text": e.original_text,
                        "injected_text": e.injected_text,
                        "rationale": e.rationale,
                    }
                    for e in perturber_out.errors
                ],
                "claims": [
                    {
                        "step_index": c.step_index,
                        "quoted_text": c.quoted_text,
                        "explanation": c.explanation,
                        "error_type": c.error_type.value if c.error_type else None,
                    }
                    for c in claims
                ],
            }
        )
        write_json(edir / "score.json", record)
        print(
            f"[score] {edir.name}: r_P={reward.perturber_reward:+.3f} r_V={reward.verifier_reward:+.3f} "
            f"recall={reward.verifier_recall:.2f} prec={reward.verifier_precision:.2f} "
            f"units={reward.num_matched_units}/{reward.num_error_units} keff={reward.k_effective}/{k}"
        )

    print(f"[score] next: python scripts/skill_selfplay.py summarize --round {args.round}")
    return 0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def cmd_summarize(args) -> int:
    root = Path(args.root)
    rdir = round_dir(root, args.round)
    manifest = read_json(rdir / "manifest.json") if (rdir / "manifest.json").exists() else {}

    scores, format_failures = [], []
    for edir in episode_dirs(rdir):
        spath = edir / "score.json"
        if spath.exists():
            rec = read_json(spath)
            scores.append(rec)
            if not rec.get("perturber_format_valid"):
                format_failures.append(
                    {
                        "episode_id": rec["episode_id"],
                        "stage": rec.get("failure_stage"),
                        "reason": (rec.get("format_violation_reason") or "")[:400],
                    }
                )

    scored = [s for s in scores if s.get("scored")]
    n_total = len(scores)

    undetected, detected, false_positives = [], [], []
    type_stats: dict[str, dict] = {}
    for s in scored:
        matched_ids = {
            d["error_id"] for d in s.get("match_details", []) if d.get("best_claim_idx") is not None
        }
        by_id = {e["error_id"]: e for e in s.get("errors", [])}
        for detail in s.get("match_details", []):
            err = by_id.get(detail["error_id"], {})
            etype = detail.get("error_type", "unknown")
            stat = type_stats.setdefault(etype, {"injected": 0, "detected": 0})
            stat["injected"] += 1
            entry = {
                "episode_id": s["episode_id"],
                "error_id": detail["error_id"],
                "error_type": etype,
                "original_text": err.get("original_text", "")[:200],
                "injected_text": err.get("injected_text", "")[:200],
                "best_overlap": detail.get("best_overlap"),
            }
            if detail["error_id"] in matched_ids:
                stat["detected"] += 1
                detected.append(entry)
            else:
                entry["closest_claims"] = detail.get("all_overlaps", {})
                undetected.append(entry)

        tp_flags = {d["best_claim_idx"] for d in s.get("match_details", []) if d.get("best_claim_idx") is not None}
        for i, claim in enumerate(s.get("claims", [])):
            if i not in tp_flags:
                false_positives.append(
                    {
                        "episode_id": s["episode_id"],
                        "quoted_text": claim["quoted_text"][:200],
                        "explanation": claim["explanation"][:300],
                    }
                )

    summary = {
        "round": args.round,
        "perturb_skill": manifest.get("perturb_skill"),
        "verify_skill": manifest.get("verify_skill"),
        "freeze": manifest.get("freeze"),
        "num_episodes": n_total,
        "num_format_valid": len(scored),
        "format_valid_rate": round(len(scored) / n_total, 4) if n_total else None,
        "metrics": {
            "mean_perturber_reward": _mean([s["perturber_reward"] for s in scores if s.get("perturber_reward") is not None]),
            "mean_perturber_reward_valid_only": _mean([s["perturber_reward"] for s in scored]),
            "mean_verifier_reward": _mean([s["verifier_reward"] for s in scored]),
            "mean_verifier_recall": _mean([s["verifier_recall"] for s in scored]),
            "mean_verifier_precision": _mean([s["verifier_precision"] for s in scored]),
            "mean_verifier_f1": _mean([s["verifier_f_beta"] for s in scored]),
            "mean_k_effective_ratio": _mean(
                [s["k_effective"] / max(1, s["k"]) for s in scored]
            ),
            "mean_units_per_episode": _mean([float(s["num_error_units"]) for s in scored]),
            "total_spam_penalty": round(sum(s.get("spam_penalty", 0.0) for s in scored), 4),
            "total_duplicate_penalty": round(sum(s.get("duplicate_penalty", 0.0) for s in scored), 4),
            "total_repetition_penalty": round(sum(s.get("repetition_penalty", 0.0) for s in scored), 4),
        },
        "error_type_detection": {
            t: {**v, "detection_rate": round(v["detected"] / v["injected"], 3)}
            for t, v in sorted(type_stats.items())
        },
        "learning_signal": {
            "format_failures": format_failures,
            "undetected_errors": undetected,
            "detected_errors": detected,
            "verifier_false_positives": false_positives,
        },
    }

    prev = rdir.parent / f"round_{args.round - 1:02d}" / "round_summary.json"
    if prev.exists():
        pm = read_json(prev)
        deltas = {"format_valid_rate": _delta(summary["format_valid_rate"], pm.get("format_valid_rate"))}
        for key, value in summary["metrics"].items():
            deltas[key] = _delta(value, pm.get("metrics", {}).get(key))
        summary["deltas_vs_previous_round"] = deltas

    write_json(rdir / "round_summary.json", summary)

    m = summary["metrics"]
    print(f"\n=== Round {args.round} summary ({summary['perturb_skill']} vs {summary['verify_skill']}) ===")
    print(f"  format-valid      : {summary['num_format_valid']}/{summary['num_episodes']} ({summary['format_valid_rate']})")
    print(f"  mean r_P          : {m['mean_perturber_reward']}  (valid only: {m['mean_perturber_reward_valid_only']})")
    print(f"  mean r_V          : {m['mean_verifier_reward']}")
    print(f"  recall / precision: {m['mean_verifier_recall']} / {m['mean_verifier_precision']}")
    print(f"  k_effective ratio : {m['mean_k_effective_ratio']}")
    print(f"  undetected errors : {len(undetected)}   false positives: {len(false_positives)}")
    if "deltas_vs_previous_round" in summary:
        d = summary["deltas_vs_previous_round"]
        print(f"  Δ recall={d.get('mean_verifier_recall')}  Δ r_P={d.get('mean_perturber_reward_valid_only')}  "
              f"Δ format-valid={d.get('format_valid_rate')}")
    print(f"  → {rdir / 'round_summary.json'}\n")
    return 0


def _delta(current, previous):
    if current is None or previous is None:
        return None
    return round(current - previous, 4)


# ---------------------------------------------------------------------------
# check-skill
# ---------------------------------------------------------------------------

SKILLS_DIR = REPO_ROOT / ".claude" / "skills"
INVARIANT_START = "## Output contract"
POLICY_START = "## Policy"
CHANGELOG_START = "## Changelog"


def _section(text: str, start: str, end: str | None) -> str:
    """Return the slice of *text* from heading *start* up to heading *end*."""
    lines = text.splitlines()
    try:
        i = next(n for n, line in enumerate(lines) if line.startswith(start))
    except StopIteration:
        return ""
    j = len(lines)
    if end is not None:
        for n in range(i + 1, len(lines)):
            if lines[n].startswith(end):
                j = n
                break
    return "\n".join(lines[i:j]).strip()


def cmd_check_skill(args) -> int:
    """Verify a newly written policy skill against the version it derives from.

    The invariant region (contract + scoring + taxonomy) must be byte-identical,
    the tuned policy section must actually differ, and the changelog must grow.
    """
    old_path = SKILLS_DIR / f"{args.role}-v{args.from_version}" / "SKILL.md"
    new_path = SKILLS_DIR / f"{args.role}-v{args.to_version}" / "SKILL.md"

    problems: list[str] = []
    for path in (old_path, new_path):
        if not path.exists():
            print(f"[check-skill] missing {path}")
            return 1

    old, new = old_path.read_text(), new_path.read_text()

    name = re.search(r"^name:\s*(\S+)\s*$", new, re.MULTILINE)
    expected = f"{args.role}-v{args.to_version}"
    if not name:
        problems.append("new skill has no `name:` in its frontmatter")
    elif name.group(1) != expected:
        problems.append(f"frontmatter name {name.group(1)!r} != directory name {expected!r}")

    old_inv = _section(old, INVARIANT_START, POLICY_START)
    new_inv = _section(new, INVARIANT_START, POLICY_START)
    if not old_inv or not new_inv:
        problems.append(f"could not locate the invariant region ({INVARIANT_START} … {POLICY_START})")
    elif old_inv != new_inv:
        import difflib

        diff = list(difflib.unified_diff(old_inv.splitlines(), new_inv.splitlines(), "old", "new", lineterm="", n=1))
        problems.append("invariant region was modified:\n  " + "\n  ".join(diff[:40]))

    old_pol = _section(old, POLICY_START, CHANGELOG_START)
    new_pol = _section(new, POLICY_START, CHANGELOG_START)
    if not new_pol:
        problems.append("new skill has no policy section")
    elif old_pol == new_pol:
        problems.append("policy section is unchanged — the update produced no learning")

    old_log = _section(old, CHANGELOG_START, None)
    new_log = _section(new, CHANGELOG_START, None)
    if len(new_log.splitlines()) <= len(old_log.splitlines()):
        problems.append("changelog did not grow — record what changed and why")

    if problems:
        print(f"[check-skill] {expected}: {len(problems)} problem(s)")
        for prob in problems:
            print(f"  - {prob}")
        return 1

    print(
        f"[check-skill] {expected}: OK — invariants preserved, policy changed "
        f"({len(old_pol.splitlines())} → {len(new_pol.splitlines())} lines), changelog updated."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Rollout root directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Sample problems and scaffold a round.")
    p_init.add_argument("--round", type=int, required=True)
    p_init.add_argument("--episodes", type=int, default=8)
    p_init.add_argument("--k", type=int, default=None, help="Fixed k; omit to sample from --k-range.")
    p_init.add_argument("--k-range", type=int, nargs=2, default=[2, 3])
    p_init.add_argument("--source", choices=["local", "hendrycks"], default="local")
    p_init.add_argument("--problems-file", default=None, help="JSONL with problem/solution fields.")
    p_init.add_argument("--topics", nargs="+", default=["algebra"])
    p_init.add_argument("--per-topic", type=int, default=50)
    p_init.add_argument("--perturb-skill", default="perturb-v1")
    p_init.add_argument("--verify-skill", default="verify-v1")
    p_init.add_argument("--freeze", choices=["none", "perturber", "verifier"], default="none")
    p_init.add_argument("--seed", type=int, default=42)
    p_init.add_argument("--allow-repeats", action="store_true", help="Allow problems seen in earlier rounds.")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_prep = sub.add_parser("prepare-verify", help="Validate perturbations and build leak-free verifier inputs.")
    p_prep.add_argument("--round", type=int, required=True)
    p_prep.set_defaults(func=cmd_prepare_verify)

    p_score = sub.add_parser("score", help="Score episodes with the repo's reward functions.")
    p_score.add_argument("--round", type=int, required=True)
    p_score.add_argument("--no-history", action="store_true", help="Disable cross-round anti-duplicate history.")
    p_score.set_defaults(func=cmd_score)

    p_sum = sub.add_parser("summarize", help="Aggregate a round into the update skills' learning signal.")
    p_sum.add_argument("--round", type=int, required=True)
    p_sum.set_defaults(func=cmd_summarize)

    p_check = sub.add_parser("check-skill", help="Verify a new policy skill version against its parent.")
    p_check.add_argument("--role", choices=["perturb", "verify"], required=True)
    p_check.add_argument("--from-version", type=int, required=True, dest="from_version")
    p_check.add_argument("--to-version", type=int, required=True, dest="to_version")
    p_check.set_defaults(func=cmd_check_skill)

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
