#!/usr/bin/env python
"""MATH-500 validation set — select the best verifier policy before testing it.

Why this exists
---------------
Self-play reward is not a usable selection signal. Measured over the ten-round
Haiku run, self-play verifier F1 correlates with held-out ProcessBench F1 at
r = +0.275 (n = 10, not significant; 56% pairwise ranking agreement against 50%
for a coin flip). Picking the policy with the best self-play reward returns one
that is *worse on held-out data than the untuned cold start*. Two confounds
cause it: every self-play round draws fresh problems, and the perturber
co-evolves, so the score mixes policy quality with problem difficulty and
opponent weakness.

This script removes both. The validation set is built **once** from
`HuggingFaceH4/MATH-500` by a **pinned** perturber, then cached. Every verifier
version is scored on byte-identical perturbed solutions, so differences are
attributable to the verifier alone.

Its role in the protocol
------------------------
MATH-500 is the **validation** set: it chooses the round. ProcessBench is the
**test** set: it reports the number, and is run on the selected policy only.
Selecting and reporting on the same benchmark would make the reported figure
meaningless, which is exactly the trap the correlation analysis above exposed.

    prepare  ->  run (per version)  ->  select  ->  full ProcessBench on winner

`scripts/eval_policies_processbench.py` still scores every round on an 80-item
ProcessBench sample. That is an **analysis** artefact for understanding the
trajectory, not the reported result, and it must not be used to pick a policy.

Metrics
-------
Perturbed episodes carry the selection metric: mean verifier F1 from the repo's
own `compute_rewards`, the same quantity self-play optimises. Clean episodes
(`--clean-frac`) are scored separately as a false-alarm rate, because
`compute_rewards` cannot distinguish correct silence from a false alarm when
there is no ground truth. The false-alarm rate does not enter the selection
score; it is reported as a guard against picking a policy that simply claims
more.

Usage
-----
    python scripts/validate_math500.py prepare --perturber hperturb-v1 --n 40
    python scripts/validate_math500.py run --prefix hverify --versions 1 2 3
    python scripts/validate_math500.py select --prefix hverify
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arappav.pipeline import scoring  # noqa: E402
from arappav.pipeline.agents import InfrastructureError  # noqa: E402
from arappav.pipeline.backends import make_backend  # noqa: E402
from arappav.pipeline.contracts import (  # noqa: E402
    Episode, render_perturb_prompt, render_verify_prompt,
)

DEFAULT_ROOT = REPO / "data" / "validation_math500"


def _w(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# prepare — build the fixed validation set once
# ---------------------------------------------------------------------------


def cmd_prepare(args) -> int:
    from datasets import load_dataset

    root = Path(args.root)
    if (root / "manifest.json").exists() and not args.force:
        print(f"[prepare] {root} already built — pass --force to rebuild.\n"
              f"          Rebuilding changes the validation set and invalidates "
              f"comparisons against versions already scored on it.")
        return 1

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    import random
    from arappav.data.categories import filter_math500, normalise
    pool = filter_math500(ds, args.category)
    if args.category and normalise(args.category) != "all":
        print(f"[prepare] restricted to {normalise(args.category)!r}: "
              f"{len(pool)} of {len(ds)} MATH-500 items")
    rng = random.Random(args.seed)
    if args.stratify:
        # Difficulty is the dimension that matters here: an unstratified draw
        # from MATH-500 can land mostly on Level 1-2 and make every policy look
        # alike. Proportional allocation by level keeps the mix fixed across
        # every version scored on this set.
        from collections import defaultdict
        by_level = defaultdict(list)
        for i in pool:
            by_level[ds[i]["level"]].append(i)
        idx, order = [], sorted(by_level)
        for lvl in order:
            share = max(1, round(args.n * len(by_level[lvl]) / len(pool)))
            idx += rng.sample(by_level[lvl], min(share, len(by_level[lvl])))
        rng.shuffle(idx)
        idx = idx[:args.n]
        counts = {lvl: sum(1 for i in idx if ds[i]["level"] == lvl) for lvl in order}
        print(f"[prepare] stratified by level: {counts}")
    else:
        idx = rng.sample(pool, min(args.n, len(pool)))
    idx.sort()
    n_clean = int(round(len(idx) * args.clean_frac))
    clean_ids = set(rng.sample(idx, n_clean))

    # The validation perturber must be independent of the policies under test.
    # Building it with the run's own evolved perturber would make validation a
    # restatement of the self-play score, which is precisely the signal we
    # already know does not track held-out ability.
    pmodel = args.perturber_model or args.model
    backend = make_backend(args.backend, model=pmodel, timeout=args.timeout,
                           skills_root=Path(args.skills_root),
                           **({"provider": args.provider} if args.backend != "claude-code" else {}))
    print(f"[prepare] {len(idx)} items ({n_clean} left clean), perturber="
          f"{args.perturber} on {pmodel} (independent of the policies under test)")

    lock = threading.Lock()
    items, failures = [], []

    def build(i: int):
        row = ds[i]
        eid = row["unique_id"].replace("/", "_").replace(".json", "")
        ep = Episode(episode_id=eid, problem=row["problem"], solution=row["solution"],
                     k=args.k, source_id=row["unique_id"],
                     topic=row.get("subject"), level=str(row.get("level")))
        rec = {"episode_id": eid, "source_id": row["unique_id"],
               "subject": row.get("subject"), "level": row.get("level"),
               "problem": row["problem"], "original_solution": row["solution"]}

        if i in clean_ids:
            # A correct solution, kept verbatim. Scored only for false alarms.
            rec.update(clean=True, k=0, solution_to_review=row["solution"], errors=[])
            with lock:
                items.append(rec)
            return

        body = render_perturb_prompt(ep, "", "").lstrip()
        res = backend.run(skill=args.perturber, user=body, step="perturb",
                          round_dir=root, episode_id=eid)
        if res.infra_failure():
            raise InfrastructureError(f"{eid}: {res.infra_failure()}")
        parsed, err, stage = scoring.parse_perturbation(res.text, args.k, row["solution"])
        if parsed is None:
            with lock:
                failures.append({"episode_id": eid, "stage": stage, "reason": str(err)[:200]})
            return
        rec.update(clean=False, k=args.k,
                   solution_to_review=parsed.perturbed_solution,
                   errors=[e.model_dump(mode="json") for e in parsed.errors])
        with lock:
            items.append(rec)

    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(build, idx))

    items.sort(key=lambda r: r["episode_id"])
    for rec in items:
        # The verifier's view: problem + text to review, and nothing else.
        _w(root / "inbox" / f"{rec['episode_id']}.json",
           {"episode_id": rec["episode_id"], "problem": rec["problem"],
            "solution_to_review": rec["solution_to_review"]})
        _w(root / "answers" / f"{rec['episode_id']}.json", rec)

    _w(root / "manifest.json", {
        "dataset": "HuggingFaceH4/MATH-500", "n_requested": args.n,
        "n_built": len(items), "n_clean": sum(1 for r in items if r["clean"]),
        "k": args.k, "clean_frac": args.clean_frac, "seed": args.seed,
        "perturber": args.perturber, "perturber_model": args.model,
        "failures": failures,
        "items": [{"id": r["episode_id"], "clean": r["clean"]} for r in items],
    })
    print(f"[prepare] built {len(items)} items "
          f"({sum(1 for r in items if r['clean'])} clean, {len(failures)} perturbation "
          f"failures) → {root}")
    return 0


# ---------------------------------------------------------------------------
# run — score verifier versions on the fixed set
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    root = Path(args.root)
    if not (root / "manifest.json").exists():
        sys.exit(f"[run] no validation set at {root} — run `prepare` first.")
    manifest = json.loads((root / "manifest.json").read_text())
    reward_cfg = scoring.load_reward_config()

    backend = make_backend(args.backend, model=args.model, timeout=args.timeout,
                           skills_root=Path(args.skills_root),
                           **({"provider": args.provider} if args.backend != "claude-code" else {}))

    for v in args.versions:
        skill = f"{args.prefix}-v{v}"
        if not (Path(args.skills_root) / skill / "SKILL.md").exists():
            print(f"[run] {skill}: no such policy, skipping")
            continue
        out = root / "outbox" / skill
        out.mkdir(parents=True, exist_ok=True)
        todo = [f for f in sorted((root / "inbox").glob("*.json"))
                if not (out / f.name).exists()]
        print(f"[run] {skill}: {len(todo)} to score "
              f"({len(manifest['items']) - len(todo)} cached)")

        abort: list[str] = []
        lock = threading.Lock()

        def one(f: Path):
            if abort:
                return
            item = json.loads(f.read_text())
            ep = Episode(episode_id=f.stem, problem=item["problem"], solution="", k=0)
            body = render_verify_prompt(ep, item["solution_to_review"], "", "").lstrip()
            res = backend.run(skill=skill, user=body, step=f"validate__{skill}",
                              round_dir=root, episode_id=f.stem)
            if res.infra_failure():
                with lock:
                    abort.append(f"{skill}/{f.stem}: {res.infra_failure()}")
                return
            (out / f.name).write_text(res.text)

        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(one, todo))
        if abort:
            raise InfrastructureError(
                f"{abort[0]} (+{len(abort)-1} more). Nothing scored for those; "
                f"re-run to resume — cached answers are kept.")

        _score_version(root, skill, reward_cfg)
    return 0


def _score_version(root: Path, skill: str, reward_cfg: dict) -> dict:
    from arappav.errors.schema_math import MathPerturberOutput

    f1s, fa_hits, fa_total, missing = [], 0, 0, 0
    for ans_path in sorted((root / "answers").glob("*.json")):
        ans = json.loads(ans_path.read_text())
        out_path = root / "outbox" / skill / ans_path.name
        if not out_path.exists():
            missing += 1
            continue
        raw = out_path.read_text()
        if ans["clean"]:
            # No ground truth: compute_rewards cannot tell silence from a false
            # alarm here, so score it directly as a false-alarm indicator.
            vout, _ = scoring.parse_verification(raw)
            fa_total += 1
            fa_hits += 1 if (vout and vout.claims) else 0
            continue
        po = MathPerturberOutput.model_validate(
            {"perturbed_solution": ans["solution_to_review"], "errors": ans["errors"]})
        s = scoring.score_episode(
            episode_id=ans["episode_id"], k=ans["k"], perturbed=po,
            failure_stage=None, failure_reason=None, verifier_raw=raw,
            config=reward_cfg, history=None)
        if s.get("verifier_reward") is not None:
            f1s.append(s["verifier_reward"])

    res = {
        "skill": skill,
        "n_perturbed": len(f1s),
        "mean_verifier_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "n_clean": fa_total,
        "false_alarm_rate": round(fa_hits / fa_total, 4) if fa_total else None,
        "missing_outputs": missing,
    }
    _w(root / "scores" / f"{skill}.json", res)
    print(f"  {skill:24s} F1={res['mean_verifier_f1']}  "
          f"false_alarm={res['false_alarm_rate']}  (n={res['n_perturbed']})")
    return res


# ---------------------------------------------------------------------------
# select — pick the winner
# ---------------------------------------------------------------------------


def cmd_select(args) -> int:
    root = Path(args.root)
    scores = []
    for f in sorted((root / "scores").glob(f"{args.prefix}-v*.json"),
                    key=lambda p: int(p.stem.rsplit("-v", 1)[1])):
        d = json.loads(f.read_text())
        if d.get("mean_verifier_f1") is not None:
            scores.append(d)
    if not scores:
        sys.exit(f"[select] no scored versions under {root/'scores'} — run `run` first.")

    print(f"{'policy':26s}{'F1':>9}{'false alarm':>14}")
    for d in scores:
        print(f"{d['skill']:26s}{d['mean_verifier_f1']:>9.4f}"
              f"{(d['false_alarm_rate'] if d['false_alarm_rate'] is not None else float('nan')):>14.4f}")

    best = max(scores, key=lambda d: d["mean_verifier_f1"])
    spread = best["mean_verifier_f1"] - min(d["mean_verifier_f1"] for d in scores)
    print(f"\nselected: {best['skill']}  (validation F1 {best['mean_verifier_f1']:.4f})")
    if best.get("false_alarm_rate") is not None and best["false_alarm_rate"] > 0.3:
        print(f"  warning: it also has a high false-alarm rate "
              f"({best['false_alarm_rate']:.2f}) — it may simply claim more.")
    if spread < 0.02:
        print(f"  warning: the field spans only {spread:.3f} F1; this selection is "
              f"close to arbitrary. Report it as such.")
    _w(root / "selected.json", {"selected": best["skill"], "by": "mean_verifier_f1",
                                "validation": best, "candidates": scores})
    print(f"→ {root/'selected.json'}")
    if args.print_only:
        print(best["skill"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--skills-root", default=".claude/skills", dest="skills_root")
    p.add_argument("--backend", choices=["api", "claude-code", "batch"], default="api")
    p.add_argument("--provider", default=None,
                   choices=["anthropic", "openai", "deepseek"])
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--category", default=None,
                   help="restrict MATH-500 to one subject; must match the "
                        "category the policies were trained on")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--timeout", type=int, default=600)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare", help="build the fixed validation set (once)")
    a.add_argument("--n", type=int, default=40, help="MATH-500 items to draw")
    a.add_argument("--k", type=int, default=3, help="errors injected per perturbed item")
    a.add_argument("--clean-frac", type=float, default=0.25, dest="clean_frac",
                   help="fraction left unperturbed, for the false-alarm guard")
    a.add_argument("--perturber", default="hperturb-v1",
                   help="PINNED perturber policy; keep it fixed for the life of "
                        "the experiment or versions stop being comparable")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--stratify", action="store_true",
                   help="allocate the sample across MATH-500 difficulty levels in "
                        "proportion to the pool, instead of drawing uniformly")
    a.add_argument("--perturber-model", default=None, dest="perturber_model",
                   help="model that builds the validation set; set it stronger "
                        "than the run's players so validation difficulty is not "
                        "bounded by the policies being tested")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_prepare)

    b = sub.add_parser("run", help="score verifier versions on the fixed set")
    b.add_argument("--prefix", default="hverify")
    b.add_argument("--versions", type=int, nargs="+", required=True)
    b.set_defaults(func=cmd_run)

    c = sub.add_parser("select", help="pick the best version by validation F1")
    c.add_argument("--prefix", default="hverify")
    c.add_argument("--print-only", action="store_true", dest="print_only",
                   help="also print the bare skill name, for shell capture")
    c.set_defaults(func=cmd_select)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except InfrastructureError as e:
        print(f"\n[validate] ABORTED — {e}", file=sys.stderr)
        raise SystemExit(2)
