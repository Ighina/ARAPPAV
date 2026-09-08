#!/usr/bin/env python
"""External held-out evaluation of the Verifier skill on **Qwen/ProcessBench**.

ProcessBench (Zheng et al., 2024) pairs a math problem with a model-generated
reasoning chain split into steps, human-annotated with the index of the **first
erroneous step** (``label``), or ``-1`` when the chain is correct throughout.
Four subsets of increasing difficulty: ``gsm8k`` < ``math`` < ``olympiadbench``
< ``omnimath``.

This is the *external validity* check for skill tuning: self-play rewards say a
policy beats this Perturber, ProcessBench says whether it detects errors humans
annotated in the wild.

Leakage rules — enforced by this script's file layout, not by good intentions:

* Items live in ``inbox/``; gold labels live in ``answers/``. The Verifier pass
  gets the inbox only.
* ``eval_summary.json`` carries item ids, hit/miss flags, and aggregate metrics
  — never problem, step, or solution text, and never gold labels. Per-item
  labels and predictions go to ``eval_details.json``, which is off-limits to
  everything but harness debugging.
* The evaluation sample is fixed by ``--seed`` alone (not by round), so every
  round is scored on the same held-out items and the numbers are comparable.
* The update skills must never read ``data/skill_evals/``. This is monitoring,
  not a training signal: selecting policy edits on these numbers is test-set
  fitting.

Metric — as in the official ProcessBench evaluation: accuracy on erroneous
samples and accuracy on correct samples, combined by their **harmonic mean**
(the paper's "F1"). The arithmetic mean (balanced accuracy) is reported
alongside it.

Usage::

    python scripts/processbench_eval.py prepare --round 1 --per-subset 25
    # fresh agent runs verify-vN over inbox/ → outbox/
    python scripts/processbench_eval.py score --round 1
    python scripts/processbench_eval.py report
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_ROOT = REPO_ROOT / "data" / "skill_evals"
SUBSETS = ["gsm8k", "math", "olympiadbench", "omnimath"]

logger = logging.getLogger("processbench_eval")


def eval_dir(root: Path, n: int) -> Path:
    return root / f"round_{n:02d}"


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Step rendering
# ---------------------------------------------------------------------------


def render_steps(steps: list[str]) -> str:
    """Render a reasoning chain as one text with explicit step boundaries.

    The whole chain is presented at once — the Verifier must judge each step in
    the context of the entire solution, as in the original ProcessBench setup —
    but every step is delimited so a claim can name the step it belongs to.
    """
    return "\n\n".join(
        f"<step_{i}>\n{step.strip()}\n</step_{i}>" for i, step in enumerate(steps)
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def cmd_prepare(args) -> int:
    from datasets import load_dataset

    root = Path(args.root)
    edir = eval_dir(root, args.round)
    if (edir / "inbox").exists() and not args.force:
        print(f"[prepare] {edir / 'inbox'} already exists — pass --force to rebuild.")
        return 1

    manifest_items = []
    for subset in args.subsets:
        ds = load_dataset("Qwen/ProcessBench", split=subset)
        erroneous = [i for i, label in enumerate(ds["label"]) if label != -1]
        correct = [i for i, label in enumerate(ds["label"]) if label == -1]

        # Seeded independently of the round: every round sees the same held-out
        # items, so cross-round deltas measure the policy, not the sample.
        rng = random.Random(f"{args.seed}:{subset}")
        if args.sampling == "balanced":
            half = args.per_subset // 2
            picks = rng.sample(erroneous, min(half, len(erroneous)))
            picks += rng.sample(correct, min(args.per_subset - half, len(correct)))
        else:
            picks = rng.sample(range(len(ds)), min(args.per_subset, len(ds)))
        picks.sort()

        for idx in picks:
            row = ds[idx]
            if args.max_steps and len(row["steps"]) > args.max_steps:
                continue
            item_id = row["id"]
            write_json(
                edir / "inbox" / f"{item_id}.json",
                {
                    "episode_id": item_id,
                    "subset": subset,
                    "problem": row["problem"],
                    "solution_to_review": render_steps(row["steps"]),
                    "step_format": "tagged",
                    "num_steps": len(row["steps"]),
                },
            )
            write_json(
                edir / "answers" / f"{item_id}.json",
                {
                    "episode_id": item_id,
                    "subset": subset,
                    "label": row["label"],
                    "num_steps": len(row["steps"]),
                    "final_answer_correct": row["final_answer_correct"],
                    "generator": row["generator"],
                },
            )
            manifest_items.append({"id": item_id, "subset": subset})

    write_json(
        edir / "manifest.json",
        {
            "round": args.round,
            "dataset": "Qwen/ProcessBench",
            "verify_skill": args.verify_skill,
            "subsets": args.subsets,
            "per_subset": args.per_subset,
            "sampling": args.sampling,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "num_items": len(manifest_items),
            "items": manifest_items,
        },
    )

    by_subset: dict[str, int] = {}
    for item in manifest_items:
        by_subset[item["subset"]] = by_subset.get(item["subset"], 0) + 1
    print(f"[prepare] round {args.round}: {len(manifest_items)} items → {edir / 'inbox'}")
    print(f"[prepare] per subset: {by_subset}")
    print(f"[prepare] gold labels in {edir / 'answers'} — the verifier pass must NOT read them.")
    print(f"[prepare] next: run {args.verify_skill} over inbox/ in a fresh agent → outbox/")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def resolve_prediction(claims: list[dict], steps_text: str, num_steps: int) -> tuple[int | None, int]:
    """Reduce Verifier claims to a ProcessBench prediction.

    Returns ``(prediction, num_recovered)`` where *prediction* is:

    * ``-1``   — no claims: the Verifier judged the chain correct;
    * ``int``  — the **earliest** step index any claim points at;
    * ``None`` — the Verifier claimed an error but no claim could be tied to a
      step. That is a real failure (it neither located an error nor declared the
      chain correct) and is scored wrong against both classes.

    A claim whose ``step_index`` is missing or out of range is recovered by
    locating its quoted text inside a step block; *num_recovered* counts those.
    """
    from arappav.reward.matcher import _normalize_for_matching

    if not claims:
        return -1, 0

    blocks = _split_step_blocks(steps_text, num_steps)
    indices: list[int] = []
    recovered = 0

    for claim in claims:
        idx = claim.get("step_index")
        if isinstance(idx, int) and 0 <= idx < num_steps:
            indices.append(idx)
            continue
        quoted = _normalize_for_matching(claim.get("quoted_text") or "")
        if not quoted:
            continue
        for i, block in enumerate(blocks):
            if quoted in block:
                indices.append(i)
                recovered += 1
                break

    if not indices:
        return None, recovered
    return min(indices), recovered


def _split_step_blocks(steps_text: str, num_steps: int) -> list[str]:
    """Return the normalized text of each ``<step_i>`` block."""
    import re

    from arappav.reward.matcher import _normalize_for_matching

    blocks = []
    for i in range(num_steps):
        match = re.search(rf"<step_{i}>(.*?)</step_{i}>", steps_text, re.DOTALL)
        blocks.append(_normalize_for_matching(match.group(1)) if match else "")
    return blocks


def _harmonic(a: float, b: float) -> float:
    return round(2 * a * b / (a + b), 4) if (a + b) else 0.0


def _subset_metrics(rows: list[dict]) -> dict:
    err = [r for r in rows if r["label"] != -1]
    ok = [r for r in rows if r["label"] == -1]
    err_acc = round(sum(r["match"] for r in err) / len(err), 4) if err else None
    ok_acc = round(sum(r["match"] for r in ok) / len(ok), 4) if ok else None
    metrics = {
        "n": len(rows),
        "n_erroneous": len(err),
        "n_correct": len(ok),
        "error_accuracy": err_acc,
        "correct_accuracy": ok_acc,
        "unresolved_predictions": sum(1 for r in rows if r["prediction"] is None),
        "recovered_step_indices": sum(r["recovered"] for r in rows),
    }
    if err_acc is not None and ok_acc is not None:
        # ProcessBench's headline metric is the harmonic mean of the two
        # accuracies; the arithmetic mean (balanced accuracy) is given too.
        metrics["processbench_f1"] = _harmonic(err_acc, ok_acc)
        metrics["balanced_accuracy"] = round((err_acc + ok_acc) / 2, 4)
    return metrics


def cmd_score(args) -> int:
    root = Path(args.root)
    edir = eval_dir(root, args.round)
    manifest = read_json(edir / "manifest.json")

    rows, missing = [], []
    for item in manifest["items"]:
        item_id = item["id"]
        answer = read_json(edir / "answers" / f"{item_id}.json")
        inbox = read_json(edir / "inbox" / f"{item_id}.json")
        out_path = edir / "outbox" / f"{item_id}.json"
        if not out_path.exists():
            missing.append(item_id)
            continue

        try:
            claims = json.loads(out_path.read_text()).get("claims", [])
        except json.JSONDecodeError as e:
            print(f"[score] {item_id}: unparseable verifier output ({e}) — counted as unresolved.")
            claims = [{"step_index": None, "quoted_text": ""}]

        prediction, recovered = resolve_prediction(
            claims, inbox["solution_to_review"], inbox["num_steps"],
        )
        rows.append(
            {
                "id": item_id,
                "subset": item["subset"],
                "label": answer["label"],
                "prediction": prediction,
                "match": prediction == answer["label"],
                "num_claims": len(claims),
                "recovered": recovered,
            }
        )

    if missing:
        print(f"[score] {len(missing)} items have no verifier output — e.g. {missing[:5]}")
    if not rows:
        print("[score] nothing to score.")
        return 1

    per_subset = {
        subset: _subset_metrics([r for r in rows if r["subset"] == subset])
        for subset in manifest["subsets"]
        if any(r["subset"] == subset for r in rows)
    }
    overall = _subset_metrics(rows)

    summary = {
        "round": args.round,
        "dataset": "Qwen/ProcessBench",
        "verify_skill": manifest["verify_skill"],
        "sampling": manifest["sampling"],
        "seed": manifest["seed"],
        "num_scored": len(rows),
        "num_missing": len(missing),
        "overall": overall,
        "per_subset": per_subset,
        # Ids and hit/miss only. No problem, step, or solution text — and no
        # gold labels: this is the one eval artifact other skills may read, and
        # it must not carry the test set's answers, even keyed by id.
        "items": [{k: r[k] for k in ("id", "subset", "match")} for r in rows],
    }
    write_json(edir / "eval_summary.json", summary)
    # Gold labels and raw predictions stay beside answers/ — for debugging the
    # harness, never for tuning a policy.
    write_json(
        edir / "eval_details.json",
        {"round": args.round, "verify_skill": manifest["verify_skill"], "rows": rows},
    )

    print(f"\n=== ProcessBench round {args.round} ({manifest['verify_skill']}) ===")
    header = f"  {'subset':<15} {'n':>4} {'err_acc':>8} {'ok_acc':>8} {'F1':>8} {'bal_acc':>8}"
    print(header)
    for subset, m in per_subset.items():
        print(
            f"  {subset:<15} {m['n']:>4} {_fmt(m['error_accuracy']):>8} {_fmt(m['correct_accuracy']):>8} "
            f"{_fmt(m.get('processbench_f1')):>8} {_fmt(m.get('balanced_accuracy')):>8}"
        )
    print(
        f"  {'OVERALL':<15} {overall['n']:>4} {_fmt(overall['error_accuracy']):>8} "
        f"{_fmt(overall['correct_accuracy']):>8} {_fmt(overall.get('processbench_f1')):>8} "
        f"{_fmt(overall.get('balanced_accuracy')):>8}"
    )
    if overall["unresolved_predictions"]:
        print(f"  unresolved (claimed an error, no locatable step): {overall['unresolved_predictions']}")
    if overall["recovered_step_indices"]:
        print(f"  step indices recovered from quoted text: {overall['recovered_step_indices']}")
    print(f"  → {edir / 'eval_summary.json'}\n")
    return 0


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args) -> int:
    root = Path(args.root)
    summaries = []
    for edir in sorted(root.glob("round_*")):
        path = edir / "eval_summary.json"
        if path.exists():
            summaries.append(read_json(path))

    if not summaries:
        print(f"[report] no eval_summary.json under {root}.")
        return 1

    subsets = sorted({s for summary in summaries for s in summary["per_subset"]},
                     key=lambda s: SUBSETS.index(s) if s in SUBSETS else 99)
    print("\n=== ProcessBench across rounds (harmonic-mean F1) ===")
    print(f"  {'round':<7} {'skill':<12} " + " ".join(f"{s:>14}" for s in subsets) + f" {'OVERALL':>9}")
    for summary in summaries:
        cells = " ".join(
            f"{_fmt(summary['per_subset'].get(s, {}).get('processbench_f1')):>14}" for s in subsets
        )
        print(
            f"  {summary['round']:<7} {summary['verify_skill']:<12} {cells} "
            f"{_fmt(summary['overall'].get('processbench_f1')):>9}"
        )
    print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Evaluation root directory.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Sample held-out ProcessBench items and build the verifier inbox.")
    p_prep.add_argument("--round", type=int, required=True)
    p_prep.add_argument("--subsets", nargs="+", default=SUBSETS, choices=SUBSETS)
    p_prep.add_argument("--per-subset", type=int, default=20)
    p_prep.add_argument("--sampling", choices=["balanced", "random"], default="balanced",
                        help="'balanced' draws equal erroneous/correct items (accuracies are per-class, so this only reduces variance).")
    p_prep.add_argument("--max-steps", type=int, default=None, help="Skip chains longer than this many steps.")
    p_prep.add_argument("--verify-skill", default="verify-v1")
    p_prep.add_argument("--seed", type=int, default=0, help="Fixes the held-out sample; keep it constant across rounds.")
    p_prep.add_argument("--force", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_score = sub.add_parser("score", help="Score verifier outputs with the ProcessBench metric.")
    p_score.add_argument("--round", type=int, required=True)
    p_score.set_defaults(func=cmd_score)

    p_report = sub.add_parser("report", help="Compare ProcessBench results across rounds.")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
