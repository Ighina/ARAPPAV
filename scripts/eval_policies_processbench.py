#!/usr/bin/env python
"""Evaluate verifier policy versions on held-out ProcessBench.

Self-play says whether `hverify-vN` beats *this* perturber. ProcessBench says
whether it finds errors humans annotated in real model-generated reasoning —
i.e. whether a self-play gain generalises at all.

Run it standalone, against policies produced by any run:

    python scripts/eval_policies_processbench.py --prefix hverify \
        --versions 1 2 3 --model claude-haiku-4-5 --per-subset 5

Design notes:

* **One fixed sample for every policy.** The items are drawn once, by `--seed`,
  and reused across versions. Comparing v1 and v3 on different draws would
  measure the draw, not the policy.
* **Same renderer, same leak guard** as the pipeline (`render_verify_prompt`),
  so an evaluation prompt cannot contain anything a self-play prompt could not.
* **Fail fast on infrastructure**, never scoring a quota error as a wrong
  answer — the bug that fabricated a whole 10-round table.
* **Resumable.** Items already answered are skipped, so this can run beside a
  live experiment and be restarted freely.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arappav.pipeline.agents import InfrastructureError, run_claude  # noqa: E402
from arappav.pipeline.contracts import Episode, render_verify_prompt  # noqa: E402


def prepare(root: Path, rnd: int, skill: str, per_subset: int, seed: int) -> Path:
    rd = root / f"round_{rnd:02d}"
    if not (rd / "inbox").is_dir():
        r = subprocess.run(
            [sys.executable, "scripts/processbench_eval.py", "--root", str(root),
             "prepare", "--round", str(rnd), "--verify-skill", skill,
             "--per-subset", str(per_subset), "--seed", str(seed)],
            capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            sys.exit(f"[prepare] failed:\n{r.stderr[-800:]}")
    return rd


def evaluate(version: int, args) -> dict:
    skill = f"{args.prefix}-v{version}"
    if not (REPO / ".claude" / "skills" / skill / "SKILL.md").exists():
        return {"version": version, "skipped": "policy does not exist yet"}

    root = Path(args.root)
    # Every version is scored on the SAME items: one shared seed, one
    # directory per version so outputs never collide.
    rd = prepare(root / skill, version, skill, args.per_subset, args.seed)
    inbox, outbox = rd / "inbox", rd / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)

    todo = [f for f in sorted(inbox.glob("*.json")) if not (outbox / f.name).exists()]
    print(f"[{skill}] {len(todo)} item(s) to run "
          f"({len(list(inbox.glob('*.json'))) - len(todo)} already done)")

    for f in todo:
        item = json.loads(f.read_text())
        ep = Episode(episode_id=f.stem, problem=item["problem"], solution="", k=0)
        prompt = render_verify_prompt(ep, item["solution_to_review"], f"/{skill}", "")
        res = run_claude(prompt, step="processbench", round_dir=rd,
                         episode_id=f.stem, model=args.model, timeout=args.timeout)
        reason = res.infra_failure()
        if reason:
            raise InfrastructureError(
                f"{skill}/{f.stem}: model never reached ({reason}). "
                f"Nothing scored. Re-run to resume — completed items are kept.")
        (outbox / f.name).write_text(res.text)

    sc = subprocess.run(
        [sys.executable, "scripts/processbench_eval.py", "--root", str(root / skill),
         "score", "--round", str(version)], capture_output=True, text=True, cwd=REPO)
    print(sc.stdout.strip())
    summ = rd / "eval_summary.json"
    out = {"version": version, "skill": skill}
    if summ.exists():
        d = json.loads(summ.read_text())
        out["overall"] = d.get("overall")
        out["per_subset"] = d.get("per_subset")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="hverify", help="verifier policy prefix")
    p.add_argument("--versions", type=int, nargs="+", required=True)
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--per-subset", type=int, default=5, dest="per_subset")
    p.add_argument("--seed", type=int, default=0,
                   help="fixes the held-out sample; keep constant across versions")
    p.add_argument("--root", default="data/policy_evals")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    results = []
    try:
        for v in args.versions:
            results.append(evaluate(v, args))
    except InfrastructureError as e:
        print(f"\n[eval] ABORTED — {e}", file=sys.stderr)
        _report(results, args)
        return 2
    _report(results, args)
    return 0


def _report(results: list[dict], args) -> None:
    if not results:
        return
    print(f"\n=== ProcessBench: {args.prefix} "
          f"(per_subset={args.per_subset}, seed={args.seed}, model={args.model}) ===")
    print(f"{'policy':16s}{'err_acc':>9s}{'ok_acc':>9s}{'F1':>9s}")
    for r in results:
        if r.get("skipped"):
            print(f"{r.get('skill', 'v' + str(r['version'])):16s}  {r['skipped']}")
            continue
        o = r.get("overall") or {}
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"
        print(f"{r['skill']:16s}{fmt(o.get('error_accuracy')):>9s}"
              f"{fmt(o.get('correct_accuracy')):>9s}{fmt(o.get('processbench_f1')):>9s}")
    out = Path(args.root) / "policy_eval_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": results},
                              indent=2) + "\n")
    print(f"\n→ {out}")


if __name__ == "__main__":
    raise SystemExit(main())
