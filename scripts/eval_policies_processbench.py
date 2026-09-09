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
import concurrent.futures as futures
import json
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from arappav.pipeline.agents import InfrastructureError  # noqa: E402
from arappav.pipeline.backends import make_backend  # noqa: E402
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


def evaluate(version: int, args, backend) -> dict:
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
    done_already = len(list(inbox.glob("*.json"))) - len(todo)
    print(f"[{skill}] {len(todo)} item(s) to run ({done_already} already done), "
          f"concurrency={args.concurrency}")

    # Items are independent, so they run in parallel. Measured sequentially at
    # ~26s median / ~57s mean per item, three quarters of the wall clock went to
    # a rate-limit tail; overlapping the waits is the entire win here.
    abort: list[str] = []
    lock = threading.Lock()
    progress = {"n": 0}
    usage: dict[str, int] = {}

    def one(f: Path) -> None:
        if abort:                      # a sibling already hit infrastructure
            return
        item = json.loads(f.read_text())
        ep = Episode(episode_id=f.stem, problem=item["problem"], solution="", k=0)
        # The policy reference is empty for the API backend, which delivers the
        # policy as a cached system prompt instead of a slash command; the same
        # renderer and the same leak guard apply either way.
        ref = "" if args.backend == "api" else f"/{skill}"
        prompt = render_verify_prompt(ep, item["solution_to_review"], ref, "").lstrip()
        res = backend.run(skill=skill, user=prompt, step="processbench",
                          round_dir=rd, episode_id=f.stem)
        reason = res.infra_failure()
        if reason:
            with lock:
                abort.append(f"{skill}/{f.stem}: {reason}")
            return
        # Written only after the guard passes, so a quota error never leaves a
        # placeholder that a later --resume would mistake for a real answer.
        (outbox / f.name).write_text(res.text)
        with lock:
            for k, v in (getattr(res, "usage", None) or {}).items():
                usage[k] = usage.get(k, 0) + v
            progress["n"] += 1
            if progress["n"] % 10 == 0 or progress["n"] == len(todo):
                print(f"[{skill}] {progress['n']}/{len(todo)}", flush=True)

    if todo:
        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(one, todo))
    if abort:
        raise InfrastructureError(
            f"{abort[0]} (+{len(abort) - 1} more). Model never reached; nothing "
            f"scored for those. Re-run to resume — completed items are kept.")

    if usage:
        # Haiku rates; cache reads bill at ~10% of the input rate.
        cost = (usage.get("input_tokens", 0) * 1.0
                + usage.get("cache_write", 0) * 1.25
                + usage.get("cache_read", 0) * 0.10
                + usage.get("output_tokens", 0) * 5.0) / 1e6
        print(f"[{skill}] tokens in={usage.get('input_tokens',0):,} "
              f"cache_read={usage.get('cache_read',0):,} "
              f"out={usage.get('output_tokens',0):,}  ≈${cost:.4f}")

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
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel items per policy (default 4)")
    p.add_argument("--backend", choices=["api", "claude-code"], default="api",
                   help="api: Messages API directly, policy sent as a cached system "
                        "prompt — ~4%% of the cost and no session-quota use, needs "
                        "ANTHROPIC_API_KEY. claude-code: `claude -p` per item, which "
                        "re-sends the whole harness each call.")
    p.add_argument("--max-tokens", type=int, default=8000, dest="max_tokens")
    p.add_argument("--skills-root", default=".claude/skills", dest="skills_root")
    args = p.parse_args()

    backend = make_backend(args.backend, model=args.model, timeout=args.timeout,
                           max_tokens=args.max_tokens, skills_root=Path(args.skills_root))
    print(f"[eval] backend={args.backend} model={args.model} "
          f"concurrency={args.concurrency}")
    results = []
    try:
        for v in args.versions:
            results.append(evaluate(v, args, backend))
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
