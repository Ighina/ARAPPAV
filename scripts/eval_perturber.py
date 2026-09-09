#!/usr/bin/env python
"""Perturber evaluation — score a perturbation policy against a frozen verifier.

The mirror of the verifier protocol. A perturber is good when the errors it
injects survive scrutiny, so it is scored by how *little* a fixed verifier
finds:

    perturber score = 1 - unit_recall(frozen verifier)

Two things are held constant so the score reflects the perturber alone:

* **The verifier is frozen** at the round-0, cold-start policy (`<prefix>-v1`,
  an empty policy body — zero-shot). In self-play the verifier co-evolves, so
  `r_P` conflates a better perturber with a weaker opponent. Pinning it removes
  that, exactly as pinning the perturber does for verifier validation.
* **The problem set is fixed** per (source, seed, n) and cached, so every
  perturber version attacks byte-identical problems.

Three problem sources, one pipeline — which is what lets analysis and
validation data be produced identically and reused:

| `--source`             | role       | drawn from                              |
|------------------------|------------|-----------------------------------------|
| `hendrycks`            | analysis   | the training distribution (MATH train)  |
| `math500`              | validation | `HuggingFaceH4/MATH-500`, held out      |
| `processbench-correct` | test       | ProcessBench chains labelled correct    |

The test source is the interesting one: those are real model-generated solutions
that human annotators marked error-free, so perturbing them asks whether the
policy can plant an error a verifier misses *in text it did not write*.

    # analysis: every round, training distribution
    python scripts/eval_perturber.py run --prefix hperturb --versions 1 2 3 \
        --source hendrycks --n 80
    # validation: every round, held out
    python scripts/eval_perturber.py run --prefix hperturb --versions 1 2 3 \
        --source math500 --n 80
    python scripts/eval_perturber.py select --prefix hperturb --source math500
    # test: the selected round only, on ProcessBench correct chains
    python scripts/eval_perturber.py run --prefix hperturb --versions 4 \
        --source processbench-correct --n 80
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import random
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

DEFAULT_ROOT = REPO / "data" / "perturber_evals"
SOURCES = ("hendrycks", "math500", "processbench-correct")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def _w(p: Path, o) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Problem pools — fixed per (source, seed, n) and cached
# ---------------------------------------------------------------------------


def build_pool(source: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    if source == "hendrycks":
        from arappav.data.ingest_math import load_math_dataset
        ds = load_math_dataset(topics=["algebra"], split="train",
                               max_examples_per_topic=400, seed=seed)
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        return [{"episode_id": f"hendrycks-{i}", "problem": ds[i]["problem"],
                 "solution": ds[i]["solution"], "meta": {"level": ds[i].get("level")}}
                for i in sorted(idx)]

    if source == "math500":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        idx = rng.sample(range(len(ds)), min(n, len(ds)))
        return [{"episode_id": ds[i]["unique_id"].replace("/", "_").replace(".json", ""),
                 "problem": ds[i]["problem"], "solution": ds[i]["solution"],
                 "meta": {"subject": ds[i].get("subject"), "level": ds[i].get("level")}}
                for i in sorted(idx)]

    # processbench-correct: chains human annotators marked error-free (label -1),
    # balanced across subsets so no single difficulty dominates the test.
    from datasets import load_dataset
    from importlib.machinery import SourceFileLoader
    pbe = SourceFileLoader("pbe", str(REPO / "scripts" / "processbench_eval.py")).load_module()
    per = max(1, n // len(SUBSETS))
    out = []
    for sub in SUBSETS:
        ds = load_dataset("Qwen/ProcessBench", split=sub)
        ok = [i for i, lab in enumerate(ds["label"]) if lab == -1]
        for i in sorted(rng.sample(ok, min(per, len(ok)))):
            out.append({"episode_id": ds[i]["id"], "problem": ds[i]["problem"],
                        "solution": pbe.render_steps(ds[i]["steps"]),
                        "meta": {"subset": sub}})
    return out[:n]


def pool_path(root: Path, source: str, n: int, seed: int) -> Path:
    return root / f"{source}_n{n}_seed{seed}" / "pool.json"


def get_pool(root: Path, source: str, n: int, seed: int) -> list[dict]:
    p = pool_path(root, source, n, seed)
    if p.exists():
        return json.loads(p.read_text())
    pool = build_pool(source, n, seed)
    _w(p, pool)
    print(f"[pool] built {len(pool)} problems from {source} → {p.parent.name}")
    return pool


# ---------------------------------------------------------------------------
# run — perturb with version N, verify with the frozen verifier, score
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    root = Path(args.root)
    pool = get_pool(root, args.source, args.n, args.seed)
    base = root / f"{args.source}_n{args.n}_seed{args.seed}"
    reward_cfg = scoring.load_reward_config()

    frozen = args.frozen_verifier or f"{args.prefix.replace('perturb', 'verify')}-v1"
    if not (Path(args.skills_root) / frozen / "SKILL.md").exists():
        sys.exit(f"[run] frozen verifier {frozen!r} not found under {args.skills_root}. "
                 f"Pass --frozen-verifier.")
    print(f"[run] frozen verifier: {frozen}  (held constant across all versions)")

    backend = make_backend(args.backend, model=args.model, timeout=args.timeout,
                           skills_root=Path(args.skills_root),
                           **({"provider": args.provider} if args.backend != "claude-code" else {}))

    for v in args.versions:
        skill = f"{args.prefix}-v{v}"
        if not (Path(args.skills_root) / skill / "SKILL.md").exists():
            print(f"[run] {skill}: no such policy, skipping")
            continue
        vdir = base / skill
        (vdir / "episodes").mkdir(parents=True, exist_ok=True)
        todo = [it for it in pool
                if not (vdir / "episodes" / f"{it['episode_id']}.json").exists()]
        print(f"[run] {skill}: {len(todo)} to run ({len(pool)-len(todo)} cached)")

        abort: list[str] = []
        lock = threading.Lock()

        def one(item: dict):
            if abort:
                return
            eid = item["episode_id"]
            ep = Episode(episode_id=eid, problem=item["problem"],
                         solution=item["solution"], k=args.k)

            pres = backend.run(skill=skill,
                               user=render_perturb_prompt(ep, "", "").lstrip(),
                               step=f"perturb__{skill}", round_dir=vdir, episode_id=eid)
            if pres.infra_failure():
                with lock:
                    abort.append(f"{skill}/{eid} perturb: {pres.infra_failure()}")
                return
            parsed, err, stage = scoring.parse_perturbation(pres.text, args.k, item["solution"])
            if parsed is None:
                _w(vdir / "episodes" / f"{eid}.json",
                   {"episode_id": eid, "format_valid": False,
                    "failure_stage": stage, "reason": str(err)[:300]})
                return

            vres = backend.run(skill=frozen,
                               user=render_verify_prompt(ep, parsed.perturbed_solution,
                                                         "", "").lstrip(),
                               step=f"verify__{skill}", round_dir=vdir, episode_id=eid)
            if vres.infra_failure():
                with lock:
                    abort.append(f"{skill}/{eid} verify: {vres.infra_failure()}")
                return

            s = scoring.score_episode(
                episode_id=eid, k=args.k, perturbed=parsed, failure_stage=None,
                failure_reason=None, verifier_raw=vres.text, config=reward_cfg,
                history=None)
            s["format_valid"] = True
            s["meta"] = item.get("meta")
            _w(vdir / "episodes" / f"{eid}.json", s)

        with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool_ex:
            list(pool_ex.map(one, todo))
        if abort:
            raise InfrastructureError(
                f"{abort[0]} (+{len(abort)-1} more). Nothing scored for those; "
                f"re-run to resume — completed episodes are kept.")

        _score(vdir, skill, frozen, args)
    return 0


def _score(vdir: Path, skill: str, frozen: str, args) -> dict:
    recalls, invalid, units, ks = [], 0, [], []
    for f in sorted((vdir / "episodes").glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("format_valid"):
            invalid += 1
            continue
        if d.get("verifier_recall") is not None:
            recalls.append(d["verifier_recall"])
            units.append(d.get("num_error_units", 0))
            ks.append(d.get("k", 0))
    n = len(recalls)
    mean_recall = sum(recalls) / n if n else None
    res = {
        "skill": skill, "frozen_verifier": frozen, "source": args.source,
        "n_scored": n, "n_format_invalid": invalid,
        # The headline: what the frozen verifier failed to find.
        "perturber_score_1_minus_recall": round(1 - mean_recall, 4) if n else None,
        "mean_verifier_recall": round(mean_recall, 4) if n else None,
        "format_valid_rate": round(n / (n + invalid), 4) if (n + invalid) else None,
        "mean_units_per_episode": round(sum(units) / n, 3) if n else None,
        "unit_collapse_rate": round(sum(1 for u, k in zip(units, ks) if u < k) / n, 3) if n else None,
    }
    _w(vdir.parent / "scores" / f"{skill}.json", res)
    print(f"  {skill:26s} 1-recall={res['perturber_score_1_minus_recall']}  "
          f"(recall={res['mean_verifier_recall']}, format_valid="
          f"{res['format_valid_rate']}, units/ep={res['mean_units_per_episode']})")
    return res


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def cmd_select(args) -> int:
    base = Path(args.root) / f"{args.source}_n{args.n}_seed{args.seed}"
    rows = []
    for f in sorted((base / "scores").glob(f"{args.prefix}-v*.json"),
                    key=lambda p: int(p.stem.rsplit("-v", 1)[1])):
        d = json.loads(f.read_text())
        if d.get("perturber_score_1_minus_recall") is not None:
            rows.append(d)
    if not rows:
        sys.exit(f"[select] nothing scored under {base/'scores'} — run first.")

    print(f"{'policy':26s}{'1-recall':>10}{'fmt valid':>11}{'units/ep':>10}")
    for d in rows:
        print(f"{d['skill']:26s}{d['perturber_score_1_minus_recall']:>10.4f}"
              f"{d['format_valid_rate']:>11.3f}{d['mean_units_per_episode']:>10.2f}")

    # A perturber that emits unparseable JSON scores no reward at all in the
    # loop, so a policy below the format floor is not a legitimate winner
    # however well its surviving episodes did.
    ok = [d for d in rows if (d["format_valid_rate"] or 0) >= args.min_format_valid]
    if not ok:
        sys.exit(f"[select] no version reaches --min-format-valid "
                 f"{args.min_format_valid}; nothing selectable.")
    best = max(ok, key=lambda d: d["perturber_score_1_minus_recall"])
    spread = (max(d["perturber_score_1_minus_recall"] for d in ok)
              - min(d["perturber_score_1_minus_recall"] for d in ok))
    print(f"\nselected: {best['skill']}  (1-recall {best['perturber_score_1_minus_recall']:.4f}, "
          f"format-valid {best['format_valid_rate']:.3f})")
    if len(ok) < len(rows):
        print(f"  {len(rows)-len(ok)} version(s) excluded below the format floor")
    if best.get("unit_collapse_rate", 0) > 0.3:
        print(f"  warning: {best['unit_collapse_rate']:.0%} of its episodes collapsed "
              f"below k units — it may be stacking one mistake, not finding hard ones.")
    if spread < 0.02:
        print(f"  warning: the field spans only {spread:.3f}; this choice is near-arbitrary.")
    _w(base / "selected.json", {"selected": best["skill"], "by": "1-recall",
                                "validation": best, "candidates": rows})
    print(f"→ {base/'selected.json'}")
    if args.print_only:
        print(best["skill"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--skills-root", default=".claude/skills", dest="skills_root")
    p.add_argument("--prefix", default="hperturb")
    p.add_argument("--source", choices=SOURCES, default="hendrycks")
    p.add_argument("--n", type=int, default=80, help="problems per version")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["api", "claude-code", "batch"], default="api")
    p.add_argument("--provider", default=None, choices=["anthropic", "openai", "deepseek"])
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--timeout", type=int, default=600)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="perturb with each version, verify with the frozen verifier")
    r.add_argument("--versions", type=int, nargs="+", required=True)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--frozen-verifier", default=None, dest="frozen_verifier",
                   help="defaults to <prefix with perturb->verify>-v1, the "
                        "round-0 zero-shot policy")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("select", help="pick the best perturber by 1-recall")
    s.add_argument("--min-format-valid", type=float, default=0.8,
                   dest="min_format_valid",
                   help="exclude versions below this format-valid rate")
    s.add_argument("--print-only", action="store_true", dest="print_only")
    s.set_defaults(func=cmd_select)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    try:
        raise SystemExit(a.func(a))
    except InfrastructureError as e:
        print(f"\n[perturber-eval] ABORTED — {e}", file=sys.stderr)
        raise SystemExit(2)
