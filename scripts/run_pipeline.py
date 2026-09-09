#!/usr/bin/env python
"""Deterministic orchestrator for the ARAPPAV self-play experiment.

This script — not a skill — owns the pipeline: dataset sampling, episode
creation, round management, policy creation/update/freezing, perturbation and
verification invocation, reward computation, optional ProcessBench, findings,
persistence and the final report.

    # cold start, 3 rounds, nothing frozen, no ProcessBench
    python scripts/run_pipeline.py --rounds 3 --episodes 8 --k 3

    # warm start, freeze the verifier, run ProcessBench each round
    python scripts/run_pipeline.py --start warm --freeze verifier \
        --processbench --processbench-per-subset 5

    # inspect every prompt without spending anything
    python scripts/run_pipeline.py --rounds 1 --episodes 2 --dry-run

See docs/REFACTOR.md for the architecture and the leakage audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arappav.pipeline.agents import InfrastructureError  # noqa: E402
from arappav.pipeline.orchestrator import Pipeline, PipelineConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("experiment")
    g.add_argument("--rounds", type=int, default=3)
    g.add_argument("--episodes", type=int, default=8)
    g.add_argument("--k", type=int, default=3, help="errors injected per episode")
    g.add_argument("--start", choices=["cold", "warm"], default="cold",
                   help="cold: empty initial policies. warm: create-policy-* skills.")
    g.add_argument("--freeze", choices=["none", "perturber", "verifier", "both"],
                   default="none", help="which policy is NOT updated between rounds")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-taxonomy", action="store_true", dest="taxonomy_free",
                   help="drop the fixed 26-category error taxonomy: the perturber "
                        "names misconceptions in its own words and a free-text "
                        "label no longer fails schema validation. The taxonomy "
                        "never affected scoring, only what the perturber could "
                        "express without voiding the episode.")

    d = p.add_argument_group("data")
    d.add_argument("--source", choices=["hendrycks", "local", "file"], default="hendrycks")
    d.add_argument("--problems-file", default=None, dest="problems_file")
    d.add_argument("--topics", nargs="+", default=["algebra"])
    d.add_argument("--per-topic", type=int, default=50, dest="per_topic")
    d.add_argument("--category", default=None,
                   help="restrict the experiment to one MATH subject (algebra, "
                        "geometry, number_theory, precalculus, prealgebra, "
                        "intermediate_algebra, counting_and_probability, or 'all'). "
                        "Training draws only that topic; MATH-500 validation is "
                        "filtered to the matching subject.")

    o = p.add_argument_group("paths")
    o.add_argument("--root", default="data/skill_rollouts")
    o.add_argument("--skills-root", default=".claude/skills", dest="skills_root")
    o.add_argument("--perturb-prefix", default="perturb", dest="perturb_prefix")
    o.add_argument("--verify-prefix", default="verify", dest="verify_prefix")
    o.add_argument("--overwrite-policies", action="store_true", dest="overwrite_policies",
                   help="replace existing <prefix>-vN skill directories")

    b = p.add_argument_group("processbench")
    b.add_argument("--processbench", action="store_true", dest="processbench_enabled",
                   help="run the held-out ProcessBench evaluation each round")
    b.add_argument("--processbench-per-subset", type=int, default=5,
                   dest="processbench_per_subset")
    b.add_argument("--processbench-root", default="data/skill_evals",
                   dest="processbench_root")
    b.add_argument("--processbench-seed", type=int, default=0, dest="processbench_seed")

    a = p.add_argument_group("agents")
    a.add_argument("--model", default=None,
                   help="model for playing episodes (perturb + verify)")
    a.add_argument("--updater-model", default=None, dest="updater_model",
                   help="model for authoring policies (create-policy-*, update-*, "
                        "final_summary). Defaults to --model. Set it higher than "
                        "--model to test whether the policy writer, rather than the "
                        "players, is the bottleneck.")
    a.add_argument("--backend", choices=["claude-code", "api"], default="claude-code",
                   help="claude-code: `claude -p` per call (Claude models only). "
                        "api: direct API — the only way to run non-Claude models.")
    a.add_argument("--provider", default=None,
                   choices=["anthropic", "openai", "deepseek"],
                   help="api backend: provider for the players; inferred from --model")
    a.add_argument("--updater-provider", default=None, dest="updater_provider",
                   choices=["anthropic", "openai", "deepseek"],
                   help="api backend: provider for the policy author; "
                        "inferred from --updater-model")
    a.add_argument("--timeout", type=int, default=900)
    a.add_argument("--max-tokens", type=int, default=16000, dest="max_tokens",
                   help="output ceiling per call; must comfortably exceed the "
                        "thinking budget or long episodes truncate")
    a.add_argument("--retry-format", type=int, default=0, dest="retry_format",
                   help="extra attempts when a reply fails to parse (default 0, "
                        "which preserves the pre-refactor penalty statistics)")
    a.add_argument("--update-mode", choices=["rewrite", "summarise", "evolve"],
                   default="rewrite", dest="update_mode",
                   help="how a policy is revised between rounds — the ablation axis. "
                        "rewrite: one updater rewrites the whole body (default). "
                        "summarise: a summariser briefs the updater first. "
                        "evolve: Trace2Skill-style — one analyst per episode proposes "
                        "typed edits in parallel, a merge step reconciles them, and "
                        "Python applies the result, which makes deletion an operation "
                        "rather than an omission.")
    a.add_argument("--accept-on-validation", action="store_true",
                   dest="accept_on_validation",
                   help="keep a revised verifier policy only if it does not regress on "
                        "a small held-out MATH-500 check; otherwise carry the previous "
                        "one forward")
    a.add_argument("--validation-n", type=int, default=12, dest="validation_n",
                   help="items in the acceptance check (default 12)")
    a.add_argument("--rich-context", action="store_true", dest="rich_context",
                   help="insert a summarise-context-* step before each policy "
                        "update, so the updater receives the TEXT of the episodes "
                        "and a legend for the metrics rather than counts, ids and "
                        "taxonomy labels")
    a.add_argument("--no-context", action="store_true", dest="no_context",
                   help="pass no cross-round history to any agent (spec 21)")
    a.add_argument("--resume", action="store_true",
                   help="skip rounds that already have a round_summary.json")
    a.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="render and persist every prompt, invoke nothing")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = PipelineConfig(
        rounds=args.rounds, episodes=args.episodes, k=args.k, start=args.start,
        freeze=args.freeze, source=args.source, problems_file=args.problems_file,
        topics=tuple(args.topics), per_topic=args.per_topic, seed=args.seed,
        category=args.category,
        root=args.root, skills_root=args.skills_root,
        perturb_prefix=args.perturb_prefix, verify_prefix=args.verify_prefix,
        model=args.model, updater_model=args.updater_model,
        backend=args.backend, provider=args.provider,
        updater_provider=args.updater_provider,
        processbench_enabled=args.processbench_enabled,
        processbench_per_subset=args.processbench_per_subset,
        processbench_root=args.processbench_root,
        processbench_seed=args.processbench_seed, no_context=args.no_context,
        overwrite_policies=args.overwrite_policies, dry_run=args.dry_run,
        timeout=args.timeout, retry_format=args.retry_format,
        max_tokens=args.max_tokens, taxonomy_free=args.taxonomy_free,
        rich_context=args.rich_context or args.update_mode == 'summarise',
        update_mode=args.update_mode,
        accept_on_validation=args.accept_on_validation,
        validation_n=args.validation_n,
        resume=args.resume,
    )
    print(f"[pipeline] backend={cfg.backend} players={cfg.model or 'default'} "
          f"updater={cfg.policy_model() or 'default'}")
    print(f"[pipeline] update-mode={cfg.update_mode}"
          + ("  accept-on-validation=on" if cfg.accept_on_validation else ""))
    print(f"[pipeline] start={cfg.start} freeze={cfg.freeze} rounds={cfg.rounds} "
          f"episodes={cfg.episodes} k={cfg.k} processbench={cfg.processbench_enabled} "
          f"no_context={cfg.no_context} dry_run={cfg.dry_run}")
    try:
        result = Pipeline(cfg).run()
    except InfrastructureError as e:
        print(f"\n[pipeline] ABORTED — {e}", file=sys.stderr)
        print("[pipeline] Completed rounds are on disk and were NOT contaminated.",
              file=sys.stderr)
        return 2
    print("\n" + result["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
