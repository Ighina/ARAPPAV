---
name: selfplay
description: DEPRECATED — the self-play experiment is now orchestrated by scripts/run_pipeline.py, not by this skill. Kept only to redirect; do not use it to run or coordinate a rollout.
---

# selfplay — deprecated

This skill used to orchestrate the whole experiment: sampling, rounds, invoking the
perturber and verifier, scoring, policy updates and reporting.

**It no longer does, and must not be used for that.** Orchestration now lives in a
deterministic Python script:

```bash
python scripts/run_pipeline.py --rounds 3 --episodes 8 --k 3
```

Why the change: when an agent orchestrates the loop, the data flow between the Perturber and
the Verifier is implicit, and the boundaries between them rest on instructions rather than on
code. Moving control into Python makes every payload explicit, auditable and reproducible —
see `docs/REFACTOR.md`, in particular the leakage audit.

If you were about to invoke this skill to run a rollout, run the script instead. Its
`--help` documents every parameter (rounds, episodes, `k`, `--start cold|warm`,
`--freeze`, `--processbench`, `--no-context`, `--dry-run`).
