---
name: eval-processbench
description: Run the held-out external evaluation of the current Verifier skill on Qwen/ProcessBench (gsm8k, math, olympiadbench, omnimath) and report the ProcessBench first-error-step metric. Use after each skill-tuning round, or when asked how well the verifier generalises to real annotated errors.
---

# eval-processbench — external validity check for the Verifier

Self-play tells you whether `verify-vN` beats *this* Perturber. **ProcessBench**
(`Qwen/ProcessBench`) tells you whether it finds errors that humans annotated in real
model-generated reasoning: a math problem, a chain split into steps, and the index of the
**first erroneous step** — or `-1` when the chain is correct throughout. Four subsets of
increasing difficulty: `gsm8k` → `math` → `olympiadbench` → `omnimath`.

Run this after every round of `selfplay`, against the verifier version that round produced.

---

## The leakage rules — non-negotiable

This is a **test set**. It is scored, never learned from.

1. **No memory of the items.** Never write a ProcessBench problem, step, chain, or label
   into a skill file, a rollout directory, a report, a commit message, or memory. The only
   thing that leaves this evaluation is aggregate numbers.
2. **The updaters never look.** `update-perturb` and `update-verify` must not read anything
   under `data/skill_evals/`. Policy edits come from the self-play round only. Tuning a
   policy on these numbers — even by choosing which edit to keep — is test-set fitting, and
   the score stops meaning anything the moment you do it.
3. **The verifier pass is isolated.** It runs in a fresh agent scoped to `inbox/`, which
   holds only the problem and the tagged chain. `answers/` and `eval_details.json` hold the
   gold labels and are off-limits to it.
4. **The orchestrator stays clean.** Do not read inbox items into the main context. Delegate
   the pass and read `eval_summary.json` — ids, hit/miss, and metrics, with no item text and
   no labels.
5. **Same items every round.** The sample is fixed by `--seed` alone, not by round, so
   cross-round movement measures the policy rather than the draw. Keep the seed constant for
   the life of an experiment.

---

## Procedure

### 1. Prepare the held-out sample

```bash
python scripts/processbench_eval.py prepare --round $R --verify-skill verify-v$V \
  --per-subset 20 --seed 0
```

Each item becomes `data/skill_evals/round_$R/inbox/<id>.json`:

```json
{"episode_id": "gsm8k-162", "subset": "gsm8k", "problem": "…",
 "solution_to_review": "<step_0>\n…\n</step_0>\n\n<step_1>\n…\n</step_1>",
 "step_format": "tagged", "num_steps": 3}
```

The whole chain is passed at once, with explicit step boundaries — the Verifier judges each
step in the context of the full solution, and names the step it is accusing. Gold labels go
to `answers/`, which the verifier pass never sees.

`--per-subset 20` (80 items) is a reasonable per-round budget; the full benchmark is 3400
items. Balanced sampling (equal erroneous/correct) is the default and only reduces variance:
the metric is computed per class, so the mix does not bias it.

### 2. Run the Verifier — in a fresh agent

Spawn one agent (`Agent`, `general-purpose`) per evaluation round, scoped exactly:

> Follow `.claude/skills/verify-v$V/SKILL.md`. For every file in
> `data/skill_evals/round_$R/inbox/`, produce the claims JSON and write it to
> `data/skill_evals/round_$R/outbox/<same filename>`. The input is tagged-step format, so
> every claim must carry the `step_index` of the block it accuses. Do not read
> `answers/`, `eval_details.json`, or any other directory.

Do not run this in a context that has seen the answers, the self-play round, or a previous
evaluation's items.

### 3. Score

```bash
python scripts/processbench_eval.py score --round $R
```

The prediction for an item is the **earliest** step index across the claims, or `-1` when
the Verifier returned no claims. A claim with a missing or out-of-range `step_index` is
recovered by locating its quoted text inside a step block; if nothing can be resolved, the
item counts as wrong against both classes — claiming an error you cannot locate is not a
detection.

### 4. Read the metric

Per subset and overall:

| Quantity | Meaning |
|----------|---------|
| `error_accuracy` | share of erroneous chains whose first bad step was identified exactly |
| `correct_accuracy` | share of correct chains reported as clean |
| `processbench_f1` | **harmonic** mean of the two — the official ProcessBench headline metric |
| `balanced_accuracy` | arithmetic mean of the two |

The harmonic mean is the one to quote. It is what makes the benchmark meaningful: a verifier
that never claims anything scores `correct_accuracy = 1.0` and `balanced_accuracy = 0.5`,
but `processbench_f1 = 0`. Watch `unresolved_predictions` too — it counts claims that
accused something unlocatable, which is a contract failure rather than a reasoning one.

### 5. Compare across rounds

```bash
python scripts/processbench_eval.py report
```

Report to the user: the per-subset F1 for this round, the delta against the previous round,
and whether the difficulty gradient holds (gsm8k ≥ math ≥ olympiadbench ≥ omnimath). Then
say plainly whether self-play gains transferred:

- **Self-play recall up, ProcessBench up** — the policy learned to verify.
- **Self-play recall up, ProcessBench flat or down** — the policy learned *this Perturber*.
  That is the outcome this evaluation exists to catch; say so directly, and treat the
  self-play gain as unproven.
- **Both flat** — the loop has converged, or the rounds are too easy to separate policies.

Never propose a policy edit from these results. Reporting the divergence is the deliverable;
fixing it happens on the next self-play round's evidence.
