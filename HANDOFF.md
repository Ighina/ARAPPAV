# Session handoff — ARAPPAV, branch `REFACTORED`

State as of 2026-09-10. Written for a fresh Claude session on another machine.
Read this first, then `docs/REFACTOR.md` and `docs/EVOLVE_MODE.md`.

---

## 1. Restore the working state

```bash
git clone <remote> && cd ARAPPAV && git checkout REFACTORED
python -m pytest tests/ -q          # expect 293 passed
```

Experiment data is **not** in git (deliberately — `data/` is gitignored). It
travels as bundles built by `scripts/package_experiment.py`:

| bundle | size | contains |
|---|---|---|
| `algebra_evolve.zip` | 4.9 MB | 10 rounds + ProcessBench (80) + PRM800K validation + MATH-500 sets |
| `algebra_evolve_b.zip` | 1.1 MB | 10 rounds, **no held-out evaluation yet** |
| `haiku_cold_10x8.zip` | 3.7 MB | 10 rounds (rewrite mode) + ProcessBench v1–v10 |
| `exp_b.zip` | 6.4 MB | 10 rounds (rewrite, api backend) + ProcessBench |
| `exp_a.zip` | 1.1 MB | 6 rounds (partial, ran out of credits) |

Copy them across out of band (scp/rsync), then:

```bash
python scripts/package_experiment.py unpack algebra_evolve.zip
# repeat per bundle; add --into <dir> to restore somewhere other than the repo
```

Secrets do **not** travel. Recreate `configs/.secrets` on the remote with
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`. It is gitignored
(`.gitignore:80`) and has never been committed.

---

## 2. What the experiments are

All four use `claude-haiku-4-5` as the **player** model. They differ in updater
and update mode.

| run | update mode | updater | gate | category | status |
|---|---|---|---|---|---|
| `algebra_evolve` | evolve | opus-5 | `--accept-on-validation` | algebra | 10 rounds, evaluated |
| `algebra_evolve_b` | evolve | haiku-4-5 | none | algebra | 10 rounds, **unevaluated** |
| `haiku_cold_10x8` | rewrite | haiku-4-5 | none | all | 10 rounds, evaluated |
| `exp_b` | rewrite | haiku-4-5 | none | all | 10 rounds, evaluated |

`algebra_evolve` vs `algebra_evolve_b` differ on **two** axes at once (updater
model and gate), so they are two independent evolve runs, not a clean ablation
of either.

---

## 3. Results established so far

### 3.1 Held-out ProcessBench, `algebra_evolve` (80 items, identical set)

| policy | v1 | v2 | **v3** | v4 | v5 | v10 |
|---|---|---|---|---|---|---|
| F1 | 0.731 | 0.840 | **0.905** | 0.874 | 0.822 | 0.831 |

Rise to a peak at round 2 (+0.175, ≈3.9 SE), then monotone decay. Driven
entirely by error accuracy (0.583 → 0.861 → 0.722); correct accuracy stayed
≈0.955 throughout.

**v6–v9 exist but were only evaluated at n=20** (0.825 / 0.714 / 0.789 / 0.825).
Mixed-n — not comparable with the rest. Extending them to 80 is queued work.

### 3.2 PRM800K as a validation set — **negative result, do not use**

Six policies, 80 items each, collected 2026-09-10:

| policy | v1 | v2 | v3 | v4 | v5 | v10 |
|---|---|---|---|---|---|---|
| PRM800K F1 | 0.7825 | 0.7969 | 0.7677 | 0.7714 | 0.8077 | 0.8031 |

Against ProcessBench: Pearson **−0.356**, Spearman **−0.657**, t(4) = −0.76
(p ≈ 0.49). Two independent failures:

- **No dynamic range.** Spread 0.040 across six policies; SE on error accuracy
  at 40 erroneous items is ≈0.07. The whole spread fits inside one SE.
- **Wrong sign.** ProcessBench's peak (v3) is PRM800K's minimum.

Cause, verified directly from the preprocessed labels: **39 of 40 erroneous
items (98%) have the error at the final step**, vs ~4% in ProcessBench. It
measures "did you flag the last step," not first-error localisation.

Data is in `data/policy_evals/prm800k_algev/`. Keep it — it is a reportable
negative result, not a failed run to delete.

### 3.3 Contract compliance degrades under `rewrite`, not under `evolve`

`format_valid_rate` per round, same player model throughout:

| run | mode | rate |
|---|---|---|
| `algebra_evolve` | evolve | 1.0 × 10 |
| `algebra_evolve_b` | evolve | 1.0 × 10 |
| `exp_b` | rewrite | 1.0 × 7, then 0.875 |
| `haiku_cold_10x8` | rewrite | 1.0 × 8, then **0.75, 0.625** |

Late-onset: eight clean rounds, then it breaks. Every failure is the same
thing — the perturber under-declaring `k` ("Expected exactly 3 errors, got 1").

That matters because under-declaring `k` is a genuine unilateral advantage:
with `r_P = 1 − unit_recall`, three errors give the verifier three independent
chances to score, one error gives it one. The k-conformance rule is the guard
against the perturber shrinking the game, and the −5/−10 penalty is what makes
it binding.

Since both evolve runs held at 1.0 (opus **and** haiku updaters) while both
rewrite runs degraded, update mode is the more likely explanation than updater
strength — though not a controlled comparison.

### 3.4 Other standing facts

- Within-round SD 0.185 → SE 0.065 at n=8. **Moves under ~0.13 are noise.**
- Self-play reward vs held-out F1: r = +0.275 (n=10, n.s.). Selecting on
  self-play reward returns a policy worse than the untuned cold start.
- **Zero `delete_rule` in 37 evolve edits** (25 append, 10 add, 2 replace).
  Accumulation survives making deletion a named, single-step operation.
- `mean_units_per_episode` sits at 2.0–2.5 against k=3 in every run, flat with
  no trend. ~20% of declared errors merge. Stable, not degrading.
- `total_spam_penalty` and `total_repetition_penalty` are **0 everywhere**.
  Those two penalties have never fired.
- Taxonomy never affects scoring (`error_type` is absent from matcher and
  reward) but is a one-sided hard constraint on the perturber.

---

## 4. Open decisions

### 4.1 Validation set — design agreed, not built

PRM800K is dead (§3.2). ProcessBench cannot be split, because baseline numbers
from the literature are on the full benchmark and there is no budget to re-run
them.

Agreed replacement: **MATH-500 perturbed once per item by Opus 5 with no
policy**, single error, to match ProcessBench's first-error-step setting.
Five design constraints, all of which come from why PRM800K failed:

1. **Sample the target step explicitly** (uniform over the chain, or matched to
   ProcessBench's empirical distribution). Left alone Opus will gravitate to the
   final step and reproduce PRM800K's exact failure. This is the single most
   important instruction in the perturber prompt.
2. **Keep a clean half** — F1 is a harmonic mean; with no correct chains
   `correct_accuracy` is undefined.
3. **Segment every solution into `<step_i>` blocks first, then perturb half.**
   The segmentation pass is needed anyway for labels, and doing it to both
   halves removes a "touched by Opus vs reference prose" stylistic tell.
4. **Go well past 80 items** — at n=80 SE ≈ 0.07 and ProcessBench's own spread
   is only 0.174. Target ~240 (SE ≈ 0.04). The perturb+segment pass is one-time;
   only verification recurs, in batch mode.
5. **Dedupe against ProcessBench by problem text.** MATH-500 and ProcessBench's
   `math` subset both derive from the Hendrycks MATH test set. A 16-vs-38 spot
   check found no overlap, which proves nothing at full scale.

**Acceptance test before trusting it as a selector:** Spearman against the
existing ProcessBench numbers. At n=6 policies that needs ρ ≈ 0.83; extending
v6–v9 to 80 items gives n=10 and needs ρ ≈ 0.65.

**Pre-commit to the design and run it once.** If it fails, do not iterate on
perturber prompts until the correlation appears — that is selecting a validation
set on the test set and would invalidate the reported ProcessBench numbers.

Note this experiment also tests the paper's central premise: if Opus-injected
errors don't predict ProcessBench, "tune on injected errors, transfer to natural
errors" is itself in question. Reportable either way.

### 4.2 Queued batch work, in recommended order

1. **Extend `algebra_evolve` v6–v9 from 20 → 80 items** (4 runs). Cheapest,
   fixes the mixed-n table, unblocks the acceptance test above.
2. **ProcessBench on `algebra_evolve_b` v1–v10** (10 runs). It currently has no
   held-out measurement at all. A second independent 10-round curve is what says
   whether the rise-and-decay shape is real or was one draw — arguably the more
   important result, since the shape *is* the finding.
3. **Build the validation set** per §4.1.

(1) and (2) can go as a single 14-run batch submission — submit all, collect
once. Do not submit sequentially; 10 policies became 10 queued batches once
already.

### 4.3 Paper framing

Safe to drop from the write-up: the **scalar reward** (in evolve mode it has no
causal path to the policy — analysts never see it), and the **anti-spam /
anti-repetition / anti-duplicate penalties** (never fired).

Cannot be dropped: **error units** (the reported recall denominator; 2.25 of 3
needs explaining) and the **format gate + k-conformance** (load-bearing on
difficulty, and §3.3 is a run where it broke).

Report §3.3 as a finding rather than omitting it — "whole-policy rewriting
degraded contract compliance to 0.625 by round 9 while structured patching held
at 1.0" argues for the method. A reviewer can find it either way.

---

## 5. Traps already hit — do not rediscover

- **Session-limit errors scored as format failures.** 171/178 calls never
  reached a model, each scored −10.0, fabricating a full results table. Fixed
  via `infra_failure()` + `_guard` abort. If a table looks too complete, check
  `attempts`.
- **Stale `eval_summary`.** v3 once reported 0.895 from a 20-item summary while
  its siblings were 80. **Always check the `n` field** before comparing.
- **`ANTHROPIC_API_KEY` shadows the claude.ai subscription** in `claude -p`,
  surfacing as "Credit balance is too low". `run_claude` strips it from the
  subprocess env.
- **`--extend` is required** to grow a ProcessBench sample; re-preparing at a
  larger `--per-subset` drops existing items.
- **haiku rejects `thinking: adaptive`** but supports `budget_tokens`. Disabling
  thinking entirely cost 0.708 vs 0.875 on identical items.
- **Batch prompts have no audit trail** — known gap, deferred.
- **`data/` must stay untracked.** 10,239 generated files were tracked once.
  Root-level `*.zip` is now gitignored too; `iclr2027.zip` and `rollout.zip`
  remain tracked from earlier commits and were left alone deliberately.

---

## 6. Uncollected / in-flight

Nothing. All six PRM800K batches were collected before this handoff; no batch
is outstanding and no background process is running.
