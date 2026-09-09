# Self-play with Claude Haiku 4.5: a negative result

**Run:** `data/skill_rollouts/haiku_cold_10x8` · 10 rounds × 8 episodes, k=3, cold start,
neither side frozen · `claude-haiku-4-5` for perturber, verifier *and* policy updates ·
Hendrycks MATH (algebra), 80 problems drawn without reuse · 2026-09-07/09.

**Result: ten rounds of self-play produced no improvement on held-out data.** The verifier
policy went from ProcessBench F1 **0.866** (cold start, *empty* policy body) to **0.831**
after ten rounds — a delta of −0.036, OLS slope −0.0027/round against a 0.062 spread. The
perturber degraded outright, its format-valid rate falling from 1.00 to 0.62.

This document records what happened and which of three candidate explanations the evidence
supports, so the result can be cited without re-deriving it.

---

## 1. Self-play trajectory

| round | format-valid | r_P (all) | r_P (valid only) | verifier F1 | precision | recall |
|---|---|---|---|---|---|---|
| 0 | 1.00 | +0.229 | +0.229 | 0.808 | 0.875 | 0.771 |
| 1 | 1.00 | +0.104 | +0.104 | 0.908 | 0.958 | 0.896 |
| 2 | 1.00 | +0.146 | +0.146 | 0.832 | 0.865 | 0.854 |
| 3 | 1.00 | +0.229 | +0.229 | 0.817 | 0.917 | 0.771 |
| 4 | 1.00 | +0.167 | +0.167 | 0.867 | 0.917 | 0.833 |
| 5 | 1.00 | +0.167 | +0.167 | 0.833 | 0.854 | 0.833 |
| 6 | 1.00 | +0.229 | +0.229 | 0.784 | 0.854 | 0.771 |
| 7 | 1.00 | +0.042 | +0.042 | 0.975 | 1.000 | 0.958 |
| 8 | 0.75 | -1.146 | +0.139 | 0.911 | 1.000 | 0.861 |
| 9 | 0.62 | -1.750 | +0.200 | 0.800 | 1.000 | 0.700 |

`r_P = (1 − unit_recall) + penalties`, so with no penalties it is exactly `1 − recall`
(verified identical to four decimals in every round). The two agents' scores are therefore
not independent measurements.

## 2. Held-out evaluation

Every policy version scored on **one identical 80-item ProcessBench sample** (20 per subset
across gsm8k / math / olympiadbench / omnimath, seed 0). Round *i* is played by policy
v*(i+1)*.

| policy | error accuracy | correct accuracy | ProcessBench F1 |
|---|---|---|---|
| hverify-v1 | 0.778 | 0.977 | **0.866** |
| hverify-v2 | 0.778 | 0.932 | **0.848** |
| hverify-v3 | 0.778 | 0.955 | **0.857** |
| hverify-v4 | 0.806 | 0.955 | **0.874** |
| hverify-v5 | 0.750 | 0.977 | **0.849** |
| hverify-v6 | 0.722 | 0.955 | **0.822** |
| hverify-v7 | 0.694 | 0.977 | **0.812** |
| hverify-v8 | 0.778 | 0.932 | **0.848** |
| hverify-v9 | 0.778 | 0.977 | **0.866** |
| hverify-v10 | 0.722 | 0.977 | **0.831** |

v1 0.866 → v10 0.831 (**−0.036**); best v4 0.874, worst v7 0.812.

Self-play F1 swings between 0.784 and 0.975 across the same rounds while the held-out line
stays flat near 0.85. That divergence is the point of holding a fixed sample out.

---

## 3. Diagnosis

### The updater is the bottleneck, not the players

`hperturb-v10`'s policy contains this rule, written by Haiku from a single round of ~24
injected errors:

> *"Prioritize incomplete_solution as the primary stealth target (0% detection this round).
> Wrong_fraction is secondary (0% detection). … **Exclude** duplication_error — it showed
> 100% detection rate this round … Also exclude proportional_reasoning_error,
> wrong_operation, operand_swap, inverse_operation_error, geometry_definition,
> whole_number_bias, variable_misconception, negative_number_error, adding_across,
> wrong_sequence_term, decimal_magnitude, and additive_thinking."*

That is a blacklist of **13 of the 26 taxonomy entries**, inferred from per-type detection
rates with n ≈ 2 per type. It is precisely the overfitting `update-perturb` instructs
against. The effect is mechanical: round 9's error mix collapsed to `incomplete_solution:4,
inversion_error:4` because nearly everything else had been banned, and 3 of 8 episodes
failed schema validation trying to satisfy a policy demanding rare types under strict
diversity constraints.

The verifier updater failed symmetrically, accumulating a literal checklist of the error
types *this perturber* happened to favour — `a − b` vs `b − a`, `(−x)²` vs `−x²`,
numerator/denominator swaps. Across all ten rounds, `wrong_operation` (52), `inversion_error`
(51) and `negative_number_error` (31) were **60% of all 225 injected errors**, so the policy
tuned to a narrow slice of the error space and gained nothing on real annotated errors.

Neither updater ever meaningfully deleted a rule. Word-level diffs between consecutive
versions: +253, −13, +31, +1, +24, +68, −11, +12, +51. Policies grew from 90 to 1717
(perturber) and 2922 (verifier) characters while the rule count stayed pinned at 5 and 8.
Haiku appends; it does not revise.

### Verification capability was never the limit

On the fixed 80 items, comparing the cold-start policy against the final one:

| | v1 | v10 |
|---|---|---|
| silent on erroneous chains | 1/36 | 0/36 |
| false alarms on clean chains | 1/44 | 1/44 |
| identified the exact first-error step | 28 | 26 |
| flagged the wrong step | 7 | 10 |

Haiku almost always notices that something is wrong and rarely cries wolf. The entire F1
loss is **step localisation**, and ten rounds of policy tuning made it slightly worse.

### The problem sample was not reused

80 episodes drew 80 distinct problems, zero repeats, spread over difficulty Levels 1–5
(8/22/12/21/17). The overfitting is to the *error-type distribution*, not to the problems.
`omnimath` sat at 0.58–0.69 for every one of the ten versions — six policy revisions moved
the hardest subset by nothing.

---

## 4. Hypotheses assessed

| Hypothesis | Verdict |
|---|---|
| Haiku too weak to write policy updates | **Supported — primary cause.** Fits per-round noise at n ≈ 2/type, blacklists half the taxonomy, never deletes. |
| Overfitting to too few MATH examples | **Rejected as stated.** 80 problems, no repeats, Levels 1–5. The overfitting is to the error distribution (60% of errors in 3 of 26 types), not the problems. |
| Haiku too weak to perturb / verify | **Rejected for verification** (0/36 missed, 1/44 false alarms; loss is step localisation only). **Partly true for perturbation**: rounds 8–9 produced 2/8 and 3/8 schema failures under a policy demanding rare error types. |

---

## 5. Threats to validity

- **8 episodes/round.** One error unit moves mean reward by ~0.04, so per-round self-play
  deltas are near-noise. This is why the held-out sample carries the conclusion.
- **An earlier 20-item ProcessBench reading suggested improvement** (0.857 → 0.933) and was
  an artifact; quadrupling the sample erased it. Any claim from 20 items should be discarded.
- **Single run, single seed.** No error bars. The direction is consistent across two
  independent measures (held-out F1 flat-to-down, perturber format validity down), but the
  magnitude is not established.
- **Two infrastructure bugs corrupted earlier readings** before being fixed: session-limit
  errors scored as format failures (fabricated a whole 10-round table), and a bare
  `json.loads` in the ProcessBench scorer that discarded fenced replies and reported 0.000
  across the board. Both now fail loudly; see the git history.

---

## 6. What this does and does not show

It shows that **an adversarial self-play loop does not improve itself when the policy writer
is weak**, even when the players are competent — and that self-play reward can move
substantially (0.784 → 0.975) while held-out ability does not move at all.

It does **not** show that the loop is unsound. The natural next experiment separates the two
roles: keep Haiku as perturber and verifier, use a stronger model for the policy updates
only (`--updater-model`). If held-out F1 then moves, the updater was the bottleneck and the
architecture is fine.
