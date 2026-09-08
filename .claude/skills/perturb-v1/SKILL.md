---
name: perturb-v1
description: Perturber policy v1 (math mode) — inject exactly k realistic, independent misconception errors into a correct math solution and emit the ARAPPAV ground-truth JSON. Use when acting as the Perturber in a skill self-play rollout, or when asked to perturb a math solution with perturb-v1.
---

# Perturber — policy v1 (math mode)

You are the **Perturber** in an ARAPPAV self-play episode. Given a math problem, its
correct step-by-step solution, and a count `k`, you rewrite the solution so it contains
exactly `k` genuine mathematical errors, and you declare those errors in machine-readable
ground truth. A frozen or co-evolving **Verifier** then tries to find them.

- **version:** 1
- **parent:** — (seed policy)
- **tuned from rounds:** — (none yet)

---

## How to run one episode

1. Read `<round_dir>/episodes/<episode_id>/problem.json` → fields `problem`, `solution`, `k`.
2. Think about the perturbation, then write **only the JSON object** to
   `<round_dir>/episodes/<episode_id>/perturb.json`. Prose outside the JSON is tolerated
   by the parser (fences are stripped, the first JSON object wins) but adds nothing.
3. Do **not** read `verify.json`, `score.json`, or any other episode's files while
   perturbing. Do not edit any file under `.claude/skills/` — only `update-perturb` does that.

---

## Output contract — INVARIANT

This section mirrors `src/arappav/errors/schema_math.py` and
`src/arappav/models/perturber.py`. **Never relax it in a later version.**

```json
{
  "perturbed_solution": "<the full solution text, with all k errors in place>",
  "errors": [
    {
      "error_id": "err_001",
      "step_index": 0,
      "original_text": "<verbatim slice of the ORIGINAL solution>",
      "injected_text": "<the erroneous replacement, verbatim in perturbed_solution>",
      "error_type": "<exact value from the taxonomy below>",
      "rationale": "<why it is wrong and what the correct step is>"
    }
  ]
}
```

Hard requirements — violating any one costs the whole episode:

| # | Rule | Enforced by |
|---|------|-------------|
| 1 | Exactly `k` entries in `errors`, unique `error_id`s | `validate_math_perturber_output` |
| 2 | `error_type` is an **exact** taxonomy string (near-misses are fuzzy-matched, but do not rely on it) | `MathInjectedError.fuzzy_match_error_type` |
| 3 | `injected_text != original_text` — a declared error that changed nothing is a *phantom* | `injected_must_differ_from_original` |
| 4 | No **redundant restatement**: injected text must not repeat the same `\boxed{X}` twice, nor join two identical statements with "and". Restating a result is not a mathematical error | `injected_must_not_be_redundant_restatement` |
| 5 | `original_text` must appear **verbatim in the original solution**, and `injected_text` **verbatim in `perturbed_solution`** | mechanical backoff + `k_effective` |
| 6 | `perturbed_solution` must differ from the original solution | `parse_and_backoff` |
| 7 | Valid JSON. LaTeX backslashes are escaped as `\\` inside JSON strings — writing `"\boxed"` produces a *backspace character*, not LaTeX | `utils/parsing._repair_invalid_json_escapes` |

On rule 5: if `original_text` cannot be located, the mechanical backoff cannot insert the
error, the error is absent from the text, and it is counted as missing (`-0.5` each) while
still inflating `k`. Quote the source **exactly**, including `&`, `\\`, and `\quad`.

---

## How you are scored — INVARIANT

Computed by `arappav.reward.reward_fns.compute_rewards` (config: `configs/reward/reward.yaml`).

```
r_P = (1 − unit_recall) + penalties
unit_recall = detected error units / error units present in the text
```

| Component | Value | Trigger |
|-----------|-------|---------|
| Format penalty (hard) | **−10** | output is not parseable JSON |
| Format penalty (soft) | **−5** | JSON parses but fails schema (phantom, restatement, wrong `k`, bad enum) |
| Missing error | **−0.5** each | declared error absent from `perturbed_solution` |
| Intra-episode duplicate | **−1.0** each | two near-verbatim injections in one episode |
| Cross-round duplicate | **−5.0** each | injection ≥0.85 similar to one from an earlier round |
| Task reward | 0 … 1 | `1 − unit_recall` |

**Error units are the crux.** Before recall is computed, causally-linked errors collapse
into one unit (`matcher.group_errors_into_units`): overlapping or containing spans, two
rewrites of the same source region, the same changed fragment propagated across lines, a
`\boxed{}` answer corrupted downstream of another error, and token-Jaccard near-duplicates
(≥0.6). Declaring one mistake `k` times therefore yields **one** unit — the Verifier catches
it once and your recall term is 0. The only way to earn reward is `k` errors that are
**independent mistakes**, each of which the Verifier must find separately.

---

## Error taxonomy — INVARIANT (use these exact strings)

```
  - whole_number_bias: Treating fraction parts as independent whole numbers
  - adding_across: Adding numerators and denominators without common denominator
  - wrong_operation: Using incorrect arithmetic operation (e.g., + instead of ×)
  - operand_swap: Swapping dividend/divisor or numerator/denominator
  - incomplete_solution: Stopping before all solution steps are complete
  - denominator_only: Changing only denominator (or only numerator) incorrectly
  - duplication_error: Incorrectly duplicating operation across both parts
  - inversion_error: Inverting wrong operand or wrong part of expression
  - wrong_fraction: Computing fraction for wrong target or reference group
  - decimal_magnitude: Misunderstanding decimal magnitude (longer ≠ larger)
  - ignores_zeroes: Ignoring zero digits' place-value contribution
  - variable_misconception: Misunderstanding what a variable represents
  - additive_thinking: Using additive reasoning where multiplicative is needed
  - wrong_sequence_term: Computing wrong term in a sequence
  - first_term_as_coefficient: Using first output as coefficient directly
  - negative_number_error: Misapplying negative number arithmetic rules
  - tacking_signs: Ignoring signs during computation, re-adding at end
  - proportional_reasoning_error: Reversing or misapplying proportional relationships
  - inverse_operation_error: Applying wrong inverse operation
  - probability_scale: Thinking probability can exceed 1 or be negative
  - probability_certainty: Believing non-1 probability means certain event
  - base_rate_fallacy: Ignoring base rates in conditional reasoning
  - geometry_definition: Using incorrect definition of shape/property
  - angle_misconception: Misapplying angle formulas or relationships
  - irrelevant_feature: Reasoning from irrelevant problem features
  - unknowable: Incorrectly claiming insufficient information to solve
```

---

## Policy — v1

> **TUNED SECTION.** `update-perturb` rewrites everything between here and the changelog
> when it produces `perturb-v2`. Everything above stays fixed.

**P1 — Read the solution as a dependency graph.** List the solution's steps and what each
one feeds. Errors placed on *parallel* branches (two independent quantities, a setup line
and an unrelated later simplification) stay independent; errors stacked along one chain
collapse into a single unit.

**P2 — One mistake per step, at most one step per unit.** Pick `k` steps that are as far
apart in the derivation as possible. Never restate, duplicate, or re-word a mistake you
already made, and never declare a downstream consequence (a recomputed final answer, a
propagated term) as a separate error — it merges, and costs you the reward you were
reaching for.

**P3 — Make each error locally plausible.** The perturbed solution should read as a
competent student's work: correct notation, coherent prose, no arithmetic that is obviously
absurd (a negative length, a probability of 7). The error should be visible only to someone
who redoes the step. Prefer errors that produce a *plausible* number over ones that produce
a glaring one.

**P4 — Match the error type to the mathematics.** Choose the type from the actual
misconception you are modelling, not from what sounds impressive: fraction arithmetic →
fraction group, sign handling → `negative_number_error` / `tacking_signs`, scaling →
`proportional_reasoning_error`, and so on. Spread types across an episode and across rounds
— identical tricks repeated between rounds trigger the −5.0 duplicate penalty.

**P5 — Do not consume the whole line.** Keep `original_text` / `injected_text` tight around
the actual change (the expression, not the paragraph). Wide spans overlap other errors and
get merged, and they make the pair less informative as ground truth.

**P6 — Self-check before writing.** Verify, in order: exactly `k` errors; every
`original_text` found by literal search in the given solution; every `injected_text` found
by literal search in your `perturbed_solution`; no two errors touching the same region or
the same value; no phantom; no restatement; every `error_type` copied from the taxonomy
list; LaTeX backslashes doubled in the JSON.

### Worked shape (k=2, independent errors)

```json
{
  "perturbed_solution": "... \\Rightarrow \\quad -2a&=12 \\\\ \\Rightarrow \\quad a &= 8/12 ...",
  "errors": [
    {"error_id": "err_001", "step_index": 1,
     "original_text": "\\Rightarrow \\quad 8a&=12", "injected_text": "\\Rightarrow \\quad -2a&=12",
     "error_type": "wrong_operation",
     "rationale": "Subtracted 5a from 3a instead of adding it to both sides; 3a+5a=8a."},
    {"error_id": "err_002", "step_index": 2,
     "original_text": "a &= 12/8", "injected_text": "a &= 8/12",
     "error_type": "operand_swap",
     "rationale": "Dividend and divisor swapped when isolating a; should be 12/8."}
  ]
}
```

The two errors sit on different lines, change different quantities, and neither is the
consequence of the other — two units, so the Verifier must find both to zero your reward.

---

## Changelog

- **v1** — seed policy. Contract and reward facts transcribed from `schema_math.py`,
  `matcher.py`, `reward_fns.py`, and `configs/reward/reward.yaml`; strategy section written
  from first principles (no rollout evidence yet).
