---
name: perturb-v4
description: Perturber policy v4 (math mode) — inject exactly k realistic, independent misconception errors into a correct math solution and emit the ARAPPAV ground-truth JSON. Use when acting as the Perturber in a skill self-play rollout, or when asked to perturb a math solution with perturb-v4.
---

# Perturber — policy v2 (math mode)

You are the **Perturber** in an ARAPPAV self-play episode. Given a math problem, its
correct step-by-step solution, and a count `k`, you rewrite the solution so it contains
exactly `k` genuine mathematical errors, and you declare those errors in machine-readable
ground truth. A frozen or co-evolving **Verifier** then tries to find them.

- **version:** 4
- **parent:** perturb-v3
- **tuned from rounds:** rounds 1-3

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

## Policy — v4

> **TUNED SECTION.** `update-perturb` rewrites everything between here and the changelog
> when it produces `perturb-v5`. Everything above stays fixed.

**P1 — Budget `k` against the solution's shape before writing anything.** Count the
derivation steps that are *not* the final answer line, and note whether the solution has
parallel branches: two independent quantities, a case split, or two solution methods. `k`
independent units need `k` such positions. A solution whose whole derivation is one line has
one position and cannot carry more, whatever you do to it — the surplus errors end up in the
same expression, a single quote covers them, and they merge or go unmatched together. When
branches exist, spend one error per branch first: they cannot merge, because nothing links
them.

**P2 — Never spend a declared error on the `\boxed{}` line.** An error that changes the
boxed value merges with the nearest earlier declared error **mechanically**, whether or not
it is independently wrong. Unless it is the *earliest* error you declare, the final answer
is worth zero as a declaration. Let it follow silently from the errors above it: propagate
the value through the text, do not list it in `errors`.

**P3 — Errors that merely *look* like propagation no longer survive.** Placing an
independently-invalid step where it reads as the mechanical consequence of an earlier error
worked once, against a Verifier that skipped such lines on sight. Against one that re-derives
each line from its predecessor as written, it fails: the step is checked precisely because it
looks routine. Keep the placement in the repertoire — it costs nothing and still catches a
careless reader — but do not spend your hardest error there, and do not expect it to carry an
episode.

**P4 — Attack the definitions, not the arithmetic. This is the primary strategy.** Any error a reader can reach by
recomputing a step will be found: sign flips, swapped operands, off-by-one indices, invalid
factorisations, dropped factors and mis-scaled fractions were caught essentially without
exception. What survives is an error in what the symbols *mean* — which quantity a variable
denotes, what a ratio compares, which of two things the problem asked for. Every line after
such an error is arithmetically consistent with it, so re-derivation reports nothing; only a
reader who checks the model against the problem statement catches it.

**P4b — Put the linked pair where one quote covers both.** Two errors that are causally
independent (so the matcher keeps them as separate units) but *thematically* the same mistake
tend to be written up as a single observation. This only pays when a reader's natural quote
spans both: the same misevaluated term on the two sides of one equation, two coordinates of
one point, two factors inside one radicand. Spread the pair across separate sentences and a
careful Verifier simply files two claims and catches both. Keep them inside one expression or
one line.

**P5 — Anchor spans on complete mathematical expressions, and stop at the mathematics.**
`original_text` and `injected_text` must be a whole equation or clause, both sides included.
Do not straddle prose and mathematics, do not stop before the right-hand side, and do not let
the span run past the expression into the sentence punctuation that follows it — a trailing
period or bracket is not part of the mistake, and no reader quoting the error will include
it. An anchor a reader would never reproduce verbatim defeats the matcher's containment test
and scores a found error as missed, which is a false win.

**P6 — One mistake per step, and never restate.** Pick steps as far apart in the derivation
as possible. Do not declare a downstream consequence, a re-worded version, or a second
rewrite of a region you already changed — all of them merge and cost you the reward you were
reaching for.

**P7 — Make each error locally plausible.** The perturbed solution should read as a
competent student's work: correct notation, coherent prose, no absurd quantities. Prefer an
error that produces a plausible number over one that produces a glaring one, and keep the
text internally consistent downstream of every error you inject.

**P8 — Self-check before writing.** In order: exactly `k` errors; **every declared span is
false as written** — re-derive each one from the line above it and confirm it actually fails,
rather than merely being the place a later error originates, because a valid equality declared
as an error is unfindable and the Verifier is charged a false positive for flagging the real
step instead; no declared error changes
a `\boxed{}` value unless it is the first; every `original_text` found by literal search in
the given solution; every `injected_text` found by literal search in your
`perturbed_solution`; every span a complete expression ending at the mathematics, not at sentence punctuation; no two errors touching the same
region or the same value; no phantom; no restatement; every `error_type` copied from the
taxonomy; LaTeX backslashes doubled in the JSON.

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
- **v2** — tuned on round 1 (8 episodes, k=3, vs `verify-v1`: mean `r_P` 0.125, format-valid
  8/8, unit recall 0.875). Rewrote the placement rules around what actually survived.
  **P1** now budgets `k` against the number of non-terminal steps and prefers parallel
  branches: ep02 and ep07 were two-line solutions and both collapsed 3 declared errors into
  2 units, while ep06's two independent solution methods carried 3 units cleanly.
  **P2** (new) forbids declaring the `\boxed{}` line at all: ep07's err_003 was a genuinely
  independent mistake and merged anyway, because any boxed-changing error unions with the
  nearest earlier one. **P3** (new) is the round's one real win — ep05's err_002 wrote an
  invalid completing-the-square factorisation at a step that looked like propagation of the
  error above it, and the Verifier skipped it by its own anti-propagation rule.
  **P4** was rewritten from the detection table: sign flips, operand swaps and index
  off-by-ones were caught 10/10, while `inverse_operation_error` went 0/2 (ep00 err_003,
  ep05 err_002). **P5** now requires whole-expression anchors — ep00's err_002 anchored
  `solve $2x - 4`, straddling the verb and omitting the right-hand side, and the Verifier's
  correct claim `$2x - 4 = 20$` scored IoU 0.368 and was recorded as a miss.
  *Rejected as exploitation:* that ep00 win is not a policy lesson — it is a span-alignment
  artifact, filed under `scorer_issues`, and P5 deliberately gives it up.
- **v3** — tuned on round 2 (8 episodes, k=3, vs `verify-v2`). **P1 and P2 are confirmed and
  unchanged**: every episode produced 3 distinct error units, 24/24 against 22/24 in round 1,
  with no boxed-line declarations and no collapse anywhere.
  **P3 is demoted.** The disguise-as-propagation placement was the headline of v2 on the
  strength of one survivor; this round six such placements were made (ep01, ep02, ep03, ep05,
  ep06, ep07) and all six were detected. `verify-v2`'s explicit propagation test — re-derive
  the line from its predecessor as written — is exactly the counter, so the rule is kept only
  as a cheap extra and no longer as the primary strategy.
  **P4 is rewritten and P4b added.** The one error that genuinely survived was ep06's
  redefinition of the unknown as its complement: a definitional error, not an arithmetic one,
  so every later line stayed consistent with it and re-derivation reported nothing. It also
  went unclaimed because the Verifier folded it into its claim about the related ratio
  mislabel in the same episode, describing both as one observation — hence P4b.
  **P5** now forbids trailing sentence punctuation in anchors.
  *Rejected as exploitation:* three of this round's four apparent wins (ep00 err_003, ep05
  err_001, ep07 err_002) were span-alignment artifacts, not survivals — the Verifier
  diagnosed all three correctly and in detail. They are filed under `scorer_issues`, and the
  P5 edit deliberately gives that reward back. Corrected for them, the Perturber scored
  0.042, worse than round 1's 0.125, and the Verifier 0.958 rather than the recorded 0.833.
- **v4** — tuned on round 3 (8 episodes, k=3, vs `verify-v3`: mean `r_P` 0.3125, the highest
  of the run, from 0.167). 23/24 units, the only collapse being a one-line solution.
  **P4 is confirmed as the primary strategy.** ep03's two definitional errors — both
  misreadings of the problem's stopping rule, leaving every later line arithmetically
  untouched — drew no claim anywhere in the episode, the cleanest misses of the whole run,
  and this against a Verifier that had just added an explicit definitions pass.
  `variable_misconception` detection fell to 0.643 across 14 instances.
  **P4b is narrowed by its own failure case.** The linked pair paid three times (ep00, ep01,
  ep02) and every time the two errors sat inside a single expression that one natural quote
  covered. In ep04 the same trick was spread across two separate sentences, and the Verifier's
  new V8 rule filed two anchored claims and took 3/3. The rule now says explicitly to keep the
  pair inside one expression or line.
  **P1** gains the one-line case: a solution with no intermediate steps has one position, and
  ep06's surplus errors landed in a single expression and were quoted together.
  **P8** now requires that every declared span be false as written.
  *Rejected as exploitation:* ep00 err_003 was not a win. It declared an error on a regrouping
  that is a valid equality given the line above it; the real mistake is the extraction that
  follows. The Verifier said so, flagged the extraction, and was charged a false positive for
  being right. That is a defect in the perturbation, fixed in P8, not a lesson. ep06's two
  tail errors were also discounted: a single claim covered both at 0.441 and 0.467 overlap,
  just under the matching threshold.

