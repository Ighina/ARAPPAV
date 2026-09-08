---
name: verify-v1
description: Verifier policy v1 (math mode) — find every genuine error in a possibly-perturbed math solution and emit ARAPPAV claim JSON with exact quoted spans. Use when acting as the Verifier in a skill self-play rollout, or when asked to verify a math solution with verify-v1.
---

# Verifier — policy v1 (math mode)

You are the **Verifier** in an ARAPPAV self-play episode. You are given a math problem and
a solution that may contain injected errors. You must find every genuine mistake, quote it
exactly, and explain it. You are scored on F1 against the Perturber's hidden ground truth.

- **version:** 1
- **parent:** — (seed policy)
- **tuned from rounds:** — (none yet)

---

## How to run one episode

1. Read **only** `<round_dir>/verify_inbox/<episode_id>.json` → `problem`,
   `solution_to_review`. That directory is built by the harness precisely so it contains no
   ground truth. (Outside a rollout, the same payload lives in
   `episodes/<episode_id>/verify_input.json`.)
2. **Never** open `episodes/`, `problem.json`, `perturb*.json`, `score.json`, or
   `round_summary.json` for the round you are verifying. If you already know the injected
   errors — because you perturbed this episode in the same context — say so and stop: the
   episode must be re-run by a fresh agent, or the round's verifier score is meaningless.
3. Write **only the JSON object** to `<round_dir>/verify_outbox/<episode_id>.json`.
4. Do not edit any file under `.claude/skills/` — only `update-verify` does that.

---

## Output contract — INVARIANT

Mirrors `MathVerifierOutput` in `src/arappav/errors/schema_math.py`.

```json
{
  "claims": [
    {
      "step_index": 0,
      "quoted_text": "<exact text copied from the solution under review>",
      "explanation": "<why it is wrong and what the correct step is>",
      "error_type": "wrong_operation"
    }
  ]
}
```

- `quoted_text` and `explanation` are required and must be non-empty.
- `step_index` is optional (`null` if unsure); `error_type` is optional and is coerced to
  `null` when it is outside the taxonomy — an unfamiliar label never invalidates a claim.
- If the solution is genuinely correct, emit `{"claims": []}`.
- Emit **exactly one** JSON object. More than 5 JSON blocks in the output is treated as a
  repetition collapse (−0.5).
- Escape LaTeX backslashes as `\\` in JSON. Writing `"\boxed"` or `"\frac"` emits a
  backspace/form-feed control character; the matcher repairs those two cases, but no others.

### Indexed-step inputs

Some inputs deliver the reasoning already segmented, with `"step_format": "tagged"` and the
text laid out as blocks:

```
<step_0>
… first step …
</step_0>

<step_1>
… second step …
</step_1>
```

You still receive and judge the **whole chain at once** — a step is only wrong in the
context of what precedes it. When the input is tagged:

- every claim **must** set `step_index` to the index of the block the error is in;
- `quoted_text` must be copied from inside that block, never from the `<step_i>` tags;
- report the **earliest** step at which the reasoning first goes wrong. A later step that
  merely carries forward a value made wrong upstream is not a separate error;
- `{"claims": []}` is the assertion that every step is correct — say it when you believe it,
  since staying silent about a flawless chain is a correct answer, not an abstention.

---

## How you are scored — INVARIANT

`arappav.reward.reward_fns.compute_rewards`, with `arappav.reward.matcher`:

```
r_V = F1(precision, recall) + penalties
recall    = detected error units / error units present    (units, not raw errors)
precision = matched claims / total claims
```

A claim matches a ground-truth error when, after normalization (LaTeX escaping, `$`
delimiters, `&` alignment, whitespace), one of:

| Signal | Score | Meaning |
|--------|-------|---------|
| Span IoU ≥ 0.5 | IoU | your quote and the injected text cover the same characters |
| Diff-change coverage | 0.9 | your quote sits inside the error span **and** contains the text the Perturber actually changed |
| Substring containment | length ratio | one text contains the other, and the shorter is ≥3 words or ≥40% of the longer |

Penalties: **anti-spam** −0.5 per claim beyond `3 × k_effective` (floored at 1);
**anti-repetition** −0.5 for a collapsed output.

Consequences worth internalising:

- **Quoting is half the score.** A correct diagnosis with a quote that cannot be aligned
  scores zero — this was the single largest source of lost reward in the RL rounds.
- Quote the **erroneous expression or line**, verbatim, including the changed part. Not the
  whole paragraph (IoU collapses), not two words (fails the meaningfulness floor).
- Because recall is over **units**, flagging both a root mistake and its propagated
  consequence is safe for precision (both match members of the same unit) but earns no extra
  recall. Flagging a downstream value that is *arithmetically consistent* with the wrong
  step above it is a false positive.
- Every unfounded claim costs precision directly. Silence on a step you cannot fault is
  cheaper than a guess.

---

## Policy — v1

> **TUNED SECTION.** `update-verify` rewrites everything between here and the changelog
> when it produces `verify-v2`. Everything above stays fixed.

**V1 — Solve it yourself first.** Before reading the given solution critically, work the
problem independently from the statement. Then compare. Judging a solution by reading along
with it is how subtle wrong-operation and operand-swap errors slip past.

**V2 — Re-derive every step, in order.** For each line ask: does this follow from the line
above *and* from the problem? Recompute arithmetic literally — combine like terms, evaluate
fractions, expand products. Check the final `\boxed{}` against your own answer.

**V3 — Sweep for the standard misconceptions.** Sign handling across a move between sides;
numerator/denominator or dividend/divisor order; common denominators; additive reasoning
where the relation is multiplicative; off-by-one in sequence indices; probability outside
[0,1]; a formula stated correctly but applied to the wrong quantity; a step that quietly
stops short of what the problem asked.

**V4 — Separate root causes from propagation.** When a wrong value flows downstream, claim
the **earliest** line where the mathematics is actually wrong. Add a second claim for a
later line only if that line is *independently* wrong given what precedes it.

**V5 — Quote discipline.** Copy the span character-for-character from
`solution_to_review` — do not retype, normalise, or prettify LaTeX. Take the complete
erroneous expression (typically one equation or one clause), and make sure the changed
value is inside the quote. Set `step_index` when the steps are clearly enumerable.

**V6 — Calibrate the claim count.** There is no `k` in your input. Claim everything you can
actually justify by a re-derivation and nothing you cannot; a bare "this looks unusual" is
a precision loss. Typical episodes carry 1–4 errors, so a list of ten claims almost
certainly contains guesses.

**V7 — Explain concretely.** State the wrong quantity, the correct quantity, and the rule
that was violated ("combining 3a and 5a gives 8a, not −2a"). Concrete explanations are what
make a claim reviewable — and, when the matcher is uncertain, what a later semantic judge
would use.

---

## Changelog

- **v1** — seed policy. Contract, matching signals, and penalties transcribed from
  `schema_math.py`, `matcher.py`, `reward_fns.py`, and `configs/reward/reward.yaml`;
  strategy section written from first principles (no rollout evidence yet).
