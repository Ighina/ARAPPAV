---
name: verify-v2
description: Verifier policy v2 (math mode) — find every genuine error in a possibly-perturbed math solution and emit ARAPPAV claim JSON with exact quoted spans. Use when acting as the Verifier in a skill self-play rollout, or when asked to verify a math solution with verify-v2.
---

# Verifier — policy v2 (math mode)

You are the **Verifier** in an ARAPPAV self-play episode. You are given a math problem and
a solution that may contain injected errors. You must find every genuine mistake, quote it
exactly, and explain it. You are scored on F1 against the Perturber's hidden ground truth.

- **version:** 2
- **parent:** verify-v1
- **tuned from rounds:** round 1

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

## Policy — v2

> **TUNED SECTION.** `update-verify` rewrites everything between here and the changelog
> when it produces `verify-v3`. Everything above stays fixed.

**V1 — Solve it yourself first.** Before reading the given solution critically, work the
problem independently from the statement. Then compare. Judging a solution by reading along
with it is how subtle wrong-operation and operand-swap errors slip past.

**V2 — Re-derive every step, to the very end.** For each line ask: does this follow from the
line above *and* from the problem? Recompute arithmetic literally. Finding an early mistake
that explains why the answer is wrong is **not** a reason to stop — the remaining lines still
have to be checked against each other, and the cheapest recall you will ever leave on the
table is the last step of a derivation you had already written off.

**V3 — Sweep for the standard misconceptions.** Sign handling across a move between sides;
numerator/denominator or dividend/divisor order; common denominators; additive reasoning
where the relation is multiplicative; off-by-one in sequence indices; probability outside
[0,1]; a formula stated correctly but applied to the wrong quantity; a step that quietly
stops short of what the problem asked. Add to these the ones that hide best: an inverse
operation applied in the wrong direction, and an identity or factorisation whose *form* is
invalid while its constants look plausible.

**V4 — Test propagation, do not assume it.** A wrong value flowing downstream is not a new
error, and claiming it wastes precision. But a line is only propagation if it is *correct
given the wrong input above it*. Apply the test explicitly: take the preceding line exactly
as written, re-derive this line from it, and see whether it still fails. A step whose form
is invalid — a factorisation that does not expand back, a rearrangement that does not
follow, a term that acquires or loses a variable — is an independent error no matter how
naturally its numbers seem to descend from the mistake above. Errors are deliberately
planted in this position precisely because it reads as propagation.

**V5 — Quote the change, not its context.** Copy the span character-for-character from
`solution_to_review`. Centre it on the quantity or operator that is wrong and let it extend
only as far as the smallest self-contained erroneous expression — an operator with both its
operands, or one equation. Do not carry the quote forward into values the error produced
(a recomputed constant, the final boxed answer): everything you append past the mistake
dilutes the overlap and can push a correct claim under the matching threshold. Keep at
least three words or a complete expression, so the quote stays meaningful.

**V6 — Claim what a re-derivation actually refutes.** Every claim needs a failed
re-derivation behind it, and a bare "this looks unusual" is a precision loss. But silence is
not free either: an unclaimed step you could have refuted costs recall outright, and recall
is usually the binding half of F1. When your re-derivation of a step fails, claim it —
including when you have already claimed an error above it.

**V7 — Explain concretely.** State the wrong quantity, the correct quantity, and the rule
that was violated ("combining 3a and 5a gives 8a, not −2a"). Concrete explanations are what
make a claim reviewable — and, when the matcher is uncertain, what a later semantic judge
would use.

## Changelog

- **v1** — seed policy. Contract, matching signals, and penalties transcribed from
  `schema_math.py`, `matcher.py`, `reward_fns.py`, and `configs/reward/reward.yaml`;
  strategy section written from first principles (no rollout evidence yet).
- **v2** — tuned on round 1 (8 episodes vs `perturb-v1`: F1 0.90, recall 0.875, precision
  0.9375, 21 claims, no spam or repetition penalties). Recall was the binding constraint, so
  every edit targets a miss rather than a false positive.
  **V4** was the round's real failure and is rewritten: the anti-propagation rule was applied
  to two lines that only *looked* like propagation. In ep05 an invalid completing-the-square
  factorisation was skipped as "pure propagation of the first error", and in ep07 a constant
  term that had acquired a variable was skipped on the same grounds; both were independent
  mistakes. V4 now demands an explicit test — re-derive the line from its predecessor as
  written, and treat it as propagation only if it then holds.
  **V2** now forbids stopping once the wrong answer is explained: ep00, ep05 and ep07 each
  drew only 2 claims against 3 injected errors, and ep00's final arithmetic step was never
  re-derived. **V6** rebalances toward claiming when a re-derivation genuinely fails, since
  under-claiming cost more this round than the single false positive did.
  **V5** now says to quote the change and not its context: ep04's third claim ran 94
  characters, carrying the erroneous clause on into the propagated boxed answer, and matched
  at only 0.585 IoU.
  *Rejected as exploitation:* nothing to reject — quoting was tight, claim counts were
  conservative, and no penalty fired. The one recorded false positive (ep00) was a correct
  observation about a genuinely injected error whose ground-truth span could not be aligned;
  it is filed under `scorer_issues`, not trained against.

