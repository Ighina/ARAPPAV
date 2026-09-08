---
name: create-policy-verifier
description: Author the initial Verifier policy for a warm-start run from first principles, returning only the policy body on stdout. Invoked once by the pipeline orchestrator when start="warm".
---

# create-policy-verifier

You are writing the **initial strategy** for the Verifier in an ARAPPAV math self-play
experiment. This is a *warm start*: the orchestrator wants your prior knowledge of how to
audit a mathematical derivation, before any rollout evidence exists.

## The task the policy governs

Given a math problem and a solution that may contain injected errors, the Verifier finds
every genuine mistake, quotes it exactly, and explains it. It is not told how many errors
there are, or whether there are any. It emits
`{"claims": [{step_index, quoted_text, explanation, error_type}]}`, and `{"claims": []}` is a
real answer when the solution is correct.

It is scored on F1 against a hidden ground truth. Two facts shape everything:

- **Only the quote is matched.** A correct diagnosis whose span cannot be aligned to the
  injected text scores zero. Matching is by span overlap or containment, so quotes that are
  too wide or too narrow both fail.
- **Recall is over error *units*, precision over claims.** Flagging a value that is merely
  the downstream consequence of an earlier mistake earns nothing and can cost precision.

## What to write

Between 5 and 8 numbered rules of **general procedure** — how to work the problem, how to
check each step, which misconceptions to sweep for, how to distinguish a root cause from
propagation, how to quote, and how to calibrate how much to claim.

Constraints:

- General procedure only. Nothing about any particular problem, and nothing that assumes
  where errors tend to be planted — that is overfitting to one adversary and collapses when
  it changes.
- Do not restate the output contract or the scoring table; the skill file carries those
  above your section.

## Output

Return **only** the markdown body: the numbered rules, nothing else. No heading, no
preamble, no code fences, no commentary. The orchestrator splices your text directly into
the policy section of a skill file whose contract you must not touch.
