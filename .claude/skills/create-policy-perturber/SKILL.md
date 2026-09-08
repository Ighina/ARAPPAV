---
name: create-policy-perturber
description: Author the initial Perturber policy for a warm-start run from first principles, returning only the policy body on stdout. Invoked once by the pipeline orchestrator when start="warm".
---

# create-policy-perturber

You are writing the **initial strategy** for the Perturber in an ARAPPAV math self-play
experiment. This is a *warm start*: the orchestrator wants your prior knowledge of how
plausible mathematical mistakes are made, before any rollout evidence exists.

## The task the policy governs

Given a math problem and a correct step-by-step solution, the Perturber rewrites **the
solution** so it contains exactly `k` genuine, independent mathematical errors, and declares
each one as `{original_text, injected_text, error_type, rationale}`. The problem statement is
read-only context and is never modified.

It is scored as `r_P = (1 - unit_recall) + penalties`, so it is rewarded for errors a careful
Verifier *fails* to find. Two facts shape everything:

- **Errors that are causally linked collapse into one unit.** Overlapping spans, token
  near-duplicates, a shared changed fragment, or an error that changes a `\boxed{}` value
  (which merges into the nearest earlier error) all count once. Only genuinely independent
  mistakes earn separate credit.
- **Declared spans must be findable.** `original_text` must appear verbatim in the given
  solution and `injected_text` verbatim in the rewritten one.

## What to write

Between 4 and 7 numbered rules of **general strategy** — where to place errors, how to keep
them independent, how to choose a misconception, how to anchor spans, and what to check
before answering. Write for a model that will read your rules cold, once, with no other
context.

Constraints:

- General strategy only. No worked examples, no specific numbers, no reference to any
  particular problem.
- Do not restate the output contract, the taxonomy, or the scoring table — the skill file
  already carries those above your section, and repeating them wastes the budget.
- Errors must be realistic student mistakes. A policy that wins by being absurd, by
  restating one mistake several times, or by gaming span matching is a bad policy.

## Output

Return **only** the markdown body: the numbered rules, nothing else. No heading, no
preamble, no code fences, no commentary. The orchestrator splices your text directly into
the policy section of a skill file whose contract you must not touch.
