---
name: update-perturb
description: Turn one scored self-play round's findings into the next Perturber policy body, returning only the revised policy rules on stdout. Invoked by the pipeline orchestrator between rounds unless the perturber is frozen.
---

# update-perturb

You revise the **Perturber policy** using the evidence from exactly one scored round.

The orchestrator passes you, inline:

- `CONTEXT FROM EARLIER ROUNDS` — the evidence it has decided you are entitled to.
  Under `--rich-context` this is a **briefing** produced by
  `summarise-context-perturber`: which injected errors survived and which were
  caught, each quoting **the actual text** of the original and the injection,
  with a concrete reading of what made the difference, plus format failures, unit
  collapse, and what carried forward from the previous round. Otherwise it is the
  older aggregate: metrics, per-error-type detection rates, and identifiers.
  Under `--no-context` the block is absent and you revise from the current policy
  alone.
- `CURRENT POLICY` — the rules that produced those results.

Reason from the **text**, not the labels. `error_type` is a taxonomy tag that
plays no part in scoring, so a rule of the form "prefer type X, avoid type Y"
derived from one round's detection rates is fitting noise over a handful of
samples — and it narrows what the Perturber will even attempt. What survived and
what was caught, as text, is the evidence.

**That is your entire input.** Do not read files, do not look for the run directory, and do
not use anything you recall from other runs. You have no tools; asking for more input is not
an option, so work with what is given.

## How to revise

1. **Diagnose from the findings.** `undetected_errors` are wins — which error types and
   which placements survived? `detected_errors` are losses — what made them conspicuous?
   `unit_collapse` (units below `k`) means declarations were merged: the policy is stacking
   one mistake as several, which earns nothing. `format_failures` are the cheapest fix of
   all, since each costs a flat penalty.
2. **Filter out anything that is not a real win.** A high reward counts only if the injected
   text is a mistake a competent mathematician would call real, and hard. Discard — do not
   learn from — reward that came from stacking one error as several, from text that is odd
   but mathematically unchanged, from span-matching games, or from errors too absurd to be
   student work.
3. **Edit, don't accrete.** At most five changes. Prefer revising an existing rule to adding
   one; delete rules the evidence contradicts. Keep the whole policy under ~60 lines and
   readable end to end.
4. **Stay general.** Rules are strategy, never episode content: no problem statements, no
   specific numbers, no "for the sequence problem, do X". A rule that only fires on this
   round's sample is memorisation.
5. **Write for a cold reader.** The next Perturber sees your rules once, with no history and
   no repository. Self-contained sentences; no file paths, no round references, and nothing
   about how the Verifier behaves — modelling one particular opponent is how a policy
   overfits.

## What you must not do

- Do not restate or modify the output contract, the taxonomy, or the scoring rules. They sit
  above your section in the skill file and are copied byte-for-byte by the orchestrator; you
  cannot change them, and repeating them wastes the policy budget.
- Do not write a changelog. The orchestrator records provenance itself.

## Output

Return **only** the markdown body of the revised policy — the numbered rules and nothing
else. No heading, no preamble, no code fences, no commentary.
