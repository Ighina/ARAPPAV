---
name: update-verify
description: Turn one scored self-play round's findings into the next Verifier policy body, returning only the revised policy rules on stdout. Invoked by the pipeline orchestrator between rounds unless the verifier is frozen.
---

# update-verify

You revise the **Verifier policy** using the evidence from exactly one scored round.

The orchestrator passes you, inline:

- `CONTEXT FROM EARLIER ROUNDS` — the findings it has decided you are entitled to: round
  metrics, per-error-type detection rates, the errors that went undetected, and the claims
  that matched nothing. Under `--no-context` this block is absent and you revise from the
  current policy alone.
- `CURRENT POLICY` — the rules that produced those results.

**That is your entire input.** Do not read files, do not look for the run directory, and do
not use anything you recall from other runs. You have no tools.

## How to revise

1. **Separate the failure classes; they need different fixes.**
   - *Reasoning miss* — nothing was claimed anywhere near the error (`best_overlap` ≈ 0).
     The procedure did not surface it. Sharpen the re-derivation or extend the sweep.
   - *Quoting miss* — the right region was flagged but the span did not align
     (`best_overlap` positive but below threshold). Pure quote discipline, and the cheapest
     recall available.
   - *Reporting miss* — one quote covered two distinct errors, so only one could match.
     Fix with a one-claim-per-error rule, not with better reasoning.
2. **Read the false positives before adding claims.** Each is either a hallucination (require
   a failed re-derivation before claiming), a correct observation about something that was
   never injected (leave it alone — it is not an error in the policy), or a downstream
   consequence claimed as independent (a root-cause rule fixes it).
3. **Optimise F1, not recall.** If precision is the binding constraint, a rule that produces
   more claims makes the policy worse even though it raises recall. Check which side is
   actually losing before you edit.
4. **Edit, don't accrete.** At most five changes; prefer revising to appending; delete what
   the evidence contradicts. Keep the policy under ~60 lines.
5. **Stay general, and do not model the adversary.** Rules are verification procedure. Never
   write "errors are usually planted in the last line" or anything else derived from this
   particular Perturber's habits: it collapses the moment the Perturber changes, and it is
   the Verifier's characteristic way of overfitting.
6. **Write for a cold reader.** The next Verifier sees your rules once, with no history and
   no repository. Self-contained sentences, no file paths, no round references.

## What you must not do

- Do not restate or modify the output contract or the scoring rules. They sit above your
  section and are copied byte-for-byte by the orchestrator.
- Do not write a changelog. The orchestrator records provenance itself.

## Output

Return **only** the markdown body of the revised policy — the numbered rules and nothing
else. No heading, no preamble, no code fences, no commentary.
