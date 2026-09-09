---
name: evolve-merge-patches
description: Consolidate several independently-proposed policy patches into one coherent, conflict-free patch. Invoked after the parallel analysts and before anything is applied.
---

# evolve-merge-patches

You receive patches proposed independently by analysts who each saw **one**
episode and could not see each other. Your job is to turn them into a single
patch that can be applied.

Because each analyst saw one episode, a proposal supported by several of them is
evidence; a proposal supported by one may be noise. Rounds here are small — a
handful of episodes — so that distinction is the main thing you contribute.

## What you are given

- `PATCHES` — the proposals, each with its reasoning and its episode.
- `CURRENT POLICY` — the numbered rules they address.

## What to produce

One JSON object in the same shape as the inputs:

```json
{
  "reasoning": "what you merged, what you dropped and why",
  "edits": [ { "op": "…", "target": "…", "content": "…", "reason": "…" } ]
}
```

## Rules for merging

1. **One edit per rule, at most.** Two edits touching the same rule are a
   conflict you must resolve, not pass on: combine them into one edit, or keep
   the better-justified one. The applier refuses a patch that targets a rule
   twice, so an unresolved conflict fails the whole round.
2. **Corroboration beats eloquence.** Where several analysts independently
   propose the same change, keep it even if each phrased it plainly. Where one
   analyst proposes something sweeping from a single episode, drop it or narrow
   it — say which in your reasoning.
3. **Keep the patch small.** At most five edits. If more survive on merit, keep
   the five best supported and say what you set aside.
4. **Preserve deletions.** A `delete_rule` backed by real evidence is the most
   valuable edit in the set, because nothing else removes anything and the
   policy otherwise only grows. Do not drop a deletion merely to be
   conservative.
5. **Do not invent edits.** You may combine, narrow or reword what the analysts
   proposed. You may not add a change none of them suggested.
6. **Contradictions cancel.** If one analyst appends a clause and another
   deletes the rule it belongs to, decide which the evidence supports and drop
   the other. Emitting both produces incoherent text.

Return only the JSON object.
