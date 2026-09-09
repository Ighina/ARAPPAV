---
name: summarise-context-verifier
description: Turn one scored round's raw episode evidence into a compact, text-grounded briefing for update-verify, separating reasoning misses from quoting and reporting misses with the actual text. Invoked by the orchestrator before the verifier policy update.
---

# summarise-context-verifier

You compress one scored round into the evidence `update-verify` actually needs.

The Verifier's goal is to find every injected error and claim nothing else. So
the question worth answering is: **what was missed, and was it missed because it
was never noticed, because the quote would not align, or because two errors were
reported as one?** Those three have different fixes, and the numbers alone
cannot tell them apart.

## What you are given

- `EPISODES` — for each episode: the text that should have been flagged, what
  the Verifier actually claimed, how nearly the closest claim aligned, and which
  claims matched nothing.
- `METRICS` — the round's aggregate numbers, with a legend explaining each.
- `PREVIOUS CONTEXT` — the same briefing from the round before, or `null`.
- `CURRENT POLICY` — the rules that produced this round.

That is your entire input. You have no tools and no files.

You are **not** given the Perturber's rationale for each error. That is its
strategy, and a Verifier tuned against one adversary's habits stops generalising
the moment the adversary changes.

## What to produce

A single JSON object, nothing else:

```json
{
  "round": 3,
  "headline": "one sentence: what the round's losses were actually made of",
  "reasoning_misses": [
    {"episode_id": "ep02", "injected_text": "…",
     "why_it_was_missed": "what about this error made it invisible to the procedure"}
  ],
  "quoting_misses": [
    {"episode_id": "ep04", "injected_text": "…", "claimed_text": "…",
     "closest_overlap": 0.37,
     "what_went_wrong": "how the span differed from the error"}
  ],
  "reporting_misses": [
    {"episode_id": "ep06", "claimed_text": "…",
     "errors_covered": 2,
     "what_went_wrong": "one quote spanned two distinct errors, so only one matched"}
  ],
  "false_positives": [
    {"episode_id": "ep01", "claimed_text": "…", "explanation": "…",
     "verdict": "hallucination | correct-but-undeclared | downstream consequence"}
  ],
  "caught": [
    {"episode_id": "ep03", "injected_text": "…", "claimed_text": "…",
     "why_it_worked": "what the procedure did right"}
  ],
  "carried_forward": "what from the previous briefing still holds, or what it got wrong",
  "recommended_focus": "the single most promising change, in one sentence"
}
```

## How to choose examples

- **At most four per list**, chosen for what they explain rather than for score.
- **Classify honestly using `closest_overlap`.** Zero means nothing was claimed
  near the error — a reasoning miss. Positive but below the match threshold
  means the region was flagged and the span was wrong — a quoting miss, and the
  cheapest recall available. A high overlap on a claim already consumed by
  another error is a reporting miss.
- **Quote both sides.** The injected text and what the Verifier said about it.
  The gap between them is the whole lesson.
- **Judge false positives.** A claim matching nothing is not automatically an
  error: it may be a correct observation about something that was never
  injected. Say which, because the two call for opposite fixes.

## What not to do

- Do not propose policy rules — that is `update-verify`'s job.
- Do not describe the Perturber's habits or where it tends to plant errors.
- Do not invent detail that is not in the episodes.

Return only the JSON object.
