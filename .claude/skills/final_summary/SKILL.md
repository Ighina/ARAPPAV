---
name: final_summary
description: Produce the final experiment report for a completed self-play run from the persisted round data supplied inline, including a round-by-round summary table. Invoked once by the pipeline orchestrator at the end of a run.
---

# final_summary

Write the final report for a completed ARAPPAV self-play experiment.

The orchestrator passes you a JSON object (`RUN DATA`) containing the run configuration,
every round summary, and the findings recorded at each round, plus a `PRECOMPUTED ROUND
TABLE` generated deterministically from the same data.

**That JSON is your only source.** Do not read files, do not infer numbers that are not
present, and do not use anything you remember from elsewhere. If a value is missing, say it
is missing rather than estimating it.

## Required contents

Produce markdown with these sections:

1. **Configuration** — dataset and source, episodes per round, number of rounds, `k`,
   initialization mode (`cold`/`warm`), freeze configuration, whether context was passed to
   agents, model, seed.
2. **Round-by-round table** — reproduce the supplied `PRECOMPUTED ROUND TABLE` verbatim.
   Do not recompute or reformat its numbers. One row per round.
3. **Policy evolution** — perturber and verifier separately: which version ran in each
   round, which rounds updated it, and which were frozen.
4. **Rewards and penalties by round** — perturber reward, verifier reward, recall,
   precision, and every penalty term, with the direction of movement between rounds.
5. **ProcessBench** — per-round results when it was run. When it was disabled, state
   explicitly that ProcessBench was not run and that no external-validity claim can be made.
6. **Findings by round** — what each round's evidence showed: which error types survived,
   which were detected, format failures, unit collapse, false positives.
7. **Final conclusions** — what the run established, stated no more strongly than the sample
   size supports. Name the number of episodes behind any trend you assert.
8. **Anomalies and failures** — format-invalid episodes, agent errors, empty policies,
   frozen sides, missing ProcessBench, or any round whose metrics look inconsistent.

## Style

- Be concrete and quantitative; cite round numbers and episode counts.
- Distinguish what was measured from what it implies. A two-round move in a single metric
  over a handful of episodes is not a trend, and you should say so when that is the case.
- Do not recommend policy edits. This is a report on what happened.

## Output

Return the markdown report on stdout and nothing else — no preamble, no fences around the
whole document, no commentary about writing it.
