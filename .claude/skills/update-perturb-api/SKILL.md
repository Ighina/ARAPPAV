---
name: update-perturb-api
description: Static updater that turns one scored API self-play round into the next Perturber prompt — reads the round's episodes and reward signal, diagnoses what worked, and writes prompts/perturb/<model>/v{N+1}/prompt.txt. Use after scoring an API round, or when asked to update the perturber prompt from rollout results.
---

# update-perturb-api — derive the next Perturber prompt

This skill is **static**: never versioned, never rewritten. It consumes one scored round of
API self-play and produces `prompts/perturb/<model_slug>/v{N+1}/prompt.txt`, plus a report.

It is the sibling of `update-perturb`, with one structural difference: the artefact is a
**self-contained prompt** sent to a model that has no repository, no tools and no memory.
Everything the Perturber needs must be *in the file*. A rule that says "see the taxonomy in
schema_math.py" is not a rule — it is a dangling pointer.

---

## Preconditions

1. `<round_dir>/round_summary.json` exists. Do not update from unscored episodes.
2. Check `manifest.freeze`. If it is `"perturber"`, write a report with `"skipped": true`,
   `"skip_reason": "perturber frozen"`, create no new version, and stop.
3. Determine the current version `N` — the highest `vN` under
   `prompts/perturb/<model_slug>/`. `manifest.perturb_skill` should end in `-v{N}`; if it
   does not, you are branching from the wrong parent. Say so and stop.

---

## Inputs

| File | What to take from it |
|------|----------------------|
| `<round_dir>/round_summary.json` | metrics, `deltas_vs_previous_round`, `error_type_detection`, `learning_signal` |
| `<round_dir>/episodes/*/score.json` | per-episode rewards, `match_details`, ground truth and claims |
| `<round_dir>/episodes/*/perturb_status.json` | format failures with the validator's message |
| `<round_dir>/episodes/*/perturb.json` | the **raw** reply — what the model actually emitted |
| `<round_dir>/api_logs/perturb_v{N}.json` | provider, stop reasons, token counts, failures, cost |
| `prompts/perturb/<model_slug>/v{N}/prompt.txt` | the prompt that produced all of it |

> **Never read the evaluation root** (`data/api_evals/…`, `data/skill_evals/…`). That is a
> test set. Selecting a policy edit on it is test-set fitting.

Read the raw `perturb.json` alongside the scores. The summary says *that* recall moved; only
the raw output shows whether the cause was strategy or a malformed reply.

---

## Step 1 — Diagnose

Bucket every episode and attach episode IDs. An observation with no episode ID behind it
does not enter the update.

- **Format failures** (`perturber_format_valid: false`). Read the validator message *and*
  the raw output. Which rule broke — phantom, restatement, wrong `k`, bad enum, unparseable
  JSON, unlocatable `original_text`? These cost −5 or −10 and are the cheapest wins.
- **Truncation and API failures** — check `api_logs` for `stop_reason: "max_tokens"` and for
  entries carrying `error`. A truncated reply is a prompt-length or `max_tokens` problem,
  not a strategy problem; never write a policy rule to fix one.
- **Wins** (`learning_signal.undetected_errors`). Which error types and which positions
  survived? A near-miss (`best_overlap` positive, the Verifier flagged the region but quoted
  it badly) is not the same as a clean miss (no claim anywhere near).
- **Losses** (`learning_signal.detected_errors`). What made them conspicuous?
- **Collapse** — compare `num_error_units` with `k` per episode. Units below `k` means the
  matcher merged declarations: the prompt is stacking, the failure this loop exists to
  prevent.
- **Penalties** — `duplicate_penalty`, `k_effective < k`.

## Step 2 — Filter out scorer exploitation

**This step is what keeps tuning from becoming reward hacking.** A high `r_P` is evidence of
a good prompt only if the reward was earned by producing a *genuinely harder error*. Ask:
would a competent mathematician reviewing this solution call the injected text a real
mistake, and a hard one?

Reject — and record under `scorer_issues`, never as a lesson — any win from:

- stacking one mistake as several declarations (root + propagated `\boxed{}`, overlapping
  rewrites, restatements);
- text that is redundant or odd but mathematically unchanged;
- exploiting the matcher rather than the mathematics: burying a small change in a very long
  `injected_text`, LaTeX escaping tricks, anchors a reader would never quote back;
- errors so severe or off-topic that they are unrealistic as student work.

Name the guard that should have caught it (`error_units`, a schema validator, an anti-*
penalty). Do not weaken the invariant region to accommodate it.

## Step 3 — Write the policy edits

- At most **5** edits per round. Each cites ≥1 episode; prefer ≥2 before promoting an
  observation to a rule.
- Edits are **general strategy**, never episode content. No problem statements, no specific
  numbers, no "for the sequence problem, do X".
- Prefer revising an existing rule to appending one. Keep the policy region under ~60 lines
  and readable end to end. Delete rules the evidence contradicts, and say so.
- **Write for a model with no context.** Self-contained sentences, no repo paths, no
  references to earlier rounds or to "the last version". The prompt is read cold, once, by a
  model that has never seen this loop.

## Step 4 — Create v{N+1}

```bash
SLUG=<model without the claude- prefix>
mkdir -p prompts/perturb/$SLUG/v$((N+1))
cp prompts/perturb/$SLUG/v$N/prompt.txt prompts/perturb/$SLUG/v$((N+1))/prompt.txt
```

Then edit **only the text between `<<<POLICY>>>` and `<<<END POLICY>>>`**. The prompt is
provider-agnostic: write policy that would hold for any model reading it cold, and never
name a provider, an SDK parameter or a model id inside it. Everything from
`<<<INVARIANT>>>` to `<<<END INVARIANT>>>` is a transcription of the code — if a round shows
the contract itself is wrong, that is a repo bug: report it, do not edit around it. Leave the
framing above and the `Return only the JSON object.` line below untouched.

Write `prompts/perturb/$SLUG/v$((N+1))/meta.json`:

```json
{"role": "perturb", "model": "<model id>", "version": 3, "parent": 2,
 "tuned_from_round": 2, "provenance": "update-perturb-api",
 "summary": "one line on what changed"}
```

## Step 5 — Verify

```bash
python scripts/selfplay_api.py check-prompt --model <model> --role perturb \
  --from-version $N --to-version $((N+1))
```

Must exit 0: the invariant region byte-identical, the policy region actually changed, and
`meta.json` present. Fix anything reported before writing the report.

## Step 6 — Decide convergence

Set `converged: true` when **either**:

- **No admissible edits** — every observation was already covered or was filtered out in
  step 2. An empty update *is* the signal; do not invent an edit to avoid one.
- **Metrics have flattened** — for two consecutive rounds,
  `|Δ mean_perturber_reward_valid_only| < 0.05`, `|Δ format_valid_rate| < 0.10`, and no new
  undetected-error pattern.

Otherwise `converged: false`, and say which axis is saturated.

## Step 7 — Write the report

`<round_dir>/update_perturb_api.json`:

```json
{
  "round": 2, "role": "perturber", "model": "claude-sonnet-5",
  "from_version": 2, "to_version": 3, "skipped": false, "skip_reason": null,
  "metrics": {"mean_perturber_reward_valid_only": 0.38, "format_valid_rate": 0.875,
              "delta_reward": 0.22, "delta_format_valid_rate": 0.125,
              "api_cost_usd": 0.42, "truncated_replies": 0, "api_errors": 0},
  "diagnosis": [{"observation": "…", "episodes": ["ep02"], "bucket": "win|loss|format|collapse|penalty|api"}],
  "policy_edits": [{"rule": "P2", "action": "revised|added|removed", "change": "…", "evidence": ["ep02", "ep05"]}],
  "scorer_issues": [{"episode": "ep04", "pattern": "…", "guard": "error_units", "action": "reported, not learned"}],
  "converged": false,
  "convergence_reason": "…"
}
```

Then report in prose: version created, the edits with their evidence, anything rejected as
exploitation, and the convergence verdict.
