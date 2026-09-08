---
name: update-verify-api
description: Static updater that turns one scored API self-play round into the next Verifier prompt — reads the round's episodes and reward signal, diagnoses misses and false positives, and writes prompts/verify/<model>/v{N+1}/prompt.txt. Use after scoring an API round, or when asked to update the verifier prompt from rollout results.
---

# update-verify-api — derive the next Verifier prompt

This skill is **static**: never versioned, never rewritten. It consumes one scored round of
API self-play and produces `prompts/verify/<model_slug>/v{N+1}/prompt.txt`, plus a report.

It is the sibling of `update-verify`, with one structural difference: the artefact is a
**self-contained prompt** sent to a model that has no repository, no tools and no memory.
Every rule must stand on its own in the file — a reference to a repo path or to "the
previous round" is a dangling pointer to a reader who has neither.

---

## Preconditions

1. `<round_dir>/round_summary.json` exists.
2. Check `manifest.freeze`. If it is `"verifier"`, write a report with `"skipped": true`,
   `"skip_reason": "verifier frozen"`, create no new version, and stop.
3. Determine the current version `N` — the highest `vN` under
   `prompts/verify/<model_slug>/`. It must match the round's `manifest.verify_skill`
   suffix; if it does not, stop and say so.

---

## Inputs

| File | What to take from it |
|------|----------------------|
| `<round_dir>/round_summary.json` | recall / precision / f1, deltas, `error_type_detection`, `learning_signal` |
| `<round_dir>/episodes/*/score.json` | per-claim outcomes, `match_details` with `best_overlap` and `all_overlaps` |
| `<round_dir>/verify_outbox/*.json` | the **raw** replies — the actual quoting behaviour |
| `<round_dir>/api_logs/verify_v{N}.json` | provider, stop reasons, token counts, failures, cost |
| `prompts/verify/<model_slug>/v{N}/prompt.txt` | the prompt that produced all of it |

> **Never read the evaluation root** (`data/api_evals/…`, `data/skill_evals/…`). It is a test
> set for this exact prompt. Tuning on it — even by choosing which edit to keep — means the
> benchmark stops measuring generalisation.

Read the raw outbox replies alongside the scores. Recall and precision say what happened;
only the raw claims show whether the cause was reasoning or quoting.

---

## Step 1 — Diagnose, separating the failure classes

For every undetected ground-truth error, classify. These drive completely different edits:

- **Reasoning miss** — no claim anywhere near it (`best_overlap ≈ 0`, nothing plausible in
  `closest_claims`). The Verifier never noticed. Fix: sharpen the re-derivation procedure or
  add the misconception to the sweep.
- **Quoting miss** — the right region was flagged but the span could not be aligned
  (`best_overlap` positive but under 0.5, or a `closest_claims` entry whose explanation is
  right). Pure quote discipline, and the cheapest recall available.
- **Reporting miss** — the reply's prose describes the error but no claim is anchored at it,
  or one quote spans two ground-truth errors and only one could match (assignment is
  one-to-one). Fix with a one-claim-per-error rule, not with better reasoning.
- **Structural miss** — the error was inside a unit already matched through another member,
  or was never in the text (`k_effective < k`). No verifier fix applies.

Then, for precision:

- **False positives**. For each, decide: a hallucination (require re-derivation before
  claiming), a correct observation about something the Perturber never declared (record it,
  do not train against it), or a downstream consequence claimed as independent (root-cause
  rule).
- **Penalties** — any nonzero `spam_penalty` or `repetition_penalty` is a format problem,
  fixable outright.
- **Truncation and API failures** — check `api_logs` for `stop_reason: "max_tokens"` and
  `error` entries. A truncated reply loses claims for mechanical reasons; fix `max_tokens`
  or prompt length, never with a policy rule.

Attach episode IDs to every observation.

## Step 2 — Filter out scorer exploitation

A high `r_V` counts only if the Verifier actually *found* the mistakes. Reject — and record
under `scorer_issues`, not as a lesson — anything that raises the score without improving
detection:

- quoting long spans, whole lines, or the entire solution to maximise overlap;
- shotgunning claims to raise recall (if precision and anti-spam did not punish it, the
  guard is miscalibrated — say so);
- boilerplate explanations attached to spans copied from suspicious-looking LaTeX rather
  than from a re-derivation;
- exploiting the Perturber's habits — "the last error is usually in the final line" — rather
  than the mathematics. That collapses the moment the Perturber's prompt changes.

If matching itself is the problem — correct diagnoses that cannot be aligned no matter how
carefully quoted — that is a matcher bug for `src/arappav/reward/matcher.py`. Report it; do
not paper over it in the policy, and never weaken the invariant region.

## Step 3 — Write the policy edits

- At most **5** edits per round; each cites ≥1 episode, preferably ≥2.
- General procedure, never episode content. No problem statements, no remembered answers,
  no "expect an operand swap in step 2".
- Prefer revising an existing rule to appending one; keep the policy region under ~60 lines.
  Delete rules the evidence contradicts and say so.
- Weight edits by what the numbers say: if precision is the binding constraint, a rule that
  adds claims makes the prompt worse even when it raises recall. Optimise F1.
- **Write for a model with no context.** Self-contained, no repo paths, no cross-round
  references. The prompt is read cold, once.

## Step 4 — Create v{N+1}

```bash
SLUG=<model without the claude- prefix>
mkdir -p prompts/verify/$SLUG/v$((N+1))
cp prompts/verify/$SLUG/v$N/prompt.txt prompts/verify/$SLUG/v$((N+1))/prompt.txt
```

Then edit **only the text between `<<<POLICY>>>` and `<<<END POLICY>>>`**. The prompt is
provider-agnostic: write policy that would hold for any model reading it cold, and never
name a provider, an SDK parameter or a model id inside it. The region from
`<<<INVARIANT>>>` to `<<<END INVARIANT>>>` transcribes the output contract, the matching
signals and the penalties; if the contract itself is wrong that is a repo bug to report, not
to edit around. Leave the framing above and the `Return only the JSON object.` line below
untouched.

Write `prompts/verify/$SLUG/v$((N+1))/meta.json`:

```json
{"role": "verify", "model": "<model id>", "version": 3, "parent": 2,
 "tuned_from_round": 2, "provenance": "update-verify-api",
 "summary": "one line on what changed"}
```

## Step 5 — Verify

```bash
python scripts/selfplay_api.py check-prompt --model <model> --role verify \
  --from-version $N --to-version $((N+1))
```

Must exit 0 before you write the report.

## Step 6 — Decide convergence

Set `converged: true` when **either**:

- **No admissible edits** — everything was already covered or filtered out in step 2. An
  empty update is a legitimate result; do not manufacture an edit.
- **Metrics have flattened** — for two consecutive rounds, `|Δ mean_verifier_f1| < 0.05` and
  `|Δ mean_verifier_recall| < 0.05`, with no new reasoning-, quoting- or reporting-miss
  pattern.

Otherwise `converged: false`. Note which class dominates: remaining reasoning misses mean
the Perturber is genuinely ahead; remaining quoting or reporting misses mean the prompt has
not internalised the contract; a clean sheet means the round was too easy.

## Step 7 — Write the report

`<round_dir>/update_verify_api.json`:

```json
{
  "round": 2, "role": "verifier", "model": "claude-sonnet-5",
  "from_version": 2, "to_version": 3, "skipped": false, "skip_reason": null,
  "metrics": {"mean_verifier_f1": 0.55, "mean_verifier_recall": 0.55,
              "mean_verifier_precision": 0.62, "delta_f1": 0.07,
              "api_cost_usd": 0.31, "truncated_replies": 0, "api_errors": 0},
  "miss_breakdown": {"reasoning": 3, "quoting": 2, "reporting": 1, "structural": 1},
  "diagnosis": [{"observation": "…", "episodes": ["ep01"], "bucket": "reasoning|quoting|reporting|structural|false_positive|penalty|api"}],
  "policy_edits": [{"rule": "V5", "action": "revised|added|removed", "change": "…", "evidence": ["ep01", "ep06"]}],
  "scorer_issues": [{"episode": "ep03", "pattern": "…", "component": "matcher", "action": "reported, not learned"}],
  "converged": false,
  "convergence_reason": "…"
}
```

Then report in prose: version created, the edits with their evidence, anything rejected as
exploitation, and the convergence verdict.
