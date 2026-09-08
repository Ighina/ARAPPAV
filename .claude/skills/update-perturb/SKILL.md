---
name: update-perturb
description: Static updater that turns one scored self-play round into the next Perturber policy — reads the round's episodes and reward signal, diagnoses what worked, and writes perturb-v{N+1}. Use after scoring a round, or when asked to update/improve the perturber skill from rollout results.
---

# update-perturb — derive the next Perturber policy

This skill is **static**: it is never versioned and never rewritten. It consumes one scored
round of skill self-play and produces `.claude/skills/perturb-v{N+1}/SKILL.md`, plus a
report recording what changed, what was rejected, and whether the policy has converged.

Analogue in the RL system: the GRPO update for the Perturber. Reward is the same reward —
`compute_rewards` in `src/arappav/reward/reward_fns.py` — but the gradient lands in prose.

---

## Preconditions

1. `<round_dir>/round_summary.json` exists (run `summarize` first). Do not update from
   unscored episodes.
2. Check `manifest.freeze`. If it is `"perturber"`, the Perturber is **frozen**: write a
   report with `"skipped": true`, `"skip_reason": "perturber frozen"`, create no new skill
   version, and stop. (`freeze: "verifier"` or `null` → proceed.)
3. Determine the current version: `ls .claude/skills | grep -E '^perturb-v[0-9]+$'`, take
   the highest `N`. The round's `manifest.perturb_skill` should equal `perturb-v{N}`; if it
   does not, you are about to branch from the wrong parent — say so and stop.

---

## Inputs

| File | What to take from it |
|------|----------------------|
| `<round_dir>/round_summary.json` | aggregate metrics, `deltas_vs_previous_round`, `error_type_detection`, `learning_signal` |
| `<round_dir>/episodes/*/score.json` | per-episode rewards, `match_details`, the injected errors and the Verifier's claims |
| `<round_dir>/episodes/*/perturb_status.json` | format failures with the validator's exact message |
| `<round_dir>/episodes/*/problem.json` | the source problem, when you need to judge whether an error was fair |
| `.claude/skills/perturb-v{N}/SKILL.md` | the policy that produced all of it |

> **Never read `data/skill_evals/`.** That directory holds the held-out ProcessBench
> evaluation — a test set. Its items, labels, and scores are outside this update's evidence,
> and selecting a policy edit on them is test-set fitting. Everything you need is in the
> round directory.

Read the **episodes**, not only the summary. The summary tells you *that* recall moved; only
the episodes tell you *why*.

---

## Step 1 — Diagnose

Bucket every episode, and attach episode IDs to every claim you make. An observation with
no episode ID behind it does not enter the update.

- **Format failures** (`perturber_format_valid: false`). Read the validator message. Which
  contract rule was broken — phantom, restatement, wrong `k`, bad enum, unparseable JSON,
  unlocatable `original_text`? These are the cheapest wins available: they cost −5 or −10
  each and are fully mechanical to prevent.
- **Wins** (`learning_signal.undetected_errors`). Which error types, which steps, which
  kinds of mathematics survived? Look at `best_overlap` and `closest_claims`: a near-miss
  (the Verifier flagged the region but quoted it badly) is not the same win as a clean miss
  (no claim anywhere near).
- **Losses** (`learning_signal.detected_errors`). Which errors were caught immediately, and
  what made them conspicuous?
- **Collapse** — compare `num_error_units` with `k` per episode. Units below `k` means the
  Perturber declared errors that the matcher merged: the policy is stacking, and stacking is
  the failure this loop exists to prevent.
- **Penalties** — `duplicate_penalty` (repeating a trick across rounds),
  `k_effective < k` (declared errors missing from the text).

## Step 2 — Filter out scorer exploitation

**This step is what keeps skill tuning from becoming reward hacking.** A high `r_P` is
evidence of a good policy only if the reward was earned by producing a *genuinely harder
error*. Before an observation becomes a policy rule, ask: would a competent human
mathematician reviewing this solution call the injected text a real mistake, and a hard one?

Reject — and record under `scorer_issues`, never as a policy lesson — any win that came from:

- stacking one mistake as several declarations (root + propagated `\boxed{}`, overlapping
  rewrites, restatements) — the exact hack found in round 3 of the RL run;
- text that is redundant or stylistically odd but mathematically unchanged;
- exploiting the matcher rather than the mathematics: burying a small change inside a very
  long `injected_text`, LaTeX escaping tricks, quoting/formatting games;
- errors so severe or so far outside the problem's topic that they are unrealistic as
  student work (they inflate reward only because the Verifier ignored them as noise).

If a scorer issue is real, name the guard that should catch it (`error_units`, a schema
validator, an anti-* penalty) in the report. Do not weaken the invariant sections of the
skill to accommodate it, and do not "teach" it to the next version.

## Step 3 — Write the policy edits

- At most **5** edits per round. Each edit must cite ≥1 episode; prefer ≥2 independent
  episodes before promoting an observation to a rule.
- Edits are **general strategy**, never episode content. No problem statements, no specific
  numbers, no "for the arithmetic-series problem, do X". A rule that only fires on this
  round's sample is overfitting — the analogue of memorising the training set.
- Prefer revising an existing rule over appending a new one. The tuned section is a policy,
  not a log; keep it under ~60 lines and readable end to end.
- Delete rules the evidence contradicts, and say so in the changelog.

## Step 4 — Create `perturb-v{N+1}`

```bash
mkdir -p .claude/skills/perturb-v$((N+1))
cp .claude/skills/perturb-v$N/SKILL.md .claude/skills/perturb-v$((N+1))/SKILL.md
```

Then edit **only**:

1. frontmatter `name:` → `perturb-v{N+1}`, and the version wording in `description:`;
2. the `version` / `parent` / `tuned from rounds` header lines;
3. everything under `## Policy` up to `## Changelog`;
4. a new changelog entry: what changed, the evidence, and what you rejected.

Everything from `## Output contract` through the taxonomy is **invariant** — it is a
transcription of the code, not a policy choice. If a round shows the contract itself is
wrong, that is a repo bug: report it, do not edit around it.

## Step 5 — Verify

```bash
python scripts/skill_selfplay.py check-skill --role perturb --from-version $N --to-version $((N+1))
```

This must exit 0: frontmatter name matches the directory, the invariant region is
byte-identical, the policy section actually changed, and the changelog grew. Fix any
reported problem before writing the report.

## Step 6 — Decide convergence

Set `converged: true` when **either**:

- **No admissible edits.** Every observation this round was already covered by the current
  policy, or was filtered out in step 2. Do not invent an edit to avoid an empty update —
  an empty update *is* the signal.
- **Metrics have flattened.** For two consecutive rounds:
  `|Δ mean_perturber_reward_valid_only| < 0.05`, `|Δ format_valid_rate| < 0.10`, and no new
  undetected-error pattern (no error type moving out of the detected set).

Otherwise `converged: false`. If you converged, say in the report which axis is saturated —
the Perturber may be out of ideas, or the Verifier may simply have gotten good, and those
call for different next experiments.

## Step 7 — Write the report

`<round_dir>/update_perturb.json`:

```json
{
  "round": 3,
  "role": "perturber",
  "from_version": 2,
  "to_version": 3,
  "skipped": false,
  "skip_reason": null,
  "metrics": {"mean_perturber_reward_valid_only": 0.38, "format_valid_rate": 0.875,
              "delta_reward": 0.22, "delta_format_valid_rate": 0.125},
  "diagnosis": [
    {"observation": "…", "episodes": ["ep02", "ep05"], "bucket": "win|loss|format|collapse|penalty"}
  ],
  "policy_edits": [
    {"rule": "P2", "action": "revised|added|removed", "change": "…", "evidence": ["ep02", "ep05"]}
  ],
  "scorer_issues": [
    {"episode": "ep04", "pattern": "…", "guard": "error_units", "action": "reported, not learned"}
  ],
  "converged": false,
  "convergence_reason": "recall still moving; two new error types undetected"
}
```

Then report back in prose: version created, the edits with their evidence, anything
rejected as exploitation, and the convergence verdict.
