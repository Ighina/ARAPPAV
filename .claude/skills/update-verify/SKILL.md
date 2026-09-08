---
name: update-verify
description: Static updater that turns one scored self-play round into the next Verifier policy — reads the round's episodes and reward signal, diagnoses misses and false positives, and writes verify-v{N+1}. Use after scoring a round, or when asked to update/improve the verifier skill from rollout results.
---

# update-verify — derive the next Verifier policy

This skill is **static**: it is never versioned and never rewritten. It consumes one scored
round of skill self-play and produces `.claude/skills/verify-v{N+1}/SKILL.md`, plus a report
recording what changed, what was rejected, and whether the policy has converged.

Analogue in the RL system: the DPO update for the Verifier — except that instead of pairing
high-`r_V` completions against low-`r_V` ones, you extract *why* the good ones were good and
write that into the policy.

---

## Preconditions

1. `<round_dir>/round_summary.json` exists (run `summarize` first).
2. Check `manifest.freeze`. If it is `"verifier"`, the Verifier is **frozen**: write a report
   with `"skipped": true`, `"skip_reason": "verifier frozen"`, create no new skill version,
   and stop. (`freeze: "perturber"` or `null` → proceed.)
3. Determine the current version from `ls .claude/skills | grep -E '^verify-v[0-9]+$'`. It
   must match the round's `manifest.verify_skill`; if it does not, stop and say so.

---

## Inputs

| File | What to take from it |
|------|----------------------|
| `<round_dir>/round_summary.json` | `mean_verifier_recall` / `precision` / `f1`, `deltas_vs_previous_round`, `error_type_detection`, `learning_signal` |
| `<round_dir>/episodes/*/score.json` | per-claim outcomes, `match_details` (with `best_overlap` and `all_overlaps`), the ground-truth errors |
| `<round_dir>/episodes/*/verify.json` | what the Verifier actually wrote — the raw quoting behaviour |
| `.claude/skills/verify-v{N}/SKILL.md` | the policy that produced all of it |

> **Never read `data/skill_evals/`.** That directory holds the held-out ProcessBench
> evaluation — a test set for this exact skill. Its items, labels, and scores are outside
> this update's evidence, and selecting a policy edit on them is test-set fitting: the
> benchmark stops measuring generalisation the moment it steers the policy. Everything you
> need is in the round directory.

Read the raw `verify.json` alongside the scores. Recall and precision say what happened;
only the raw claims show whether the cause was reasoning or quoting.

---

## Step 1 — Diagnose, separating the three failure classes

For every undetected ground-truth error (`learning_signal.undetected_errors`), classify —
this distinction drives completely different edits:

- **Reasoning miss** — no claim anywhere near the error (`best_overlap ≈ 0`, no plausible
  entry in `closest_claims`). The Verifier never noticed the mistake. Fix: sharpen the
  re-derivation procedure, or add the specific misconception to the sweep list.
- **Quoting miss** — the Verifier *did* flag the right region but the span could not be
  aligned (`best_overlap` positive but under 0.5, or a claim in `closest_claims` whose
  explanation is right). This is a pure quote-discipline failure and the cheapest recall
  available. Fix: tighten the quoting rules.
- **Structural miss** — the error was inside a unit already detected through another member,
  or the error was never in the text (`k_effective < k`). No verifier fix applies; note it
  and move on.

Then, for precision:

- **False positives** (`learning_signal.verifier_false_positives`). For each, decide: was it
  a hallucinated mistake (fix by requiring re-derivation before claiming), a correct
  observation about text the Perturber never declared (a real but undeclared error — record
  it, do not train against it), or a downstream consequence claimed as an independent error
  (fix with the root-cause rule)?
- **Spam / repetition penalties** — any nonzero `spam_penalty` or `repetition_penalty` is a
  format problem, fixable outright.

Attach episode IDs to every observation. An observation with no episode behind it does not
enter the update.

## Step 2 — Filter out scorer exploitation

A high `r_V` counts only if the Verifier actually *found* the mistakes. Reject — and record
under `scorer_issues`, not as a policy lesson — anything that raises the score without
improving detection:

- quoting long spans, whole lines, or the entire solution to maximise overlap;
- shotgunning claims to raise recall (precision and anti-spam are supposed to punish this;
  if they did not, the guard is miscalibrated — say so);
- boilerplate explanations attached to spans copied mechanically from suspicious-looking
  LaTeX, rather than from a re-derivation;
- exploiting the Perturber's habits — "the last error is usually in the final line" — rather
  than the mathematics. This is the Verifier's version of overfitting: it collapses the
  moment the Perturber's policy changes.

Never weaken the invariant sections of the skill. If matching itself is the problem (correct
diagnoses that cannot be aligned no matter how carefully quoted), that is a matcher bug for
`src/arappav/reward/matcher.py` — report it, do not paper over it in the policy.

## Step 3 — Write the policy edits

- At most **5** edits per round; each cites ≥1 episode, preferably ≥2 independent ones.
- Edits are general procedure, never episode content. No problem statements, no remembered
  answers, no "expect an operand swap in step 2" tied to this round's sample.
- Prefer revising an existing rule to appending a new one; keep the tuned section under
  ~60 lines. Delete rules the evidence contradicts and say so.
- Weight the edits by what the numbers say: if precision is the binding constraint, a rule
  that adds claims makes the policy worse even when it raises recall. Optimise F1.

## Step 4 — Create `verify-v{N+1}`

```bash
mkdir -p .claude/skills/verify-v$((N+1))
cp .claude/skills/verify-v$N/SKILL.md .claude/skills/verify-v$((N+1))/SKILL.md
```

Then edit **only**: the frontmatter `name:` (and version wording in `description:`), the
`version` / `parent` / `tuned from rounds` header lines, everything under `## Policy` up to
`## Changelog`, and a new changelog entry (what changed, the evidence, what you rejected).

Everything from `## Output contract` through the scoring table is **invariant** — a
transcription of `schema_math.py`, `matcher.py`, and `reward_fns.py`. If the contract itself
is wrong, that is a repo bug: report it, do not edit around it.

## Step 5 — Verify

```bash
python scripts/skill_selfplay.py check-skill --role verify --from-version $N --to-version $((N+1))
```

Must exit 0 before you write the report.

## Step 6 — Decide convergence

Set `converged: true` when **either**:

- **No admissible edits** — every observation was already covered by the current policy or
  was filtered out in step 2. An empty update is a legitimate result; do not manufacture an
  edit to avoid one.
- **Metrics have flattened** — for two consecutive rounds, `|Δ mean_verifier_f1| < 0.05` and
  `|Δ mean_verifier_recall| < 0.05`, with no new reasoning-miss or quoting-miss pattern.

Otherwise `converged: false`. Note which class of miss still dominates: remaining reasoning
misses mean the Perturber is genuinely ahead; remaining quoting misses mean the policy has
not internalised the contract; a clean sheet means the round was too easy.

## Step 7 — Write the report

`<round_dir>/update_verify.json`:

```json
{
  "round": 3,
  "role": "verifier",
  "from_version": 2,
  "to_version": 3,
  "skipped": false,
  "skip_reason": null,
  "metrics": {"mean_verifier_f1": 0.55, "mean_verifier_recall": 0.55,
              "mean_verifier_precision": 0.62, "delta_f1": 0.07},
  "miss_breakdown": {"reasoning": 3, "quoting": 2, "structural": 1},
  "diagnosis": [
    {"observation": "…", "episodes": ["ep01", "ep06"], "bucket": "reasoning|quoting|structural|false_positive|penalty"}
  ],
  "policy_edits": [
    {"rule": "V5", "action": "revised|added|removed", "change": "…", "evidence": ["ep01", "ep06"]}
  ],
  "scorer_issues": [
    {"episode": "ep03", "pattern": "…", "component": "matcher", "action": "reported, not learned"}
  ],
  "converged": false,
  "convergence_reason": "quoting misses eliminated, reasoning misses still rising"
}
```

Then report back in prose: version created, the edits with their evidence, anything rejected
as exploitation, and the convergence verdict.
