---
name: selfplay
description: Run the skill-tuning self-play loop — alternate perturb-vN and verify-vN over scored rounds, then update the skills with update-perturb / update-verify until the round budget is spent or the policies converge. Use when asked to run skill self-play, a skill-tuning rollout, or to improve the perturber/verifier skills through self-play.
---

# selfplay — skill-tuning rollout loop

The RL loop in `README.md` updates model weights (GRPO for the Perturber, DPO for the
Verifier). This loop keeps the same game, the same episodes, and the same reward function,
but the thing being updated is a pair of **skills**:

```
  perturb-vN ─┐                                  ┌─ update-perturb ─→ perturb-v(N+1)
              ├─→ episodes ─→ compute_rewards ─→ ┤
  verify-vN  ─┘        (scripts/skill_selfplay.py) └─ update-verify  ─→ verify-v(N+1)
```

Rewards come from the repo's own `compute_rewards` — error units, graded format penalties,
anti-spam, anti-duplicate — so a skill that scores well here is optimising the same
objective the RL trainers do.

---

## Parameters

Accept these from the invocation; otherwise use the defaults.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `rounds` | 3 | maximum rounds; the loop may stop earlier on convergence |
| `episodes` | 8 | episodes per round |
| `k` | 3 | errors per episode (or `--k-range LO HI` to sample) |
| `freeze` | `none` | `perturber` or `verifier` freezes that side, exactly as `self_play.freeze` does in the RL config |
| `source` | `local` | `local` replays problems stored in past rollout logs (offline); `hendrycks` downloads from HF |
| `root` | `data/skill_rollouts` | rollout root |
| `eval` | `on` | run the held-out ProcessBench evaluation after each round |
| `eval_per_subset` | 20 | ProcessBench items per subset (4 subsets) |
| `eval_seed` | 0 | fixes the held-out sample; keep constant across rounds |

Starting versions are the highest existing `perturb-vN` / `verify-vN` under
`.claude/skills/` (v1 on a fresh repo). Announce the plan — rounds, episodes, k, freeze,
starting versions — before round 1.

---

## Per-round procedure

### 1. Scaffold

```bash
python scripts/skill_selfplay.py init --round $R --episodes $E --k $K \
  --perturb-skill perturb-v$P --verify-skill verify-v$V --freeze $FREEZE --source $SOURCE
```

Problems are drawn without reuse across rounds, so each round is a fresh test set.

### 2. Perturber pass

For each `episodes/<id>/problem.json`, follow **`perturb-v$P`** and write
`episodes/<id>/perturb.json`. Work episode by episode; do not look ahead at other episodes'
outputs. Batch the file writes, but reason about each perturbation separately.

### 3. Validate and build the verifier's inputs

```bash
python scripts/skill_selfplay.py prepare-verify --round $R
```

This parses each perturbation exactly as the RL path does (`parse_and_backoff`: fence
stripping, LaTeX escape repair, mechanical injection backoff), records format validity, and
writes `verify_inbox/<id>.json` containing only the problem and the text to review.

### 4. Verifier pass — **in an isolated context**

The Verifier must not know what was injected. You have just written the ground truth, so
**you cannot verify these episodes yourself**. Run this pass in a fresh agent (`Agent` tool,
`general-purpose`) — invoking this skill is the request for those spawns, and the round's
verifier score is meaningless without the isolation. One agent per round is enough.

Give the agent exactly this scope:

> Follow `.claude/skills/verify-v$V/SKILL.md`. For every file in
> `<round_dir>/verify_inbox/`, produce the claims JSON and write it to
> `<round_dir>/verify_outbox/<same filename>`. Do not read anything under
> `<round_dir>/episodes/` — it contains the answers.

If subagents are unavailable, run the pass in a separate `claude` session on the same inbox.
Never fall back to verifying in this context: a leaked round teaches both skills the wrong
lesson, and the reward signal that follows is fiction.

### 5. Score and summarize

```bash
python scripts/skill_selfplay.py score --round $R
python scripts/skill_selfplay.py summarize --round $R
```

`score` applies graded format penalties, error-unit recall, and cross-round anti-duplicate
history. `summarize` writes `round_summary.json` with the aggregate metrics, per-error-type
detection rates, the deltas against the previous round, and the learning signal (format
failures, undetected errors, detected errors, false positives).

### 6. Update the skills

Honour `freeze` — it decides which updater runs, and both updaters check it again:

| freeze | update-perturb | update-verify |
|--------|----------------|---------------|
| `none` | runs | runs |
| `perturber` | **skipped** (frozen) | runs |
| `verifier` | runs | **skipped** (frozen) |

Invoke the applicable updaters via the `Skill` tool with the round directory. Each writes a
new skill version, verifies it with `check-skill`, and reports a convergence verdict to
`update_perturb.json` / `update_verify.json`.

### 7. External evaluation — ProcessBench

Unless `eval=off`, invoke **`eval-processbench`** with the round number and the verifier
version that this round produced (the new `verify-v{V+1}`, or the unchanged version if the
Verifier is frozen). It samples held-out `Qwen/ProcessBench` items, runs the verifier skill
over them in its own fresh agent, and reports the first-error-step metric per subset.

Three rules bind you here:

- Never read the evaluation's items into this context — delegate the pass and read only
  `eval_summary.json` (ids, hit/miss, aggregate metrics; no item text, no gold labels).
- Never feed evaluation results into `update-perturb` or `update-verify`. They are
  monitoring, not signal; the updaters are forbidden from reading `data/skill_evals/`.
- Never let evaluation results decide whether to continue, or which policy version to keep.
  Convergence is judged on self-play only.

The one thing this evaluation is for: saying whether self-play gains generalise. A round
where self-play recall rises and ProcessBench does not is a policy that learned this
Perturber, and you must report it as such.

### 8. Advance

Set `$P` / `$V` to the versions that were just created (a frozen side keeps its version) and
start the next round.

---

## Stopping

Stop the loop when **any** of these holds, and say which one fired:

1. `rounds` rounds are complete.
2. Every updater that ran this round reported `"converged": true` — no admissible policy
   edits, or flat metrics across two consecutive rounds. Under `freeze`, only the active
   side's verdict counts.
3. A round produced no usable signal — e.g. every episode was format-invalid, or the
   verifier pass could not be isolated. Stop and report; do not update a skill from a round
   that carries no information.

Do not keep spending rounds on a converged policy: a version that changes nothing but the
number is noise in the history.

---

## Final report

- Rounds run, and which stop condition fired.
- Per-round table: format-valid rate, mean `r_P`, mean `r_V`, recall, precision.
- Version lineage (`perturb-v1 → v3`, `verify-v1 → v2`) and the substantive edit behind each
  version bump.
- ProcessBench per round: per-subset and overall F1 for each verifier version, and an
  explicit verdict on whether the self-play gains transferred to the held-out benchmark.
- Anything the updaters filed under `scorer_issues` — those are candidate repo bugs
  (matcher, error units, penalty calibration), not skill problems, and they are the most
  valuable output of a round that otherwise looked good.
