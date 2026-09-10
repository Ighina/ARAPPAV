# `--update-mode evolve`: Trace2Skill-style policy updates

The ablation axis for *how* a policy is revised between rounds. Three modes,
selected with `--update-mode`:

| mode | mechanism |
|---|---|
| `rewrite` (default) | one updater rewrites the whole policy body |
| `summarise` | a `summarise-context-*` step briefs that updater with the episodes' text first |
| `evolve` | one analyst per episode proposes typed edits in parallel, a merge step reconciles them, and Python applies the result |

`evolve` follows Trace2Skill (Qwen, arXiv 2603.25158): parallel trajectory-local
proposals, hierarchical consolidation, structured patch operations.

---

## Why it exists

Whole-policy rewriting accumulates. Measured over ten rounds of
`haiku_cold_10x8`, verifier policy bodies grew from 90 characters (empty cold
start) to 2,922, with consecutive word-level diffs of +253, −13, +31, +1, +24,
+68, −11, +12, +51 — while held-out ProcessBench F1 stayed flat. The rule count
never moved: 8 rules throughout, each getting longer.

The mechanism is that **a rewrite can only drop a rule by forgetting it**.
Omission is not a salient action, so a model asked to "return the revised
policy" reproduces what it was given and appends. Naming the operation is meant
to change that.

---

## The three stages

### 1. Parallel analysts, one per episode

`evolve-analyse-perturber` / `evolve-analyse-verifier` each see **exactly one**
episode and the current policy, and never each other's work. Each proposes at
most two edits.

The isolation is the point: a change proposed independently by several analysts
is *corroborated*, while one proposed by a single analyst may be noise. With 8
episodes a round and a measured within-round standard deviation of 0.185, that
distinction is the main defence against fitting sampling variation.

### 2. Merge

`evolve-merge-patches` receives every proposal and reconciles them into one
patch: deduplicating, resolving contradictions (an append to a rule another
analyst wants deleted), capping at five edits, and — explicitly — preserving
deletions, which are otherwise the first thing a cautious merger drops.

### 3. Deterministic application

`arappav.pipeline.patches.apply_patch` applies the merged patch in Python. The
LLM never writes the policy file.

| op | effect |
|---|---|
| `append_to_rule` | add a clause to an existing rule |
| `replace_in_rule` | swap specific text inside a rule |
| `rewrite_rule` | replace one rule's body |
| `add_rule` | introduce a new rule |
| `delete_rule` | **remove a rule** |

Edits resolve against the *original* rule list and apply in one pass, so two
edits cannot interact through a target a previous edit moved. A patch that
targets the same rule twice is **refused** — that is an unresolved merge
conflict, and failing the round is better than silently producing mangled text.
Patches see only the tuned policy body; the invariant region (output contract,
scoring, taxonomy) is never exposed to them.

---

## How the reward is used — and where it is not

**The scalar reward never reaches an analyst.** Verified against the prompts
actually sent in `algebra_evolve`: `perturber_reward`, `verifier_reward` and the
round means appear nowhere in an analyst's context.

What the analyst sees instead is the reward's *constituents*, per error:

```json
{"error_id": "err_001",
 "original_text": "$(3y + 8)(y - 3)$",
 "injected_text": "$(3y + 7)(y - 3)$",
 "detected": true,
 "closest_overlap": 0.517,
 "closest_claim": "…", "closest_claim_explanation": "…"}
```

plus `k`, `num_error_units`, `verifier_recall` and `verifier_precision` for that
one episode.

So the reward is used **three ways, none of them as a number an analyst optimises**:

1. **As the matcher's verdict.** `detected` is exactly the match decision that
   `compute_rewards` aggregates into recall. The analyst reasons over the same
   evidence the reward is computed from, one episode at a time, in text.
2. **As `closest_overlap`.** The continuous alignment score, reported whether or
   not it cleared the matching threshold, which is what separates a reasoning
   miss (nothing claimed nearby) from a quoting miss (right region, wrong span).
3. **As the round's reported score**, for the summary and the plots — a
   measurement, not a signal into the update.

This is deliberate. Handing an updater a scalar and asking it to infer which
textual change caused it is a credit-assignment problem across ~24 error units
per round, and the measured correlation between self-play reward and held-out
ability is r = +0.275 (n = 10, not significant); selecting on it returns a
policy *worse* than the untuned cold start. A number carrying that little signal
is not worth optimising against, whereas "this specific injected text went
undetected, and here is what the verifier said instead" is actionable.

**The consequence to keep in view:** in `evolve` mode the loop is no longer
reward-driven in its update path. The reward defines the game and scores it, but
the policy changes are driven by per-episode textual evidence. `rewrite` and
`summarise` do pass the aggregate metrics through, so this is a real difference
between the modes and not only an interface change.

Optional gating (`--accept-on-validation`) is the one place a score decides
anything — and it deliberately uses a **held-out** score, never the self-play
reward.

---

## What it did in practice

`algebra_evolve` (10 rounds, algebra-only, opus updater, haiku players):

| op | count across 9 patches, 37 edits |
|---|---|
| `append_to_rule` | 25 |
| `add_rule` | 10 |
| `replace_in_rule` | 2 |
| `rewrite_rule` | 0 |
| **`delete_rule`** | **0** |

**Not one deletion.** Making removal a single documented operation away did not
produce any. Accumulation therefore is not an artifact of whole-policy
rewriting — it survives when deleting is trivially available, which points at
the model's disposition rather than the interface.

Held-out ProcessBench over the same run (80 items, identical set):

| policy | v1 | v2 | v3 | v4 | v5 | v10 |
|---|---|---|---|---|---|---|
| F1 | 0.731 | 0.840 | **0.905** | 0.874 | 0.822 | 0.831 |

A real rise to a peak at round 2 (+0.175, ≈3.9 SE), then monotone decay. Three
rounds help; everything after does not. Combined with zero deletions, the
reading is that later rounds append to a policy that was already right, and the
additions dilute it.

---

## Cost

`evolve` is call-hungry by construction: one analyst per episode per unfrozen
role, plus one merge each. At 8 episodes and both sides active that is 18 calls
per round on top of the 16 the episodes themselves cost — roughly double a
`rewrite` round.
