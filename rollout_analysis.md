# ARAPPAV Rollout Analysis — Round 1

**Date:** 2026-07-13
**Mode:** Math (Hendrycks MATH — algebra subset)
**Models:** Untuned Perturber + Frozen Verifier
**Status:** ✅ All 7 recommendations implemented — 49/49 tests passing

---

## Overview

Analyzed 7 perturber episodes and 2 verifier episodes from Round 1 in math mode. The system uses an adversarial self-play loop: the Perturber (P) injects errors into correct math solutions, the frozen Verifier (V) tries to detect them, and rewards flow based on V's recall against P's ground truth.

---

## 1. Perturber Performance

| Metric | Value |
|--------|-------|
| Total episodes | 7 |
| Format-valid | **2 (29%)** |
| Format-invalid | **5 (71%)** |

### Format failure breakdown

| Episode | Failure Type | Detail |
|---------|-------------|--------|
| `algebra_Level 3_0_ep0` | **Phantom errors** | All 3 errors had `injected_text == original_text` — P "claimed" errors without modifying the text. A reward-hacking attempt blocked by the phantom validator. |
| `algebra_Level 1_2_ep2` | **Taxonomy violation** | Used `'duplicated_operation'` instead of valid `'duplication_error'` |
| `algebra_Level 4_3_ep3` | **JSON parse error** | Malformed/incomplete JSON — comma delimiter expected at column 357 |
| `algebra_Level 3_4_ep4` | **Taxonomy violation** | Used `'sign_error'` — not in the 26-type math taxonomy |
| `algebra_Level 4_5_ep5` | **Taxonomy violation** | Used `'addition_across'` instead of valid `'adding_across'` — a near-miss typo |

**Key observation:** 3 of 5 failures (60%) are taxonomy-enum mismatches where the model used semantically reasonable but syntactically invalid error type names. The near-misses (`'addition_across'` → `'adding_across'`, `'duplicated_operation'` → `'duplication_error'`) indicate the model **understands the taxonomy conceptually** but doesn't reproduce the exact enum strings. The strict Pydantic enum validation is too brittle for an untuned model.

---

## 2. Overlapping Error Injection Bug (Critical)

In **both** format-valid episodes, the perturber generated overlapping errors that couldn't all be mechanically inserted:

### Episode `ep1` (`algebra_Level 2_1`)

- `err_002` modifies the final sum to `\boxed{60} + 8`
- `err_003` tries to replace `"The sum is $(4+11)/2 \cdot 8 = \boxed{60}$."` → `"The sum is $4/2 \cdot 8 = \boxed{16}$."`
- But `err_002` **already changed that text region**, so `err_003`'s `original_text` no longer exists in the perturbed output
- Result: **1 of 3 injected errors is physically missing from the text**

### Episode `ep6` (`algebra_Level 1_6`)

- `err_001` tries to duplicate `15\cdot 2x` in the distribution, `err_003` tries to replace the entire align environment
- Both target overlapping regions; `err_001`'s injection is missing from the final text
- Result: **1 of 3 injected errors is physically missing from the text**

The mechanical backoff (`apply_error_injections`) does sequential `original_text → injected_text` string replacement, but when errors overlap, the later error's `original_text` substring is gone. **Both "valid" episodes actually only contain 2 detectable errors**, not the k=3 that was requested.

---

## 3. Verifier Performance (Frozen)

The verifier was run at `n_completions=3` per episode. Only 2 perturber rollouts were format-valid, so only 2 were scored.

### Episode `ep1` (`algebra_Level 2_1`)

| Response | Claims | Errors Identified | Reward |
|----------|--------|-------------------|--------|
| 0 | 2 | 2/2 detectable errors found | **0.0000** |
| 1 | 2 | 2/2 detectable errors found | **0.0000** |
| 2 | 2 | 2/2 detectable errors found | **0.0000** |

> ⚠️ **The verifier correctly identified both detectable errors in all 3 responses**, yet received **zero reward** (F1=0, recall=0, precision=0). This is a **matcher failure**, not a verifier failure.

Additionally, **Response 2 contains 164 repeated JSON blocks** — the model collapsed into a repetition loop within a single generation, a known failure mode for smaller LLMs.

### Episode `ep6` (`algebra_Level 1_6`)

| Response | Claims | Errors Identified | Reward |
|----------|--------|-------------------|--------|
| 0 | 1 | 1/2 detectable errors | **0.5000** |
| 1 | 1 | 1/2 detectable errors | **0.5000** |
| 2 | 1 | 1/2 detectable errors | **0.5000** |

The verifier consistently catches the extra `+15` (the most visible error) but misses the broader structural error (`err_003`, the wrong operation claim). Precision=1.0 (no false positives) but recall=0.33.

---

## 4. Matching System Failure (Critical Bug)

The reward computation reports **0 matched errors** for `ep1` despite the verifier perfectly identifying both detectable errors. Two root causes:

### Cause A: Length asymmetry between `injected_text` and `quoted_text`

```
injected_text (174 chars): "The sum of an arithmetic series is equal to the
  average of the first and last term, multiplied by the number of terms.
  The average is just the sum of the first and last term."

quoted_text  (55 chars):  "The average is just the sum of the first and last term."
```

The perturber outputs the **entire modified sentence** as `injected_text`, including surrounding correct text. The verifier outputs **only the erroneous fragment**. The span-IoU is `55/174 = 0.316` — below the 0.5 matching threshold. The verifier's correct identification **cannot be matched** to the ground truth.

### Cause B: LaTeX escaping mismatch

```
Verifier outputs:  "(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8"    ← JSON double-escaped
Perturbed text:    "$(4 + 11)/2 \cdot 8 = \boxed{60} + 8$"    ← single backslash in text
```

The verifier's `quoted_text` uses JSON-escaped LaTeX (`\\\\cdot`, `\\\\boxed`), which doesn't literally appear in the perturbed text. The exact-match span finder returns `None`, yielding **IoU = 0.0**.

Even the fuzzy matcher (`_normalize_for_matching`) only handles `&` alignment characters and whitespace — **not LaTeX backslash escaping mismatches**.

---

## 5. Reward Signal Analysis

| Episode | P Base Reward | P Final Reward | V Reward | Signal Quality |
|---------|--------------|----------------|----------|----------------|
| `ep0` | -10 (format) | -10 | — | ✅ Correctly penalized phantom errors |
| `ep1` | 1.0 | 1.0 | 0.0 | ❌ P rewarded despite V finding both errors |
| `ep2` | -10 (format) | -10 | — | ⚠️ Penalized for taxonomy typo, not bad errors |
| `ep3` | -10 (format) | -10 | — | ✅ Correctly penalized JSON failure |
| `ep4` | -10 (format) | -10 | — | ⚠️ Penalized for taxonomy typo |
| `ep5` | -10 (format) | -10 | — | ⚠️ Penalized for taxonomy typo |
| `ep6` | 0.67 | 0.67 | 0.50 | ⚠️ Ok signal but P still wins (1 undetected + 1 missing) |

**The perturber gets a free pass on `ep1`**: its reward is 1.0 (maximum) because the matcher can't align the verifier's correct claims to its ground truth. This means the GRPO update would **reinforce the perturber's strategy of embedding errors within long `injected_text` spans** — exactly the wrong direction for training.

---

## 6. Systemic Issues Summary

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | **Span-IoU matcher fails on length-asymmetric texts** | 🔴 Critical | Verifier gets zero credit for correct detections; P gets rewarded for being caught |
| 2 | **LaTeX escaping mismatch between V output and P output** | 🔴 Critical | Secondary cause of false-zero matches; quoted text can't be found in perturbed output |
| 3 | **Overlapping error injection drops errors silently** | 🟠 High | 33% of k=3 errors never make it into the text; P is evaluated on k=3 but only delivers 2 |
| 4 | **71% format failure rate** | 🟠 High | Most episodes produce no training signal; -10 penalty dominates reward landscape |
| 5 | **Taxonomy enum too strict for untuned models** | 🟡 Medium | Near-miss type names (`'addition_across'`) cause format failures; model understands semantics but not exact strings |
| 6 | **Verifier repetition collapse** | 🟡 Medium | Response 2 in `ep1` generated 164 identical JSON blocks — wastes compute, signals model instability |
| 7 | **Verifier only catches surface-level errors** | 🟡 Medium | Consistently finds obvious errors (extra `+15`, `+8`) but misses structural errors (wrong formula derivation, operand swaps) |
| 8 | **No reward signal from format-invalid episodes** | 🟡 Medium | When P fails format, V gets no reward at all — can't learn from 71% of episodes |

---

## 7. Recommendations

### 1. Fix the matcher (highest priority)

Add substring-containment as a match criterion. If `quoted_text` is fully contained within `injected_text` (or vice versa), treat it as a match regardless of IoU. This alone would fix the `ep1` false-zero.

**File:** `src/arappav/reward/matcher.py` — `match_claims_to_errors()` and `_compute_char_span_overlap()`

### 2. Add LaTeX normalization to the matcher

Unescape `\\\\` → `\\` in both `quoted_text` and `injected_text` before span comparison. Extend `_normalize_for_matching()` to handle JSON escaping of LaTeX commands.

### 3. Add fuzzy enum matching for taxonomy

Accept near-miss error type names (Levenshtein distance ≤ 2) with a warning instead of hard-failing. `'addition_across'` → `'adding_across'` should be auto-corrected.

**File:** `src/arappav/errors/schema_math.py` — `MathInjectedError.error_type` validator

### 4. Prevent overlapping error injections

After inserting each error, update subsequent errors' `original_text` references, or require the perturber to target non-overlapping text spans via prompt engineering.

**File:** `src/arappav/utils/parsing.py` — `apply_error_injections()`

### 5. Count only insertable errors for k

If an error can't be mechanically inserted, decrement the effective k so the verifier's recall denominator is correct. Track `k_effective` separately from `k_requested`.

**File:** `src/arappav/reward/reward_fns.py` — `compute_rewards()`

### 6. Add a repetition penalty to the verifier

Detect and penalize outputs with >3 repeated JSON blocks to discourage collapse. Add a `repetition_penalty` field to `RewardOutput`.

**File:** `src/arappav/reward/reward_fns.py` — `compute_rewards()`

### 7. Soften format penalties during early training

Consider a warmup schedule where taxonomy near-misses get a warning instead of -10. The model needs to learn the exact enum strings through training, not be killed on first contact.

---

## Appendix: Raw Data Summary

### Perturber episodes

| Episode | Problem | Format | Errors in text | Failure reason |
|---------|---------|--------|----------------|----------------|
| `ep0` | Ratio b/a=3, find a | ❌ | 0/3 | Phantom (all 3) |
| `ep1` | Sum 4+5+...+11 | ✅ | 2/3 | err_003 overlapping |
| `ep2` | 2x+4=\|-17+3\| | ❌ | 0/3 | `'duplicated_operation'` |
| `ep3` | x²+bx+2008 factor | ❌ | 0/3 | JSON parse error |
| `ep4` | x²+6x+k=0, ratio 2:1 | ❌ | 0/3 | `'sign_error'` |
| `ep5` | \|x-2\| ≤ 5.6 integers | ❌ | 0/3 | `'addition_across'` |
| `ep6` | Expand (13x+15)·2x | ✅ | 2/3 | err_001 overlapping |

### Verifier episodes

| Paper | k | V claims (avg) | V recall | V precision | V reward |
|-------|---|----------------|----------|-------------|----------|
| `algebra_Level 2_1` | 3 | 2.0 | 0.00* | 0.00* | 0.00* |
| `algebra_Level 1_6` | 3 | 1.0 | 0.33 | 1.00 | 0.50 |

> \* Matcher failure — verifier correctly identified both detectable errors but got zero credit due to span-IoU and LaTeX escaping issues.

---

## 8. Implemented Fixes

All seven recommendations have been implemented. Here is a summary of the changes and their impact:

### 8.1 Matcher: substring containment + LaTeX normalization

**Files:** `src/arappav/reward/matcher.py`

- Added `_normalize_latex_escapes()` — normalises JSON double-escaped LaTeX (`\\\\cdot` → `\\cdot`) for span matching.
- Added `_is_substring_match()` — detects when one text is a **meaningful** substring of another (≥3 words or ≥40% length ratio), preventing trivial fragments like `"rate was"` from matching.
- Extended `_normalize_for_matching()` to strip `$` math delimiters (verifiers often drop these when quoting).
- In `match_claims_to_errors()`: when span IoU is below threshold but substring containment is detected, the score is boosted to just above the threshold.

**Before → After:**

| Episode | V recall | V reward | P reward |
|---------|----------|----------|----------|
| `ep1` | 0.00 → **1.00** | 0.00 → **1.00** | 1.00 → **-0.50** |
| `ep6` | 0.33 → **0.50** | 0.50 → **0.67** | 0.67 → **0.00** |

### 8.2 Fuzzy enum matching for error types

**Files:** `src/arappav/errors/schema_math.py`, `src/arappav/errors/schema.py`

- Added `_levenshtein_distance()` and `_fuzzy_match_*_error_type()` helpers.
- Two-tier matching: Levenshtein distance ≤ 5 (handles typos), OR word-level overlap + distance ≤ 10 (handles alternative wordings like `'addition_across'` → `'adding_across'`).
- Added `fuzzy_match_error_type` field validators on `InjectedError` and `MathInjectedError`.

**Result:** `'addition_across'`, `'wrong_opration'` now auto-correct with a warning instead of hard-failing. Completely unknown types (e.g., `'completely_wrong_name_xyz'`) still fail as expected.

### 8.3 Overlapping error injection detection

**Files:** `src/arappav/utils/parsing.py`

- `apply_error_injections()` now detects overlapping `original_text` spans before applying replacements.
- When two errors target the same text region, only the one with the longer `original_text` is applied; the shorter is skipped with a warning.
- This prevents the silent loss of error injections and makes the issue visible in logs.

### 8.4 Effective k, repetition penalty, missing error penalty

**Files:** `src/arappav/reward/reward_fns.py`

- **`k_effective`**: `compute_rewards()` now counts how many `injected_text` values are actually present in the perturbed text. Recall uses `k_effective` as denominator instead of `k_actual`.
- **Missing error penalty**: Perturber is penalised `-0.5` per error declared but not present in the text (e.g., overlapping errors that couldn't be injected).
- **Repetition penalty**: Verifier is penalised `-2.0` when it generates >5 repeated JSON blocks (detected via `_count_json_blocks()`), addressing the collapse observed in `ep1` response 2 (164 blocks).
- New config sections: `anti_repetition`, `anti_missing`.
- New fields in `RewardOutput`: `k_effective`, `repetition_penalty`.

### Test results

```
49 passed in 0.04s — all existing tests pass with the new behavior.
```

---

## 9. Round 1.1 — Review corrections (2026-07-13)

A verification pass over the §8 fixes found that several were incomplete or inaccurate; all are now corrected (79 tests passing, including 30 new regression tests pinned to the real Round 1 episodes in `tests/fixtures/round1_math_episodes.jsonl`).

### Corrections to §8 claims

- **§4 Cause B / §8.1 LaTeX normalization was never actually working.** The original `_normalize_latex_escapes` had a loop bug (each iteration replaced on the input, discarding prior replacements) and collapsed 4 literal backslashes → 2, one escaping level away from the real mismatch (parsed double-escaped JSON = 2 literal backslashes vs 1 in the text). Rewritten to collapse any 2+ backslash run before a letter to a single backslash.
- **ep1's err_002 false-zero was not caused by escaping** — the stored parsed claims have the same escaping as the text. The real cause was its span IoU of **0.494**, just under the 0.5 threshold. The substring-containment boost was what fixed ep1.
- **The repetition penalty was dead code** — no caller passed `verifier_raw_output`. Now threaded through `grpo_trainer`, `preference_builder`, `metrics`, and `run_single_rollout`. Default penalty softened -2.0 → -0.5 so a perfect-but-repetitive verifier (now 0.5) still ranks between clean-perfect (1.0) and a miss (0.0).
- **Fuzzy enum matching could silently mislabel** (`'sign_error'` → `'inversion_error'` via the generic shared word "error") while missing the actual ep2 failure (`'duplicated_operation'`). Rewritten in shared `arappav/errors/fuzzy.py`: strict tier distance ≤ 3; relaxed tier requires a shared **non-generic** word (exact or inflected) across all candidates. Now: `'duplicated_operation'` → `'duplication_error'`, `'sign_error'` → clean rejection.

### Additional fixes

- **Diff-based change coverage in the matcher**: a claim that quotes the error region *and* contains the text the diff of `original_text`→`injected_text` identifies as changed scores 0.9 — above any substring boost, so greedy assignment prefers claims that quote the actual mistake over ones quoting an unchanged clause.
- **Substring boost now scales with length ratio** (was a flat `threshold + 0.01`), removing arbitrary tie-breaking.
- **`k_effective` uses normalized containment** (`error_present_in_text`), so escaping/whitespace drift no longer counts a real error as missing.
- **Anti-spam floor**: `max_allowed` is now based on `max(1, k_effective)` — the verifier is no longer penalized for any claim when all injections were dropped.
- **Partial-config KeyError fixed** (`config["anti_spam"]` / `config["anti_duplicate"]` after a `.get()` guard).
- **`configs/reward/reward.yaml` updated** with the `anti_phantom`, `anti_repetition`, `anti_missing` sections that §8.4 claimed but never added.
- **`RewardOutput.match_details`** now exposes per-error match info (printed by `run_single_rollout.py`), so future analyses can distinguish matcher failures from verifier failures directly.

### Still open

- §7 recommendation 7 (format-penalty warmup schedule) — needs an episode/round counter in the training loop; deferred until verifier training exists. *(Partially superseded by the graded format penalties in §10.)*
- §6 issue 8 (verifier gets no signal when P fails format) — would require running the verifier on the unperturbed text in those episodes; also deferred (the verifier is currently frozen, so no signal is lost yet).

---

## 10. Round 2 analysis & fixes (2026-07-13)

Analysis of `data/rollouts_math/` (rounds 1–2, perturber + verifier) after GRPO training with loss 0.03 → ~0.00002.

### Key finding: the near-zero loss signals vanishing gradients, not success

GRPO learns only from **within-group reward variance**. All four persistently-failing perturber episodes (`3_0` phantom, `1_2` unmodified solution, `3_4` `sign_error`, `2_7` invalid JSON escape) produced **byte-identical outputs in rounds 1 and 2** — every completion in those groups scores a flat −10, advantage = 0, zero gradient. Format-valid rate moved only 3/8 → 4/8. The matcher itself was clean throughout: every match/miss inspected was genuine (e.g. round-2 `2_1`'s recall 0.33 reflects real verifier misses against a genuinely subtler perturbation, keff 2→3).

### Fixes implemented (95 tests passing; new: `tests/test_round2_fixes.py`)

1. **Backoff ordering bug** (`perturber.py`): the identical-to-original rejection ran *before* the mechanical injection backoff, killing exactly the episodes the backoff was built to save (`1_2`, both rounds). Parsing is now unified in `parse_and_backoff()` — validate without the unchanged check, apply backoff, reject only if still unchanged. Used by both `PerturberModel.generate` and the GRPO reward path (which previously had **no backoff at all**, so the trainer scored episodes the rollout collector would have salvaged).
2. **Invalid-JSON-escape repair** (`utils/parsing.py`): raw LaTeX in JSON strings (`\,`, `\cdot` → invalid escapes) is now repaired by doubling lone backslashes while consuming valid escapes left-to-right. Verified against `2_7`'s actual output prefix: the char-160 `Invalid \escape` failure is gone.
3. **Graded format penalties** (`grpo_trainer.py`, `reward.yaml`): unparseable JSON −10 vs parseable-but-schema-invalid −5 (`format_penalty_soft`) — all-failure groups now carry a gradient across failure severities.
4. **Within-group variance diagnostics** (`grpo_trainer.py`): each reward batch logs mean reward and the count of zero-variance groups; warns loudly when an entire batch produces no learning signal. Watch this instead of the loss.
5. **`sign_error` alias** (`errors/fuzzy.py`, `schema_math.py`): curated alias map (checked before fuzzy matching) resolves `sign_error` → `negative_number_error`; recovers `3_4`-class episodes without weakening the generic-word safeguard.
6. **Decode-time repetition bounds** (`models/generation_utils.py`): config generation kwargs are now translated per backend — previously HF-style keys (`max_new_tokens`, `do_sample`, `n_completions`) were passed verbatim to vLLM/HF, which is why collapsed verifier outputs ran to ~24k chars despite `max_new_tokens: 2048`. The verifier additionally stops at the first closed fenced JSON object (vLLM `stop`), and `repetition_penalty: 1.05` was added to `verifier.yaml`.
7. **Failed episodes now record `raw_output`** (`selfplay_loop.py`) so future rounds can be replayed against improved parsing (this round's invalid episodes stored only a 500-char prefix inside `format_reason`).

### Expected round-3 impact

`3_4` and `2_7` should become format-valid (alias + escape repair), `1_2` should be salvaged by the backoff, and `3_0` (phantom) should keep failing but at −5 — projected format-valid ≥ 6/8, with the remaining failures finally differentiated in reward.

### Deferred (by choice)

- Propagated-error policy (perturber counting a root error and its propagated final answer as two errors, seen in round-2 `2_1`) — revisit if observed at scale.

---

## 11. Round 3 analysis & fixes (2026-07-16)

Analysis of `new_rollouts/` (rounds 1–2 of the perturber-only run, verifier frozen), rescored with the repo's own `compute_rewards` + matcher against all 3 verifier samples per episode.

### Mechanics confirmed working

- **Round-2 fixes held**: format-valid 6/8 → 7/8 (beats the ≥6/8 projection); the graded −10/−5 penalties applied as designed; no byte-identical failure repeats.
- **Matcher clean**: every unmatched ground-truth error had ~zero overlap with any claim — genuine verifier misses, not matcher false-zeros.
- **Reward moved with real variance**: mean r_P −1.76 → −0.29 (valid-only 0.16 → 0.38), verifier recall 0.59 → 0.55, round-2 rewards spread across {−5 … 0.67}.

### Key finding: the deferred propagated/stacked-error hack went live

The perturber's highest-reward episodes won by **stacking**: declaring a root mistake plus its downstream consequences as separate errors, capping verifier recall at 1/k → r_P ≥ 0.67 without subtlety. Observed patterns (round-2): a wrong line plus its propagated `\boxed{}` answer (`2_7`), overlapping rewrites of one derivation block (`1_6`), and pure textual redundancy that changes no math (`4_3`: "$b=251+8=\boxed{259}$ and $b=8+251=\boxed{259}$"). `duplication_error` reached ~33% of injected errors. Additionally, verifier claims quoting `\boxed`/`\frac` with single backslashes in JSON parse to control characters (`\b`, `\f`), silently killing matches (round-1 `3_4` resp2).

### Fixes implemented (120 tests passing; new: `tests/test_round3_fixes.py`, pinned to `tests/fixtures/round3_math_episodes.jsonl`)

1. **Error units for recall** (`matcher.group_errors_into_units`, `reward_fns`, `reward.yaml error_units`): causally-linked errors collapse into one unit before recall — merge rules: overlapping spans / meaningful containment (injected↔injected, original↔original, and cross injected↔original), shared original→injected diff fragment (the same phantom term propagated across lines), changed-`\boxed{}` propagation into the nearest upstream error, and token-Jaccard near-duplicates (≥0.6). Recall = detected units / present units; a verifier flagging both root and propagation keeps full precision. Per-error recall remains available via `error_units.enabled: false`.
2. **Intra-episode anti-duplicate penalty** (`reward_fns`, `anti_duplicate.intra_episode_*`): near-verbatim re-injection within one episode costs −1.0 per duplicate.
3. **Redundant-restatement schema rejection** (`schema_math.MathInjectedError`): injected text repeating the same `\boxed{X}` twice, or joining identical statements with "and", now fails validation (→ graded −5) — restating content is not a math error. Genuine term-duplication (e.g. `+15\cdot 2x+15\cdot 2x`) is unaffected.
4. **JSON control-char repair in matcher normalization**: `\x08`→`\b`, `\x0c`→`\f` restored before matching (backspace/form-feed never appear in real math text).
5. **GRPO reward diagnostics persisted** (`make_perturber_reward_fn(diagnostics_path=…)`, wired in `selfplay_loop`): per-batch mean reward + within-group variance stats now land in `grpo_reward_diag_<mode>_round<N>.jsonl` next to the rollouts, so training-health data survives runs whose console logs aren't synced.
6. **Perturber prompt updated** (rules 4–5): errors must be independent; propagated/overlapping errors score as one; restatements are rejected.

### Effect on the observed episodes (rescored)

Round-1 `2_7` (propagated `+2(1)` chain): 0.67 → 0.0. Round-2 `1_6`/`2_7`: 0.67 → 0.5 (independent missed errors keep credit). Round-2 `4_3`: schema-rejected → −5. Legitimate wins (`3_0` resp2 0.67, `3_4` 0.33) unchanged. Round-1 pinned ep6 expectation updated: its three overlapping declarations are now one unit (recall 1.0 under units; 0.5 pinned with units disabled).

### Policy notes / still open

- Unit merging deliberately favors the (frozen, trusted) verifier when declarations overlap — the perturber avoids collapse by injecting genuinely independent errors, which is the incentive we want.
- The "X and Y and Y-swapped" restatement variant (round-2 `4_3` err_002) passes schema (only exact-duplicate segments are rejected) but is neutralized at reward level via unit collapse.
- Format-penalty warmup and verifier-signal-on-format-failure remain deferred (verifier still frozen).
