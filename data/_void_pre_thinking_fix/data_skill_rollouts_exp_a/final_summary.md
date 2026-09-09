# ARAPPAV Self-Play Run Report — `exp_a`

## 1. Configuration

| Setting | Value |
|---|---|
| Dataset / source | `hendrycks`, topic `algebra`, `per_topic = 300` |
| Problems file | none (`null`) — problems drawn from the source above |
| Episodes per round | 8 |
| Rounds | 10 (rounds 0–9) |
| `k` (errors requested per episode) | 3 |
| Initialization | `cold` |
| Freeze configuration | `none` (both perturber and verifier updatable every round) |
| Context passed to agents | Yes (`no_context = false`) |
| Episode model / backend | `claude-haiku-4-5`, backend `api`, provider `null` |
| Updater model / provider | `claude-opus-5`, provider `null` |
| Seed | 42 |
| Run root | `data/skill_rollouts/exp_a`; skills root `.claude/skills` |
| Policy prefixes | `exp_a_perturb`, `exp_a_verify` |
| ProcessBench | **disabled** (`processbench_enabled = false`); configured but unused values: 5 per subset, root `data/skill_evals`, seed 0 |
| Other | `overwrite_policies = false`, `dry_run = false`, `timeout = 900`, `retry_format = 1`, `resume = true` |

Total episodes run: 80 (10 rounds × 8 episodes).

## 2. Round-by-round table

| round | perturb policy | verify policy | format-valid | mean r_P | mean r_V | recall | precision | units/ep | penalties | ProcessBench |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | exp_a_perturb-v1 | exp_a_verify-v1 | 1.0 | 0.4167 | 0.5208 | 0.4583 | 0.6458 | 1.875 | 0 | disabled |
| 1 | exp_a_perturb-v2 | exp_a_verify-v2 | 1.0 | 0.2292 | 0.6458 | 0.5833 | 0.8125 | 2.125 | 0 | disabled |
| 2 | exp_a_perturb-v3 | exp_a_verify-v3 | 1.0 | 0.5 | 0.2375 | 0.25 | 0.2292 | 2.0 | 0 | disabled |
| 3 | exp_a_perturb-v4 | exp_a_verify-v4 | 1.0 | 0.4792 | 0.5417 | 0.4583 | 0.75 | 2.5 | 0 | disabled |
| 4 | exp_a_perturb-v5 | exp_a_verify-v5 | 0.875 | -0.3542 | 0.6048 | 0.5476 | 0.7143 | 2.0 | 0 | disabled |
| 5 | exp_a_perturb-v6 | exp_a_verify-v6 | 0.875 | -0.0625 | 0.4381 | 0.3571 | 0.619 | 2.0 | 0 | disabled |
| 6 | exp_a_perturb-v7 | exp_a_verify-v7 | 0.75 | -1.2917 | 0.1944 | 0.1389 | 0.3333 | 2.5 | 0 | disabled |
| 7 | exp_a_perturb-v8 | exp_a_verify-v8 | 0.875 | -0.25 | 0.3048 | 0.2857 | 0.381 | 2.0 | 0 | disabled |
| 8 | exp_a_perturb-v9 | exp_a_verify-v9 | 0.625 | -1.4375 | 0.3 | 0.3 | 0.3 | 2.0 | 0 | disabled |
| 9 | exp_a_perturb-v10 | exp_a_verify-v10 | 1.0 | 0.5 | 0.4125 | 0.375 | 0.5208 | 2.375 | 0 | disabled |

## 3. Policy evolution

Freeze was set to `none`, and both sides advanced by exactly one version every round, so every round's update was applied to both policies.

**Perturber:**

| Round | Version running |
|---|---|
| 0 | `exp_a_perturb-v1` (cold start) |
| 1 | `exp_a_perturb-v2` |
| 2 | `exp_a_perturb-v3` |
| 3 | `exp_a_perturb-v4` |
| 4 | `exp_a_perturb-v5` |
| 5 | `exp_a_perturb-v6` |
| 6 | `exp_a_perturb-v7` |
| 7 | `exp_a_perturb-v8` |
| 8 | `exp_a_perturb-v9` |
| 9 | `exp_a_perturb-v10` |

**Verifier:**

| Round | Version running |
|---|---|
| 0 | `exp_a_verify-v1` (cold start) |
| 1 | `exp_a_verify-v2` |
| 2 | `exp_a_verify-v3` |
| 3 | `exp_a_verify-v4` |
| 4 | `exp_a_verify-v5` |
| 5 | `exp_a_verify-v6` |
| 6 | `exp_a_verify-v7` |
| 7 | `exp_a_verify-v8` |
| 8 | `exp_a_verify-v9` |
| 9 | `exp_a_verify-v10` |

No round ran a frozen side; no round reused a prior version. The version numbering implies an update was produced after each of rounds 0–8 for both sides (v10 was the last version exercised, at round 9; whether a v11 was produced after round 9 is not recorded in the run data).

## 4. Rewards and penalties by round

All four penalty totals — duplicate, spam, repetition, phantom — are **0 in every one of the 10 rounds**. No penalty term ever fired, so none of the reward movement below is attributable to penalties as reported in the metrics block. (Note the tension with round 4's format failure, discussed in §8.)

**Perturber reward (mean over all 8 episodes) and valid-only reward:**

| Round | mean r_P | Δ vs prior | mean r_P (valid only) | Δ vs prior |
|---|---|---|---|---|
| 0 | 0.4167 | — | 0.4167 | — |
| 1 | 0.2292 | ↓ 0.1875 | 0.2292 | ↓ 0.1875 |
| 2 | 0.5000 | ↑ 0.2708 | 0.5000 | ↑ 0.2708 |
| 3 | 0.4792 | ↓ 0.0208 | 0.4792 | ↓ 0.0208 |
| 4 | −0.3542 | ↓ 0.8334 | 0.3095 | ↓ 0.1697 |
| 5 | −0.0625 | ↑ 0.2917 | 0.6429 | ↑ 0.3334 |
| 6 | −1.2917 | ↓ 1.2292 | 0.7778 | ↑ 0.1349 |
| 7 | −0.2500 | ↑ 1.0417 | 0.4286 | ↓ 0.3492 |
| 8 | −1.4375 | ↓ 1.1875 | 0.7000 | ↑ 0.2714 |
| 9 | 0.5000 | ↑ 1.9375 | 0.5000 | ↓ 0.2000 |

The all-episode and valid-only means diverge sharply from round 4 onward. The all-episode perturber reward is negative in rounds 4–8 while the valid-only reward stays positive throughout (range 0.2292–0.7778). The negative all-episode means therefore track format failures, not degraded error-injection quality on the episodes that parsed.

**Verifier reward, recall, precision:**

| Round | mean r_V | Δ | recall | Δ | precision | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.5208 | — | 0.4583 | — | 0.6458 | — |
| 1 | 0.6458 | ↑ 0.1250 | 0.5833 | ↑ 0.1250 | 0.8125 | ↑ 0.1667 |
| 2 | 0.2375 | ↓ 0.4083 | 0.2500 | ↓ 0.3333 | 0.2292 | ↓ 0.5833 |
| 3 | 0.5417 | ↑ 0.3042 | 0.4583 | ↑ 0.2083 | 0.7500 | ↑ 0.5208 |
| 4 | 0.6048 | ↑ 0.0631 | 0.5476 | ↑ 0.0893 | 0.7143 | ↓ 0.0357 |
| 5 | 0.4381 | ↓ 0.1667 | 0.3571 | ↓ 0.1905 | 0.6190 | ↓ 0.0953 |
| 6 | 0.1944 | ↓ 0.2437 | 0.1389 | ↓ 0.2182 | 0.3333 | ↓ 0.2857 |
| 7 | 0.3048 | ↑ 0.1104 | 0.2857 | ↑ 0.1468 | 0.3810 | ↑ 0.0477 |
| 8 | 0.3000 | ↓ 0.0048 | 0.3000 | ↑ 0.0143 | 0.3000 | ↓ 0.0810 |
| 9 | 0.4125 | ↑ 0.1125 | 0.3750 | ↑ 0.0750 | 0.5208 | ↑ 0.2208 |

The verifier peaks at round 1 (r_V 0.6458, precision 0.8125) and round 4 (r_V 0.6048, recall 0.5476), and bottoms at round 6 (r_V 0.1944, recall 0.1389, precision 0.3333). Round 2 is a single-round collapse (precision 0.2292) that recovers fully by round 3.

**Units per episode** stayed in the band 1.875–2.5 across all 10 rounds, never reaching `k = 3`. Highest: rounds 3 and 6 (2.5). Lowest: round 0 (1.875).

## 5. ProcessBench

**ProcessBench was not run in any of the 10 rounds.** `processbench_enabled` is `false` in the run configuration, and every round's `processbench` block records `enabled: false` with the note "ProcessBench was not run for this round."

Consequently **no external-validity claim can be made from this run.** All recall, precision, and reward numbers above are self-play quantities measured against the perturber's own declared error list. Nothing here establishes that the verifier's behavior transfers to an independent, externally-labeled step-error benchmark.

## 6. Findings by round

Counts of injected errors below are as recorded in each round's `error_type_detection` block.

**Round 0 (8/8 format-valid, 9 detected / 15 undetected).**
Perfect detection on the singleton types `wrong_fraction` (1/1), `geometry_definition` (1/1), `irrelevant_feature` (1/1). The dominant type `wrong_operation` was injected 10 times and detected 3 (0.30). `wrong_sequence_term` 1/4 (0.25). Zero detection on `variable_misconception` (0/1), `additive_thinking` (0/1), `inverse_operation_error` (0/1). Three false positives (ep00, ep05, ep06) — all cases where the verifier flagged a verification/consistency line and produced an explanation that contradicts itself or restates the flagged computation as correct. Unit collapse in 7 of 8 episodes: `k = 3` requested but 2 units produced in ep01–ep05 and only 1 unit in ep06 and ep07.

**Round 1 (8/8 format-valid, 9 detected / 15 undetected).** Best verifier round of the run.
Full detection on `wrong_fraction` (2/2) and `variable_misconception` (1/1); `operand_swap` 2/3. Survivors: `whole_number_bias` 0/3, plus singletons `denominator_only`, `negative_number_error`, `incomplete_solution` all 0/1. `wrong_operation` again the largest bucket at 3/9. Only 2 false positives — one of which (ep01) is a flag on a LaTeX typo (`oxed{-9}`) rather than a mathematical error, and one (ep03) a substantive disagreement about the smallest `n`. Unit collapse in 6 episodes (2 units in five of them, 1 unit in ep02).

**Round 2 (8/8 format-valid, 5 detected / 19 undetected).** Verifier's worst precision (0.2292).
Detection was concentrated entirely in ep01, ep03, ep06; ep00, ep02, ep04, ep05, ep07 yielded zero detections between them. `variable_misconception` 0/3, `additive_thinking` 0/2, `wrong_sequence_term` 0/2, `duplication_error` 0/1, `decimal_magnitude` 0/1 all survived completely. Seven false positives — the largest count of the run — including two chained flags in ep03 (`16x = 64` then `x = 8`), two in ep05, two in ep06, and one in ep07 where the verifier flagged a speed value while conceding the resulting distance was correct. Unit collapse in 7 of 8 episodes.

**Round 3 (8/8 format-valid, 8 detected / 16 undetected).** Highest units/ep of the run (2.5), tied with round 6.
`operand_swap` 3/4 (0.75), `decimal_magnitude` 1/1, `wrong_fraction` 1/1, `negative_number_error` 1/2. Complete survivors: `whole_number_bias` 0/4, `incomplete_solution` 0/3, `inversion_error` 0/2, `duplication_error` 0/1, `variable_misconception` 0/1. Only 1 false positive (ep04, a geometric-series indexing claim).

**Round 4 (7/8 format-valid, 7 detected / 14 undetected).** First format failure of the run.
ep03 failed schema validation with a **phantom error**: `injected_text` equal to `original_text` for `err_001`, i.e. no modification was actually made. Detection reversed relative to earlier rounds: `whole_number_bias`, previously a total survivor, was detected 4/8 (0.50); `operand_swap` 1/1; `duplication_error` 1/2. `wrong_operation` fell to 1/4 (0.25). Survivors: `variable_misconception` 0/2, `denominator_only` 0/1, `incomplete_solution` 0/1, `additive_thinking` 0/1, `wrong_sequence_term` 0/1. Two false positives (ep00 factorization/α-β claim, ep02 sign of the root). Unit collapse in 6 episodes.

**Round 5 (7/8 format-valid, 6 detected / 15 undetected).**
Format failure in ep01: schema rejected 2 errors when exactly 3 were required. `incomplete_solution` 1/1, `variable_misconception` 1/2, `negative_number_error` 1/2, `whole_number_bias` 2/5. `wrong_operation` dropped to its lowest yet, 1/7 (0.143). Complete survivors: `duplication_error`, `inversion_error`, `operand_swap`, `wrong_fraction`, all 0/1. Two false positives, both in ep02 and both on completing-the-square arithmetic; the second false-positive explanation visibly contradicts itself mid-text ("Wait, let me recalculate… Actually this is correct if we proceed from t…"). Unit collapse in all 7 valid episodes, every one at 2 units.

**Round 6 (6/8 format-valid, 2 detected / 16 undetected).** Worst verifier round (r_V 0.1944, recall 0.1389).
Two format failures: ep03 failed at the **JSON parse stage** — the response began mid-content (`{19440}$\n\nThese are 6 separable computational places…`), i.e. reasoning text leaked into the JSON payload — and ep07 returned only 1 error where 3 were required. Only two errors were detected in the entire round: one `wrong_operation` (ep04) and one `variable_misconception` (ep06). Everything else survived: `whole_number_bias` 0/4, `additive_thinking` 0/4, `wrong_sequence_term` 0/3, `incomplete_solution` 0/1, `duplication_error` 0/1, `negative_number_error` 0/1. **Zero false positives** — the verifier failed by silence, not by over-flagging. Unit collapse in 3 episodes.

**Round 7 (7/8 format-valid, 4 detected / 17 undetected).**
Format failure in ep02: 1 error returned where 3 were required. Detection: `wrong_sequence_term` 1/2, `negative_number_error` 1/2, `whole_number_bias` 1/3, `operand_swap` 1/5 (0.20). Complete survivors: `wrong_operation` 0/4, `variable_misconception` 0/2, `additive_thinking` 0/2, `incomplete_solution` 0/1. One false positive (ep07, an arithmetic claim of 91 vs 92 in `wx + yz`). All four detections came from just two episodes (ep03, ep07); ep00, ep04, ep05, ep06 each had all three errors survive. Unit collapse in 6 episodes.

**Round 8 (5/8 format-valid — worst of the run; 2 detected / 13 undetected).**
Three format failures, all schema, all the same shape: ep01 and ep04 returned 1 error, ep06 returned 2, where 3 were required. Only 2 detections: `inversion_error` 1/1 (ep00) and one `wrong_operation` (ep03, overlap 0.519). Complete survivors: `whole_number_bias` 0/4, `additive_thinking` 0/2, `operand_swap` 0/2, `wrong_fraction` 0/2, and 3 of 4 `wrong_operation`. One false positive (ep00, inverse-variation multiplier). Verifier recall and precision were both exactly 0.30.

**Round 9 (8/8 format-valid — recovery; 7 detected / 17 undetected).**
The perturber concentrated heavily on `negative_number_error` (8 injections, more than any other type in this round), of which 2 were detected (0.25); ep03 had all three of its negative-number errors survive. Full detection on `variable_misconception` (2/2) and `inversion_error` (1/1); `wrong_sequence_term` 1/2. `wrong_operation` 1/6 (0.167). Complete survivors: `denominator_only`, `operand_swap`, `incomplete_solution`, `whole_number_bias`, `proportional_reasoning_error`, each 0/1. Two false positives (ep01 sign of the root, ep06 restatement of a rate definition). Unit collapse in 5 episodes, all at 2 units.

**Cross-round patterns in the findings.**
- The perturber never once produced the requested `k = 3` separable error units on average; mean units/ep sat between 1.875 and 2.5 in all 10 rounds, and unit collapse appears in every round's findings (3 to 7 episodes per round).
- `wrong_operation` was the most heavily injected type across the run and its detection rate was below 0.40 in every round in which it appeared (0.30, 0.333, 0.375, 0.40, 0.25, 0.143, 0.333, 0.00, 0.25, 0.167 for rounds 0–9).
- `whole_number_bias` is the clearest example of non-monotonic detection: 0.50 (r0), 0.00 (r1), 0.333 (r2), 0.00 (r3), 0.50 (r4), 0.40 (r5), 0.00 (r6), 0.333 (r7), 0.00 (r8), 0.00 (r9).
- All 10 format failures in the run were perturber-side output failures; none was a verifier-side failure.

## 7. Final conclusions

What this run established, over 80 episodes total and only 8 episodes per round:

1. **The pipeline executed all 10 rounds under `cold` start with no freezing**, producing 10 perturber versions and 10 verifier versions. No round ran an empty or reused policy.
2. **No penalty term ever fired.** Duplicate, spam, repetition, and phantom penalties were 0 in all 10 rounds as recorded in the metrics.
3. **The perturber never met the `k = 3` target on average.** Units per episode ranged 1.875–2.5 across all 10 rounds. This is the single most consistent observation in the run: it holds in 10/10 rounds, i.e. all 80 episodes.
4. **Format validity degraded in the middle of the run and recovered at the end.** 1.0 in rounds 0–3, then 0.875 / 0.875 / 0.75 / 0.875 / 0.625, then 1.0 in round 9. This is 10 failed episodes out of 80. The rank correlation between format validity and all-episode perturber reward is visually strong (r_P negative exactly in the five rounds with failures, positive in the five rounds without), but the valid-only perturber reward stayed positive throughout — so the negative reward means reflect format penalties, not weaker perturbations on parsed episodes.
5. **Verifier performance did not trend.** r_V went 0.5208 → 0.6458 → 0.2375 → 0.5417 → 0.6048 → 0.4381 → 0.1944 → 0.3048 → 0.3000 → 0.4125. It ends (0.4125) below where it started (0.5208), but with 8 episodes per round the round-to-round swings — a 0.41 drop from round 1 to 2, a 0.30 rise from round 2 to 3 — are on the order of one or two episodes' worth of outcome. **No claim of verifier improvement or degradation over the run is supported by this sample.** The apparent "decline" from round 4 to round 6 is a two-round move over 16 episodes and should not be called a trend.
6. **Some error types survived detection far more reliably than others, but per-round counts are tiny.** `wrong_operation` was below 0.40 detection in all 10 rounds; that is the one type-level pattern with enough injections behind it (roughly 5–10 injections per round) to be worth noting. Types like `geometry_definition`, `irrelevant_feature`, `proportional_reasoning_error`, and `decimal_magnitude` appeared 1–2 times total; their 0.0 or 1.0 detection rates carry essentially no information.
7. **Nothing about external validity was measured.** ProcessBench was disabled for the whole run. Every number here is internal to self-play.

Framed strictly: this run demonstrates that the loop runs end-to-end and produces stable, penalty-free measurements, that the perturber systematically under-delivers on `k`, and that the perturber's output-format reliability was unstable in rounds 4–8. It does not demonstrate that either policy got better or worse at its task.

## 8. Anomalies and failures

**Format-invalid episodes (10 total across the run, all perturber-side):**

| Round | Episode | Stage | Nature |
|---|---|---|---|
| 4 | ep03 | schema | Phantom error: `injected_text == original_text` for `err_001` — no modification made |
| 5 |
