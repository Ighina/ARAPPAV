# PAPER_PLAN.md — ICLR 2027 submission

**Working title:** *Adversarial Skill Tuning: Self-Play Between Natural-Language Policies for
Mathematical Error Detection*

**Status (2026-09-07):** complete first draft, compiles clean.
`iclr2027/arappav.tex` → `arappav.pdf`, 10 pages (8 body + references + appendix), 0 errors,
0 overfull boxes, 26 citations all resolved. Anonymous (ICLR style auto-anonymises while
`\iclrfinalcopy` stays commented).

Build: `cd iclr2027 && pdflatex arappav && bibtex arappav && pdflatex arappav && pdflatex arappav`

---

## 1. The claim

Adversarial self-play for verifier training is brittle in weight space — three RL rounds
produced three successive reward hacks (phantom errors, stacked errors, redundant
restatement), each costing a round of GPU time to find. This paper asks whether the same game
is easier to steer when the policies are **prose**: a perturber and a verifier implemented as
versioned agent skills, rewritten between rounds by a static updater, scored by the same
programmatic reward an RL implementation would use.

The headline finding is not that it works. It is that **two rounds transferred and one did
not**, and nothing inside the game distinguished them.

---

## 2. Section-by-section content

### §1 Introduction
Motivation from published-literature error checking; the supervision bottleneck; the
GAN/self-play analogy; our own RL reward-hacking history as the problem statement.
Three contributions: the formalisation, the three guardrails, the external-validity protocol.

### §2 Related work
Four paragraphs: process supervision and error identification (ProcessBench, PRM800K,
Math-Shepherd); automated error detection in scientific writing; self-play and
self-improvement; optimising prompts/skills instead of weights.

Both requested references are load-bearing rather than decorative:

- **Gibney (2025), Nature** — grounds the applied motivation, and specifically why
  unconfirmed flags make this a *precision* problem, which is why both our reward and the
  benchmark metric punish false alarms as hard as misses.
- **Si et al. (2026), Ctx2Skill, arXiv 2604.27660** — closest prior work, given an explicit
  three-way contrast: (i) our reward is programmatic over machine-checkable ground truth, not
  a model judge, so credit assignment is exact and the failure mode to guard is scorer
  exploitation rather than judge bias; (ii) both our players are skills and both update, so
  curriculum and solver co-evolve; (iii) they prevent adversarial collapse with Cross-Time
  Replay on curated in-domain probes, we use an external human-annotated benchmark the update
  mechanism is structurally forbidden to read — and §5 shows that distinction is load-bearing,
  because the round where our loop over-fits is invisible from inside the game.

### §3 Method — the equations
| Eq. | Content |
|-----|---------|
| 1–2 | The episode: perturber emits $(\hat s, E)$, verifier sees only $(p,\hat s)$ and emits $C$ |
| 3 | Alignment score $\mu(e,c)$ = max(span IoU, diff-change coverage $\gamma{=}0.9$, containment ratio) |
| 4–5 | Error units $\mathcal{U}=E/\!\sim$, effective units, recall over **units** and precision over **claims** |
| 6–7 | $r_V = F_\beta - \lambda_{\text{spam}}(\cdot) - \lambda_{\text{rep}}(\cdot)$; $r_P = (1-\text{rec}) - \lambda_{\text{miss}} - \lambda_{\text{dup}} - \lambda_{\text{hist}}$ |
| 8 | Dominating two-level format penalty ($-10$ unparseable / $-5$ schema-invalid) |
| 9–12 | **The new part:** skill split $\sigma=(\iota,\varpi)$; scored round $\mathcal{E}^t$; update $\varpi^{t+1}=\Lambda(\varpi^t,\mathcal{A}(\mathcal{E}^t))$ as the textual analogue of a policy-gradient step (sampled batch → credit assignment → trust region); alternating updates as coordinate ascent |
| 13 | Faithfulness filter $\mathcal{A}(\mathcal{E}^t)=\mathcal{E}^t\setminus X^t$ — scorer-exploiting wins go to a bug report, never into the policy |
| 14–15 | ProcessBench reduction (earliest claimed step, $-1$ if no claims) and the metric |

Plus prose paragraphs on enforced context isolation and the convergence rule.

### §4 Experimental setup
Claude Code harness, Claude Opus 5; 8 episodes/round, $k=3$, Hendrycks MATH, neither player
frozen; ProcessBench protocol (20/subset balanced, round-independent seed, tagged
`<step_i>` boundaries, whole chain at once); the five leakage controls.

### §5 Results
Self-play table, ProcessBench table, regression anatomy, published baselines, filter output.

### §6 Limitations · §7 Conclusion · Appendix A
Appendix A prints the actual V3 rule text before and after the round-3 update, so a reader
can see the "also check the model" → "give it the same weight as re-derivation" promotion
that caused the regression.

---

## 3. Numbers reported (all read from artifacts on disk)

**Self-play** (`data/skill_rollouts/round_0*/round_summary.json`):

| Round | P | V | Fmt | $r_P$ | $r_V$ | Recall | Prec | Units/ep | FP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | v1 | v1 | 8/8 | 0.125 | 0.900 | 0.875 | 0.938 | 2.75 | 1 |
| 2 | v2 | v2 | 8/8 | 0.167 | 0.833 | 0.833 | 0.833 | 3.00 | 4 |
| 3 | v3 | v3 | 8/8 | 0.313 | 0.750 | 0.688 | 0.875 | 2.88 | 2 |

**ProcessBench** (`data/skill_evals/round_0*/eval_summary.json`, $n=80$, same items throughout):

| Verifier | gsm8k | math | olympiad | omnimath | Macro | Pooled |
|---|---|---|---|---|---|---|
| verify-v2 (after R1) | 0.889 | 0.889 | 0.889 | 0.800 | 0.867 | 0.869 |
| verify-v3 (after R2) | 0.889 | 0.947 | 0.947 | 0.800 | 0.896 | **0.897** |
| verify-v4 (after R3) | 0.889 | 0.947 | 0.847 | 0.800 | 0.871 | 0.874 |

Two additions beyond the run's own verdict:

- **Item-level flips.** v2→v3 is exactly two items, both wrong→right, no regressions.
  v3→v4 is exactly two, both right→wrong (both OlympiadBench), no gains. Unanimous in both
  directions — which is why the movements are reportable despite the sample size.
- **Macro alongside pooled F1.** ProcessBench's Table 3 macro-averages subsets; our harness
  pools all items. Both are given so the two are not conflated.

**Regression anatomy.** Error accuracy unchanged at 0.85 between v3 and v4; the entire loss is
OlympiadBench correct-accuracy 1.00 → 0.80. v4 kept every detection v3 had and added two false
alarms on clean chains — the predictable consequence of the two v3→v4 edits, which both pushed
the same way.

**Published baselines** (ProcessBench Table 3, full 3,400 cases): PRM800K 56.5, GPT-4o-0806
61.9, QwQ-32B-Preview 71.5, o1-mini 87.9. Included **for scale only**, with the
non-comparability stated in the text: 80 balanced items vs 3,400 natural, and 2024 single-pass
critics vs a 2026 frontier model following a tuned multi-step policy in an agent harness. No
SOTA claim is made.

---

## 4. Corrections and honesty flags carried in the draft

- **Corrected:** the faithfulness filter logged **18** entries across the six update reports,
  not 14 — roughly ten distinct defects, since both updaters independently flag the same
  matcher artifacts. §5.5 says so.
- **Stated in §6:** each reported transition is a two-item change; Wilson intervals on the
  pooled accuracies span roughly ±0.12; neither +0.029 nor −0.023 is individually significant.
  They are reported because the item-level direction is unanimous, not because $n=80$ supports
  a statistical claim.
- **Stated in §6:** there is no `verify-v1` ProcessBench point (the first eval ran after round
  1), so the draft cannot separate seed-policy quality from tuning gain.
- **Stated in §5.5:** round 2's recorded −0.067 self-play verifier regression is ≈ +0.058 on
  the updater's corrected reading, because of span-alignment artifacts. We report the
  uncorrected numbers and treat the gap as a measurement limitation rather than adjusting
  results.
- **Stated in §4:** round 4 is scaffolded but unscored; the paper analyses three completed
  rounds.

---

## 5. Planned experiments

Two families, plus the two scale-ups they subsume. Everything below runs on the **same fixed
80-item ProcessBench sample (seed 0)** used in §3, so every number in the paper stays on one
axis.

### What already exists

- `scripts/selfplay_api.py` runs the identical loop through stateless API calls, with
  `PROVIDERS` for **anthropic**, **openai** and **deepseek**. Artifacts are prompts
  (`prompts/<role>/<model>/vN/prompt.txt`) instead of skills, tuned by
  `update-{perturb,verify}-api`.
- **`selfplay_api.py verify --inbox/--outbox` already accepts any directory of
  `{problem, solution_to_review}` records — exactly what `processbench_eval.py prepare`
  emits.** So an API model can be evaluated on ProcessBench today, through the same renderer
  and the same leak guard as the skill path. No new evaluation code is needed.
- **DeepSeek is three rounds in.** `data/api_rollouts/deepseek-v4-pro/round_01..03`, prompts
  v1–v4, 8/8 format-valid every round; $r_V$ 0.933 → 0.707 → 0.835 and $r_P$ 0.042 → 0.292 →
  −0.042. Non-monotone, and a different shape from the Opus skill trajectory — already
  interesting.
- **A DeepSeek ProcessBench eval is in flight:**
  `data/api_evals/deepseek-v4-pro/round_03` (`verify-deepseek-v4-pro-v4`, seed 0, same 80
  items), 60/80 answered at time of writing. Finish with
  `processbench_eval.py score --root data/api_evals/deepseek-v4-pro --round 3`.
- `sonnet-5` is seeded (v1 prompts) with no rounds run.

### What needs building

1. **MiniMax provider** (note: the lab spells it *MiniMax*). One entry in `PROVIDERS` with the
   OpenAI-compatible `base_url` and `MINIMAX_API_KEY`, one prefix in the provider-inference
   map, then `selfplay_api.py seed --model <id>`. This is the only code change either
   experiment family requires.
2. **A single-pass critic-prompt runner** for E1(b) — ProcessBench's own prompt shape rather
   than our contract. Can be a `--prompt-file` flag on `selfplay_api.py verify`.

### E1 — Contemporary baselines

The published Table 3 numbers are 2024-era and not comparable to ours (§5.4 of the draft says
so). We need our own baselines on our own protocol. **Two distinct baseline notions, and the
distinction is the point:**

| Point | What it is | What the gap above it measures |
|---|---|---|
| (b) **Critic prompt** | ProcessBench's own single-pass critique prompt | — |
| (a) **Seed policy v1** | our untuned contract + hand-written strategy | what the *contract and harness* buy |
| (c) **Tuned vN** | after $N$ rounds of self-play | what *tuning* buys |

Reporting all three per model separates the two mechanisms the paper conflates today, and
(a) also fills the missing-`verify-v1` hole already flagged in §4. Models: Opus 5, Sonnet 5,
DeepSeek V4-pro, MiniMax, and a GPT-5.1-class point if `OPENAI_API_KEY` is available (the
`openai` provider already works).

**Deliverable:** one table, rows = models, column groups = {critic prompt, seed v1, tuned vN},
cells = ProcessBench F1 per subset + overall.

### E2 — Three tiers of open access

Run the full loop independently at each tier, identical problems, seed, $k$, reward, and
evaluation sample:

| Tier | Model | Access | Artifact |
|---|---|---|---|
| Closed frontier | Claude Opus 5 | closed weights, closed API | skill (harness) |
| Open frontier-scale | DeepSeek V4-pro | open weights, self-hostable | prompt (API) |
| Open mid-scale | MiniMax | open weights, smaller/cheaper | prompt (API) |

Questions the tiering answers:

- **Does the loop work below the frontier?** Or does adversarial skill tuning need a model
  strong enough to diagnose its own failures — i.e. is $\Lambda$ the capability bottleneck?
- **Who gains most?** More headroom at the small end, but less ability to execute a
  sophisticated policy. The sign of that trade-off is a real result either way.
- **Does the round-3 overfit reproduce?** If the same over-generalisation appears at every
  tier, it is a property of the *loop*; if only at the frontier, it is a property of a model
  strong enough to over-fit its opponent. This is the single most valuable thing the tiering
  can tell us.

**Confound to control:** tier and harness type are currently entangled — Opus runs as a skill
in the agent harness (files, tools, multi-step), DeepSeek and MiniMax as stateless prompts.
**Run Opus through the API path as well**, so there is at least one model measured both ways.
Without that, any tier difference is uninterpretable.

**Bonus experiment, nearly free:** *cross-model policy transplant.* Take the Opus-tuned
verifier policy and run it unchanged on MiniMax, and vice versa. Ctx2Skill claims skills
distilled from frontier models let smaller models beat unaided larger ones; we hold exactly
the artifacts needed to test that claim on a human-annotated benchmark. A 2×3 transplant
matrix costs one evaluation pass per cell.

### E3 — Scale the evaluation

Everything in §3 rests on 10 items per class per subset. Move to several hundred per subset,
or the full 3,400, at least for the final version of each model's tuned policy. Until then no
individual transition is statistically significant, and the paper says so.

### E4 — Freeze ablation

`freeze=perturber` and `freeze=verifier` isolate whether the verifier's gains need a
co-evolving opponent at all. Currently unmeasured, and an obvious reviewer question. Cheap:
the flag is already plumbed through both loops.

---

## 6. Other work before submission

1. **Fix the span-alignment artifacts** the faithfulness filter logged — trailing punctuation
   falling outside normalisation, and the $\theta = 0.5$ IoU gate rejecting claims that scored
   0.441/0.467. These distort the self-play numbers the paper prints (round 2's recorded
   −0.067 is ≈ +0.058 corrected).
2. **More rounds and a second seed** on at least the Opus trajectory, to show whether the
   round-3 overfit is systematic or incidental. Round 4 is already scaffolded.
3. **Decide the scope question:** fold paper-mode (academic-paper perturbation) into this
   submission, or keep it as future work. The draft currently scopes to math mode only.
4. **Decide how to frame the API variant.** With E2 done it stops being a footnote and
   becomes a main result — the paper would then be about adversarial policy tuning in general,
   with skills and prompts as two instantiations. That likely changes the title and the
   framing of §3.3.

---

## 7. File map

| Path | Role |
|---|---|
| `iclr2027/arappav.tex` | the paper |
| `iclr2027/arappav.bib` | 26 references |
| `iclr2027/arappav.pdf` | compiled output |
| `iclr2027/iclr2027_conference.{sty,bst}` | unmodified template |
| `data/skill_rollouts/round_0*/round_summary.json` | §5.1 numbers |
| `data/skill_rollouts/round_0*/update_*.json` | §5.1 qualitative record, §5.5 defect log |
| `data/skill_evals/round_0*/eval_summary.json` | §5.2 numbers |
| `.claude/skills/{perturb,verify}-v*/SKILL.md` | the skill policies; changelogs are the lineage |
| `scripts/skill_selfplay.py`, `scripts/processbench_eval.py` | the skill-path harnesses (§4) |
| `scripts/selfplay_api.py` | the API-path harness — E1/E2 run through this |
| `prompts/<role>/<model>/vN/prompt.txt` | the API-path policies |
| `data/api_rollouts/<model>/round_0*/` | API self-play rounds (DeepSeek: 3 done) |
| `data/api_evals/<model>/round_0*/` | API ProcessBench evals (DeepSeek round 3 in flight) |
