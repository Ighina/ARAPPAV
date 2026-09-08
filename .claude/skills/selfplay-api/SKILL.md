---
name: selfplay-api
description: Run the API-based self-play loop — alternate a Perturber prompt and a Verifier prompt through stateless API calls (Anthropic, OpenAI or DeepSeek) over scored rounds, then tune them with update-perturb-api / update-verify-api into prompts/<role>/<model>/vN/prompt.txt. Use when asked to run API self-play, lesson tuning, or to improve the perturber/verifier prompts for a specific model.
---

# selfplay-api — lesson-tuning rollout loop

Same game, same episodes, same reward function as `selfplay`. Two things differ, and they
are the whole point.

**The players are API calls, not the harness.** Each role is one stateless request:
one system prompt, one user message, one JSON reply. In `selfplay` both roles are played by
this agent, and their separation rests on instructions plus a subagent boundary — a
convention. Here they share no context *because there is no shared context to share*. A
round is also replayable: prompt file + episode input is the entire cause of an output, with
no harness system prompt, tool results, memory or skill files in the way.

**The tuned artefact is a prompt, not a skill.** Nothing is written under `.claude/skills/`.
Versions live at:

```
prompts/perturb/<model_slug>/v<N>/prompt.txt
prompts/verify/<model_slug>/v<N>/prompt.txt
```

`<model_slug>` is the model id without its `claude-` prefix (`claude-sonnet-5` → `sonnet-5`),
so several models can be tuned side by side without colliding.

```
  perturb vN ─┐                                    ┌─ update-perturb-api ─→ v(N+1)
              ├─→ episodes ─→ compute_rewards ─→ ──┤
  verify  vN ─┘   (API calls)   (repo's own)       └─ update-verify-api  ─→ v(N+1)
```

Scoring is **not** reimplemented. `scripts/skill_selfplay.py` does the sampling, parsing,
error units, penalties, anti-duplicate history and summaries, pointed at a separate rollout
root, so results are directly comparable with the harness loop.

---

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | `claude-opus-5` | any supported model id; sets `<model_slug>` |
| `provider` | inferred | `anthropic` / `openai` / `deepseek`; only needed when the id is ambiguous |
| `rounds` | 3 | maximum rounds; may stop earlier on convergence |
| `episodes` | 8 | episodes per round |
| `k` | 3 | errors per episode (or `--k-range LO HI`) |
| `freeze` | `none` | `perturber` or `verifier` freezes that side |
| `source` | `local` | `local` replays problems from past rollout logs; `hendrycks` downloads |
| `root` | `data/api_rollouts/<model_slug>` | rollout root |
| `effort` | `high` | `output_config.effort` for both roles |
| `concurrency` | 4 | parallel API calls per pass |
| `eval` | `off` | ProcessBench check of the Verifier prompt (see step 7) |

Starting versions are the highest existing `vN` under each role's prompt directory.
Announce the plan — model, rounds, episodes, k, freeze, starting versions — before round 1.

**Cost.** Every pass spends real money: roughly `2 × episodes` calls per round, plus 80 per
ProcessBench eval. Report the round's spend from the API log, and get the user's agreement
before a run substantially larger than the defaults.

---

## Providers

Three backends, chosen automatically from the model id (`claude-*` → Anthropic,
`deepseek*` → DeepSeek, `gpt-*`/`o1`/`o3`/`o4`/`chatgpt*` → OpenAI). Pass `--provider`
for anything else.

| Provider | SDK | Credential | Notes |
|----------|-----|------------|-------|
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` or `ant auth login` | adaptive thinking + `output_config.effort` |
| `openai` | `openai` | `OPENAI_API_KEY` | `max_completion_tokens`; `reasoning_effort` sent only to `gpt-5`/`o1`/`o3`/`o4` |
| `deepseek` | `openai` (base URL `https://api.deepseek.com`) | `DEEPSEEK_API_KEY` | `max_tokens`; no effort control |

Each model gets its own prompt tree and its own rollout root, so tuning DeepSeek does not
touch the Claude prompts and the two can be compared on identical episodes.

Two things to keep in mind when comparing providers:

- `effort` is not a common scale. Anthropic takes `low…max`; OpenAI's reasoning models take
  `low/medium/high` (this loop clamps `xhigh`/`max` onto `high`); DeepSeek takes nothing. A
  cross-provider table is comparing *configurations*, not equal compute.
- `--json-mode` (OpenAI/DeepSeek only) forces `response_format=json_object` and will improve
  their format-valid rate. It has no Anthropic equivalent, so switching it on makes format
  metrics incomparable. It is off by default for that reason.

Only Anthropic rates are built in. For OpenAI or DeepSeek, pass `--price-in` / `--price-out`
(USD per 1M tokens) if you want costed logs; otherwise cost is reported as `n/a` rather than
guessed.

## Setup (once per model)

```bash
python scripts/selfplay_api.py seed --model <model>
```

Writes `v1` for both roles from `prompts/_seed/{perturb,verify}.txt`. Add
`--from-skill N` to import the tuned policy from `.claude/skills/<role>-vN` instead of the
first-principles seed — useful for a head-to-head, but note that those policies were tuned
under the harness this loop exists to get away from, so the default is the clean start.

Credentials: `ANTHROPIC_API_KEY`, or `ant auth login`. The runner exits with a clear message
if neither is present.

Each `prompt.txt` has two marked regions:

- `<<<INVARIANT>>> … <<<END INVARIANT>>>` — input format, output contract, taxonomy, reward
  and matching rules. A transcription of the code. **Never edited by an updater.**
- `<<<POLICY>>> … <<<END POLICY>>>` — the strategy. This is what gets tuned.

---

## Per-round procedure

Let `$R` be the round, `$M` the model, `$ROOT` the rollout root, and `$P` / `$V` the current
prompt versions. Note that `skill_selfplay.py` takes `--root` **before** its subcommand.

### 1. Scaffold

```bash
python scripts/skill_selfplay.py --root $ROOT init --round $R --episodes $E --k $K \
  --source $SOURCE --freeze $FREEZE \
  --perturb-skill perturb-$SLUG-v$P --verify-skill verify-$SLUG-v$V
```

Problems are drawn without reuse across rounds within a root.

### 2. Perturber pass

```bash
python scripts/selfplay_api.py perturb --model $M --round $R --version $P --root $ROOT
```

One call per episode, in parallel. Each writes `episodes/<id>/perturb.json`.
Add `--dry-run 1` first if you have changed a prompt and want to see the rendered request
before spending anything.

### 3. Validate and build the verifier's inputs

```bash
python scripts/skill_selfplay.py --root $ROOT prepare-verify --round $R
```

Parses each perturbation exactly as the RL path does (fence stripping, LaTeX escape repair,
mechanical injection backoff), records format validity, and writes `verify_inbox/<id>.json`
containing only the problem and the text to review.

### 4. Verifier pass

```bash
python scripts/selfplay_api.py verify --model $M --round $R --version $V --root $ROOT
```

**Do not spawn a subagent and do not verify anything yourself.** Isolation here is
structural: the runner builds the verifier message from a two-field allowlist
(`problem`, `solution_to_review`), refuses to send a message containing ground-truth
markers, and never passes `k`. That guarantee is the reason this loop exists — reproducing
the pass by hand throws it away.

### 5. Score and summarize

```bash
python scripts/skill_selfplay.py --root $ROOT score --round $R
python scripts/skill_selfplay.py --root $ROOT summarize --round $R
```

### 6. Update the prompts

Honour `freeze` — it decides which updater runs, and both updaters check it again:

| freeze | update-perturb-api | update-verify-api |
|--------|--------------------|-------------------|
| `none` | runs | runs |
| `perturber` | **skipped** | runs |
| `verifier` | runs | **skipped** |

Invoke the applicable updaters via the `Skill` tool with the round directory and the model.
Each writes a new prompt version, verifies it with `check-prompt`, and reports a convergence
verdict to `update_perturb_api.json` / `update_verify_api.json`.

### 7. External evaluation — ProcessBench (optional, `eval=on`)

The same prompt can be scored on held-out human-annotated errors, through the same renderer:

```bash
EROOT=data/api_evals/$SLUG
python scripts/processbench_eval.py --root $EROOT prepare --round $R \
  --verify-skill verify-$SLUG-v$V2 --per-subset 20 --seed 0
python scripts/selfplay_api.py verify --model $M --version $V2 \
  --inbox $EROOT/round_$RR/inbox --outbox $EROOT/round_$RR/outbox
python scripts/processbench_eval.py --root $EROOT score --round $R
```

Use `$V2`, the version this round produced. Keep `--seed` constant for the life of the
experiment. Three rules bind you:

- Never read the evaluation's items into this context — read only `eval_summary.json`.
- Never feed evaluation results into either updater. They are monitoring, not signal, and
  the updaters are forbidden from reading the eval root.
- Never let evaluation results decide whether to continue or which version to keep.
  Convergence is judged on self-play only.

A round where self-play recall rises and ProcessBench does not is a prompt that learned this
Perturber. Report it as such.

### 8. Advance

Set `$P` / `$V` to the versions just created (a frozen side keeps its version) and continue.

---

## Stopping

Stop when **any** of these holds, and say which one fired:

1. `rounds` rounds are complete.
2. Every updater that ran reported `"converged": true`. Under `freeze`, only the active
   side's verdict counts.
3. A round produced no usable signal — every episode format-invalid, or a pass that failed
   on API errors for most episodes. Stop and report; do not tune a prompt from a round that
   carries no information.

---

## Final report

- Model, rounds run, and which stop condition fired.
- Per-round table: format-valid rate, mean `r_P`, mean `r_V`, recall, precision, and the
  round's API cost.
- Version lineage (`perturb v1 → v3`, `verify v1 → v2`) and the substantive edit behind each
  bump.
- ProcessBench per round if it was run, with an explicit verdict on transfer.
- Anything the updaters filed under `scorer_issues` — candidate repo bugs, and the most
  valuable output of a round that otherwise looked good.
- Any episode that failed with an API error, and whether it was retried.
