# Evaluation and play backends

A policy can be run three ways. They send the same policy and the same episode
and produce the same artefacts; they differ in transport, and the difference is
large enough to decide whether an experiment is affordable.

```bash
--backend api          # default: Messages API directly
--backend batch        # Message Batches API, 50% off, asynchronous
--backend claude-code  # `claude -p` per item, no API key needed
```

---

## Why this exists

Profiling one real ProcessBench item through `claude -p` with `claude-haiku-4-5`:

| | measured |
|---|---|
| input tokens | **17,209** |
| …Claude Code harness system prompt + tool definitions | ~13,150 (**76%**) |
| …actual content (policy + item) | ~2,620 |
| output tokens | 6,263 — to produce a 30-byte answer |
| cost | **$0.048** |
| latency | 44.6 s |

Every `claude -p` invocation boots a fresh agent session and re-sends the whole
harness. A trivial `claude -p "Say ok"` still costs 13,140 input tokens and
$0.0086. That overhead, not the model, is why a single round of ProcessBench
evaluation exhausted an interactive session limit.

The API backends send only the policy and the item.

---

## The three backends

| | `claude-code` | `api` (default) | `batch` |
|---|---|---|---|
| transport | `claude -p` per item | Messages / Chat Completions | Message Batches |
| policy delivered as | `/skill` slash command | (cached) system prompt | cached system prompt |
| input tokens per item | ~17,200 | ~2,600 | ~2,600 |
| cost per item (haiku) | $0.048 | ~$0.002 | ~$0.001 |
| latency | ~45 s/item | ~26 s/item | minutes–24 h, all items at once |
| quota consumed | **interactive session limit** | API key | API key |
| providers | Claude only | Anthropic / OpenAI / DeepSeek | Anthropic only |
| credentials | none | provider API key | `ANTHROPIC_API_KEY` |

Projected on the same work:

| workload | `claude-code` | `api` | `batch` |
|---|---|---|---|
| 800 calls (10 policies × 80 items) | $38.28 | $1.36 | ~$0.70 |
| 3,400 calls (full ProcessBench, 1 policy) | $162.69 | $5.79 | ~$2.90 |

### `api` — the default

Sends the policy as a system prompt and the item as the user turn. On Anthropic
the policy carries `cache_control: ephemeral`; since it is byte-identical across
every item of a run, it bills at roughly a tenth of the input rate after the
first call. OpenAI and DeepSeek cache long prefixes automatically, so the same
saving applies with no flag.

### `batch` — cheapest, asynchronous

Submits every outstanding item as one job at 50% of the standard rate, which
stacks with caching. Suited to a full-benchmark sweep, not an interactive check.

Two properties make it safe to leave unattended:

- **The batch id is persisted before polling begins** (`batch_id.txt` in the
  round directory). An interrupted run reattaches to the in-flight batch rather
  than submitting a second copy and paying twice.
- **Failed requests produce no output file.** Anything the server reports as
  errored, expired or cancelled is retried on the next run instead of being
  scored as an empty answer.

Results are keyed by `custom_id` throughout, never by position, because a batch
returns them in arbitrary order.

Anthropic only: OpenAI's batch API is a different, file-upload shape and
DeepSeek has none. Requesting `--backend batch` with a non-Anthropic model fails
immediately with that explanation rather than silently mis-routing.

### `claude-code` — the parity reference

The original path. Needs no API key, so it always works; keep it for checking
that an API result matches, and for machines with no key configured.

---

## Providers (`api` backend)

The provider is inferred from the model id, or named with `--provider`.

| provider | SDK | credential | inferred from |
|---|---|---|---|
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` (or `ant auth login`) | `claude-*` |
| `openai` | `openai` | `OPENAI_API_KEY` | `gpt-*`, `o1`, `o3`, `o4`, `chatgpt*` |
| `deepseek` | `openai`, base URL `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `deepseek*` |

DeepSeek rides the OpenAI SDK because that is DeepSeek's own documented
integration path. Claude uses the official Anthropic SDK, never a compatibility
shim.

Per-provider request differences are handled automatically:

- **Anthropic** — `thinking: adaptive` + `output_config.effort`, except on
  `claude-haiku-4-5`, which rejects both.
- **OpenAI** — `max_completion_tokens`; `reasoning_effort` is sent only to the
  reasoning families (`gpt-5`, `o1`, `o3`, `o4`), since it errors on the others.
  Anthropic's `xhigh`/`max` effort levels clamp to `high`, which is OpenAI's top.
- **DeepSeek** — `max_tokens`, no effort control.

An unrecognised model id fails loudly rather than guessing a provider.

**Comparability caveat.** `effort` is not a common scale across providers, and
DeepSeek has none at all. A cross-provider table compares *configurations*, not
equal compute.

---

## Usage

```bash
# default: Anthropic API, cached policy
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/eval_policies_processbench.py --prefix hverify --versions 1 10 \
    --model claude-haiku-4-5 --per-subset 20 --concurrency 8

# cheapest: batch (asynchronous; re-run the same command to reattach/resume)
python scripts/eval_policies_processbench.py --prefix hverify --versions 1 10 \
    --model claude-haiku-4-5 --per-subset 20 --backend batch

# OpenAI
export OPENAI_API_KEY=sk-...
python scripts/eval_policies_processbench.py --prefix hverify --versions 1 \
    --model gpt-5 --per-subset 20

# DeepSeek
export DEEPSEEK_API_KEY=sk-...
python scripts/eval_policies_processbench.py --prefix hverify --versions 1 \
    --model deepseek-chat --per-subset 20

# no key available, or checking parity against the CLI path
python scripts/eval_policies_processbench.py --prefix hverify --versions 1 \
    --model claude-haiku-4-5 --backend claude-code
```

Every backend is resumable: items with an existing output file are skipped, so
an interrupted run never re-pays for completed work.

---

## Invariants across backends

- The same renderer (`contracts.render_verify_prompt`) and the same leak guard
  build every payload, so an evaluation prompt cannot contain anything a
  self-play prompt could not.
- The same fail-fast check classifies quota, auth, timeout and empty replies as
  *infrastructure*, aborting rather than scoring them. A malformed model answer
  is still data and is scored.
- The same artefact layout (`inbox/`, `outbox/`, `prompts/`, `eval_summary.json`).

---

## Status

The `claude-code` path has been exercised end to end on real runs. The `api` and
`batch` paths are covered by 22 tests against a stubbed SDK — request shape,
caching, provider routing, id-keyed collection, failure classification — but at
the time of writing had **not** been run against a live API key. Send two items
before committing to a full sweep.

---

## Running the pipeline on non-Claude models

`scripts/run_pipeline.py` also takes `--backend`. It defaults to `claude-code`,
which invokes `claude -p` and therefore reaches **Claude models only**;
`--backend api` is the only way to run the self-play loop on OpenAI or DeepSeek.

Players and policy author are configured independently, and may use different
providers:

```bash
python scripts/run_pipeline.py --rounds 10 --episodes 8 \
    --backend api --model gpt-5.6-terra --updater-model gpt-5.6-sol
```

`run_experiments.sh` wraps the four comparisons this repository cares about —
strong-updater vs weak-updater, on Claude, OpenAI and DeepSeek — picking the
backend per model family and skipping any run whose credential is absent.
