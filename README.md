# ARAPPAV — Adversarial Reward for Academic Paper Perturbation and Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-play reinforcement learning system where two adversarial LLMs are trained in a **min-max game**
across two domains:

- **Paper mode**: Perturb/verify academic paper sections with factual, logical, or statistical errors.
- **Math mode**: Perturb/verify math problem solutions with misconception-based errors (derived from the Hendrycks MATH dataset).

In both modes:

- **Perturber (P)**: given a text (or problem+solution), inserts exactly `k` errors, producing a modified version plus a machine-readable ground-truth log.
- **Verifier (V)**: given the (possibly) perturbed text, must locate the `k` errors and explain what's wrong.

The two models are trained adversarially:
- **P is rewarded** when V fails to catch the injected errors.
- **V is rewarded** when it correctly identifies P's errors (and penalized for false positives).

This setup is analogous to GAN training, adapted to LLMs via RL on discrete text tokens.

---

## Reward Definition

For a single episode with `k` injected errors:

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Verifier recall | `matched_errors / k` | Fraction of true errors caught |
| Verifier precision | `matched_errors / num_claims` | Fraction of claims that are real errors |
| Verifier reward (r_V) | `F_beta(precision, recall)` | Default F1; configurable beta |
| Perturber reward (r_P) | `1 − recall` (base) | Successfully fooled = higher reward |

Plus:
- **Format penalty**: applied to P if it doesn't inject exactly `k` valid errors (dominates task reward).
- **Anti-spam penalty**: applied to V if it floods with excessive claims.
- **Anti-duplicate penalty**: applied to P if it repeats the same trick across rounds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Self-Play Orchestrator                         │
│       (alternates P and V training over rounds)                  │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
        ┌───────▼────────┐              ┌───────▼────────┐
        │   Perturber     │              │    Verifier     │
        │  (GRPO-trained) │              │  (DPO-trained)  │
        └───────┬────────┘              └───────┬────────┘
                │  perturbed_paper +            │
                │  ground_truth_errors          │
                ▼                               ▼
        ┌─────────────────────────────────────────────┐
        │           Reward / Judge Module              │
        │  - Span-overlap matching + reward comp       │
        │  - r_P and r_V per episode                   │
        └─────────────────────────────────────────────┘
```

### Training Algorithms

| Model | Algorithm | Library |
|-------|-----------|---------|
| Perturber | **GRPO** (Group Relative Policy Optimization) | `trl.GRPOTrainer` |
| Verifier | **DPO** (Direct Preference Optimization) | `trl.DPOTrainer` |

> **Note:** The original design brief used the abbreviation "GRDP", which was interpreted as **GRPO** (Group Relative Policy Optimization), the correct name for this algorithm in the TRL library.

GRPO is a natural fit for the Perturber: sample `G` perturbations per input paper, the Verifier scores them via `r_P`, and GRPO computes relative advantage within each group to update the policy.

DPO is used for the Verifier: for the same perturbed input, multiple Verifier completions are sampled; higher-`r_V` responses are paired as "chosen" against lower-`r_V` "rejected" responses.

---

## Operating Modes

ARAPPAV supports two operating modes, toggled via the `mode` config flag or the `--mode` CLI argument:

| | Paper Mode | Math Mode |
|---|---|---|
| **Config** | `mode: paper` (default) | `mode: math` |
| **Input** | Paper section text | Math problem + step-by-step solution |
| **Dataset** | Local `.txt`/`.md` files | [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (HF Hub) |
| **Error types** | 6 types (numerical, citation, logical, …) | 26 types (whole_number_bias, wrong_operation, …) |
| **Perturber task** | Inject errors into academic prose | Inject misconception errors into solution steps |
| **Verifier task** | Find errors in paper text | Find errors in math solution, given the problem |
| **Schema module** | `errors/schema.py` | `errors/schema_math.py` |
| **Taxonomy module** | `errors/taxonomy.py` | `errors/taxonomy_math.py` |
| **Prompt module** | `models/perturber.py` (paper prompts) | `models/perturber.py` (math prompts) |
| **Data loader** | `data/ingest.py` | `data/ingest_math.py` |

### Paper mode

```bash
# Ingest local papers
python scripts/run_ingest.py --raw_dir data/raw --output_dir data/processed

# Single rollout
python scripts/run_single_rollout.py --k 3

# Full training
python scripts/run_selfplay.py mode=paper
```

### Math mode

```bash
# Single rollout (auto-samples from Hendrycks MATH if no --text provided)
python scripts/run_single_rollout.py --mode math --k 2 --topic algebra

# Full training
python scripts/run_selfplay.py mode=math self_play.num_rounds=10

# Or use the end-to-end notebook
jupyter notebook notebooks/end_to_end_math.ipynb
```

The math mode error taxonomy is derived from:
> Rittle-Johnson et al., *"Detecting Math Misconceptions: An AI Benchmark Dataset"*, AIME-Con 2025.

---

## Repository Structure

```
arappav/
├── README.md
├── pyproject.toml
├── mathematical-errors.pdf           # Reference paper for math error taxonomy
├── configs/                          # Hydra YAML configs
│   ├── default.yaml                  # Top-level (includes mode flag)
│   ├── models/{perturber,verifier}.yaml
│   ├── training/{grpo,dpo}.yaml
│   ├── reward/reward.yaml
│   └── data/corpus.yaml
├── data/
│   ├── raw/                          # source papers (gitignored)
│   ├── processed/                    # chunked/cleaned JSON or parquet
│   └── rollouts/                     # self-play episode logs (gitignored)
├── src/arappav/
│   ├── data/                         # ingest, ingest_math, chunking, HF dataset wrappers
│   ├── models/                       # Perturber & Verifier wrappers (paper + math prompts)
│   ├── serving/                      # vLLM engine wrapper
│   ├── errors/                       # taxonomy, taxonomy_math, schema, schema_math
│   ├── reward/                       # Matcher + reward functions (paper + math)
│   ├── training/                     # GRPO, DPO, preference builder, self-play loop
│   ├── eval/                         # Metrics + evaluation harness
│   └── utils/                        # Logging, seeding
├── scripts/                          # Entry points (all support --mode)
│   ├── run_ingest.py
│   ├── run_selfplay.py
│   ├── run_eval.py
│   └── run_single_rollout.py
├── tests/                            # pytest suite (49 tests)
└── notebooks/
    ├── explore_rollouts.ipynb        # Paper mode exploration
    └── end_to_end_math.ipynb         # Math mode end-to-end training
```

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Ingest papers (paper mode)

Place `.txt` or `.md` papers in `data/raw/`, then:

```bash
python scripts/run_ingest.py --raw_dir data/raw --output_dir data/processed
```

For **math mode**, no ingestion is needed — the Hendrycks MATH dataset is downloaded automatically from Hugging Face Hub on first use.

### 3. Sanity-check a single episode

**Paper mode:**
```bash
python scripts/run_single_rollout.py \
  --perturber_model Qwen/Qwen2.5-3B-Instruct \
  --verifier_model Qwen/Qwen2.5-7B-Instruct \
  --k 3
```

**Math mode** (auto-samples a problem from Hendrycks MATH):
```bash
python scripts/run_single_rollout.py --mode math --k 2 --topic algebra
```

This runs one P→V→Reward cycle and prints everything, so you can verify prompt templates and structured output parsing **before** committing GPU-hours to training.

### 4. Run the self-play training loop

```bash
# Paper mode (default)
python scripts/run_selfplay.py --config-name default

# Math mode
python scripts/run_selfplay.py mode=math self_play.num_rounds=10
```

Override any config value from the command line (Hydra):

```bash
python scripts/run_selfplay.py \
  self_play.num_rounds=50 \
  self_play.episodes_per_round=256 \
  perturber.model_name_or_path=Qwen/Qwen2.5-3B-Instruct \
  verifier.model_name_or_path=Qwen/Qwen2.5-7B-Instruct
```

### 5. Run held-out evaluation

```bash
python scripts/run_eval.py \
  --perturber_checkpoint outputs/perturber_final \
  --verifier_checkpoint outputs/verifier_final \
  --processed_dir data/processed
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Error Taxonomy

### Paper Mode (6 types)

The Perturber can inject errors of the following types (defined in `src/arappav/errors/taxonomy.py`):

| Type | Description |
|------|-------------|
| `numerical` | Altered statistic, wrong percentage, incorrect table/figure value |
| `citation` | Misattributed claim, fabricated/altered citation, wrong year |
| `logical` | Non-sequitur, reversed causality, invalid inference |
| `methodological` | Swapped experimental setup detail (dataset split, hyperparameter) |
| `terminology` | Technical term swapped with plausible-but-wrong alternative |
| `negation` | Inserted/removed negation flipping a claim's meaning |

The taxonomy is extensible — add new types to the `ErrorType` enum.

### Math Mode (26 types)

Derived from Rittle-Johnson et al. (2025), *"Detecting Math Misconceptions: An AI Benchmark Dataset"*
(see `mathematical-errors.pdf`). Defined in `src/arappav/errors/taxonomy_math.py`.

| Group | Types |
|-------|-------|
| **Fractions** | `whole_number_bias`, `adding_across`, `denominator_only`, `duplication_error`, `inversion_error`, `wrong_fraction`, `incomplete_solution` |
| **Decimals** | `decimal_magnitude`, `ignores_zeroes` |
| **Algebra** | `variable_misconception`, `additive_thinking`, `wrong_operation`, `operand_swap`, `wrong_sequence_term`, `first_term_as_coefficient` |
| **Negatives** | `negative_number_error`, `tacking_signs` |
| **Proportions** | `proportional_reasoning_error`, `inverse_operation_error`, `base_rate_fallacy` |
| **Probability** | `probability_scale`, `probability_certainty` |
| **Geometry** | `geometry_definition`, `angle_misconception` |
| **Cross-cutting** | `irrelevant_feature`, `unknowable` |

Error types are organized into topic groups via `MathErrorType.topic_groups()` for contextual filtering during perturbation.

---

## Configuration

All settings are in YAML files under `configs/`, composed by [Hydra](https://hydra.cc/):

| File | Controls |
|------|----------|
| `configs/default.yaml` | Top-level: **mode** (`paper` or `math`), **freeze** (selective training), rounds, episodes, vLLM, wandb, output paths, math dataset settings |
| `configs/models/perturber.yaml` | Perturber model ID, LoRA, generation params |
| `configs/models/verifier.yaml` | Verifier model ID, LoRA, generation params |
| `configs/training/grpo.yaml` | GRPO hyperparameters (lr, beta, group size, etc.) |
| `configs/training/dpo.yaml` | DPO hyperparameters (lr, beta, loss type, etc.) |
| `configs/reward/reward.yaml` | k range, format penalty, matching thresholds, anti-spam/duplicate |
| `configs/data/corpus.yaml` | Corpus source, chunking strategy, split ratios (paper mode) |

### Freeze Mode

The `self_play.freeze` parameter allows selectively **deactivating the finetuning** of one model
so you can train the other in isolation — useful for ablation studies and controlled experiments.

| Value | Effect |
|-------|--------|
| `null` (default) | Both Perturber (GRPO) and Verifier (DPO) are trained normally |
| `"perturber"` | Perturber is **frozen** — only the Verifier is trained (DPO) |
| `"verifier"` | Verifier is **frozen** — only the Perturber is trained (GRPO) |

> **Assertion:** At most one model can be frozen at a time. Setting `freeze` to `"perturber"`
> or `"verifier"` guarantees the other model is still trained — it is not possible to freeze both.

**CLI examples:**

```bash
# Train only the Perturber (keep Verifier fixed)
python scripts/run_selfplay.py self_play.freeze=verifier

# Train only the Verifier (keep Perturber fixed)
python scripts/run_selfplay.py self_play.freeze=perturber

# Train both (default)
python scripts/run_selfplay.py
```

When a model is frozen, its rollouts are still collected (they're needed to compute rewards for
the other model), but the weight update step is skipped.

---

## Hardware Assumptions

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Perturber (3B) | 1× GPU, 16 GB VRAM | 1× A100-40GB |
| Verifier (7B) | 1× GPU, 24 GB VRAM | 1× A100-40GB or 2× A100-40GB |
| vLLM serving | 1× GPU, 24+ GB VRAM | 2× A100-40GB (tensor parallelism) |
| Full self-play | 2× GPUs (one per model) | 2–4× A100-40GB or 4× A100-80GB |

With LoRA (default), VRAM requirements are significantly lower. Full fine-tuning is available via config override.

---

## Key Design Decisions

1. **Alternating (not simultaneous) self-play** keeps the moving-target problem tractable. P trains against a frozen V, then V trains against a frozen P.
2. **Paper-level splits** (not chunk-level) prevent leakage: all chunks from a paper go to the same split.
3. **Pure reward functions** — `compute_rewards(ground_truth, verifier_claims, config)` has no hidden state, making it fully testable.
4. **Format penalty dominates** — ensures instruction-following (exactly `k` errors) before optimizing the adversarial objective.
5. **Build milestone 4 (reward/matcher) first** — validated with 49 unit tests before any GPU time is spent on RL training.
6. **Mode dispatch via config** — a single `mode` flag (`paper` or `math`) switches dataset loading, prompt templates, error taxonomy, and output schemas. The reward module uses duck-typing so the same matching/reward logic works across both modes.

---

## Build Order (Milestones)

1. ✅ Repo scaffold + config system + pydantic schemas + taxonomy
2. ✅ Data ingestion + chunking
3. ✅ Perturber/Verifier prompt templates + vLLM wrapper
4. ✅ **Reward/matcher + tests** ← solid before RL
5. ✅ GRPO loop for Perturber
6. ✅ DPO loop for Verifier
7. ✅ Alternating self-play loop + checkpointing + eval
8. ✅ Held-out evaluation harness + qualitative report
9. ✅ **Math mode** — Hendrycks MATH dataset, 26-error taxonomy, end-to-end notebook
10. 🔲 Stretch: PDF/LaTeX ingestion, reward shaping, degenerate-strategy detection tuning

---

## Dependencies

Core:
- `torch >= 2.3.0`
- `transformers >= 4.44.0`
- `trl >= 0.11.0` — **API moves quickly; pin your version**
- `peft >= 0.12.0`
- `accelerate >= 0.33.0`
- `vllm >= 0.5.0`
- `pydantic >= 2.8.0`
- `datasets >= 2.20.0`
- `hydra-core >= 1.3.0`

Dev: `pytest`, `black`, `isort`, `ruff`, `jupyter`

---

## Note on TRL Versioning

The `trl` library's GRPOTrainer and DPOTrainer APIs evolve quickly across minor versions. This project was scaffolded targeting `trl >= 0.11.0`. If you encounter API mismatches, pin to the exact version in your environment and check the [TRL release notes](https://github.com/huggingface/trl/releases).

---

## License

MIT
