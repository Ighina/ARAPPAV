"""Self-Play Orchestrator.

Implements the alternating training loop:
1. Freeze V, roll out N episodes with P → collect rewards → update P (GRPO).
2. Freeze P, roll out M episodes with V → build DPO pairs → update V (DPO).
3. Repeat for ``num_rounds``.

Supports both ``paper`` and ``math`` modes:
- Paper mode: Perturb/verify academic paper sections.
- Math mode: Perturb/verify math problem solutions (Hendrycks MATH dataset).

The orchestrator manages:
- Model loading/unloading (swapping between P and V in GPU memory).
- vLLM engine lifecycle for fast rollouts.
- Rollout logging to disk for later analysis.
- Checkpointing and periodic held-out evaluation.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from arappav.utils.logging import setup_logging
from arappav.utils.seeding import set_seed

logger = logging.getLogger(__name__)


class SelfPlayLoop:
    """Orchestrates alternating Perturber / Verifier training rounds."""

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Full Hydra/OmegaConf config (composed from default.yaml and sub-configs).
        """
        self.cfg = cfg
        self.mode = cfg.get("mode", "paper")
        self.freeze = cfg.self_play.get("freeze", None)  # None, "perturber", or "verifier"
        self.output_dir = Path(cfg.get("output_dir", "./outputs"))
        self.rollout_dir = Path(cfg.get("rollout_dir", "./data/rollouts"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.round: int = 0
        self.perturber_model = None
        self.verifier_model = None
        self.vllm_engine = None
        self.perturber_trainer = None
        self.verifier_trainer = None

        # History for anti-duplicate
        self._perturbation_history: list = []

        # Math dataset cache (loaded once)
        self._math_dataset_dict = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Initialise models, tokenizers, vLLM engine, and trainers."""
        set_seed(self.cfg.self_play.seed)
        setup_logging()

        # vLLM engine (shared across P and V, swapped as needed)
        if self.cfg.get("vllm"):
            from arappav.serving.vllm_engine import VLLMEngine

            self.vllm_engine = VLLMEngine(
                model_name_or_path="",  # will be set per-model
                tensor_parallel_size=self.cfg.vllm.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.vllm.gpu_memory_utilization,
                max_model_len=self.cfg.vllm.max_model_len,
                dtype=self.cfg.vllm.dtype,
                enforce_eager=self.cfg.vllm.enforce_eager,
            )

        self._load_perturber()
        self._load_verifier()

        self._validate_freeze()

        logger.info(f"Self-play loop initialised (mode={self.mode}).")

    def _load_perturber(self):
        """Load (or reload) the Perturber model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.cfg.perturber.model_name_or_path
        logger.info(f"Loading Perturber: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        if self.cfg.perturber.use_peft:
            from peft import LoraConfig, get_peft_model

            lora_cfg = self.cfg.perturber.lora
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            logger.info("Applied LoRA to Perturber.")

        self.perturber_model = model
        self.perturber_tokenizer = tokenizer

    def _load_verifier(self):
        """Load (or reload) the Verifier model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.cfg.verifier.model_name_or_path
        logger.info(f"Loading Verifier: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        if self.cfg.verifier.use_peft:
            from peft import LoraConfig, get_peft_model

            lora_cfg = self.cfg.verifier.lora
            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=list(lora_cfg.target_modules),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            logger.info("Applied LoRA to Verifier.")

        self.verifier_model = model
        self.verifier_tokenizer = tokenizer

    def _validate_freeze(self):
        """Validate the freeze config and log the active mode."""
        allowed = (None, "perturber", "verifier")
        if self.freeze not in allowed:
            raise ValueError(
                f"Invalid self_play.freeze={self.freeze!r}. "
                f"Must be null, 'perturber', or 'verifier'."
            )
        if self.freeze is None:
            logger.info("Freeze mode: none — both Perturber and Verifier will be trained.")
        else:
            logger.info(
                "Freeze mode: '%s' — %s will NOT be trained.",
                self.freeze,
                "Perturber" if self.freeze == "perturber" else "Verifier",
            )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_corpus(self):
        """Load the dataset (paper corpus or math problems) with splits."""
        if self.mode == "math":
            return self._load_math_corpus()
        else:
            from arappav.data.dataset import load_corpus_split

            corpus_cfg = self.cfg.corpus
            return load_corpus_split(
                processed_dir=corpus_cfg.processed_dir,
                train_split=corpus_cfg.train_split,
                val_split=corpus_cfg.val_split,
                test_split=corpus_cfg.test_split,
                seed=corpus_cfg.split_seed,
            )

    def _load_math_corpus(self):
        """Load the Hendrycks MATH dataset with splits."""
        if self._math_dataset_dict is not None:
            return self._math_dataset_dict

        from arappav.data.ingest_math import build_math_splits

        math_cfg = self.cfg.get("math", {})
        self._math_dataset_dict = build_math_splits(
            topics=list(math_cfg.get("topics", ["algebra"])),
            max_train_per_topic=math_cfg.get("max_train_per_topic"),
            max_val_per_topic=math_cfg.get("max_val_per_topic", 100),
            max_test_per_topic=math_cfg.get("max_test_per_topic", 100),
            seed=self.cfg.self_play.seed,
        )
        return self._math_dataset_dict

    # ------------------------------------------------------------------
    # Episode rollout helpers
    # ------------------------------------------------------------------

    def _sample_k(self) -> int:
        """Sample the number of errors to inject for an episode."""
        reward_cfg = self.cfg.reward
        if reward_cfg.k_fixed is not None:
            return reward_cfg.k_fixed
        # Math mode: use tighter k range
        if self.mode == "math":
            math_cfg = self.cfg.get("math", {})
            lo, hi = math_cfg.get("k_range", [1, 4])
        else:
            lo, hi = reward_cfg.k_range
        return random.randint(lo, hi)

    def _run_perturber_episode(
        self, chunk_or_problem: str, k: int, chunk_id: str,
        solution: str | None = None,
    ) -> dict | None:
        """Run one Perturber generation episode.

        Args:
            chunk_or_problem: Paper text (paper mode) or problem statement (math mode).
            k: Number of errors to inject.
            chunk_id: Identifier for logging.
            solution: Required in math mode — the correct solution.

        Returns:
            Dict with episode data, or None if generation failed.
        """
        from arappav.models.perturber import PerturberModel

        perturber_wrapper = PerturberModel(
            model_name_or_path=self.cfg.perturber.model_name_or_path,
            mode=self.mode,
            use_vllm=self.vllm_engine is not None and self.vllm_engine.is_loaded(),
            vllm_engine=self.vllm_engine,
            generation_kwargs=dict(self.cfg.perturber.generation),
        )

        perturber_out, parse_err = perturber_wrapper.generate(
            chunk_or_problem, k, solution=solution,
        )

        if perturber_out is None:
            logger.warning(f"Perturber parse failure for {chunk_id}: {parse_err}")
            return {
                "chunk_id": chunk_id,
                "perturbed_text": None,
                "ground_truth": [],
                "k": k,
                "format_valid": False,
                "format_reason": parse_err,
            }

        # Normalize output: paper mode has perturbed_text, math mode has perturbed_solution
        result_text = (
            perturber_out.perturbed_solution
            if self.mode == "math"
            else perturber_out.perturbed_text
        )

        return {
            "chunk_id": chunk_id,
            "perturbed_text": result_text,
            "ground_truth": [e.model_dump() for e in perturber_out.errors],
            "k": k,
            "format_valid": True,
            "format_reason": None,
        }

    def _run_verifier_episode(
        self,
        text_to_review: str,
        problem: str | None = None,
        n_completions: int = 3,
    ) -> list[dict]:
        """Run Verifier generation on a perturbed text/solution.

        Args:
            text_to_review: Perturbed text (paper mode) or perturbed solution (math mode).
            problem: Required in math mode — the original problem statement.
            n_completions: Number of completions to sample.

        Returns:
            List of dicts with raw output and parsed claims.
        """
        from arappav.models.verifier import VerifierModel

        verifier_wrapper = VerifierModel(
            model_name_or_path=self.cfg.verifier.model_name_or_path,
            mode=self.mode,
            use_vllm=self.vllm_engine is not None and self.vllm_engine.is_loaded(),
            vllm_engine=self.vllm_engine,
            generation_kwargs=dict(self.cfg.verifier.generation),
        )

        results = verifier_wrapper.generate(
            text_to_review, problem=problem, n_completions=n_completions,
        )

        return [
            {
                "raw_text": "",  # raw text reconstructed from output
                "claims": [c.model_dump() for c in vout.claims] if vout else [],
                "parse_valid": vout is not None,
                "parse_error": err,
            }
            for (vout, err) in results
        ]

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_perturber_rollouts(
        self, split: str = "train", num_episodes: int | None = None
    ) -> list[dict]:
        """Collect Perturber rollouts: sample data, generate perturbations."""
        if num_episodes is None:
            num_episodes = self.cfg.self_play.episodes_per_round

        dataset_dict = self._load_corpus()
        dataset = dataset_dict[split]

        rollouts = []
        for i in range(num_episodes):
            idx = i % len(dataset)
            row = dataset[idx]

            if self.mode == "math":
                problem_text = row["problem"]
                solution_text = row["solution"]
                paper_id = f"{row['topic']}_{row['level']}_{idx}"
                k = self._sample_k()
                chunk_id = f"{paper_id}_ep{i}"
                episode = self._run_perturber_episode(
                    problem_text, k, chunk_id, solution=solution_text,
                )
                if episode:
                    episode["paper_id"] = paper_id
                    episode["original_text"] = problem_text
                    episode["original_solution"] = solution_text
                    episode["rollout_id"] = f"r{self.round}_p{i}"
                    episode["mode"] = "math"
                    rollouts.append(episode)
            else:
                paper_text = row["text"]
                paper_id = row["id"]
                k = self._sample_k()
                chunk_id = f"{paper_id}_ep{i}"
                episode = self._run_perturber_episode(paper_text, k, chunk_id)
                if episode:
                    episode["paper_id"] = paper_id
                    episode["original_text"] = paper_text
                    episode["rollout_id"] = f"r{self.round}_p{i}"
                    episode["mode"] = "paper"
                    rollouts.append(episode)

        logger.info(f"Collected {len(rollouts)} Perturber rollouts (round {self.round})")
        return rollouts

    def collect_verifier_rollouts(
        self,
        perturber_rollouts: list[dict],
        n_completions: int = 3,
    ) -> list[dict]:
        """Run Verifier on Perturber-generated texts to collect preference data."""
        verifier_rollouts = []

        for rollout in perturber_rollouts:
            if not rollout.get("format_valid"):
                continue

            perturbed_text = rollout["perturbed_text"]
            problem = rollout.get("original_text") if rollout.get("mode") == "math" else None

            verifier_results = self._run_verifier_episode(
                perturbed_text, problem=problem, n_completions=n_completions,
            )

            verifier_rollouts.append(
                {
                    "prompt": perturbed_text,  # used as context for DPO
                    "responses": [
                        (r["raw_text"],
                         None if not r["parse_valid"] else _dict_to_verifier_output(r["claims"], self.mode),
                         r["parse_error"])
                        for r in verifier_results
                    ],
                    "perturbed_text": perturbed_text,
                    "ground_truth": [
                        _dict_to_injected_error(e, self.mode) for e in rollout["ground_truth"]
                    ],
                    "k": rollout["k"],
                    "paper_id": rollout["paper_id"],
                    "mode": self.mode,
                }
            )

        logger.info(
            f"Collected {len(verifier_rollouts)} Verifier rollouts (round {self.round})"
        )
        return verifier_rollouts

    # ------------------------------------------------------------------
    # Training round
    # ------------------------------------------------------------------

    def train_round(self):
        """Execute one complete self-play round: update P, then update V."""
        self.round += 1
        logger.info(f"{'='*60}\n  SELF-PLAY ROUND {self.round} / {self.cfg.self_play.num_rounds}  (mode={self.mode})\n{'='*60}")

        # --- Phase 1: Perturber update (GRPO) ---
        freeze_perturber = self.freeze == "perturber"

        if freeze_perturber:
            logger.info("Phase 1: Collecting Perturber rollouts (GRPO update SKIPPED — perturber frozen)")
        else:
            logger.info("Phase 1: Collecting Perturber rollouts + GRPO update")

        p_rollouts = self.collect_perturber_rollouts()

        if not freeze_perturber:
            grpo_dataset = self._build_grpo_dataset(p_rollouts)
            if grpo_dataset is not None and len(grpo_dataset) > 0:
                self._train_perturber(grpo_dataset)

        self._save_rollouts(p_rollouts, "perturber")

        # --- Phase 2: Verifier update (DPO) ---
        freeze_verifier = self.freeze == "verifier"

        if freeze_verifier:
            logger.info("Phase 2: Verifier frozen — skipping DPO update")
        else:
            logger.info("Phase 2: Collecting Verifier rollouts + DPO update")
            v_rollouts = self.collect_verifier_rollouts(p_rollouts)

            from arappav.training.preference_builder import build_preference_pairs, pairs_to_dataset

            pairs = build_preference_pairs(
                v_rollouts,
                reward_config=OmegaConf.to_container(self.cfg.reward, resolve=True),
                min_reward_gap=0.05,
                max_pairs=self.cfg.self_play.pairs_per_update,
            )

            if pairs:
                dpo_dataset = pairs_to_dataset(pairs)
                self._train_verifier(dpo_dataset)

            self._save_rollouts(v_rollouts, "verifier")

        # --- Checkpoint ---
        if self.round % self.cfg.self_play.checkpoint_frequency == 0:
            self._save_checkpoint()

        # --- Held-out eval ---
        if self.round % self.cfg.self_play.eval_frequency == 0:
            self._run_eval()

        # Update perturbation history for anti-duplicate
        for r in p_rollouts:
            if r.get("format_valid"):
                self._perturbation_history.extend(
                    [_dict_to_injected_error(e, self.mode) for e in r.get("ground_truth", [])]
                )

        logger.info(f"Round {self.round} complete.")

    def run(self):
        """Run the full self-play loop for ``num_rounds``."""
        self.setup()

        for _ in range(self.cfg.self_play.num_rounds):
            self.train_round()

        # Final save
        self._save_checkpoint(final=True)
        logger.info("Self-play loop finished.")

    # ------------------------------------------------------------------
    # Internal: training
    # ------------------------------------------------------------------

    def _build_grpo_dataset(self, rollouts: list[dict]):
        """Build a HF Dataset for GRPO training from Perturber rollouts."""
        prompts = []
        for r in rollouts:
            if r.get("format_valid"):
                if self.mode == "math":
                    from arappav.models.perturber import build_math_perturber_prompt
                    prompt = build_math_perturber_prompt(
                        r["original_text"], r.get("original_solution", ""), r["k"],
                    )
                else:
                    from arappav.models.perturber import build_perturber_prompt
                    prompt = build_perturber_prompt(r["original_text"], r["k"])
                prompts.append(prompt)

        if not prompts:
            return None

        from datasets import Dataset
        return Dataset.from_dict({"prompt": prompts})

    def _train_perturber(self, dataset):
        """Run one GRPO update on the Perturber."""
        if self.perturber_trainer is None:
            from arappav.training.grpo_trainer import GRPOConfig, GRPOPerturberTrainer

            grpo_cfg = GRPOConfig(
                learning_rate=self.cfg.grpo.learning_rate,
                per_device_batch_size=self.cfg.grpo.per_device_batch_size,
                gradient_accumulation_steps=self.cfg.grpo.gradient_accumulation_steps,
                num_epochs=self.cfg.grpo.num_epochs,
                max_grad_norm=self.cfg.grpo.max_grad_norm,
                warmup_ratio=self.cfg.grpo.warmup_ratio,
                lr_scheduler_type=self.cfg.grpo.lr_scheduler_type,
                optim=self.cfg.grpo.optim,
                beta=self.cfg.grpo.beta,
                num_generations=self.cfg.grpo.num_generations,
                temperature=self.cfg.grpo.temperature,
                max_prompt_length=self.cfg.grpo.max_prompt_length,
                use_vllm_for_rollouts=self.cfg.grpo.use_vllm_for_rollouts,
            )

            from arappav.training.grpo_trainer import make_perturber_reward_fn

            reward_fn = make_perturber_reward_fn(
                verifier_model=None,
                reward_config=OmegaConf.to_container(self.cfg.reward, resolve=True),
                k_sampler=self._sample_k,
            )

            self.perturber_trainer = GRPOPerturberTrainer(
                model=self.perturber_model,
                tokenizer=self.perturber_tokenizer,
                config=grpo_cfg,
                reward_function=reward_fn,
                output_dir=self.output_dir / f"perturber_round{self.round}",
                use_peft=self.cfg.perturber.use_peft,
            )

        metrics = self.perturber_trainer.train(dataset)
        logger.info(f"GRPO metrics: {metrics}")
        return metrics

    def _train_verifier(self, dataset):
        """Run one DPO update on the Verifier."""
        if self.verifier_trainer is None:
            from arappav.training.dpo_trainer import DPOConfig, DPOVerifierTrainer

            dpo_cfg = DPOConfig(
                learning_rate=self.cfg.dpo.learning_rate,
                per_device_batch_size=self.cfg.dpo.per_device_batch_size,
                gradient_accumulation_steps=self.cfg.dpo.gradient_accumulation_steps,
                num_epochs=self.cfg.dpo.num_epochs,
                max_grad_norm=self.cfg.dpo.max_grad_norm,
                warmup_ratio=self.cfg.dpo.warmup_ratio,
                lr_scheduler_type=self.cfg.dpo.lr_scheduler_type,
                optim=self.cfg.dpo.optim,
                beta=self.cfg.dpo.beta,
                max_prompt_length=self.cfg.dpo.max_prompt_length,
                max_length=self.cfg.dpo.max_length,
                loss_type=self.cfg.dpo.loss_type,
            )

            self.verifier_trainer = DPOVerifierTrainer(
                model=self.verifier_model,
                ref_model=None,
                tokenizer=self.verifier_tokenizer,
                config=dpo_cfg,
                output_dir=self.output_dir / f"verifier_round{self.round}",
                use_peft=self.cfg.verifier.use_peft,
            )

        metrics = self.verifier_trainer.train(dataset)
        logger.info(f"DPO metrics: {metrics}")
        return metrics

    # ------------------------------------------------------------------
    # Internal: checkpointing & logging
    # ------------------------------------------------------------------

    def _save_rollouts(self, rollouts: list[dict], prefix: str):
        """Save rollouts to disk as JSONL."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.rollout_dir / f"{prefix}_{self.mode}_round{self.round}_{timestamp}.jsonl"
        with open(path, "w") as f:
            for r in rollouts:
                f.write(json.dumps(r, default=str) + "\n")
        logger.info(f"Saved {len(rollouts)} {prefix} rollouts to {path}")

    def _save_checkpoint(self, final: bool = False):
        """Save both model checkpoints."""
        label = "final" if final else f"round{self.round}"
        p_dir = self.output_dir / f"perturber_{self.mode}_{label}"
        v_dir = self.output_dir / f"verifier_{self.mode}_{label}"

        self.perturber_model.save_pretrained(str(p_dir))
        self.perturber_tokenizer.save_pretrained(str(p_dir))

        self.verifier_model.save_pretrained(str(v_dir))
        self.verifier_tokenizer.save_pretrained(str(v_dir))

        logger.info(f"Checkpoints saved: {p_dir}, {v_dir}")

    def _run_eval(self):
        """Run held-out evaluation."""
        from arappav.eval.evaluate import run_evaluation

        dataset_dict = self._load_corpus()
        metrics = run_evaluation(
            perturber_model=None,
            verifier_model=None,
            eval_dataset=dataset_dict["val"],
            reward_config=OmegaConf.to_container(self.cfg.reward, resolve=True),
        )
        logger.info(f"Eval metrics (round {self.round}): {metrics}")


# ---------------------------------------------------------------------------
# Dict → schema conversion helpers
# ---------------------------------------------------------------------------


def _dict_to_injected_error(d: dict, mode: str = "paper"):
    """Convert a dict to an InjectedError or MathInjectedError, tolerating missing fields."""
    try:
        if mode == "math":
            from arappav.errors.schema_math import MathInjectedError
            from arappav.errors.taxonomy_math import MathErrorType
            d = dict(d)
            d["error_type"] = MathErrorType(d.get("error_type", "wrong_operation"))
            return MathInjectedError(**d)
        else:
            from arappav.errors.schema import InjectedError
            from arappav.errors.taxonomy import ErrorType
            d = dict(d)
            d["error_type"] = ErrorType(d.get("error_type", "numerical"))
            return InjectedError(**d)
    except Exception:
        return None


def _dict_to_verifier_output(claims: list[dict], mode: str = "paper"):
    """Convert a list of claim dicts to a VerifierOutput or MathVerifierOutput."""
    try:
        if mode == "math":
            from arappav.errors.schema_math import MathVerifierClaim, MathVerifierOutput
            parsed_claims = [MathVerifierClaim(**c) for c in claims]
            return MathVerifierOutput(claims=parsed_claims)
        else:
            from arappav.errors.schema import VerifierClaim, VerifierOutput
            parsed_claims = [VerifierClaim(**c) for c in claims]
            return VerifierOutput(claims=parsed_claims)
    except Exception:
        return None
