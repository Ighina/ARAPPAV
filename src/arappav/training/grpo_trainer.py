"""GRPO Trainer for the Perturber.

Wraps TRL's ``GRPOTrainer`` with ARAPPAV-specific reward computation.
GRPO (Group Relative Policy Optimization) samples multiple perturbations
per input paper (group size G), computes ``r_P`` for each, ranks them,
and updates the policy using relative advantage within each group.

This is a natural fit for the Perturber: sample G perturbations per paper,
the Verifier evaluates all of them, and Perturber is rewarded when its
injected errors fool the Verifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO Perturber training."""

    learning_rate: float = 5e-6
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    beta: float = 0.04
    epsilon: float = 0.2
    num_generations: int = 8  # G
    temperature: float = 0.9
    max_prompt_length: int = 4096
    use_vllm_for_rollouts: bool = True
    missing_eos_penalty: float = 1.0


class GRPOPerturberTrainer:
    """Trainer for the Perturber model using GRPO.

    Delegates to TRL's ``GRPOTrainer`` for the core RL update, while providing
    ARAPPAV-specific reward computation and rollout management.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: GRPOConfig,
        reward_function: callable,
        output_dir: str | Path = "./checkpoints/perturber",
        use_peft: bool = True,
        peft_config: dict | None = None,
    ):
        """
        Args:
            model: The Perturber policy model (may be PEFT-wrapped).
            tokenizer: Tokenizer matching the model.
            config: GRPO training hyperparameters.
            reward_function: Callable ``(prompt, completion) -> float`` that
                computes the Perturber's reward for one rollout. The caller
                (self-play loop) is responsible for wiring this up so the
                reward function has access to the Verifier, matcher, and config.
            output_dir: Directory for checkpoints.
            use_peft: Whether the model uses PEFT/LoRA.
            peft_config: PEFT config dict (if applicable).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.reward_function = reward_function
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_peft = use_peft
        self.peft_config = peft_config

        self._grpo_trainer = None

    def _build_grpo_trainer(self, train_dataset: Dataset):
        """Construct the underlying TRL GRPOTrainer."""
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer

        trl_config = TRLGRPOConfig(
            output_dir=str(self.output_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            optim=self.config.optim,
            beta=self.config.beta,
            epsilon=self.config.epsilon,
            num_generations=self.config.num_generations,
            temperature=self.config.temperature,
            max_completion_length=self.config.max_prompt_length,
            use_vllm=self.config.use_vllm_for_rollouts,
            logging_steps=10,
            save_steps=500,
            report_to="wandb" if False else "none",  # config-driven in practice
            remove_unused_columns=False,
        )

        self._grpo_trainer = GRPOTrainer(
            model=self.model,
            args=trl_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            reward_funcs=[self.reward_function],
        )

    def train(self, train_dataset: Dataset) -> dict:
        """Run one round of GRPO training on the Perturber.

        Args:
            train_dataset: HF Dataset where each row contains at minimum:
                - ``"prompt"``: the perturber prompt string (includes paper text).
                - ``"paper_id"``: paper identifier.
                - ``"k"``: number of errors to inject.

        Returns:
            Training metrics dict.
        """
        if self._grpo_trainer is None:
            self._build_grpo_trainer(train_dataset)

        logger.info(
            f"Starting GRPO training: {self.config.num_epochs} epoch(s), "
            f"G={self.config.num_generations}, lr={self.config.learning_rate}"
        )

        train_result = self._grpo_trainer.train()
        metrics = train_result.metrics

        # Save checkpoint
        checkpoint_dir = self.output_dir / f"grpo_step_{train_result.metrics.get('step', 'end')}"
        self._grpo_trainer.save_model(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        logger.info(f"Perturber checkpoint saved to {checkpoint_dir}")

        return metrics

    def save_final(self):
        """Save the final model state."""
        final_dir = self.output_dir / "final"
        if self._grpo_trainer is not None:
            self._grpo_trainer.save_model(str(final_dir))
        else:
            self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Final Perturber model saved to {final_dir}")


# ---------------------------------------------------------------------------
# Reward function factory for GRPO
# ---------------------------------------------------------------------------


def make_perturber_reward_fn(
    verifier_model,
    reward_config: dict,
    k_sampler: callable | None = None,
    mode: str = "paper",
):
    """Build a reward function suitable for TRL's GRPOTrainer.

    The returned function has signature ``(prompts, completions, **kwargs) -> list[float]``
    as expected by TRL. Internally it:
    1. Parses the Perturber's structured output from each completion.
    2. Runs the Verifier on each perturbed text.
    3. Computes r_P via ``compute_rewards``.

    Args:
        verifier_model: The (currently frozen) Verifier model wrapper.
        reward_config: Reward configuration dict.
        k_sampler: Optional callable ``() -> int`` for sampling k per episode.
        mode: ``"paper"`` or ``"math"`` — determines which validation schema to use.

    Returns:
        A reward function compatible with TRL's GRPOTrainer.
    """
    from arappav.reward.reward_fns import compute_rewards

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        import json

        # Extract per-prompt original_solution values from the dataset columns
        # (passed through by TRL's GRPOTrainer as kwargs).
        original_solutions = kwargs.get("original_solution", [None] * len(prompts))

        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Parse perturber output from completion
            try:
                # The completion should be JSON
                data = json.loads(completion)
            except json.JSONDecodeError:
                rewards.append(reward_config.get("format_penalty", -10.0))
                continue

            k = k_sampler() if k_sampler else 3

            # Use the mode-appropriate validator
            original = (
                original_solutions[i]
                if i < len(original_solutions) and original_solutions[i]
                else None
            )
            if mode == "math":
                from arappav.errors.schema_math import validate_math_perturber_output

                perturber_out, err = validate_math_perturber_output(
                    data, k, original_solution=original,
                )
            else:
                from arappav.errors.schema import validate_perturber_output

                perturber_out, err = validate_perturber_output(
                    data, k, original_text=original,
                )

            if perturber_out is None:
                logger.debug("GRPO reward: Perturber parse failed: %s", err)
                rewards.append(reward_config.get("format_penalty", -10.0))
                continue

            # Run Verifier — math mode uses perturbed_solution, paper mode uses perturbed_text
            if mode == "math":
                perturbed_text = perturber_out.perturbed_solution
            else:
                perturbed_text = perturber_out.perturbed_text

            verifier_results = verifier_model.generate(perturbed_text, n_completions=1)
            _, verifier_out, _ = verifier_results[0] if verifier_results else ("", None, "no output")

            if verifier_out is None:
                # Verifier failed → Perturber "wins" by default
                reward_out = compute_rewards(
                    ground_truth=perturber_out.errors,
                    verifier_claims=[],
                    perturbed_text=perturbed_text,
                    k=k,
                    config=reward_config,
                    perturber_format_valid=True,
                )
            else:
                reward_out = compute_rewards(
                    ground_truth=perturber_out.errors,
                    verifier_claims=verifier_out.claims,
                    perturbed_text=perturbed_text,
                    k=k,
                    config=reward_config,
                    perturber_format_valid=True,
                )

            rewards.append(reward_out.perturber_reward)

        return rewards

    return reward_fn
