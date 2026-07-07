"""DPO Trainer for the Verifier.

Wraps TRL's ``DPOTrainer``. The Verifier is trained via Direct Preference
Optimization using preference pairs constructed from self-play rollouts:
for the same perturbed input, a higher-r_V verifier response is "chosen"
and a lower-r_V response is "rejected".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO Verifier training."""

    learning_rate: float = 5e-6
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    beta: float = 0.1
    max_prompt_length: int = 4096
    max_length: int = 6144
    loss_type: str = "sigmoid"
    precompute_ref_log_probs: bool = False


class DPOVerifierTrainer:
    """Trainer for the Verifier model using DPO.

    Expects a dataset of preference pairs built by ``PreferenceBuilder``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizer,
        config: DPOConfig,
        output_dir: str | Path = "./checkpoints/verifier",
        use_peft: bool = True,
        peft_config: dict | None = None,
    ):
        """
        Args:
            model: The Verifier policy model.
            ref_model: Reference model for DPO (usually a frozen copy of the
                initial policy). If None, TRL will use ``disable_dropout`` on
                the model itself.
            tokenizer: Tokenizer.
            config: DPO training hyperparameters.
            output_dir: Directory for checkpoints.
            use_peft: Whether the model uses PEFT/LoRA.
            peft_config: PEFT config dict.
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_peft = use_peft
        self.peft_config = peft_config

        self._dpo_trainer = None

    def _build_dpo_trainer(self, train_dataset: Dataset):
        """Construct the underlying TRL DPOTrainer."""
        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer

        trl_config = TRLDPOConfig(
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
            max_prompt_length=self.config.max_prompt_length,
            max_length=self.config.max_length,
            loss_type=self.config.loss_type,
            precompute_ref_log_probs=self.config.precompute_ref_log_probs,
            logging_steps=10,
            save_steps=500,
            report_to="none",
            remove_unused_columns=False,
        )

        self._dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=trl_config,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

    def train(self, train_dataset: Dataset) -> dict:
        """Run one round of DPO training on the Verifier.

        Args:
            train_dataset: HF Dataset with columns:
                - ``"prompt"``: the verifier prompt (includes perturbed text).
                - ``"chosen"``: the better verifier response.
                - ``"rejected"``: the worse verifier response.

        Returns:
            Training metrics dict.
        """
        if self._dpo_trainer is None:
            self._build_dpo_trainer(train_dataset)

        logger.info(
            f"Starting DPO training: {self.config.num_epochs} epoch(s), "
            f"beta={self.config.beta}, lr={self.config.learning_rate}"
        )

        train_result = self._dpo_trainer.train()
        metrics = train_result.metrics

        checkpoint_dir = self.output_dir / f"dpo_step_{train_result.metrics.get('step', 'end')}"
        self._dpo_trainer.save_model(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        logger.info(f"Verifier checkpoint saved to {checkpoint_dir}")

        return metrics

    def save_final(self):
        """Save the final model state."""
        final_dir = self.output_dir / "final"
        if self._dpo_trainer is not None:
            self._dpo_trainer.save_model(str(final_dir))
        else:
            self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Final Verifier model saved to {final_dir}")
