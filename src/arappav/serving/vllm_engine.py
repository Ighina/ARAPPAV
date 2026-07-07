"""vLLM engine wrapper for fast rollout generation.

Both Perturber and Verifier use this during self-play rollouts to generate
large batches of completions efficiently.

The engine is initialised once and reused across rounds to avoid costly
model reloads.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VLLMEngine:
    """Thin wrapper around a vLLM ``LLM`` instance for batch generation.

    Usage::

        engine = VLLMEngine("Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=1)
        outputs = engine.generate(["prompt 1", "prompt 2"], temperature=0.8, max_tokens=1024)
    """

    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        dtype: str = "auto",
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
    ):
        """Initialise the vLLM engine.

        Args:
            model_name_or_path: HF model ID or local path.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_len: Maximum context length.
            dtype: Data type for model weights.
            enforce_eager: Disable CUDA graph (saves memory, slower).
            trust_remote_code: Allow custom modeling code.
        """
        self.model_name_or_path = model_name_or_path
        self._llm = None
        self._init_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "trust_remote_code": trust_remote_code,
        }

    @property
    def llm(self):
        """Lazy-load the vLLM LLM instance."""
        if self._llm is None:
            from vllm import LLM

            logger.info(f"Initialising vLLM engine for: {self.model_name_or_path}")
            self._llm = LLM(**self._init_kwargs)
        return self._llm

    def generate(
        self,
        prompts: list[str],
        sampling_params: dict | None = None,
        **kwargs,
    ) -> list[str]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of prompt strings.
            sampling_params: vLLM SamplingParams kwargs. If None, reasonable
                defaults are used.
            **kwargs: Additional SamplingParams or generate arguments.

        Returns:
            List of generated text strings (one per prompt).
        """
        from vllm import SamplingParams

        if sampling_params is None:
            sampling_params = {}

        # Merge sampling_params dict with explicit kwargs (kwargs win)
        sp_kwargs = {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 4096,
            **sampling_params,
            **kwargs,
        }
        sp = SamplingParams(**sp_kwargs)

        outputs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text for o in outputs]

    def generate_single(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate a single completion.

        Args:
            prompt: Single prompt string.
            **kwargs: Passed to ``generate()``.

        Returns:
            Generated text string.
        """
        return self.generate([prompt], **kwargs)[0]

    def is_loaded(self) -> bool:
        """Check if the vLLM model has been loaded."""
        return self._llm is not None

    def unload(self):
        """Release GPU memory by deleting the vLLM instance."""
        if self._llm is not None:
            logger.info("Unloading vLLM engine to free GPU memory.")
            del self._llm
            self._llm = None
            import torch

            torch.cuda.empty_cache()
