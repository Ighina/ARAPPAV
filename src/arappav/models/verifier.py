"""Verifier model wrapper with prompt templates and generation logic.

The Verifier receives a (possibly) perturbed paper or math solution and must
locate and explain all errors, outputting structured claims.

Supports two modes:
- ``paper``: Verify academic paper sections for factual/logical/statistical errors.
- ``math``: Verify math solutions for misconception-based errors.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from arappav.errors.schema import VerifierOutput, validate_verifier_output
from arappav.errors.schema_math import MathVerifierOutput, validate_math_verifier_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — Paper mode
# ---------------------------------------------------------------------------


def build_verifier_prompt(
    text: str,
    template_name: str = "verifier_default",
) -> str:
    """Build the prompt for the Verifier model (paper mode).

    Args:
        text: The (possibly) perturbed paper section text.
        template_name: Which prompt template to use.

    Returns:
        Formatted prompt string.
    """
    if template_name == "verifier_default":
        return _verifier_default_prompt(text)
    else:
        raise ValueError(f"Unknown verifier prompt template: {template_name}")


def _verifier_default_prompt(text: str) -> str:
    """Default Verifier prompt (paper mode)."""
    return f"""You are an expert reviewer of academic papers with a sharp eye for
factual, logical, and statistical errors.

Your task: carefully read the academic text below and identify ALL errors
that may have been introduced — such as altered statistics, misattributed
citations, logical fallacies, methodological mistakes, swapped terminology,
or flipped negations.

## Rules
1. Identify every error you can find. Be thorough but precise.
2. For each error, quote the exact erroneous text and explain WHY it is wrong.
3. If you believe the text is error-free, return an empty claims list.
4. Output ONLY the JSON object described below — no preamble, no markdown fences.

## Required JSON output format
```json
{{
  "claims": [
    {{
      "location": "paragraph 2, sentence 3",
      "quoted_text": "<the exact erroneous text from the paper>",
      "explanation": "<why this text is wrong>"
    }}
  ]
}}
```

## Academic text to review
{text}

## Your response (JSON only):"""


# ---------------------------------------------------------------------------
# Prompt templates — Math mode
# ---------------------------------------------------------------------------


def build_math_verifier_prompt(
    problem: str,
    solution: str,
    template_name: str = "verifier_math_default",
) -> str:
    """Build the prompt for the Verifier in math mode.

    Args:
        problem: The math problem statement.
        solution: The (possibly perturbed) solution to review.
        template_name: Which prompt template to use.

    Returns:
        Formatted prompt string.
    """
    if template_name == "verifier_math_default":
        return _verifier_math_default(problem, solution)
    else:
        raise ValueError(f"Unknown math verifier prompt template: {template_name}")


def _verifier_math_default(problem: str, solution: str) -> str:
    """Default Verifier prompt for math mode."""
    return f"""You are an expert mathematics reviewer who can identify subtle errors
in mathematical solutions — including arithmetic mistakes, misapplied concepts,
incorrect operations, skipped steps, and common student misconceptions.

Your task: carefully review the math problem and its solution below. Identify
ALL errors in the solution. Be thorough but precise.

## Rules
1. Identify every error you can find in the solution.
2. For each error, quote the exact erroneous text, indicate which step it
   appears in, and explain WHY it is wrong, including what the correct
   approach would be.
3. If you believe the solution is completely correct, return an empty claims list.
4. Output ONLY the JSON object described below — no preamble, no markdown fences.

## Required JSON output format
```json
{{
  "claims": [
    {{
      "step_index": 0,
      "quoted_text": "<the exact erroneous text from the solution>",
      "explanation": "<why this is wrong and what the correct approach would be>",
      "error_type": "wrong_operation"
    }}
  ]
}}
```

## Math Problem
{problem}

## Solution to Review
{solution}

## Your response (JSON only):"""


# ---------------------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------------------


class VerifierModel:
    """Wrapper around a Verifier policy model.

    Handles prompt construction, generation, and structured-output parsing.
    Supports both paper mode and math mode via the ``mode`` parameter.

    Designed to work both with a local HF model/pipeline and with vLLM.
    """

    def __init__(
        self,
        model_name_or_path: str,
        mode: str = "paper",
        use_vllm: bool = False,
        vllm_engine=None,  # : VLLMEngine | None
        generation_kwargs: dict | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.use_vllm = use_vllm
        self.vllm_engine = vllm_engine
        self.generation_kwargs = generation_kwargs or {}

        self._hf_pipeline = None

    def _get_hf_pipeline(self):
        if self._hf_pipeline is None:
            from transformers import pipeline

            logger.info(f"Loading Verifier model: {self.model_name_or_path}")
            self._hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name_or_path,
                device_map="auto",
                trust_remote_code=True,
            )
        return self._hf_pipeline

    def generate(
        self,
        text: str,
        problem: str | None = None,
        template_name: str | None = None,
        n_completions: int = 1,
        **override_kwargs,
    ) -> list[tuple[VerifierOutput | MathVerifierOutput | None, str | None]]:
        """Generate verifier claims for a given text/solution.

        Args:
            text: The text to review. In paper mode, this is the perturbed text.
                  In math mode, this is the **solution** to review.
            problem: Required in math mode — the problem statement.
            template_name: Prompt template to use (auto-selected per mode if None).
            n_completions: Number of completions to sample (for DPO pair construction).
            **override_kwargs: Override default generation kwargs.

        Returns:
            List of ``(Output, None)`` or ``(None, error_msg)`` tuples.
            Output is ``VerifierOutput`` in paper mode, ``MathVerifierOutput`` in math mode.
        """
        # Auto-select template based on mode
        if template_name is None:
            template_name = "verifier_math_default" if self.mode == "math" else "verifier_default"

        if self.mode == "math":
            if problem is None:
                return [(None, "Math mode requires a 'problem' argument.")]
            prompt = build_math_verifier_prompt(problem, text, template_name)
        else:
            prompt = build_verifier_prompt(text, template_name)

        gen_kwargs = {**self.generation_kwargs, **override_kwargs}

        if self.use_vllm and self.vllm_engine is not None:
            raw_outputs = self.vllm_engine.generate(
                [prompt] * n_completions, **gen_kwargs
            )
        else:
            pipeline = self._get_hf_pipeline()
            results = []
            for _ in range(n_completions):
                result = pipeline(prompt, **gen_kwargs)
                results.append(result[0]["generated_text"])
            raw_outputs = results

        parsed = []
        for raw in raw_outputs:
            if raw.startswith(prompt):
                raw = raw[len(prompt):].strip()
            parsed.append(_parse_verifier_response(raw, self.mode))

        return parsed


def _parse_verifier_response(
    raw_output: str, mode: str = "paper"
) -> tuple[VerifierOutput | MathVerifierOutput | None, str | None]:
    """Attempt to parse the Verifier's raw string output as JSON."""
    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}\nRaw output (first 500 chars): {raw_output[:500]}"

    if mode == "math":
        return validate_math_verifier_output(data)
    else:
        return validate_verifier_output(data)
