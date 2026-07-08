"""Perturber model wrapper with prompt templates and generation logic.

The Perturber is given a paper (or math problem+solution) and must insert exactly
``k`` errors, returning a perturbed version plus a machine-readable ground-truth log.

Supports two modes:
- ``paper``: Perturb academic paper sections with factual/logical/statistical errors.
- ``math``: Perturb math solutions with misconception-based errors.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from arappav.errors.schema import PerturberOutput, validate_perturber_output
from arappav.errors.schema_math import MathPerturberOutput, validate_math_perturber_output
from arappav.errors.taxonomy import ErrorType
from arappav.errors.taxonomy_math import MathErrorType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates — Paper mode
# ---------------------------------------------------------------------------


def build_perturber_prompt(
    text: str,
    k: int,
    template_name: str = "perturber_default",
    error_type_descriptions: str | None = None,
) -> str:
    """Build the prompt for the Perturber model (paper mode).

    Args:
        text: The paper section text to perturb.
        k: Number of errors to inject.
        template_name: Which prompt template to use.
        error_type_descriptions: Pre-formatted error type list.

    Returns:
        Formatted prompt string.
    """
    if error_type_descriptions is None:
        error_type_descriptions = ErrorType.prompt_list()

    if template_name == "perturber_default":
        return _perturber_default_prompt(text, k, error_type_descriptions)
    else:
        raise ValueError(f"Unknown perturber prompt template: {template_name}")


def _perturber_default_prompt(text: str, k: int, error_types: str) -> str:
    """Default Perturber prompt (paper mode)."""
    return f"""You are an expert at identifying and inserting subtle factual, logical,
and statistical errors into academic text.

Your task: given the academic text below, insert exactly **{k}** distinct errors.
Each error must be one of the following types:

{error_types}

## Rules
1. You MUST inject exactly {k} errors — no more, no less.
2. Errors should be subtle and blend naturally with the surrounding text.
   They should NOT be trivially obvious (e.g., "the sky is green").
3. Each error must be of a different type if possible; if the text is short,
   you may reuse types.
4. Keep the overall structure and formatting of the original text intact.
5. Output ONLY the JSON object described below — no preamble, no markdown fences.

## Required JSON output format
```json
{{
  "perturbed_text": "<the full text with all {k} errors injected>",
  "errors": [
    {{
      "error_id": "err_001",
      "location": "paragraph 2, sentence 3",
      "original_text": "<the original correct text>",
      "injected_text": "<the erroneous replacement text>",
      "error_type": "numerical",
      "rationale": "<why this injected text is wrong>"
    }}
  ]
}}
```

## Academic text to perturb
{text}

## Your response (JSON only):"""


# ---------------------------------------------------------------------------
# Prompt templates — Math mode
# ---------------------------------------------------------------------------


def build_math_perturber_prompt(
    problem: str,
    solution: str,
    k: int,
    template_name: str = "perturber_math_default",
    error_type_descriptions: str | None = None,
) -> str:
    """Build the prompt for the Perturber in math mode.

    Args:
        problem: The math problem statement.
        solution: The correct step-by-step solution.
        k: Number of errors to inject.
        template_name: Which prompt template to use.
        error_type_descriptions: Pre-formatted error type list.

    Returns:
        Formatted prompt string.
    """
    if error_type_descriptions is None:
        error_type_descriptions = MathErrorType.prompt_list()

    if template_name == "perturber_math_default":
        return _perturber_math_default(problem, solution, k, error_type_descriptions)
    else:
        raise ValueError(f"Unknown math perturber prompt template: {template_name}")


def _perturber_math_default(
    problem: str, solution: str, k: int, error_types: str
) -> str:
    """Default Perturber prompt for math mode."""
    return f"""You are an expert at identifying and injecting realistic mathematical
errors into step-by-step solutions, mimicking common student misconceptions.

Your task: given a math problem and its correct solution below, insert exactly
**{k}** distinct errors into the solution. Each error must be one of the
following types (derived from the literature on math misconceptions):

{error_types}

## Rules
1. Inject EXACTLY {k} errors — no more, no less.
2. Errors should be **subtle and realistic** — they should look like genuine
   student mistakes, not obvious nonsense.
3. Each error should target a different step of the solution if possible.
4. Preserve the overall structure and LaTeX formatting of the solution.
5. The perturbed solution should still read as a coherent (but flawed)
   attempt at solving the problem.
6. Output ONLY the JSON object described below — no preamble, no markdown fences.

## Required JSON output format
```json
{{
  "perturbed_solution": "<the full solution with all {k} errors injected>",
  "errors": [
    {{
      "error_id": "err_001",
      "step_index": 0,
      "original_text": "<the correct text from the solution>",
      "injected_text": "<the erroneous replacement text>",
      "error_type": "wrong_operation",
      "rationale": "<why this is wrong and what the correct approach would be>"
    }}
  ]
}}
```

## Math Problem
{problem}

## Correct Solution
{solution}

## Your response (JSON only):"""


# ---------------------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------------------


class PerturberModel:
    """Wrapper around a Perturber policy model.

    Handles prompt construction, generation, and structured-output parsing.
    Supports both paper mode and math mode via the ``mode`` parameter.

    Designed to work both with a local HF model/pipeline and with vLLM.
    """

    def __init__(
        self,
        model_name_or_path: str,
        mode: str = "paper",
        use_vllm: bool = False,
        vllm_engine=None,  # : VLLMEngine | None — don't want circular import
        generation_kwargs: dict | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.use_vllm = use_vllm
        self.vllm_engine = vllm_engine
        self.generation_kwargs = generation_kwargs or {}

        # Lazy-loaded HF pipeline (for non-vLLM local inference)
        self._hf_pipeline = None

    def _get_hf_pipeline(self):
        if self._hf_pipeline is None:
            from transformers import pipeline

            logger.info(f"Loading Perturber model: {self.model_name_or_path}")
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
        k: int,
        solution: str | None = None,
        template_name: str = "perturber_default",
        **override_kwargs,
    ) -> tuple[PerturberOutput | MathPerturberOutput | None, str | None]:
        """Generate a perturbed version of the input with k errors.

        Args:
            text: The text to perturb. In paper mode, this is the paper section.
                  In math mode, this is the **problem** statement.
            k: Number of errors to inject.
            solution: Required in math mode — the correct solution to perturb.
            template_name: Prompt template to use.
            **override_kwargs: Override default generation kwargs.

        Returns:
            ``(Output, None)`` on success, ``(None, error_msg)`` on failure.
            Output is ``PerturberOutput`` in paper mode, ``MathPerturberOutput`` in math mode.
        """
        if self.mode == "math":
            if solution is None:
                return None, "Math mode requires a 'solution' argument."
            prompt = build_math_perturber_prompt(text, solution, k, template_name)
        else:
            prompt = build_perturber_prompt(text, k, template_name)

        gen_kwargs = {**self.generation_kwargs, **override_kwargs}

        if self.use_vllm and self.vllm_engine is not None:
            raw_output = self.vllm_engine.generate_single(prompt, **gen_kwargs)
        else:
            pipeline = self._get_hf_pipeline()
            result = pipeline(prompt, **gen_kwargs)
            raw_output = result[0]["generated_text"]

        # Remove the prompt prefix if echoed back
        if raw_output.startswith(prompt):
            raw_output = raw_output[len(prompt):].strip()

        return _parse_response(raw_output, k, self.mode)


def _parse_response(
    raw_output: str, k: int, mode: str = "paper"
) -> tuple[PerturberOutput | MathPerturberOutput | None, str | None]:
    """Attempt to parse the Perturber's raw string output as JSON.

    Handles common failure modes: markdown code fences, trailing commas,
    missing outer braces.
    """
    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try to extract JSON object if there's surrounding text
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
        return validate_math_perturber_output(data, k)
    else:
        return validate_perturber_output(data, k)
