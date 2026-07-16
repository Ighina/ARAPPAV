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
from arappav.models.generation_utils import prepare_generation_kwargs
from arappav.utils.parsing import apply_error_injections, extract_first_json_object, strip_json_fences

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
4. Errors must be **independent mistakes**. Do NOT count the downstream
   consequence of an earlier error (e.g. a final answer that is wrong only
   because of a previous mistake, or the same wrong term repeated on later
   lines) as a separate error — propagated or overlapping errors are scored
   as ONE error.
5. Restating existing content is NOT an error: never repeat the same
   statement or the same \\boxed{{...}} result twice — such outputs are
   rejected outright.
6. Preserve the overall structure and LaTeX formatting of the solution.
7. The perturbed solution should still read as a coherent (but flawed)
   attempt at solving the problem.
8. Output ONLY the JSON object described below — no preamble, no markdown fences.

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

        #: Raw text of the most recent generation — lets rollout collectors
        #: record the unparsed output of failed episodes for later replay.
        self.last_raw_output: str | None = None

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
        template_name: str | None = None,
        **override_kwargs,
    ) -> tuple[PerturberOutput | MathPerturberOutput | None, str | None]:
        """Generate a perturbed version of the input with k errors.

        Args:
            text: The text to perturb. In paper mode, this is the paper section.
                  In math mode, this is the **problem** statement.
            k: Number of errors to inject.
            solution: Required in math mode — the correct solution to perturb.
            template_name: Prompt template to use (auto-selected per mode if None).
            **override_kwargs: Override default generation kwargs.

        Returns:
            ``(Output, None)`` on success, ``(None, error_msg)`` on failure.
            Output is ``PerturberOutput`` in paper mode, ``MathPerturberOutput`` in math mode.
        """
        # Auto-select template based on mode
        if template_name is None:
            template_name = "perturber_math_default" if self.mode == "math" else "perturber_default"

        if self.mode == "math":
            if solution is None:
                return None, "Math mode requires a 'solution' argument."
            prompt = build_math_perturber_prompt(text, solution, k, template_name)
        else:
            prompt = build_perturber_prompt(text, k, template_name)

        gen_kwargs = {**self.generation_kwargs, **override_kwargs}

        if self.use_vllm and self.vllm_engine is not None:
            raw_output = self.vllm_engine.generate_single(
                prompt, **prepare_generation_kwargs(gen_kwargs, "vllm")
            )
        else:
            pipeline = self._get_hf_pipeline()
            result = pipeline(prompt, **prepare_generation_kwargs(gen_kwargs, "hf"))
            raw_output = result[0]["generated_text"]

        # Remove the prompt prefix if echoed back
        if raw_output.startswith(prompt):
            raw_output = raw_output[len(prompt):].strip()

        self.last_raw_output = raw_output

        # Determine original text and the target field name
        if self.mode == "math":
            original_for_check = solution
        else:
            original_for_check = text

        parsed, err, _stage = parse_and_backoff(
            raw_output, k, self.mode, original_for_check
        )
        return parsed, err


def parse_and_backoff(
    raw_output: str,
    k: int,
    mode: str = "paper",
    original_text: str | None = None,
) -> tuple[PerturberOutput | MathPerturberOutput | None, str | None, str | None]:
    """Parse Perturber output, applying the mechanical injection backoff.

    Handles common failure modes: markdown code fences (including multiple
    blocks), extra data after the first JSON object, invalid LaTeX escapes.

    Crucially, the identical-to-original check runs **after** the mechanical
    backoff: a model that defines valid errors in JSON but returns the
    unmodified text is salvaged by injecting the errors mechanically, and is
    only rejected if the text is still unchanged afterwards (e.g. all errors
    are phantom or their original_text can't be located).

    Args:
        raw_output: Raw model output.
        k: Expected number of errors.
        mode: ``"paper"`` or ``"math"``.
        original_text: If provided, the original text/solution. Used to
            reject outputs whose perturbed text is unchanged after backoff.

    Returns:
        ``(parsed, None, None)`` on success, or ``(None, error_msg, stage)``
        on failure, where *stage* is ``"json"`` when no JSON object could be
        extracted and ``"schema"`` for validation failures. Callers can use
        the stage to grade format penalties.
    """
    cleaned = strip_json_fences(raw_output)

    data, err = extract_first_json_object(cleaned)
    if data is None:
        return None, f"{err}\nRaw output (first 500 chars): {raw_output[:500]}", "json"

    # Validate WITHOUT the identical-to-original check — that runs after the
    # backoff below, which can rescue unmodified-text outputs.
    if mode == "math":
        parsed, err = validate_math_perturber_output(data, k, original_solution=None)
    else:
        parsed, err = validate_perturber_output(data, k, original_text=None)
    if parsed is None:
        return None, err, "schema"

    text_field = "perturbed_solution" if mode == "math" else "perturbed_text"
    perturbed = getattr(parsed, text_field)

    error_dicts = [
        {"error_id": e.error_id, "original_text": e.original_text,
         "injected_text": e.injected_text}
        for e in parsed.errors
    ]
    found = sum(1 for e in error_dicts if e["injected_text"] in perturbed)
    unchanged = original_text is not None and perturbed == original_text

    if found < len(error_dicts) or unchanged:
        logger.warning(
            "Only %d/%d injected_text values found in perturbed output%s — "
            "applying mechanical backoff.",
            found, len(error_dicts),
            " (text unchanged from original)" if unchanged else "",
        )
        mechanically_perturbed, warnings = apply_error_injections(
            perturbed, error_dicts,
        )
        for w in warnings:
            logger.warning("Mechanical injection: %s", w)

        setattr(parsed, text_field, mechanically_perturbed)
        perturbed = mechanically_perturbed
    else:
        logger.info(
            "All %d injected_text values found in perturbed output — "
            "using model output as-is.", len(error_dicts),
        )

    if original_text is not None and perturbed == original_text:
        return None, (
            "Perturber returned perturbed text identical to the original, and "
            f"mechanical injection could not apply any of the {len(error_dicts)} "
            "declared error(s) — no detectable errors exist in the text."
        ), "schema"

    return parsed, None, None
