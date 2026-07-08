"""Math-specific prompt templates for Perturber and Verifier.

In math mode:
- The Perturber receives a math problem + correct solution, and must inject
  exactly k errors into the solution steps.
- The Verifier receives the problem + (possibly) perturbed solution, and must
  identify and explain all errors.
"""

from __future__ import annotations

from arappav.errors.taxonomy_math import MathErrorType


# ---------------------------------------------------------------------------
# Perturber math prompts
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
        problem: The math problem statement (LaTeX).
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
# Verifier math prompts
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
