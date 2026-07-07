"""Math-specific Pydantic schemas for structured I/O contracts.

In math mode, the Perturber receives a math problem + correct solution and must
inject errors into the solution. The Verifier receives the problem + (possibly)
perturbed solution and must identify errors.

These are separate from the paper-mode schemas because the input/output shapes
differ: math mode works with (problem, solution) pairs, not free-form text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from arappav.errors.taxonomy_math import MathErrorType


# ---------------------------------------------------------------------------
# Perturber math output schemas
# ---------------------------------------------------------------------------


class MathInjectedError(BaseModel):
    """A single error injected by the Perturber into a math solution."""

    error_id: str = Field(..., description="Unique identifier for this error (e.g., 'err_001')")
    step_index: int = Field(
        ..., ge=0, description="Which solution step contains the error (0-indexed)."
    )
    original_text: str = Field(
        ..., description="The correct text/step from the original solution."
    )
    injected_text: str = Field(
        ..., description="The erroneous replacement text/step."
    )
    error_type: MathErrorType = Field(..., description="Category of the injected error.")
    rationale: str = Field(
        ..., description="Why the injected text constitutes an error — the ground-truth explanation."
    )

    @field_validator("error_id")
    @classmethod
    def error_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("error_id must not be empty")
        return v.strip()

    @field_validator("original_text", "injected_text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text fields must not be empty")
        return v


class MathPerturberOutput(BaseModel):
    """Full structured output from the Perturber in math mode."""

    perturbed_solution: str = Field(
        ..., description="The full solution text with all k errors injected."
    )
    errors: list[MathInjectedError] = Field(
        ..., description="List of injected errors; must have exactly k entries."
    )

    @field_validator("perturbed_solution")
    @classmethod
    def solution_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("perturbed_solution must not be empty")
        return v


# ---------------------------------------------------------------------------
# Verifier math output schemas
# ---------------------------------------------------------------------------


class MathVerifierClaim(BaseModel):
    """A single claim by the Verifier about an error in a math solution."""

    step_index: int | None = Field(
        default=None, ge=0, description="Which solution step contains the error (0-indexed, if known)."
    )
    quoted_text: str = Field(
        ..., description="The exact erroneous text from the solution."
    )
    explanation: str = Field(
        ..., description="Why the Verifier believes this text is wrong, including the correct approach."
    )
    error_type: MathErrorType | None = Field(
        default=None,
        description="Optional: the category of error the Verifier believes this is.",
    )

    @field_validator("quoted_text", "explanation")
    @classmethod
    def fields_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("quoted_text and explanation must not be empty")
        return v


class MathVerifierOutput(BaseModel):
    """Full structured output from the Verifier in math mode."""

    claims: list[MathVerifierClaim] = Field(
        default_factory=list,
        description="List of error claims; empty if Verifier finds no errors.",
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_math_perturber_output(
    data: dict, expected_k: int
) -> tuple[MathPerturberOutput | None, str | None]:
    """Validate and parse Perturber output in math mode.

    Args:
        data: Raw parsed JSON from the Perturber.
        expected_k: Required number of injected errors.

    Returns:
        (MathPerturberOutput, None) on success, or (None, error_message) on failure.
    """
    try:
        output = MathPerturberOutput.model_validate(data)
    except Exception as e:
        return None, f"Schema validation failed: {e}"

    if len(output.errors) != expected_k:
        return None, (
            f"Expected exactly {expected_k} errors, got {len(output.errors)}. "
            f"Error IDs: {[e.error_id for e in output.errors]}"
        )

    ids = [e.error_id for e in output.errors]
    if len(ids) != len(set(ids)):
        return None, f"Duplicate error_ids detected: {ids}"

    return output, None


def validate_math_verifier_output(
    data: dict,
) -> tuple[MathVerifierOutput | None, str | None]:
    """Validate and parse Verifier output in math mode.

    Args:
        data: Raw parsed JSON from the Verifier.

    Returns:
        (MathVerifierOutput, None) on success, or (None, error_message) on failure.
    """
    try:
        output = MathVerifierOutput.model_validate(data)
    except Exception as e:
        return None, f"Schema validation failed: {e}"

    return output, None
