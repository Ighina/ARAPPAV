"""Pydantic schemas for structured I/O contracts.

Both Perturber and Verifier must emit JSON conforming to these models.
Parse failures are hard-penalized rather than crashing the pipeline.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator

from arappav.errors.taxonomy import ErrorType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturber output schemas
# ---------------------------------------------------------------------------


class InjectedError(BaseModel):
    """A single error injected by the Perturber into the paper text."""

    error_id: str = Field(..., description="Unique identifier for this error (e.g., 'err_001')")
    location: str = Field(
        ...,
        description="Rough location descriptor (e.g., paragraph index, section name, "
        "or character span like 'chars 450-520')",
    )
    original_text: str = Field(
        ..., description="The original (correct) text that was replaced or modified."
    )
    injected_text: str = Field(
        ..., description="The injected (erroneous) text that replaced the original."
    )
    error_type: ErrorType = Field(..., description="Category of the injected error.")
    rationale: str = Field(
        ..., description="Why the injected text constitutes an error (the ground-truth explanation)."
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


class PerturberOutput(BaseModel):
    """Full structured output from the Perturber."""

    perturbed_text: str = Field(
        ..., description="The full paper/section text with all k errors injected."
    )
    errors: list[InjectedError] = Field(
        ..., description="List of injected errors; must have exactly k entries."
    )

    @field_validator("perturbed_text")
    @classmethod
    def perturbed_text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("perturbed_text must not be empty")
        return v


# ---------------------------------------------------------------------------
# Verifier output schemas
# ---------------------------------------------------------------------------


class VerifierClaim(BaseModel):
    """A single claim by the Verifier about an error in the paper."""

    location: str = Field(
        ...,
        description="Rough location of the claimed error (paragraph index, section, or char span).",
    )
    quoted_text: str = Field(
        ..., description="The exact text the Verifier identifies as erroneous."
    )
    explanation: str = Field(
        ..., description="Why the Verifier believes this text is wrong."
    )

    @field_validator("quoted_text", "explanation")
    @classmethod
    def fields_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("quoted_text and explanation must not be empty")
        return v


class VerifierOutput(BaseModel):
    """Full structured output from the Verifier."""

    claims: list[VerifierClaim] = Field(
        default_factory=list,
        description="List of error claims; may be empty if Verifier finds no errors.",
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_perturber_output(
    data: dict, expected_k: int, original_text: str | None = None
) -> tuple[PerturberOutput | None, str | None]:
    """Validate and parse Perturber output, enforcing exactly `expected_k` errors.

    Args:
        data: Raw parsed JSON from the Perturber.
        expected_k: Required number of injected errors.
        original_text: If provided, the original paper text.  Used to verify
            that the perturbed text actually differs from the input.

    Returns:
        (PerturberOutput, None) on success, or (None, error_message) on failure.
    """
    try:
        output = PerturberOutput.model_validate(data)
    except Exception as e:
        return None, f"Schema validation failed: {e}"

    if len(output.errors) != expected_k:
        return None, (
            f"Expected exactly {expected_k} errors, got {len(output.errors)}. "
            f"Error IDs: {[e.error_id for e in output.errors]}"
        )

    # Check for duplicate error IDs
    ids = [e.error_id for e in output.errors]
    if len(ids) != len(set(ids)):
        return None, f"Duplicate error_ids detected: {ids}"

    # Reject if the perturbed text is identical to the original
    if original_text is not None and output.perturbed_text == original_text:
        return None, (
            "Perturber returned perturbed_text identical to the original — "
            "no errors were actually injected into the text. "
            f"The model defined {len(output.errors)} error(s) in JSON but "
            "did not modify the text."
        )

    # Soft warning: check each injected_text appears somewhere in the output
    for error in output.errors:
        if error.injected_text not in output.perturbed_text:
            logger.warning(
                "Injected text for %s not found in perturbed_text — "
                "the error may not be detectable. Injected: %r",
                error.error_id,
                error.injected_text[:120],
            )

    return output, None


def validate_verifier_output(data: dict) -> tuple[VerifierOutput | None, str | None]:
    """Validate and parse Verifier output.

    Args:
        data: Raw parsed JSON from the Verifier.

    Returns:
        (VerifierOutput, None) on success, or (None, error_message) on failure.
    """
    try:
        output = VerifierOutput.model_validate(data)
    except Exception as e:
        return None, f"Schema validation failed: {e}"

    return output, None
