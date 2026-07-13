"""Math-specific Pydantic schemas for structured I/O contracts.

In math mode, the Perturber receives a math problem + correct solution and must
inject errors into the solution. The Verifier receives the problem + (possibly)
perturbed solution and must identify errors.

These are separate from the paper-mode schemas because the input/output shapes
differ: math mode works with (problem, solution) pairs, not free-form text.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, field_validator, model_validator

from arappav.errors.fuzzy import fuzzy_match_enum
from arappav.errors.taxonomy_math import MathErrorType

logger = logging.getLogger(__name__)


#: Curated aliases for error-type names models persistently emit that fuzzy
#: matching (correctly) refuses to guess. Observed in Round 1-2 rollouts.
MATH_ERROR_TYPE_ALIASES = {
    "sign_error": "negative_number_error",
}


def _fuzzy_match_math_error_type(name: str) -> MathErrorType | None:
    """Find the closest MathErrorType to *name* (see ``fuzzy.fuzzy_match_enum``)."""
    return fuzzy_match_enum(name, MathErrorType, aliases=MATH_ERROR_TYPE_ALIASES)


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

    @field_validator("error_type", mode="before")
    @classmethod
    def fuzzy_match_error_type(cls, v: str | MathErrorType) -> MathErrorType:
        """Allow near-miss error type names via fuzzy matching.

        If the exact enum value is not found, auto-correct typos (small edit
        distance) and alternative wordings that share a non-generic word with
        a taxonomy member (e.g. ``'addition_across'`` → ``'adding_across'``).
        See ``arappav.errors.fuzzy.fuzzy_match_enum`` for the exact rules.
        """
        if isinstance(v, MathErrorType):
            return v
        # Try exact match first
        try:
            return MathErrorType(v)
        except ValueError:
            pass
        # Fuzzy match
        best = _fuzzy_match_math_error_type(v)
        if best is not None:
            logger.warning(
                "Fuzzy-matched error_type %r → %r. "
                "The Perturber should be encouraged to use exact enum values.",
                v, best.value,
            )
            return best
        raise ValueError(
            f"Unknown error_type {v!r} — not in MathErrorType taxonomy and no "
            f"close match found. "
            f"Valid types: {[e.value for e in MathErrorType]}"
        )

    @model_validator(mode="after")
    def injected_must_differ_from_original(self):
        """Reject phantom errors where injected_text equals original_text."""
        inj = self.injected_text.strip()
        orig = self.original_text.strip()
        if inj == orig:
            raise ValueError(
                f"Phantom error detected for {self.error_id}: "
                f"injected_text equals original_text — no actual modification was made. "
                f"Text: {inj[:120]!r}"
            )
        return self


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

    @field_validator("error_type", mode="before")
    @classmethod
    def coerce_unknown_to_none(cls, v: str | MathErrorType | None) -> MathErrorType | None:
        """If the Verifier outputs an error type not in the taxonomy, treat it as
        ``None`` rather than failing validation.  The Verifier should not be
        constrained to the Perturber's fixed taxonomy."""
        if v is None:
            return None
        if isinstance(v, MathErrorType):
            return v
        # v is a string — check if it's a valid enum member
        try:
            return MathErrorType(v)
        except ValueError:
            logger.warning(
                "Unknown error_type %r — coercing to None.", v,
            )
            return None


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
    data: dict, expected_k: int, original_solution: str | None = None
) -> tuple[MathPerturberOutput | None, str | None]:
    """Validate and parse Perturber output in math mode.

    Args:
        data: Raw parsed JSON from the Perturber.
        expected_k: Required number of injected errors.
        original_solution: If provided, the original correct solution.  Used to
            verify that the perturbed solution actually differs from the input.

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

    # Reject if the perturbed solution is identical to the original
    if original_solution is not None and output.perturbed_solution == original_solution:
        return None, (
            "Perturber returned perturbed_solution identical to the original — "
            "no errors were actually injected into the solution. "
            f"The model defined {len(output.errors)} error(s) in JSON but "
            "did not modify the solution text."
        )

    # Soft warning: check each injected_text appears somewhere in the output
    for error in output.errors:
        if error.injected_text not in output.perturbed_solution:
            logger.warning(
                "Injected text for %s not found in perturbed_solution — "
                "the error may not be detectable. Injected: %r",
                error.error_id,
                error.injected_text[:120],
            )

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
