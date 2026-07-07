"""Enumerated error taxonomy for Perturber injection types.

Each injected error must be tagged with one of these types in the structured output.
This enables per-category reward breakdown and evaluation.
"""

from enum import Enum


class ErrorType(str, Enum):
    """Taxonomy of errors the Perturber can inject into academic papers."""

    NUMERICAL = "numerical"
    """Altered statistic, wrong percentage, incorrect table/figure value."""

    CITATION = "citation"
    """Misattributed claim, fabricated/altered citation, wrong year."""

    LOGICAL = "logical"
    """Non-sequitur, reversed causality, invalid inference from stated premises."""

    METHODOLOGICAL = "methodological"
    """Swapped experimental setup detail (e.g., wrong dataset split, wrong hyperparameter)."""

    TERMINOLOGY = "terminology"
    """Swapped technical term with a plausible-but-wrong one."""

    NEGATION = "negation"
    """Inserted or removed negation flipping a claim's meaning."""

    # Extensible: add new error types here as the taxonomy evolves.
    # Examples for future expansion:
    # EQUATION = "equation"       — altered mathematical formula or derivation step
    # REFERENCE = "reference"     — broken cross-reference to figure/table/equation
    # OMISSION = "omission"       — deleted a key caveat or limitation statement
    # FABRICATION = "fabrication" — invented result that sounds plausible

    @classmethod
    def descriptions(cls) -> dict[str, str]:
        """Return a human-readable description for each error type."""
        return {
            cls.NUMERICAL: "Altered statistic, percentage, or table/figure value",
            cls.CITATION: "Misattributed claim, fabricated/altered citation, or wrong year",
            cls.LOGICAL: "Non-sequitur, reversed causality, or invalid inference",
            cls.METHODOLOGICAL: "Swapped experimental setup detail (dataset split, hyperparameter, etc.)",
            cls.TERMINOLOGY: "Technical term swapped with a plausible-but-wrong alternative",
            cls.NEGATION: "Inserted/removed negation flipping a claim's meaning",
        }

    @classmethod
    def prompt_list(cls) -> str:
        """Return a formatted list of error types for inclusion in prompts."""
        lines = []
        for error_type in cls:
            desc = cls.descriptions().get(error_type.value, "")
            lines.append(f"  - {error_type.value}: {desc}")
        return "\n".join(lines)
