"""Math-specific error taxonomy for Perturber injection.

Derived from the error categories documented in:
  Rittle-Johnson et al., "Detecting Math Misconceptions: An AI Benchmark Dataset"
  (AIME-Con 2025, Proceedings pages 20-24)

Each error type corresponds to a class of student misconception or
instructionally relevant error observed in middle-school mathematics.
"""

from enum import Enum


class MathErrorType(str, Enum):
    """Taxonomy of mathematical errors the Perturber can inject into solutions."""

    # --- Fraction / Rational Number Errors ---
    WHOLE_NUMBER_BIAS = "whole_number_bias"
    """Treating numerator and denominator of a fraction as two independent whole
    numbers rather than a single value (e.g., believing 2/3 and 3/4 are close
    because both have small numbers)."""

    ADDING_ACROSS = "adding_across"
    """Adding numerators together and denominators together without finding a
    common denominator first."""

    WRONG_OPERATION = "wrong_operation"
    """Applying an incorrect arithmetic or algebraic operation (e.g., dividing
    when multiplication is required, or adding instead of multiplying)."""

    OPERAND_SWAP = "operand_swap"
    """Swapping the positions of operands — e.g., swapping dividend and divisor
    in a division problem, or numerator and denominator in a fraction operation."""

    INCOMPLETE_SOLUTION = "incomplete_solution"
    """Stopping the solution before all required steps are complete — e.g.,
    failing to simplify a final fraction, or computing a unit fraction without
    the final answer."""

    DENOMINATOR_ONLY = "denominator_only"
    """Operating on the denominator while leaving the numerator unchanged
    (or vice versa) during fraction arithmetic."""

    DUPLICATION_ERROR = "duplication_error"
    """Incorrectly duplicating an operation across both numerator and denominator
    when it should apply to only one (e.g., multiplying both by a whole number
    instead of just the numerator)."""

    INVERSION_ERROR = "inversion_error"
    """Inverting the wrong operand or the wrong part of an expression —
    e.g., inverting the whole-number multiplier instead of the fraction divisor."""

    WRONG_FRACTION = "wrong_fraction"
    """Computing a fraction for the wrong target quantity or reference group."""

    # --- Decimal Errors ---
    DECIMAL_MAGNITUDE = "decimal_magnitude"
    """Misunderstanding the magnitude of decimal numbers — e.g., believing a
    longer decimal representation means a larger number, or that fewer digits
    after the decimal point means a larger number."""

    IGNORES_ZEROES = "ignores_zeroes"
    """Treating zero digits as not contributing to place value — e.g., believing
    0.5 = 0.05 or that 1.02 < 1.1 because it has a zero."""

    # --- Variable / Algebraic Errors ---
    VARIABLE_MISCONCEPTION = "variable_misconception"
    """Misunderstanding what a variable represents — e.g., treating a variable
    as a missing digit rather than a quantity that can take any value."""

    ADDITIVE_THINKING = "additive_thinking"
    """Using additive reasoning where multiplicative reasoning is required —
    e.g., finding the difference between numbers instead of a ratio or product."""

    WRONG_SEQUENCE_TERM = "wrong_sequence_term"
    """Computing the wrong term in a sequence — e.g., calculating the n+1 term
    when the n+2 term was requested."""

    FIRST_TERM_AS_COEFFICIENT = "first_term_as_coefficient"
    """Using the first output term as the coefficient of the functional rule
    rather than deriving the correct slope/coefficient."""

    # --- Negative Number Errors ---
    NEGATIVE_NUMBER_ERROR = "negative_number_error"
    """Misapplying negative number rules — e.g., incorrectly applying 'two
    negatives make a positive', or ignoring/tacking on negative signs."""

    TACKING_SIGNS = "tacking_signs"
    """Ignoring negative signs during computation and adding them back at the
    end as an afterthought rather than carrying them through."""

    # --- Proportional Reasoning Errors ---
    PROPORTIONAL_REASONING_ERROR = "proportional_reasoning_error"
    """Reversing or misapplying proportional relationships — e.g., multiplying
    instead of dividing when scaling, or applying the wrong scale factor."""

    INVERSE_OPERATION_ERROR = "inverse_operation_error"
    """Applying the wrong inverse operation — e.g., multiplying instead of
    dividing by the same factor, or thinking the inverse of multiplication
    by 2 is multiplication by 1/4."""

    # --- Probability / Statistics Errors ---
    PROBABILITY_SCALE = "probability_scale"
    """Misunderstanding that probability values must lie between 0 and 1
    (or 0% and 100%)."""

    PROBABILITY_CERTAINTY = "probability_certainty"
    """Believing that an event with non-1 probability is certain to occur,
    or misunderstanding randomness."""

    BASE_RATE_FALLACY = "base_rate_fallacy"
    """Ignoring or misapplying base rates when reasoning about conditional
    probabilities or proportional comparisons."""

    # --- Geometry Errors ---
    GEOMETRY_DEFINITION = "geometry_definition"
    """Using an incorrect definition for a geometric shape or property —
    e.g., believing a polygon must have exactly 5 or 6 sides."""

    ANGLE_MISCONCEPTION = "angle_misconception"
    """Misapplying angle formulas or relationships — e.g., dividing total
    interior angle sum by one interior angle rather than using the correct
    polygon angle formula."""

    # --- General / Cross-cutting ---
    IRRELEVANT_FEATURE = "irrelevant_feature"
    """Basing reasoning on a feature of the problem that is irrelevant to the
    solution, while ignoring the relevant structure."""

    UNKNOWABLE = "unknowable"
    """Incorrectly concluding that there is not enough information to solve
    the problem when sufficient information is provided."""

    @classmethod
    def descriptions(cls) -> dict[str, str]:
        """Return a human-readable description for each math error type."""
        return {
            cls.WHOLE_NUMBER_BIAS: "Treating fraction parts as independent whole numbers",
            cls.ADDING_ACROSS: "Adding numerators and denominators without common denominator",
            cls.WRONG_OPERATION: "Using incorrect arithmetic operation (e.g., + instead of ×)",
            cls.OPERAND_SWAP: "Swapping dividend/divisor or numerator/denominator",
            cls.INCOMPLETE_SOLUTION: "Stopping before all solution steps are complete",
            cls.DENOMINATOR_ONLY: "Changing only denominator (or only numerator) incorrectly",
            cls.DUPLICATION_ERROR: "Incorrectly duplicating operation across both parts",
            cls.INVERSION_ERROR: "Inverting wrong operand or wrong part of expression",
            cls.WRONG_FRACTION: "Computing fraction for wrong target or reference group",
            cls.DECIMAL_MAGNITUDE: "Misunderstanding decimal magnitude (longer ≠ larger)",
            cls.IGNORES_ZEROES: "Ignoring zero digits' place-value contribution",
            cls.VARIABLE_MISCONCEPTION: "Misunderstanding what a variable represents",
            cls.ADDITIVE_THINKING: "Using additive reasoning where multiplicative is needed",
            cls.WRONG_SEQUENCE_TERM: "Computing wrong term in a sequence",
            cls.FIRST_TERM_AS_COEFFICIENT: "Using first output as coefficient directly",
            cls.NEGATIVE_NUMBER_ERROR: "Misapplying negative number arithmetic rules",
            cls.TACKING_SIGNS: "Ignoring signs during computation, re-adding at end",
            cls.PROPORTIONAL_REASONING_ERROR: "Reversing or misapplying proportional relationships",
            cls.INVERSE_OPERATION_ERROR: "Applying wrong inverse operation",
            cls.PROBABILITY_SCALE: "Thinking probability can exceed 1 or be negative",
            cls.PROBABILITY_CERTAINTY: "Believing non-1 probability means certain event",
            cls.BASE_RATE_FALLACY: "Ignoring base rates in conditional reasoning",
            cls.GEOMETRY_DEFINITION: "Using incorrect definition of shape/property",
            cls.ANGLE_MISCONCEPTION: "Misapplying angle formulas or relationships",
            cls.IRRELEVANT_FEATURE: "Reasoning from irrelevant problem features",
            cls.UNKNOWABLE: "Incorrectly claiming insufficient information to solve",
        }

    @classmethod
    def prompt_list(cls) -> str:
        """Return a formatted list of math error types for inclusion in prompts."""
        lines = []
        for error_type in cls:
            desc = cls.descriptions().get(error_type.value, "")
            lines.append(f"  - {error_type.value}: {desc}")
        return "\n".join(lines)

    @classmethod
    def topic_groups(cls) -> dict[str, list["MathErrorType"]]:
        """Return error types grouped by math topic for contextual filtering."""
        return {
            "fractions": [
                cls.WHOLE_NUMBER_BIAS, cls.ADDING_ACROSS, cls.DENOMINATOR_ONLY,
                cls.DUPLICATION_ERROR, cls.INVERSION_ERROR, cls.WRONG_FRACTION,
                cls.INCOMPLETE_SOLUTION,
            ],
            "decimals": [
                cls.DECIMAL_MAGNITUDE, cls.IGNORES_ZEROES,
            ],
            "algebra": [
                cls.VARIABLE_MISCONCEPTION, cls.ADDITIVE_THINKING,
                cls.WRONG_OPERATION, cls.OPERAND_SWAP,
                cls.WRONG_SEQUENCE_TERM, cls.FIRST_TERM_AS_COEFFICIENT,
            ],
            "negatives": [
                cls.NEGATIVE_NUMBER_ERROR, cls.TACKING_SIGNS,
            ],
            "proportions": [
                cls.PROPORTIONAL_REASONING_ERROR, cls.INVERSE_OPERATION_ERROR,
                cls.BASE_RATE_FALLACY,
            ],
            "probability": [
                cls.PROBABILITY_SCALE, cls.PROBABILITY_CERTAINTY,
            ],
            "geometry": [
                cls.GEOMETRY_DEFINITION, cls.ANGLE_MISCONCEPTION,
            ],
            "general": [
                cls.IRRELEVANT_FEATURE, cls.UNKNOWABLE,
            ],
        }
