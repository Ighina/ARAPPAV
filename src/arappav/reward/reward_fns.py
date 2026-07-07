"""Reward functions for Perturber and Verifier.

All reward functions are **pure functions** of ``(ground_truth, verifier_claims, config)``
with no hidden state — for testability and reproducibility.

Reward design:
- Perturber is rewarded when Verifier fails to catch injected errors.
- Verifier is rewarded for correct detection, penalized for false positives.
- Format penalties dominate task rewards to enforce instruction-following.
- Degenerate-strategy guards prevent reward hacking.

Supports both paper mode (InjectedError/VerifierClaim) and math mode
(MathInjectedError/MathVerifierClaim) via duck-typing on ``injected_text``
and ``quoted_text`` attributes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from arappav.errors.schema import InjectedError, PerturberOutput, VerifierClaim, VerifierOutput
from arappav.reward.matcher import MatchResult, match_claims_to_errors

logger = logging.getLogger(__name__)


@dataclass
class RewardOutput:
    """Result of reward computation for one episode."""

    perturber_reward: float
    verifier_reward: float

    # Decomposed metrics
    verifier_recall: float
    verifier_precision: float
    verifier_f_beta: float

    # Counts
    k: int
    num_verifier_claims: int
    num_matched: int

    # Penalties
    format_penalty_applied: bool = False
    format_violation_reason: str | None = None
    duplicate_penalty: float = 0.0
    spam_penalty: float = 0.0

    # For logging / debugging
    perturber_base_reward: float = 0.0  # before penalties
    verifier_base_reward: float = 0.0


# ---------------------------------------------------------------------------
# Core reward computation
# ---------------------------------------------------------------------------


def compute_rewards(
    ground_truth: list[InjectedError],
    verifier_claims: list[VerifierClaim],
    perturbed_text: str,
    k: int,
    config: dict | None = None,
    perturber_format_valid: bool = True,
    perturber_format_reason: str | None = None,
    historical_perturbations: list[InjectedError] | None = None,
) -> RewardOutput:
    """Compute rewards for one self-play episode.

    Args:
        ground_truth: Actual injected errors from the Perturber.
        verifier_claims: Claims made by the Verifier.
        perturbed_text: The full perturbed text (for span matching).
        k: Expected number of errors.
        config: Reward configuration dict (from ``reward.yaml``). Uses defaults if None.
        perturber_format_valid: Whether the Perturber produced valid output.
        perturber_format_reason: If invalid, why.
        historical_perturbations: Previous errors from this Perturber (for anti-duplicate).

    Returns:
        ``RewardOutput`` with all reward components.
    """
    if config is None:
        config = _default_config()

    # --- Format penalty ---
    if not perturber_format_valid:
        fmt_penalty = config.get("format_penalty", -10.0)
        return RewardOutput(
            perturber_reward=fmt_penalty,
            verifier_reward=0.0,
            verifier_recall=0.0,
            verifier_precision=0.0,
            verifier_f_beta=0.0,
            k=k,
            num_verifier_claims=len(verifier_claims),
            num_matched=0,
            format_penalty_applied=True,
            format_violation_reason=perturber_format_reason,
        )

    # --- Match claims to errors ---
    match_result = match_claims_to_errors(
        ground_truth=ground_truth,
        verifier_claims=verifier_claims,
        perturbed_text=perturbed_text,
        span_overlap_threshold=config.get("span_overlap_threshold", 0.5),
        use_semantic_match=config.get("use_semantic_match", False),
    )

    # --- Verifier metrics ---
    k_actual = len(ground_truth)
    recall = match_result.num_matched_errors / max(1, k_actual)
    precision = match_result.num_true_positives / max(1, len(verifier_claims))
    f_beta = _compute_f_beta(precision, recall, config.get("precision_recall_beta", 1.0))

    # --- Base rewards ---
    verifier_formula = config.get("verifier_reward_formula", "f_beta")
    if verifier_formula == "f_beta":
        r_V_base = f_beta
    elif verifier_formula == "recall_only":
        r_V_base = recall
    else:
        r_V_base = f_beta

    perturber_formula = config.get("perturber_reward_formula", "one_minus_recall")
    if perturber_formula == "one_minus_recall":
        r_P_base = 1.0 - recall
    else:
        r_P_base = 1.0 - recall  # fallback

    # --- Anti-spam penalty (Verifier) ---
    spam_penalty = 0.0
    if config.get("anti_spam", {}).get("enabled", True):
        max_ratio = config["anti_spam"].get("max_claims_ratio", 3.0)
        penalty_per = config["anti_spam"].get("penalty_per_excess", -0.5)
        max_allowed = int(max_ratio * k_actual)
        if len(verifier_claims) > max_allowed:
            excess = len(verifier_claims) - max_allowed
            spam_penalty = excess * penalty_per

    # --- Anti-duplicate penalty (Perturber) ---
    dup_penalty = 0.0
    if config.get("anti_duplicate", {}).get("enabled", True) and historical_perturbations:
        dup_penalty = _compute_duplicate_penalty(
            ground_truth, historical_perturbations, config["anti_duplicate"]
        )

    # --- Final rewards ---
    r_P = r_P_base + dup_penalty
    r_V = r_V_base + spam_penalty

    return RewardOutput(
        perturber_reward=r_P,
        verifier_reward=r_V,
        verifier_recall=recall,
        verifier_precision=precision,
        verifier_f_beta=f_beta,
        k=k_actual,
        num_verifier_claims=len(verifier_claims),
        num_matched=match_result.num_matched_errors,
        duplicate_penalty=dup_penalty,
        spam_penalty=spam_penalty,
        perturber_base_reward=r_P_base,
        verifier_base_reward=r_V_base,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_f_beta(precision: float, recall: float, beta: float) -> float:
    """Compute F-beta score.

    Args:
        precision: Precision in [0, 1].
        recall: Recall in [0, 1].
        beta: Controls precision-recall tradeoff (beta > 1 weights recall more).

    Returns:
        F-beta score in [0, 1].
    """
    if precision <= 0 and recall <= 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def _compute_duplicate_penalty(
    errors: list[InjectedError],
    history: list[InjectedError],
    config: dict,
) -> float:
    """Check current errors against historical perturbations for near-duplicates.

    Uses simple text-based Jaccard similarity on injected_text as a proxy for
    embedding similarity. In production, replace with actual embedding cosine-sim.

    Args:
        errors: Current round's errors.
        history: Errors from previous rounds.
        config: Anti-duplicate config section.

    Returns:
        Total duplicate penalty (≤ 0).
    """
    threshold = config.get("similarity_threshold", 0.85)
    penalty_per = config.get("penalty", -5.0)

    total_penalty = 0.0
    for error in errors:
        for hist_error in history:
            sim = _jaccard_similarity(error.injected_text, hist_error.injected_text)
            if sim >= threshold:
                total_penalty += penalty_per
                break  # one match is enough

    return total_penalty


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two strings.

    Simple proxy for embedding cosine similarity — avoids needing a sentence
    transformer in the reward module.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard similarity in [0, 1].
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Convenience: episode-level reward from structured outputs
# ---------------------------------------------------------------------------


def compute_episode_rewards(
    perturber_output: PerturberOutput,
    verifier_output: VerifierOutput,
    k: int,
    config: dict | None = None,
    historical_perturbations: list[InjectedError] | None = None,
) -> RewardOutput:
    """High-level convenience: compute rewards from structured outputs.

    Args:
        perturber_output: Validated Perturber structured output.
        verifier_output: Validated Verifier structured output.
        k: Expected number of errors.
        config: Reward configuration dict.
        historical_perturbations: Previous perturbations for anti-duplicate check.

    Returns:
        ``RewardOutput``.
    """
    return compute_rewards(
        ground_truth=perturber_output.errors,
        verifier_claims=verifier_output.claims,
        perturbed_text=perturber_output.perturbed_text,
        k=k,
        config=config,
        perturber_format_valid=True,
        historical_perturbations=historical_perturbations,
    )


def compute_math_episode_rewards(
    perturber_output: Any,  # MathPerturberOutput
    verifier_output: Any,   # MathVerifierOutput
    k: int,
    config: dict | None = None,
    historical_perturbations: list | None = None,
) -> RewardOutput:
    """Compute rewards for a math-mode self-play episode.

    Works via duck-typing: ``MathInjectedError`` has ``injected_text`` and
    ``MathVerifierClaim`` has ``quoted_text``, matching the interface expected
    by ``compute_rewards``.

    Args:
        perturber_output: Validated ``MathPerturberOutput``.
        verifier_output: Validated ``MathVerifierOutput``.
        k: Expected number of errors.
        config: Reward configuration dict.
        historical_perturbations: Previous perturbations for anti-duplicate.

    Returns:
        ``RewardOutput``.
    """
    return compute_rewards(
        ground_truth=perturber_output.errors,
        verifier_claims=verifier_output.claims,
        perturbed_text=perturber_output.perturbed_solution,
        k=k,
        config=config,
        perturber_format_valid=True,
        historical_perturbations=historical_perturbations,
    )


def _default_config() -> dict:
    """Return sensible default reward config (used when no config is provided)."""
    return {
        "format_penalty": -10.0,
        "span_overlap_threshold": 0.5,
        "precision_recall_beta": 1.0,
        "verifier_reward_formula": "f_beta",
        "perturber_reward_formula": "one_minus_recall",
        "use_semantic_match": False,
        "anti_duplicate": {"enabled": True, "similarity_threshold": 0.85, "penalty": -5.0},
        "anti_spam": {"enabled": True, "max_claims_ratio": 3.0, "penalty_per_excess": -0.5},
    }
