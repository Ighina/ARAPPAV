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
import re
from dataclasses import dataclass, field
from typing import Any

from arappav.errors.schema import InjectedError, PerturberOutput, VerifierClaim, VerifierOutput
from arappav.reward.matcher import MatchResult, error_present_in_text, match_claims_to_errors

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
    k_effective: int = 0  # errors actually present in the perturbed text

    # Penalties
    format_penalty_applied: bool = False
    format_violation_reason: str | None = None
    duplicate_penalty: float = 0.0
    spam_penalty: float = 0.0
    phantom_penalty: float = 0.0
    repetition_penalty: float = 0.0

    # For logging / debugging
    perturber_base_reward: float = 0.0  # before penalties
    verifier_base_reward: float = 0.0
    match_details: list[dict] = field(default_factory=list)
    """Per-error match info from the matcher (best claim, overlap scores) —
    lets rollout analyses distinguish matcher failures from verifier failures."""


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
    verifier_raw_output: str | None = None,
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
        verifier_raw_output: Raw Verifier output text (for repetition detection).

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
            phantom_penalty=0.0,
        )

    # --- Effective k: count errors actually present in the text ---------------
    # Uses normalized containment so that formatting drift (LaTeX escaping,
    # whitespace) between the error record and the text doesn't count a real
    # error as missing.
    k_actual = len(ground_truth)
    k_effective = sum(
        1 for e in ground_truth
        if error_present_in_text(getattr(e, "injected_text", ""), perturbed_text)
    )
    # Use effective k for recall denominator (but never zero)
    k_for_recall = max(1, k_effective)

    # --- Match claims to errors ---
    match_result = match_claims_to_errors(
        ground_truth=ground_truth,
        verifier_claims=verifier_claims,
        perturbed_text=perturbed_text,
        span_overlap_threshold=config.get("span_overlap_threshold", 0.5),
        use_semantic_match=config.get("use_semantic_match", False),
    )

    # --- Verifier metrics (using effective k) ---
    recall = match_result.num_matched_errors / k_for_recall
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
    anti_spam_cfg = config.get("anti_spam", {})
    if anti_spam_cfg.get("enabled", True):
        max_ratio = anti_spam_cfg.get("max_claims_ratio", 3.0)
        penalty_per = anti_spam_cfg.get("penalty_per_excess", -0.5)
        # Floor at 1: if every injection was dropped (k_effective == 0) the
        # Verifier can't know that, and shouldn't be penalized for any claim.
        max_allowed = int(max_ratio * max(1, k_effective))
        if len(verifier_claims) > max_allowed:
            excess = len(verifier_claims) - max_allowed
            spam_penalty = excess * penalty_per

    # --- Anti-repetition penalty (Verifier) ---
    # Detect when the Verifier collapses into generating the same JSON block
    # repeatedly — a known failure mode for smaller LLMs.
    # The penalty is milder than a full miss (default -0.5) so that a
    # verifier that detects everything but repeats still ranks between a
    # clean perfect detection (1.0) and a total miss (0.0) in GRPO groups.
    repetition_penalty = 0.0
    anti_repetition_cfg = config.get("anti_repetition", {})
    if anti_repetition_cfg.get("enabled", True) and verifier_raw_output:
        json_block_count = _count_json_blocks(verifier_raw_output)
        max_blocks = anti_repetition_cfg.get("max_json_blocks", 5)
        repetition_penalty_per = anti_repetition_cfg.get("penalty", -0.5)
        if json_block_count > max_blocks:
            repetition_penalty = repetition_penalty_per
            logger.warning(
                "Repetition penalty applied: Verifier generated %d JSON blocks "
                "(max allowed: %d). Penalty: %.1f",
                json_block_count, max_blocks, repetition_penalty,
            )

    # --- Anti-duplicate penalty (Perturber) ---
    dup_penalty = 0.0
    anti_duplicate_cfg = config.get("anti_duplicate", {})
    if anti_duplicate_cfg.get("enabled", True) and historical_perturbations:
        dup_penalty = _compute_duplicate_penalty(
            ground_truth, historical_perturbations, anti_duplicate_cfg
        )

    # --- Anti-phantom penalty (Perturber) ---
    phantom_penalty = 0.0
    anti_phantom_cfg = config.get("anti_phantom", {})
    if anti_phantom_cfg.get("enabled", True) and k_actual > 0:
        phantom_count = sum(
            1 for e in ground_truth
            if getattr(e, "injected_text", "").strip() == getattr(e, "original_text", "").strip()
        )
        phantom_ratio = phantom_count / k_actual
        phantom_threshold = anti_phantom_cfg.get("max_phantom_ratio", 0.0)
        phantom_penalty_per = anti_phantom_cfg.get("penalty", -10.0)

        if phantom_ratio > phantom_threshold:
            phantom_penalty = phantom_penalty_per
            logger.warning(
                "Phantom error penalty applied: %d/%d errors (%.0f%%) are phantom "
                "(injected_text == original_text). Penalty: %.1f",
                phantom_count, k_actual, phantom_ratio * 100, phantom_penalty,
            )

    # --- Penalise perturber for missing errors (overlapping / not injected) ---
    # Errors declared but not present in the text count against the Perturber.
    missing_error_penalty = 0.0
    anti_missing_cfg = config.get("anti_missing", {})
    if anti_missing_cfg.get("enabled", True) and k_effective < k_actual:
        missing_count = k_actual - k_effective
        missing_penalty_per = anti_missing_cfg.get("penalty_per_missing", -0.5)
        missing_error_penalty = missing_count * missing_penalty_per
        if missing_error_penalty != 0:
            logger.warning(
                "Missing error penalty: %d/%d errors not found in perturbed text. "
                "Penalty: %.1f",
                missing_count, k_actual, missing_error_penalty,
            )

    # --- Final rewards ---
    r_P = r_P_base + dup_penalty + phantom_penalty + missing_error_penalty
    r_V = r_V_base + spam_penalty + repetition_penalty

    return RewardOutput(
        perturber_reward=r_P,
        verifier_reward=r_V,
        verifier_recall=recall,
        verifier_precision=precision,
        verifier_f_beta=f_beta,
        k=k_actual,
        k_effective=k_effective,
        num_verifier_claims=len(verifier_claims),
        num_matched=match_result.num_matched_errors,
        duplicate_penalty=dup_penalty,
        spam_penalty=spam_penalty,
        phantom_penalty=phantom_penalty,
        repetition_penalty=repetition_penalty,
        perturber_base_reward=r_P_base,
        verifier_base_reward=r_V_base,
        match_details=match_result.match_details,
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


def _count_json_blocks(text: str) -> int:
    """Count the number of ``{"claims": [...]}`` blocks in a Verifier output.

    A high count (>5) indicates the model collapsed into repetition.
    """
    return len(re.findall(r'\{\s*"claims"\s*:\s*\[', text))


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
    verifier_raw_output: str | None = None,
) -> RewardOutput:
    """High-level convenience: compute rewards from structured outputs.

    Args:
        perturber_output: Validated Perturber structured output.
        verifier_output: Validated Verifier structured output.
        k: Expected number of errors.
        config: Reward configuration dict.
        historical_perturbations: Previous perturbations for anti-duplicate check.
        verifier_raw_output: Raw Verifier text, for repetition detection.

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
        verifier_raw_output=verifier_raw_output,
    )


def compute_math_episode_rewards(
    perturber_output: Any,  # MathPerturberOutput
    verifier_output: Any,   # MathVerifierOutput
    k: int,
    config: dict | None = None,
    historical_perturbations: list | None = None,
    verifier_raw_output: str | None = None,
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
        verifier_raw_output=verifier_raw_output,
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
        "anti_phantom": {"enabled": True, "max_phantom_ratio": 0.0, "penalty": -10.0},
        "anti_repetition": {"enabled": True, "max_json_blocks": 5, "penalty": -0.5},
        "anti_missing": {"enabled": True, "penalty_per_missing": -0.5},
    }
