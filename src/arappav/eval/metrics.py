"""Evaluation metrics for ARAPPAV.

Tracks:
- Verifier recall, precision, F1 (overall and per error-category).
- Perturber "trick rate" = 1 − verifier recall.
- Format-compliance rate for both models.
- Per-category breakdown.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from arappav.errors.taxonomy import ErrorType
from arappav.reward.reward_fns import compute_rewards


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics across episodes."""

    num_episodes: int = 0
    num_format_valid_perturber: int = 0
    num_format_valid_verifier: int = 0

    # Verifier metrics
    verifier_recall_mean: float = 0.0
    verifier_precision_mean: float = 0.0
    verifier_f1_mean: float = 0.0

    # Perturber metrics
    perturber_reward_mean: float = 0.0
    perturber_trick_rate_mean: float = 0.0  # 1 - recall

    # Per-category breakdown
    per_category: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Raw per-episode metrics (for detailed analysis)
    per_episode: list[EpisodeMetrics] = field(default_factory=list)


@dataclass
class CategoryMetrics:
    """Metrics for a single error category."""

    total_errors: int = 0
    matched_errors: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0


@dataclass
class EpisodeMetrics:
    """Per-episode evaluation metrics."""

    paper_id: str
    chunk_id: str
    k: int
    perturber_format_valid: bool
    verifier_format_valid: bool
    verifier_recall: float
    verifier_precision: float
    verifier_f1: float
    perturber_reward: float
    num_errors: int
    num_claims: int
    num_matched: int
    per_category_recall: dict[str, float] = field(default_factory=dict)


def compute_episode_metrics(
    ground_truth: list,
    verifier_claims: list,
    perturbed_text: str,
    k: int,
    paper_id: str,
    chunk_id: str,
    perturber_format_valid: bool,
    verifier_format_valid: bool,
    reward_config: dict | None = None,
) -> EpisodeMetrics:
    """Compute metrics for a single evaluation episode.

    Args:
        ground_truth: List of InjectedError objects.
        verifier_claims: List of VerifierClaim objects.
        perturbed_text: The full perturbed text.
        k: Expected number of errors.
        paper_id: Paper identifier.
        chunk_id: Chunk identifier.
        perturber_format_valid: Whether Perturber output was valid.
        verifier_format_valid: Whether Verifier output was valid.
        reward_config: Reward configuration.

    Returns:
        EpisodeMetrics dataclass.
    """
    reward_out = compute_rewards(
        ground_truth=ground_truth,
        verifier_claims=verifier_claims,
        perturbed_text=perturbed_text,
        k=k,
        config=reward_config,
        perturber_format_valid=perturber_format_valid,
    )

    # Per-category recall
    per_cat = _compute_per_category_recall(ground_truth, verifier_claims, perturbed_text)

    return EpisodeMetrics(
        paper_id=paper_id,
        chunk_id=chunk_id,
        k=k,
        perturber_format_valid=perturber_format_valid,
        verifier_format_valid=verifier_format_valid,
        verifier_recall=reward_out.verifier_recall,
        verifier_precision=reward_out.verifier_precision,
        verifier_f1=reward_out.verifier_f_beta,
        perturber_reward=reward_out.perturber_reward,
        num_errors=len(ground_truth),
        num_claims=reward_out.num_verifier_claims,
        num_matched=reward_out.num_matched,
        per_category_recall=per_cat,
    )


def _compute_per_category_recall(
    ground_truth: list, verifier_claims: list, perturbed_text: str
) -> dict[str, float]:
    """Compute recall broken down by error type category.

    Returns:
        Dict mapping error_type string → recall in [0, 1].
    """
    from arappav.reward.matcher import match_claims_to_errors

    # Group errors by type
    by_type: dict[str, list] = defaultdict(list)
    for err in ground_truth:
        by_type[err.error_type.value].append(err)

    # Match all claims (not per-type — the matcher is global)
    match_result = match_claims_to_errors(ground_truth, verifier_claims, perturbed_text)

    # Compute per-type recall from match details
    per_type = {}
    for error_type, errors in by_type.items():
        matched = 0
        for err in errors:
            if match_result.matched_claim_indices.get(err.error_id) is not None:
                matched += 1
        per_type[error_type] = matched / max(1, len(errors))

    return per_type


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> EvalMetrics:
    """Aggregate per-episode metrics into summary statistics.

    Args:
        episodes: List of EpisodeMetrics from evaluation.

    Returns:
        Aggregated EvalMetrics.
    """
    if not episodes:
        return EvalMetrics()

    n = len(episodes)
    fmt_p = sum(1 for e in episodes if e.perturber_format_valid)
    fmt_v = sum(1 for e in episodes if e.verifier_format_valid)

    # Aggregate numeric metrics
    recall_sum = sum(e.verifier_recall for e in episodes)
    precision_sum = sum(e.verifier_precision for e in episodes)
    f1_sum = sum(e.verifier_f1 for e in episodes)
    p_reward_sum = sum(e.perturber_reward for e in episodes)
    trick_sum = sum(1.0 - e.verifier_recall for e in episodes)

    # Per-category aggregation
    cat_data: dict[str, dict] = defaultdict(lambda: {"total": 0, "matched": 0})
    for ep in episodes:
        for cat, recall in ep.per_category_recall.items():
            cat_data[cat]["total"] += 1
            cat_data[cat]["matched"] += recall  # accumulate recall values

    per_category = {}
    for cat, data in cat_data.items():
        avg_recall = data["matched"] / max(1, data["total"])
        per_category[cat] = CategoryMetrics(
            total_errors=data["total"],
            matched_errors=int(data["matched"]),
            recall=avg_recall,
            precision=0.0,  # per-category precision requires claim-type labels
            f1=0.0,
        )

    return EvalMetrics(
        num_episodes=n,
        num_format_valid_perturber=fmt_p,
        num_format_valid_verifier=fmt_v,
        verifier_recall_mean=recall_sum / n,
        verifier_precision_mean=precision_sum / n,
        verifier_f1_mean=f1_sum / n,
        perturber_reward_mean=p_reward_sum / n,
        perturber_trick_rate_mean=trick_sum / n,
        per_category=per_category,
        per_episode=episodes,
    )


def format_metrics_report(metrics: EvalMetrics) -> str:
    """Format evaluation metrics as a human-readable string.

    Args:
        metrics: Aggregated metrics.

    Returns:
        Multi-line string report.
    """
    lines = [
        "=" * 60,
        "  ARAPPAV Evaluation Report",
        "=" * 60,
        f"  Episodes:                {metrics.num_episodes}",
        f"  Perturber format rate:   {metrics.num_format_valid_perturber / max(1, metrics.num_episodes):.2%}",
        f"  Verifier format rate:    {metrics.num_format_valid_verifier / max(1, metrics.num_episodes):.2%}",
        "",
        "  --- Verifier ---",
        f"  Recall:                  {metrics.verifier_recall_mean:.4f}",
        f"  Precision:               {metrics.verifier_precision_mean:.4f}",
        f"  F1:                      {metrics.verifier_f1_mean:.4f}",
        "",
        "  --- Perturber ---",
        f"  Mean reward:             {metrics.perturber_reward_mean:.4f}",
        f"  Trick rate (1-recall):   {metrics.perturber_trick_rate_mean:.4f}",
        "",
        "  --- Per Category Recall ---",
    ]

    for cat_name, cat_metrics in sorted(metrics.per_category.items()):
        lines.append(f"  {cat_name:20s}: {cat_metrics.recall:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)
