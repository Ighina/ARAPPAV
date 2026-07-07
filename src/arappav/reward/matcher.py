"""Matcher: aligns Verifier claims against Perturber ground-truth errors.

The core matching logic: for each ground-truth error, find the best-matching
Verifier claim (if any). For each Verifier claim, determine whether it matches
a real error (true positive) or is a hallucination (false positive).

Matching uses:
1. **Span overlap** (char-level IoU of quoted_text vs injected_text).
2. **Optional semantic match** via an LLM judge (for fuzzy/near-miss matching).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from arappav.errors.schema import InjectedError, VerifierClaim

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of matching Verifier claims to ground-truth errors."""

    # Per-error matching
    matched_claim_indices: dict[str, int | None] = field(default_factory=dict)
    """error_id → verifier_claim_index (or None if unmatched)"""

    # Per-claim classification
    claim_is_true_positive: list[bool] = field(default_factory=list)
    """For each claim, whether it matched a real error."""

    # Counts
    num_matched_errors: int = 0
    num_unmatched_errors: int = 0
    num_true_positives: int = 0
    num_false_positives: int = 0

    # Detailed match info (for logging/debugging)
    match_details: list[dict] = field(default_factory=list)
    """Per-error dicts with overlap scores and best-match info."""


def _compute_char_span_overlap(
    text_a: str, text_b: str, full_text: str
) -> float:
    """Compute character-span overlap between two substrings within a full text.

    Uses the first occurrence of each substring in ``full_text`` to determine
    character spans, then computes **intersection-over-union (IoU)** of the spans.

    Args:
        text_a: First substring (e.g., injected_text from ground truth).
        text_b: Second substring (e.g., quoted_text from Verifier claim).
        full_text: The full perturbed text.

    Returns:
        IoU overlap in [0, 1]. 0 = no overlap; 1 = perfect span match.
    """
    def _find_span(sub: str) -> tuple[int, int] | None:
        idx = full_text.find(sub)
        if idx == -1:
            return None
        return (idx, idx + len(sub))

    span_a = _find_span(text_a)
    span_b = _find_span(text_b)

    if span_a is None or span_b is None:
        return 0.0

    a_start, a_end = span_a
    b_start, b_end = span_b

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_len = overlap_end - overlap_start
    union_len = (a_end - a_start) + (b_end - b_start) - overlap_len

    if union_len <= 0:
        return 0.0

    return overlap_len / union_len


def match_claims_to_errors(
    ground_truth: list[InjectedError],
    verifier_claims: list[VerifierClaim],
    perturbed_text: str,
    span_overlap_threshold: float = 0.5,
    use_semantic_match: bool = False,
    semantic_match_fn: callable | None = None,
) -> MatchResult:
    """Match Verifier claims to ground-truth errors.

    Greedy matching: each ground-truth error is assigned the best-overlapping
    (unmatched) Verifier claim above the span-overlap threshold. A claim can
    match at most one error.

    Args:
        ground_truth: List of actual injected errors from the Perturber.
        verifier_claims: List of claims from the Verifier.
        perturbed_text: The full perturbed text (used for span computation).
        span_overlap_threshold: Minimum IoU for a claim to be considered a match.
        use_semantic_match: If True, also use a semantic matching function.
        semantic_match_fn: Callable ``(error: InjectedError, claim: VerifierClaim, text: str) -> float``
            returning a similarity score in [0, 1]. Used as a fallback or
            complement to span overlap.

    Returns:
        A ``MatchResult`` with full matching information.
    """
    result = MatchResult()
    num_errors = len(ground_truth)
    num_claims = len(verifier_claims)

    result.claim_is_true_positive = [False] * num_claims

    if num_errors == 0:
        result.num_false_positives = num_claims
        return result

    # Build an overlap matrix: errors × claims
    overlap_matrix = []
    for error in ground_truth:
        row = []
        for claim in verifier_claims:
            iou = _compute_char_span_overlap(
                error.injected_text, claim.quoted_text, perturbed_text
            )
            # If span overlap is low and semantic matching is enabled, try that
            if iou < span_overlap_threshold and use_semantic_match and semantic_match_fn is not None:
                semantic_score = semantic_match_fn(error, claim, perturbed_text)
                # Use max of span and semantic (with semantic typically being more generous)
                iou = max(iou, semantic_score * 0.8)  # slightly discount pure semantic matches
            row.append(iou)
        overlap_matrix.append(row)

    # Greedy assignment: for each error (in order), pick the best unmatched claim.
    used_claims: set[int] = set()

    for err_idx, row in enumerate(overlap_matrix):
        error_id = ground_truth[err_idx].error_id
        best_claim_idx: int | None = None
        best_score = 0.0

        # Consider claims in descending order of overlap
        sorted_indices = sorted(range(num_claims), key=lambda i: row[i], reverse=True)
        for claim_idx in sorted_indices:
            if claim_idx in used_claims:
                continue
            if row[claim_idx] >= span_overlap_threshold:
                best_claim_idx = claim_idx
                best_score = row[claim_idx]
                break

        result.matched_claim_indices[error_id] = best_claim_idx
        result.match_details.append(
            {
                "error_id": error_id,
                "best_claim_idx": best_claim_idx,
                "best_overlap": best_score,
                "error_type": ground_truth[err_idx].error_type.value,
                "all_overlaps": {
                    verifier_claims[i].quoted_text[:80]: round(row[i], 3)
                    for i in range(num_claims)
                },
            }
        )

        if best_claim_idx is not None:
            used_claims.add(best_claim_idx)
            result.claim_is_true_positive[best_claim_idx] = True
            result.num_matched_errors += 1
        else:
            result.num_unmatched_errors += 1

    result.num_true_positives = result.num_matched_errors
    result.num_false_positives = num_claims - result.num_true_positives

    return result


def exact_text_match(
    ground_truth: list[InjectedError],
    verifier_claims: list[VerifierClaim],
) -> MatchResult:
    """Simpler exact-match matcher (no span overlap computation).

    A verifier claim matches a ground-truth error if ``quoted_text`` is a
    substring of ``injected_text`` or vice versa. Faster but less robust
    than span-overlap matching.

    Args:
        ground_truth: Actual injected errors.
        verifier_claims: Verifier claims.

    Returns:
        MatchResult.
    """
    result = MatchResult()
    num_errors = len(ground_truth)
    num_claims = len(verifier_claims)

    result.claim_is_true_positive = [False] * num_claims
    used_claims: set[int] = set()

    for error in ground_truth:
        best_idx: int | None = None
        for i, claim in enumerate(verifier_claims):
            if i in used_claims:
                continue
            if error.injected_text in claim.quoted_text or claim.quoted_text in error.injected_text:
                best_idx = i
                break

        result.matched_claim_indices[error.error_id] = best_idx
        if best_idx is not None:
            used_claims.add(best_idx)
            result.claim_is_true_positive[best_idx] = True
            result.num_matched_errors += 1
        else:
            result.num_unmatched_errors += 1

    result.num_true_positives = result.num_matched_errors
    result.num_false_positives = num_claims - result.num_true_positives
    return result
