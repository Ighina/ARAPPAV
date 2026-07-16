"""Matcher: aligns Verifier claims against Perturber ground-truth errors.

The core matching logic: for each ground-truth error, find the best-matching
Verifier claim (if any). For each Verifier claim, determine whether it matches
a real error (true positive) or is a hallucination (false positive).

Matching uses (in order of strength):
1. **Span overlap** (char-level IoU of quoted_text vs injected_text).
2. **Diff-based change coverage** — the claim quotes the region the Perturber
   actually changed (per the original_text→injected_text diff).
3. **Substring containment** — quoted_text is a meaningful fragment of
   injected_text (or vice versa), scaled by length ratio.
4. **Optional semantic match** via an LLM judge (for fuzzy/near-miss matching).
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

from arappav.errors.schema import InjectedError, VerifierClaim

logger = logging.getLogger(__name__)

#: Score assigned when a claim demonstrably covers the changed region of an
#: error (diff-based match). Stronger than a plain substring match, weaker
#: than a perfect span IoU, so greedy assignment prefers exact spans first.
CHANGE_COVERAGE_SCORE = 0.9


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


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy span matching.

    Handles common formatting mismatches between injected/claimed text and the
    actual perturbed text, especially LaTeX alignment characters.

    Transformations:
    - Repair control characters produced by JSON-valid escapes eating LaTeX
      commands (``\\b`` → backspace in ``\\boxed``, ``\\f`` → form feed in
      ``\\frac``).
    - Normalise LaTeX backslash escaping (``\\\\`` → ``\\``).
    - Strip LaTeX math delimiters (``$``, ``$$``) that verifiers often drop.
    - Strip LaTeX alignment markers (``&``) from align/align* environments.
    - Collapse consecutive whitespace and strip leading/trailing whitespace.
    """
    # A model quoting "\boxed{...}" with a single backslash inside JSON emits
    # the *valid* JSON escape \b, which parses to a backspace character —
    # invisible corruption that never appears in real math text, so restoring
    # the LaTeX backslash is always safe. Same for \f (\frac).
    normalized = text.replace("\x08", "\\b").replace("\x0c", "\\f")
    normalized = _normalize_latex_escapes(normalized)

    # Strip LaTeX math delimiters — verifiers often drop $ signs when quoting
    # math expressions, so "$x^2$" vs "x^2" should still match.
    normalized = normalized.replace("$$", "")
    normalized = normalized.replace("$", "")

    # Remove LaTeX alignment & characters (inside align environments these
    # are formatting, not content).
    normalized = normalized.replace("&=", "=")
    normalized = normalized.replace("& ", " ")
    # Also handle standalone & within math mode
    normalized = re.sub(r"(?<=\s)&(?=\s)", "", normalized)

    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def _normalize_latex_escapes(text: str) -> str:
    """Normalise LaTeX backslash escaping mismatches.

    When models double-escape LaTeX commands in their JSON output
    (``\\\\cdot`` in the raw JSON), the parsed Python string contains two
    literal backslashes while the reference text contains one. Collapsing
    any run of 2+ backslashes before a letter down to a single backslash
    makes both escaping levels compare equal, and is idempotent.

    Caveat: a LaTeX line break (two backslashes) immediately followed by a
    command with no separating whitespace would also be collapsed. Both sides
    of a comparison are normalised identically, so matching is unaffected.
    """
    return re.sub(r"\\{2,}(?=[a-zA-Z])", "\\\\", text)


def _changed_fragments(
    original_text: str,
    injected_text: str,
    min_len: int = 6,
    context: int = 10,
) -> list[str]:
    """Extract the fragments of ``injected_text`` that differ from ``original_text``.

    Uses a character-level diff to isolate what the Perturber actually
    changed, so matching can target the erroneous region instead of the whole
    (mostly correct) sentence the error is embedded in.

    Fragments shorter than ``min_len`` are expanded with surrounding context
    so that trivial edits (a flipped sign, a single digit) don't match
    anywhere in the text by accident.

    Args:
        original_text: The correct text before injection.
        injected_text: The erroneous replacement text.
        min_len: Minimum fragment length; shorter fragments are expanded.
        context: Characters of context to add per expansion step.

    Returns:
        List of changed fragments (possibly empty for pure deletions).
    """
    matcher = difflib.SequenceMatcher(None, original_text, injected_text, autojunk=False)
    fragments: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("replace", "insert") or j2 <= j1:
            continue
        start, end = j1, j2
        while end - start < min_len and (start > 0 or end < len(injected_text)):
            start = max(0, start - context)
            end = min(len(injected_text), end + context)
        fragments.append(injected_text[start:end])
    return fragments


def _claim_covers_change(error, claim) -> bool:
    """Check whether a claim's quoted text covers an error's changed region.

    Two conditions must hold (after normalisation):
    1. The quoted text is anchored to the error region — one of
       ``quoted_text`` / ``injected_text`` contains the other.
    2. The quoted text contains at least one fragment that the diff of
       ``original_text`` vs ``injected_text`` identifies as changed.

    This is a much stronger signal than plain substring containment: a claim
    quoting an *unchanged* clause of the injected sentence fails condition 2.
    """
    original = getattr(error, "original_text", "") or ""
    if not original:
        return False

    quoted_norm = _normalize_for_matching(claim.quoted_text)
    injected_norm = _normalize_for_matching(error.injected_text)
    if not quoted_norm or not injected_norm:
        return False
    if quoted_norm not in injected_norm and injected_norm not in quoted_norm:
        return False

    fragments = _changed_fragments(original, error.injected_text)
    return any(
        _normalize_for_matching(frag) in quoted_norm for frag in fragments
    )


def _substring_match_score(text_a: str, text_b: str) -> float:
    """Score how meaningfully one text is a substring of the other.

    Returns the length ratio ``len(shorter) / len(longer)`` in (0, 1] when
    the shorter (normalized) text is contained in the longer and is
    meaningful — at least 3 words long or at least 40% the length of the
    longer text. Returns 0.0 otherwise. The ratio lets callers break ties
    between multiple substring-matched claims (a longer quoted fragment is
    better evidence than a shorter one).

    Args:
        text_a: First substring (e.g., injected_text).
        text_b: Second substring (e.g., quoted_text).
    """
    norm_a = _normalize_for_matching(text_a)
    norm_b = _normalize_for_matching(text_b)

    shorter = norm_a if len(norm_a) <= len(norm_b) else norm_b
    longer = norm_b if len(norm_a) <= len(norm_b) else norm_a

    if not shorter or shorter not in longer:
        return 0.0

    word_count = len(shorter.split())
    length_ratio = len(shorter) / max(1, len(longer))

    if word_count >= 3 or length_ratio >= 0.4:
        return length_ratio
    return 0.0


def _is_substring_match(text_a: str, text_b: str) -> bool:
    """Check if one text is a **meaningful** substring of the other."""
    return _substring_match_score(text_a, text_b) > 0.0


def error_present_in_text(injected_text: str, text: str) -> bool:
    """Check whether an injected error is actually present in a text.

    Tries exact containment first, then falls back to normalized containment
    (LaTeX escaping, ``$`` delimiters, alignment markers, whitespace) so that
    formatting drift between the error record and the perturbed text doesn't
    count a real error as missing.
    """
    if injected_text in text:
        return True
    return _normalize_for_matching(injected_text) in _normalize_for_matching(text)


def _locate_span(sub: str, full_text: str) -> tuple[int, int] | None:
    """Locate a substring's character span within a full text.

    Tries exact containment first, then a normalized search (LaTeX backslash
    escaping, alignment characters, whitespace collapsing) mapped back to
    approximate positions in the original text.
    """
    idx = full_text.find(sub)
    if idx != -1:
        return (idx, idx + len(sub))

    sub_norm = _normalize_for_matching(sub)
    text_norm = _normalize_for_matching(full_text)
    idx = text_norm.find(sub_norm)
    if idx == -1:
        return None
    orig_pos = _map_norm_pos_to_original(idx, text_norm, full_text)
    return (orig_pos, orig_pos + len(sub_norm))


def _compute_char_span_overlap(
    text_a: str, text_b: str, full_text: str
) -> float:
    """Compute character-span overlap between two substrings within a full text.

    Uses the first occurrence of each substring in ``full_text`` to determine
    character spans, then computes **intersection-over-union (IoU)** of the spans.

    Falls back to a normalized text search when exact matching fails, handling
    LaTeX alignment characters and whitespace differences.

    Args:
        text_a: First substring (e.g., injected_text from ground truth).
        text_b: Second substring (e.g., quoted_text from Verifier claim).
        full_text: The full perturbed text.

    Returns:
        IoU overlap in [0, 1]. 0 = no overlap; 1 = perfect span match.
    """
    span_a = _locate_span(text_a, full_text)
    span_b = _locate_span(text_b, full_text)

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


def _map_norm_pos_to_original(norm_pos: int, norm_text: str, orig_text: str) -> int:
    """Map a character position in normalized text back to the original text.

    Walks through both strings in parallel, skipping whitespace differences and
    LaTeX alignment characters that were removed during normalization.

    Args:
        norm_pos: Character index in the normalized text.
        norm_text: The normalized version of orig_text.
        orig_text: The original (unnormalized) text.

    Returns:
        Approximate character index in orig_text corresponding to norm_pos.
    """
    orig_idx = 0
    norm_idx = 0

    while norm_idx < norm_pos and orig_idx < len(orig_text):
        # Skip characters in original that were removed during normalization
        orig_char = orig_text[orig_idx]
        # LaTeX alignment & before =
        if orig_text[orig_idx:orig_idx + 2] == "&=":
            orig_idx += 1  # skip the &
            continue
        if orig_char == "&" and (
            orig_idx + 1 >= len(orig_text) or orig_text[orig_idx + 1].isspace()
        ):
            orig_idx += 1
            continue
        # Collapsed whitespace: skip extra whitespace in original
        if orig_char.isspace():
            # Check if this is part of a collapsed whitespace run
            if norm_idx > 0 and norm_text[norm_idx - 1] == " ":
                # Already accounted for one space in normalized; skip extras
                while orig_idx < len(orig_text) and orig_text[orig_idx].isspace():
                    orig_idx += 1
                continue

        norm_idx += 1
        orig_idx += 1

    return orig_idx


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

            if iou < span_overlap_threshold:
                # --- Diff-based change coverage (strong) ---
                # The claim quotes the error region AND includes the text the
                # Perturber actually changed (per original→injected diff).
                if _claim_covers_change(error, claim):
                    iou = max(iou, CHANGE_COVERAGE_SCORE)
                else:
                    # --- Substring-containment boost (weak fallback) ---
                    # Handles the common case where injected_text is a long
                    # sentence and quoted_text is just a fragment of it. The
                    # boost scales with the length ratio so that greedy
                    # assignment prefers claims quoting more of the error.
                    sub_score = _substring_match_score(
                        error.injected_text, claim.quoted_text
                    )
                    if sub_score > 0:
                        iou = max(
                            iou,
                            span_overlap_threshold + 0.01
                            + 0.08 * sub_score,
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


def _token_jaccard(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity on normalized text, in [0, 1]."""
    tokens_a = set(_normalize_for_matching(text_a).lower().split())
    tokens_b = set(_normalize_for_matching(text_b).lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _boxed_contents(text: str) -> list[str]:
    """Extract the (normalized) contents of every ``\\boxed{...}`` in a text."""
    normalized = _normalize_latex_escapes(text)
    return [re.sub(r"\s+", "", m) for m in _BOXED_RE.findall(normalized)]


def group_errors_into_units(
    ground_truth: list[InjectedError],
    perturbed_text: str,
    near_duplicate_threshold: float = 0.6,
    merge_overlapping_spans: bool = True,
    merge_propagated_boxed: bool = True,
    merge_shared_change_fragment: bool = True,
) -> list[list[int]]:
    """Group injected errors into distinct **error units** for unit-level recall.

    The Perturber can stack a root error plus its downstream consequences (a
    corrupted final answer, an overlapping rewrite of the same region, a
    near-duplicate injection) and declare them as separate errors. A Verifier
    that flags the root line once then caps out at 1/k recall — a reward hack.
    This collapses causally-linked errors so the Verifier is scored on
    *distinct* mistakes:

    1. **Overlapping spans** — two errors whose injected_texts occupy
       overlapping regions of the perturbed text, contain one another, or
       whose original_texts meaningfully contain one another (two rewrites
       declared over the same source region) are the same mistake.
    2. **Near-duplicates** — injected_texts (or original_texts) with token
       Jaccard above ``near_duplicate_threshold``.
    3. **Shared changed fragment** — two errors whose original→injected
       diffs introduce the *same* text (e.g. the same phantom ``+ 2(1)``
       term appended to three consecutive derivation lines) are one change
       propagated, not independent mistakes.
    4. **Propagated final answer** — an error that changes a ``\\boxed{...}``
       result merges with the nearest earlier error (by step_index, then
       declaration order): a wrong boxed answer alongside upstream errors is
       treated as propagation, not an independent mistake.

    Returns:
        List of units, each a list of indices into ``ground_truth``. Units
        preserve first-appearance order; indices within a unit are sorted.
    """
    n = len(ground_truth)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    spans = [
        _locate_span(getattr(e, "injected_text", ""), perturbed_text)
        for e in ground_truth
    ]
    change_fragments: list[set[str]] = []
    for e in ground_truth:
        frags = _changed_fragments(
            getattr(e, "original_text", "") or "",
            getattr(e, "injected_text", "") or "",
        )
        change_fragments.append(
            {f for f in (_normalize_for_matching(fr) for fr in frags) if len(f) >= 4}
        )

    for i in range(n):
        for j in range(i + 1, n):
            e_i, e_j = ground_truth[i], ground_truth[j]
            inj_i = getattr(e_i, "injected_text", "")
            inj_j = getattr(e_j, "injected_text", "")

            if merge_overlapping_spans:
                if spans[i] is not None and spans[j] is not None:
                    lo = max(spans[i][0], spans[j][0])
                    hi = min(spans[i][1], spans[j][1])
                    if lo < hi:
                        union(i, j)
                        continue
                # Containment catches overlapping rewrites even when one
                # variant is no longer locatable in the final text; original
                # and cross (injected↔original) containment catch two errors
                # declared over the same source region (e.g. a line plus the
                # block containing it, or a rewrite that quotes the region
                # another error separately modified).
                orig_i = getattr(e_i, "original_text", "") or ""
                orig_j = getattr(e_j, "original_text", "") or ""
                if (
                    _is_substring_match(inj_i, inj_j)
                    or (orig_i and orig_j and _is_substring_match(orig_i, orig_j))
                    or (orig_j and _is_substring_match(inj_i, orig_j))
                    or (orig_i and _is_substring_match(orig_i, inj_j))
                ):
                    union(i, j)
                    continue

            if merge_shared_change_fragment and change_fragments[i] & change_fragments[j]:
                union(i, j)
                continue

            if (
                _token_jaccard(inj_i, inj_j) >= near_duplicate_threshold
                or _token_jaccard(
                    getattr(e_i, "original_text", ""),
                    getattr(e_j, "original_text", ""),
                )
                >= near_duplicate_threshold
            ):
                union(i, j)

    if merge_propagated_boxed and n > 1:
        # Paper-mode InjectedError has no step_index — fall back to
        # declaration order (duck-typing, like the rest of the reward path).
        order = sorted(
            range(n), key=lambda i: (getattr(ground_truth[i], "step_index", 0), i)
        )
        for pos, i in enumerate(order):
            if pos == 0:
                continue
            boxed_inj = _boxed_contents(getattr(ground_truth[i], "injected_text", ""))
            boxed_orig = _boxed_contents(getattr(ground_truth[i], "original_text", ""))
            # Only errors that actually *change* a boxed result are treated as
            # propagation; quoting an unchanged boxed answer is incidental.
            if boxed_inj and set(boxed_inj) != set(boxed_orig):
                union(order[pos - 1], i)

    units: dict[int, list[int]] = {}
    for i in range(n):
        units.setdefault(find(i), []).append(i)
    return [sorted(members) for _, members in sorted(units.items(), key=lambda kv: min(kv[1]))]


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
