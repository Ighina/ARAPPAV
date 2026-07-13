"""Shared JSON-parsing helpers for model outputs.

Both the Perturber and Verifier models output JSON that may be wrapped in
markdown code fences or contain multiple JSON blocks. These helpers handle
those cases robustly.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def strip_json_fences(text: str) -> str:
    """Remove **all** markdown code fences from text.

    Handles nested / repeated fences that arise when a model outputs multiple
    JSON blocks (e.g.  ```json {…} ``` ```json {…} ```).  After stripping,
    the caller should use :func:`extract_first_json_object` to extract only
    the first complete JSON object.

    Args:
        text: Raw model output, possibly with markdown fences.

    Returns:
        Fence-free text.
    """
    cleaned = text.strip()

    # Strip leading fences (handle ```json and ```)
    while cleaned.startswith("```"):
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        cleaned = cleaned.strip()

    # Strip trailing fences
    while cleaned.endswith("```"):
        cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return cleaned


def extract_first_json_object(text: str) -> tuple[dict | None, str | None]:
    """Extract the first complete JSON object from text, ignoring trailing data.

    This is more robust than ``json.loads()`` because it tolerates extra
    content after the first JSON object (e.g. a second ``{"claims": []}``
    block that some models append as a fallback).

    Args:
        text: Fence-stripped text that should start with a JSON object.

    Returns:
        ``(parsed_dict, None)`` on success, or ``(None, error_message)``
        on failure.
    """
    cleaned = text.strip()

    if not cleaned:
        return None, "Empty input after stripping fences."

    # If the text doesn't start with '{', try to locate one.
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        if start == -1:
            return None, f"No JSON object found in output. First 200 chars: {cleaned[:200]}"
        cleaned = cleaned[start:]

    # Use raw_decode to parse just the first JSON object — ignores trailing
    # data (e.g. a second JSON block or stray backticks).
    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(cleaned)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error after fence stripping: {e}\nFirst 500 chars: {cleaned[:500]}"


# ---------------------------------------------------------------------------
# Mechanical error injection (Perturber backoff)
# ---------------------------------------------------------------------------


def apply_error_injections(
    text: str,
    errors: list[dict[str, str]],
) -> tuple[str, list[str]]:
    """Mechanically apply ``original_text → injected_text`` replacements.

    Used as a backoff when the Perturber model defines valid error objects in
    JSON but fails to actually modify the solution text.  Processes errors in
    **reverse position order** so earlier replacements don't shift the spans
    of later ones.

    When errors target **overlapping spans** (one ``original_text`` contains
    another), only the longest-span error is applied; the shorter is skipped
    with a warning, since both modifications cannot coexist in the same text
    region.  The Perturber is expected to learn to spread errors across
    disjoint regions through training rewards.

    Args:
        text: The text to perturb (original solution or paper text).
        errors: List of error dicts, each with ``"original_text"`` and
            ``"injected_text"`` keys (as produced by the Perturber's JSON).

    Returns:
        ``(perturbed_text, warnings)`` where *warnings* is a list of
        human-readable messages about errors that could not be mechanically
        injected.
    """
    warnings: list[str] = []
    perturbed = text

    # --- Step 1: locate each original_text in the text -----------------------
    located: list[dict] = []
    phantom_count = 0
    for err in errors:
        orig = err["original_text"]
        injected = err["injected_text"]

        # Skip phantom errors — no actual modification to apply
        if orig.strip() == injected.strip():
            phantom_count += 1
            warnings.append(
                f"Phantom error {err.get('error_id', '?')}: "
                f"injected_text equals original_text — skipping (no change to apply)."
            )
            continue

        pos = perturbed.find(orig)

        # Fallback: try with stripped whitespace
        if pos == -1:
            pos = perturbed.find(orig.strip())

        if pos == -1:
            warnings.append(
                f"Could not find original_text for {err.get('error_id', '?')} "
                f"in solution. Original: {orig[:100]!r}"
            )
            continue

        located.append(
            {
                "error_id": err.get("error_id", "?"),
                "original_text": orig,
                "injected_text": injected,
                "pos": pos,
                "end": pos + len(orig),
            }
        )

    if phantom_count > 0:
        logger.warning(
            "Mechanical injection: %d phantom error(s) skipped (injected == original).",
            phantom_count,
        )

    # --- Step 2: detect overlapping spans and resolve conflicts --------------
    # Sort by position ascending for overlap detection
    located.sort(key=lambda e: e["pos"])

    resolved: list[dict] = []
    skipped_overlap: list[str] = []
    for i, loc in enumerate(located):
        # Check overlap with the last accepted replacement
        if resolved and loc["pos"] < resolved[-1]["end"]:
            # Overlap detected: keep the one with the longer original_text
            prev = resolved[-1]
            prev_len = prev["end"] - prev["pos"]
            curr_len = loc["end"] - loc["pos"]
            if curr_len > prev_len:
                # Current is longer — replace previous
                skipped_overlap.append(
                    f"Overlapping error {prev['error_id']}: original_text overlaps "
                    f"with {loc['error_id']} — skipping shorter replacement. "
                    f"Prev original: {prev['original_text'][:80]!r}"
                )
                resolved[-1] = loc
            else:
                skipped_overlap.append(
                    f"Overlapping error {loc['error_id']}: original_text overlaps "
                    f"with {prev['error_id']} — skipping shorter replacement. "
                    f"Current original: {loc['original_text'][:80]!r}"
                )
        else:
            resolved.append(loc)

    if skipped_overlap:
        for msg in skipped_overlap:
            warnings.append(msg)
            logger.warning("Mechanical injection: %s", msg)

    # --- Step 3: sort descending by position (replace from end to start) -----
    resolved.sort(key=lambda e: e["pos"], reverse=True)

    # --- Step 4: apply replacements ------------------------------------------
    for loc in resolved:
        pos = loc["pos"]
        orig = loc["original_text"]
        injected = loc["injected_text"]
        perturbed = perturbed[:pos] + injected + perturbed[pos + len(orig):]

    total_warnings = len([w for w in warnings if w.startswith("Phantom") or w.startswith("Overlapping") or w.startswith("Could not")])
    if total_warnings > 0:
        logger.warning(
            "Mechanical injection: %d/%d errors applied (%d warning(s)).",
            len(resolved),
            len(errors),
            total_warnings,
        )
    else:
        logger.info(
            "Mechanical injection: all %d errors applied successfully.",
            len(resolved),
        )

    return perturbed, warnings
