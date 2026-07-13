"""Fuzzy matching of model-produced error-type names against taxonomy enums.

Untuned models frequently emit semantically reasonable but syntactically
invalid enum values (``'addition_across'`` for ``'adding_across'``). Hard-
failing on those wastes training episodes on a -10 format penalty, so we
auto-correct near-misses with a warning instead.

Shared by the paper-mode (``schema.py``) and math-mode (``schema_math.py``)
schemas.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

E = TypeVar("E", bound=Enum)

#: Words too generic to count as evidence that two error-type names refer to
#: the same concept — nearly half the taxonomy contains "error".
GENERIC_WORDS = frozenset({"error", "wrong", "mistake", "incorrect", "misconception"})


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion, deletion, substitution
            curr_row.append(min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + (c1 != c2),
            ))
        prev_row = curr_row
    return prev_row[-1]


def _words_match(a: str, b: str) -> bool:
    """Check whether two underscore-delimited words refer to the same concept.

    Exact match always counts. For words of 4+ characters, a small edit
    distance (scaled to word length) also counts, so inflections like
    ``'duplicated'`` / ``'duplication'`` or ``'sign'`` / ``'signs'`` match.
    """
    if a == b:
        return True
    min_len = min(len(a), len(b))
    if min_len < 4:
        return False
    return levenshtein_distance(a, b) <= max(1, min_len // 3)


def fuzzy_match_enum(
    name: str,
    enum_cls: type[E],
    max_distance: int = 3,
    relaxed_distance: int = 10,
    aliases: dict[str, str] | None = None,
) -> E | None:
    """Find the enum member closest to *name*, or None if nothing is close.

    ``aliases`` maps known alternative names to enum values, checked before
    any fuzzy matching. Use it for names models persistently emit that fuzzy
    matching (correctly) refuses to guess.

    Two-tier strategy:

    1. **Strict:** the member with the smallest full-string Levenshtein
       distance, if that distance is ≤ ``max_distance`` (catches typos).
    2. **Relaxed:** among ALL members within ``relaxed_distance``, those
       sharing at least one non-generic word with *name* (exact or inflected
       — see ``_words_match``); the closest such member wins. Generic words
       like "error"/"wrong" are excluded as evidence, so ``'sign_error'``
       cannot match ``'inversion_error'`` on the word "error" alone.

    Args:
        name: The candidate error type string.
        enum_cls: The taxonomy enum to match against.
        max_distance: Maximum edit distance for a strict match.
        relaxed_distance: Maximum edit distance for a word-overlap match.

    Returns:
        The closest enum member, or ``None`` if no match within distance.
    """
    name_lower = name.lower().strip()

    if aliases and name_lower in aliases:
        member = enum_cls(aliases[name_lower])
        logger.warning("Aliased error_type %r → %r.", name, member.value)
        return member

    scored = sorted(
        ((levenshtein_distance(name_lower, member.value), member) for member in enum_cls),
        key=lambda pair: pair[0],
    )
    best_dist, best = scored[0]

    if best_dist <= max_distance:
        return best

    name_words = [w for w in name_lower.split("_") if w and w not in GENERIC_WORDS]
    for dist, member in scored:
        if dist > relaxed_distance:
            break
        member_words = [w for w in member.value.split("_") if w not in GENERIC_WORDS]
        if any(_words_match(a, b) for a in name_words for b in member_words):
            logger.warning(
                "Relaxed fuzzy match for %r → %r (distance=%d, word overlap).",
                name, member.value, dist,
            )
            return member

    return None
