"""Canonical MATH subject categories, and the names each dataset uses for them.

A restricted experiment trains and evaluates within one subject. That only works
if "the same category" means the same thing on both sides, and it does not
naturally: the Hendrycks training split calls it ``counting_and_probability``
while MATH-500 calls it ``Counting & Probability``. Matching by string would
silently yield an empty validation set — which looks like a small sample rather
than a bug.

This module is the single place those names are reconciled.
"""

from __future__ import annotations

#: canonical name -> (Hendrycks training topic, MATH-500 `subject` value)
CATEGORIES: dict[str, tuple[str, str]] = {
    "algebra": ("algebra", "Algebra"),
    "counting_and_probability": ("counting_and_probability", "Counting & Probability"),
    "geometry": ("geometry", "Geometry"),
    "intermediate_algebra": ("intermediate_algebra", "Intermediate Algebra"),
    "number_theory": ("number_theory", "Number Theory"),
    "prealgebra": ("prealgebra", "Prealgebra"),
    "precalculus": ("precalculus", "Precalculus"),
}

#: The category used when a restricted run does not name one.
DEFAULT_CATEGORY = "algebra"

ALL = "all"


def normalise(name: str | None) -> str:
    """Accept a canonical name, a dataset spelling, or None; return canonical."""
    if not name or name.lower() == ALL:
        return ALL
    key = name.strip().lower().replace(" & ", "_and_").replace(" ", "_").replace("&", "and")
    if key in CATEGORIES:
        return key
    for canon, (train, subject) in CATEGORIES.items():
        if name.strip().lower() in (train.lower(), subject.lower()):
            return canon
    raise ValueError(
        f"unknown MATH category {name!r}; choose from "
        f"{sorted(CATEGORIES)} or {ALL!r}"
    )


def train_topics(category: str | None) -> list[str]:
    """Hendrycks topic list for a category (all topics when unrestricted)."""
    c = normalise(category)
    if c == ALL:
        return [t for t, _ in CATEGORIES.values()]
    return [CATEGORIES[c][0]]


def math500_subject(category: str | None) -> str | None:
    """MATH-500 `subject` value to filter on, or None when unrestricted."""
    c = normalise(category)
    return None if c == ALL else CATEGORIES[c][1]


def filter_math500(rows, category: str | None):
    """Filter a MATH-500 split to one category. Empty means a naming mismatch."""
    subject = math500_subject(category)
    if subject is None:
        return list(range(len(rows)))
    idx = [i for i, s in enumerate(rows["subject"]) if s == subject]
    if not idx:
        raise ValueError(
            f"no MATH-500 items with subject {subject!r} — the category mapping "
            f"is wrong, not the sample"
        )
    return idx
