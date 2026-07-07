"""Paper chunking: split papers into trainable sections.

Two strategies:
- ``section``: Split by markdown headings (##, ###, etc.). Preserves section metadata.
- ``fixed_window``: Split into fixed-length token windows with configurable overlap.

The chunk level is the unit of perturbation/verification since injecting or
detecting errors over an entire paper may exceed model context windows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk (section or window) of a paper."""

    paper_id: str
    chunk_id: str  # e.g., "paper_01_sec_2" or "paper_01_win_0"
    text: str
    section_title: str | None = None  # heading text if available
    start_char: int = 0
    end_char: int = 0

    # Arbitrary metadata
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Section-based chunking
# ---------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _find_headings(text: str) -> list[tuple[int, int, str, str]]:
    """Find markdown headings in text.

    Returns:
        List of (start, end, level_marker, title) tuples.
    """
    headings = []
    for match in _HEADING_PATTERN.finditer(text):
        headings.append(
            (match.start(), match.end(), match.group(1), match.group(2).strip())
        )
    return headings


def chunk_by_section(
    paper: dict,
    min_tokens: int = 256,
    approximate_token_length: bool = True,
) -> list[Chunk]:
    """Split a paper into sections based on markdown headings.

    Sections that are too short (below ``min_tokens``) are merged with the
    preceding section to avoid tiny fragments.

    Args:
        paper: Paper dict with ``id`` and ``text`` keys.
        min_tokens: Minimum approximate token count for a section to stand alone.
        approximate_token_length: If True, estimate tokens as ``len(text)//4``.
            If False, uses exact whitespace-split word count (cruder).

    Returns:
        List of Chunk objects, one per section.
    """
    text = paper["text"]
    paper_id = paper["id"]
    headings = _find_headings(text)

    if not headings:
        # No headings found — treat the whole paper as one chunk.
        return [
            Chunk(
                paper_id=paper_id,
                chunk_id=f"{paper_id}_sec_0",
                text=text.strip(),
                section_title=None,
                start_char=0,
                end_char=len(text),
            )
        ]

    # Build sections: each heading spans from its start to the next heading start.
    sections = []
    for i, (start, end, level, title) in enumerate(headings):
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        section_text = text[start:next_start].strip()
        sections.append(
            {
                "title": title,
                "level": len(level),
                "text": section_text,
                "start_char": start,
                "end_char": next_start,
            }
        )

    # Merge short sections with the preceding one.
    chunks = []
    buffer = None

    for i, sec in enumerate(sections):
        est_len = len(sec["text"]) // 4 if approximate_token_length else len(sec["text"].split())

        if buffer is None:
            buffer = sec
        else:
            buffer_est = len(buffer["text"]) // 4 if approximate_token_length else len(buffer["text"].split())
            if buffer_est < min_tokens:
                # Merge current into buffer
                buffer["text"] += "\n\n" + sec["text"]
                buffer["end_char"] = sec["end_char"]
            else:
                chunks.append(buffer)
                buffer = sec

    if buffer is not None:
        chunks.append(buffer)

    # Convert to Chunk objects
    result = []
    for i, sec in enumerate(chunks):
        result.append(
            Chunk(
                paper_id=paper_id,
                chunk_id=f"{paper_id}_sec_{i}",
                text=sec["text"],
                section_title=sec.get("title"),
                start_char=sec["start_char"],
                end_char=sec["end_char"],
                metadata={"heading_level": sec.get("level"), "section_index": i},
            )
        )

    return result


# ---------------------------------------------------------------------------
# Fixed-window chunking
# ---------------------------------------------------------------------------


def chunk_by_fixed_window(
    paper: dict,
    chunk_size: int = 2048,
    chunk_overlap: int = 256,
    approximate_token_length: bool = True,
) -> list[Chunk]:
    """Split a paper into fixed-length token windows with overlap.

    Args:
        paper: Paper dict with ``id`` and ``text`` keys.
        chunk_size: Target window size in tokens (approximate).
        chunk_overlap: Overlap between consecutive windows in tokens.
        approximate_token_length: If True, estimate tokens as ``len(text)//4``.

    Returns:
        List of Chunk objects.
    """
    text = paper["text"]
    paper_id = paper["id"]

    if approximate_token_length:
        char_chunk_size = chunk_size * 4
        char_overlap = chunk_overlap * 4
    else:
        # Very rough: assume 1 token ≈ 0.75 words
        char_chunk_size = int(chunk_size * 0.75 * 5)
        char_overlap = int(chunk_overlap * 0.75 * 5)

    step = char_chunk_size - char_overlap
    if step <= 0:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + char_chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    paper_id=paper_id,
                    chunk_id=f"{paper_id}_win_{idx}",
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )
            idx += 1

        start += step

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def chunk_papers(
    papers: list[dict],
    strategy: str = "section",
    chunk_size: int = 2048,
    chunk_overlap: int = 256,
    min_chunk_tokens: int = 512,
) -> list[Chunk]:
    """Dispatch chunking across a list of papers.

    Args:
        papers: List of paper dicts.
        strategy: ``"section"`` or ``"fixed_window"``.
        chunk_size: Window size for fixed_window strategy.
        chunk_overlap: Overlap for fixed_window strategy.
        min_chunk_tokens: Minimum token length for a chunk to be kept.

    Returns:
        Flat list of all Chunks across all papers.
    """
    all_chunks = []

    for paper in papers:
        if strategy == "section":
            chunks = chunk_by_section(paper, min_tokens=min_chunk_tokens)
        elif strategy == "fixed_window":
            chunks = chunk_by_fixed_window(paper, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            raise ValueError(f"Unknown chunk strategy: {strategy}")

        # Filter out chunks that are too short
        chunks = [c for c in chunks if len(c.text) // 4 >= min_chunk_tokens]

        all_chunks.extend(chunks)

    logger.info(
        f"Chunked {len(papers)} papers into {len(all_chunks)} chunks (strategy={strategy})"
    )
    return all_chunks
