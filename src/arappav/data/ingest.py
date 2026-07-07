"""Paper ingestion: download/parse papers from various sources.

Supports:
- Local plain text / markdown files (primary for v1).
- arXiv API (stretch goal — stubbed).
- Hugging Face datasets (stretch goal — stubbed).

PDF/LaTeX parsing is a stretch goal and is stubbed with clear TODO markers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ingest_local(raw_dir: str | Path, max_papers: int | None = None) -> list[dict]:
    """Ingest papers from a local directory of text/markdown files.

    Each file is treated as one paper. The filename (without extension) is used as
    the paper ID.

    Args:
        raw_dir: Path to directory containing .txt or .md files.
        max_papers: If set, limit to this many papers (useful for debugging).

    Returns:
        List of dicts with keys: ``id``, ``text``, ``source_path``, ``format``.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    papers = []
    supported_suffixes = {".txt", ".md", ".markdown"}

    for filepath in sorted(raw_dir.iterdir()):
        if filepath.suffix not in supported_suffixes:
            continue
        if filepath.name.startswith("."):
            continue

        try:
            text = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Skipping {filepath}: not valid UTF-8")
            continue

        if not text.strip():
            logger.warning(f"Skipping {filepath}: empty file")
            continue

        papers.append(
            {
                "id": filepath.stem,
                "text": text,
                "source_path": str(filepath),
                "format": filepath.suffix.lstrip("."),
            }
        )

        if max_papers and len(papers) >= max_papers:
            break

    logger.info(f"Ingested {len(papers)} papers from {raw_dir}")
    return papers


def ingest_arxiv(
    query: str = "cs.CL",
    max_results: int = 100,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Ingest papers from arXiv API.  **Stretch goal — not implemented in v1.**

    Args:
        query: arXiv category or search query.
        max_results: Maximum number of papers to fetch.
        start_date: Earliest submission date (YYYY-MM-DD).
        end_date: Latest submission date (YYYY-MM-DD).

    Returns:
        List of dicts with keys: ``id``, ``text``, ``title``, ``authors``, ``abstract``.
    """
    # TODO: Implement arXiv API ingestion.
    # - Use the `arxiv` Python package or direct API calls to export.arxiv.org.
    # - Fetch PDF, convert to text via `pymupdf` or `grobid`.
    # - Or fetch arXiv's LaTeX source when available and convert via `pandoc`.
    raise NotImplementedError(
        "arXiv ingestion is a stretch goal and is not yet implemented. "
        "Use `ingest_local` with pre-processed text files for v1."
    )


def ingest_huggingface(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_papers: int | None = None,
) -> list[dict]:
    """Ingest papers from a Hugging Face dataset.  **Stretch goal — not implemented in v1.**

    Args:
        dataset_name: HF dataset identifier (e.g., 'arxiv_dataset').
        split: Dataset split to load.
        text_column: Column name containing the paper text.
        max_papers: Maximum number of papers to load.

    Returns:
        List of dicts with keys: ``id``, ``text``, ``source``.
    """
    # TODO: Implement HF dataset ingestion.
    # - Use `datasets.load_dataset(dataset_name, split=split)`.
    # - Map the relevant columns to our internal format.
    raise NotImplementedError(
        "Hugging Face dataset ingestion is a stretch goal and is not yet implemented. "
        "Use `ingest_local` with pre-processed text files for v1."
    )


def save_processed(papers: list[dict], output_dir: str | Path) -> list[Path]:
    """Save ingested papers as JSONL in the processed directory.

    Args:
        papers: List of paper dicts.
        output_dir: Directory to write JSONL files into.

    Returns:
        List of paths written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for paper in papers:
        out_path = output_dir / f"{paper['id']}.json"
        out_path.write_text(json.dumps(paper, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(out_path)

    logger.info(f"Saved {len(paths)} papers to {output_dir}")
    return paths


def load_processed(processed_dir: str | Path) -> list[dict]:
    """Load previously processed papers from JSONL files.

    Args:
        processed_dir: Directory containing JSON paper files.

    Returns:
        List of paper dicts.
    """
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    papers = []
    for filepath in sorted(processed_dir.glob("*.json")):
        paper = json.loads(filepath.read_text(encoding="utf-8"))
        papers.append(paper)

    logger.info(f"Loaded {len(papers)} processed papers from {processed_dir}")
    return papers
