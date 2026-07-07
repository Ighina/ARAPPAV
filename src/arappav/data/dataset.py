"""Hugging Face ``datasets`` wrappers for the ARAPPAV corpus.

Provides utilities to build train/val/test splits at the paper level (not chunk
level) and to stream chunks with their associated metadata during self-play.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from arappav.data.chunking import Chunk, chunk_papers
from arappav.data.ingest import load_processed

logger = logging.getLogger(__name__)


def build_split(
    papers: list[dict],
    train_split: float = 0.80,
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Split papers at the **paper level** into train/val/test sets.

    Paper-level splitting avoids leakage: all chunks from a given paper go to the
    same split.

    Args:
        papers: List of paper dicts.
        train_split: Fraction for training.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        seed: Random seed for reproducible splits.

    Returns:
        A ``DatasetDict`` with ``"train"``, ``"val"``, ``"test"`` keys.
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    paper_ids = [p["id"] for p in papers]
    paper_texts = [p["text"] for p in papers]

    # First split: train vs rest
    train_ids, rest_ids, train_texts, rest_texts = train_test_split(
        paper_ids, paper_texts, test_size=(val_split + test_split), random_state=seed
    )

    # Second split: val vs test (relative to rest)
    val_frac_of_rest = val_split / (val_split + test_split)
    val_ids, test_ids, val_texts, test_texts = train_test_split(
        rest_ids, rest_texts, test_size=(1 - val_frac_of_rest), random_state=seed
    )

    def _make_dataset(ids: list[str], texts: list[str]) -> Dataset:
        return Dataset.from_dict({"id": ids, "text": texts})

    logger.info(
        f"Split {len(papers)} papers → train={len(train_ids)}, "
        f"val={len(val_ids)}, test={len(test_ids)}"
    )

    return DatasetDict(
        {
            "train": _make_dataset(train_ids, train_texts),
            "val": _make_dataset(val_ids, val_texts),
            "test": _make_dataset(test_ids, test_texts),
        }
    )


def iter_chunks(
    dataset_split: Dataset,
    chunk_strategy: str = "section",
    chunk_size: int = 2048,
    chunk_overlap: int = 256,
    min_chunk_tokens: int = 512,
) -> Iterator[Chunk]:
    """Yield Chunks from a dataset split, one paper at a time.

    This is the main entry point for the self-play loop: it streams chunks
    without materialising all of them in memory.

    Args:
        dataset_split: A HF Dataset (e.g., ``dataset_dict["train"]``).
        chunk_strategy: Chunking strategy to use.
        chunk_size: Window size for fixed_window.
        chunk_overlap: Overlap for fixed_window.
        min_chunk_tokens: Skip chunks shorter than this.

    Yields:
        Chunk objects.
    """
    for row in dataset_split:
        paper = {"id": row["id"], "text": row["text"]}
        chunks = chunk_papers(
            [paper],
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_tokens=min_chunk_tokens,
        )
        yield from chunks


def load_corpus_split(
    processed_dir: str | Path,
    train_split: float = 0.80,
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """Convenience: load processed papers and return a DatasetDict with splits.

    Args:
        processed_dir: Directory of JSON paper files.
        train_split, val_split, test_split: Split ratios (must sum to 1.0).
        seed: Random seed.

    Returns:
        DatasetDict with train/val/test splits.
    """
    papers = load_processed(processed_dir)
    return build_split(papers, train_split, val_split, test_split, seed)
