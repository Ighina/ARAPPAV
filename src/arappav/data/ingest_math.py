"""Math dataset pipeline: load and process the Hendrycks MATH dataset.

The EleutherAI/hendrycks_math dataset contains competition-level math problems
with step-by-step solutions across 7 topics (algebra, geometry, etc.) and 5
difficulty levels.

Each example provides:
- ``problem``: LaTeX-formatted problem statement
- ``solution``: step-by-step solution with reasoning
- ``level``: difficulty 1-5
- ``type``: topic category

In ARAPPAV math mode, the Perturber receives a (problem, solution) pair and
injects errors into the solution. The Verifier receives the problem and the
(possibly perturbed) solution and must identify errors.

Reference: Hendrycks et al., "Measuring Mathematical Problem Solving With the
MATH Dataset" (NeurIPS 2021).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, get_dataset_config_names

logger = logging.getLogger(__name__)

# All available math topics in the Hendrycks MATH dataset
MATH_TOPICS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_math_dataset(
    topics: list[str] | None = None,
    split: str = "train",
    max_examples_per_topic: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load problems from the Hendrycks MATH dataset.

    Args:
        topics: List of topics to load. If None, loads all topics.
        split: Which split to load ('train' or 'test').
        max_examples_per_topic: If set, randomly subsample this many examples per topic.
        seed: Random seed for subsampling.

    Returns:
        A HF Dataset with columns: ``problem``, ``solution``, ``level``, ``type``, ``topic``.
    """
    if topics is None:
        topics = MATH_TOPICS

    all_rows = []
    for topic in topics:
        try:
            ds = _load_single_topic(topic, split)
        except Exception as e:
            logger.warning(f"Could not load topic '{topic}': {e}")
            continue

        if max_examples_per_topic and len(ds) > max_examples_per_topic:
            ds = ds.shuffle(seed=seed).select(range(max_examples_per_topic))

        for row in ds:
            all_rows.append(
                {
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "level": row["level"],
                    "type": row["type"],
                    "topic": topic,
                }
            )

    logger.info(
        f"Loaded {len(all_rows)} math problems across {len(topics)} topics "
        f"(split='{split}')"
    )
    return Dataset.from_list(all_rows)


def _load_single_topic(topic: str, split: str) -> Dataset:
    """Load a single topic from the Hendrycks MATH dataset."""
    from datasets import load_dataset

    ds_dict = load_dataset("EleutherAI/hendrycks_math", topic, split=split)
    return ds_dict


def build_math_splits(
    topics: list[str] | None = None,
    train_topics: list[str] | None = None,
    val_topics: list[str] | None = None,
    test_topics: list[str] | None = None,
    max_train_per_topic: int | None = None,
    max_val_per_topic: int | None = 100,
    max_test_per_topic: int | None = 100,
    seed: int = 42,
) -> DatasetDict:
    """Build train/val/test splits from the Hendrycks MATH dataset.

    By default, uses the official train/test splits provided by the dataset
    and further splits the training set into train/val. Topics can be assigned
    to specific splits for held-out topic evaluation.

    Args:
        topics: All topics to load (default: all 7).
        train_topics: Topics for training (default: algebra, counting, geometry,
            intermediate_algebra, number_theory, prealgebra).
        val_topics: Topics for validation (default: same as train, from test split).
        test_topics: Topics for testing (default: all).
        max_train_per_topic: Max training examples per topic (None = all).
        max_val_per_topic: Max validation examples per topic.
        max_test_per_topic: Max test examples per topic.
        seed: Random seed.

    Returns:
        DatasetDict with 'train', 'val', 'test' keys.
    """
    if topics is None:
        topics = MATH_TOPICS
    if train_topics is None:
        train_topics = topics
    if val_topics is None:
        val_topics = topics
    if test_topics is None:
        test_topics = topics

    train_ds = load_math_dataset(
        topics=train_topics, split="train",
        max_examples_per_topic=max_train_per_topic, seed=seed,
    )
    val_ds = load_math_dataset(
        topics=val_topics, split="test",
        max_examples_per_topic=max_val_per_topic, seed=seed,
    )
    test_ds = load_math_dataset(
        topics=test_topics, split="test",
        max_examples_per_topic=max_test_per_topic, seed=seed + 1,  # different seed for test
    )

    logger.info(
        f"Math splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    return DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})


def save_math_dataset(dataset_dict: DatasetDict, output_dir: str | Path) -> None:
    """Save a math DatasetDict to disk as Parquet files.

    Args:
        dataset_dict: DatasetDict with train/val/test splits.
        output_dir: Directory to save into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ds in dataset_dict.items():
        path = output_dir / f"math_{split_name}.parquet"
        ds.to_parquet(path)
        logger.info(f"Saved {split_name} split ({len(ds)} rows) to {path}")


def load_math_dataset_from_disk(processed_dir: str | Path) -> DatasetDict:
    """Load a previously saved math DatasetDict from disk.

    Args:
        processed_dir: Directory containing math_*.parquet files.

    Returns:
        DatasetDict with train/val/test splits.
    """
    from datasets import load_from_disk
    import pyarrow.parquet as pq

    processed_dir = Path(processed_dir)
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = processed_dir / f"math_{split_name}.parquet"
        if path.exists():
            splits[split_name] = Dataset.from_parquet(str(path))
            logger.info(f"Loaded {split_name} split ({len(splits[split_name])} rows)")

    if not splits:
        raise FileNotFoundError(f"No math_*.parquet files found in {processed_dir}")

    return DatasetDict(splits)
