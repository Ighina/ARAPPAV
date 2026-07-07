#!/usr/bin/env python
"""Run held-out evaluation on trained ARAPPAV models.

Usage:
    python scripts/run_eval.py --perturber_checkpoint checkpoints/perturber_final \\
                               --verifier_checkpoint checkpoints/verifier_final \\
                               --processed_dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAPPAV models.")
    parser.add_argument("--perturber_checkpoint", type=str, required=True, help="Path to Perturber checkpoint.")
    parser.add_argument("--verifier_checkpoint", type=str, required=True, help="Path to Verifier checkpoint.")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Directory of processed papers.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of eval episodes.")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval", help="Output directory for reports.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    from arappav.data.dataset import load_corpus_split
    from arappav.eval.evaluate import run_evaluation
    from arappav.models.perturber import PerturberModel
    from arappav.models.verifier import VerifierModel
    from arappav.utils.logging import setup_logging
    from arappav.utils.seeding import set_seed

    setup_logging()
    set_seed(args.seed)

    # Load models
    perturber = PerturberModel(
        model_name_or_path=args.perturber_checkpoint,
        use_vllm=False,
    )
    verifier = VerifierModel(
        model_name_or_path=args.verifier_checkpoint,
        use_vllm=False,
    )

    # Load eval data
    dataset_dict = load_corpus_split(
        processed_dir=args.processed_dir,
        seed=args.seed,
    )

    logger.info(f"Running evaluation on {args.num_episodes} episodes...")
    metrics = run_evaluation(
        perturber_model=perturber,
        verifier_model=verifier,
        eval_dataset=dataset_dict["test"],
        num_episodes=args.num_episodes,
        qualitative_samples=5,
        output_dir=args.output_dir,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
