#!/usr/bin/env python
"""Entry point for the full ARAPPAV self-play training loop.

Usage:
    python scripts/run_selfplay.py --config-name default
    python scripts/run_selfplay.py --config-name default self_play.num_rounds=10

Requires Hydra (hydra-core) for config management.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Ensure the package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../configs",
    config_name="default",
    version_base=None,
)
def main(cfg: DictConfig):
    """Run the self-play training loop."""
    from arappav.training.selfplay_loop import SelfPlayLoop
    from arappav.utils.logging import setup_logging

    setup_logging()

    logger.info("=" * 60)
    logger.info("ARAPPAV Self-Play Training Loop")
    logger.info(f"Perturber: {cfg.perturber.model_name_or_path}")
    logger.info(f"Verifier:  {cfg.verifier.model_name_or_path}")
    logger.info(f"Rounds:    {cfg.self_play.num_rounds}")
    logger.info("=" * 60)

    loop = SelfPlayLoop(cfg)
    loop.run()


if __name__ == "__main__":
    main()
