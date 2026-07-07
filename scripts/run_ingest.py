#!/usr/bin/env python
"""Ingest papers from a local directory and save processed versions.

Usage:
    python scripts/run_ingest.py --raw_dir data/raw --output_dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arappav.data.ingest import ingest_local, save_processed
from arappav.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Ingest papers for ARAPPAV.")
    parser.add_argument("--raw_dir", type=str, default="./data/raw", help="Directory of raw text/md files.")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Output directory for processed JSON.")
    parser.add_argument("--max_papers", type=int, default=None, help="Limit number of papers (for debugging).")
    args = parser.parse_args()

    logger = setup_logging()

    logger.info(f"Ingesting papers from {args.raw_dir}")
    papers = ingest_local(args.raw_dir, max_papers=args.max_papers)

    if not papers:
        logger.warning("No papers found. Place .txt or .md files in the raw directory.")
        return

    paths = save_processed(papers, args.output_dir)
    logger.info(f"Done. {len(paths)} papers saved to {args.output_dir}")


if __name__ == "__main__":
    main()
