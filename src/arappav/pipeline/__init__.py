"""Deterministic Python orchestration for the ARAPPAV self-play pipeline.

This package replaces the `selfplay` *skill* as the owner of the experiment.
Python decides what happens; skills are invoked as pure functions over
explicitly constructed inputs. See `docs/REFACTOR.md`.
"""
