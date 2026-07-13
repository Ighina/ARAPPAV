"""Preference pair builder for DPO Verifier training.

Constructs DPO preference pairs from self-play rollouts:
- For each perturbed text, sample multiple Verifier completions (n > 1).
- Compute r_V for each completion against the ground truth.
- Pair: higher-r_V response = "chosen", lower-r_V response = "rejected".
- Filter out pairs where the reward gap is negligible (noise).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from datasets import Dataset

from arappav.errors.schema import InjectedError, VerifierClaim, VerifierOutput
from arappav.reward.reward_fns import compute_rewards

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single preference pair for DPO training."""

    prompt: str
    chosen: str  # raw response text
    rejected: str  # raw response text
    chosen_reward: float
    rejected_reward: float
    paper_id: str
    perturbed_text: str
    ground_truth: list[InjectedError]
    k: int


def build_preference_pairs(
    rollouts: list[dict],
    reward_config: dict,
    min_reward_gap: float = 0.05,
    max_pairs: int | None = None,
) -> list[PreferencePair]:
    """Build DPO preference pairs from a batch of self-play rollouts.

    Each rollout dict must contain:
    - ``"prompt"``: the verifier prompt string.
    - ``"responses"``: list of dicts, each with keys ``"raw_text"`` (str),
      ``"parsed"`` (dict | None with ``"claims"`` key), ``"parse_error"``
      (str | None).
    - ``"perturbed_text"``: the perturbed paper/solution text.
    - ``"ground_truth"``: list of InjectedError objects.
    - ``"k"``: expected error count.
    - ``"paper_id"``: paper identifier.
    - ``"mode"``: ``"paper"`` or ``"math"`` (used to reconstruct verifier output).

    Args:
        rollouts: List of rollout dicts as described above.
        reward_config: Reward configuration.
        min_reward_gap: Minimum reward difference to form a pair (filter noise).
        max_pairs: If set, randomly subsample to this many pairs.

    Returns:
        List of PreferencePair objects.
    """
    pairs = []

    for rollout in rollouts:
        responses = rollout.get("responses", [])
        if len(responses) < 2:
            continue

        mode = rollout.get("mode", "paper")

        # Score each response
        scored = []
        for r in responses:
            raw_text = r["raw_text"]

            # Reconstruct VerifierOutput from the serialised ``parsed`` dict
            verifier_out = _parsed_to_verifier_output(r.get("parsed"), mode)

            if verifier_out is not None:
                reward_out = compute_rewards(
                    ground_truth=rollout["ground_truth"],
                    verifier_claims=verifier_out.claims,
                    perturbed_text=rollout["perturbed_text"],
                    k=rollout["k"],
                    config=reward_config,
                    perturber_format_valid=True,
                    verifier_raw_output=raw_text,
                )
                r_v = reward_out.verifier_reward
            else:
                # Parse failure → treat as minimum reward
                r_v = -10.0

            scored.append((raw_text, verifier_out, r_v))

        # Sort by reward (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        # Build pairs from best vs worst within the group
        # Strategy: pair each adjacent pair that meets the gap threshold,
        # and also pair (best, worst) to maximise contrast.
        best_text, _, best_r = scored[0]
        worst_text, _, worst_r = scored[-1]

        if best_r - worst_r >= min_reward_gap:
            pairs.append(
                PreferencePair(
                    prompt=rollout["prompt"],
                    chosen=best_text,
                    rejected=worst_text,
                    chosen_reward=best_r,
                    rejected_reward=worst_r,
                    paper_id=rollout["paper_id"],
                    perturbed_text=rollout["perturbed_text"],
                    ground_truth=rollout["ground_truth"],
                    k=rollout["k"],
                )
            )

        # Also pair adjacent positions with sufficient gap
        for i in range(len(scored) - 1):
            _, _, r_high = scored[i]
            _, _, r_low = scored[i + 1]
            if r_high - r_low >= min_reward_gap:
                pairs.append(
                    PreferencePair(
                        prompt=rollout["prompt"],
                        chosen=scored[i][0],
                        rejected=scored[i + 1][0],
                        chosen_reward=r_high,
                        rejected_reward=r_low,
                        paper_id=rollout["paper_id"],
                        perturbed_text=rollout["perturbed_text"],
                        ground_truth=rollout["ground_truth"],
                        k=rollout["k"],
                    )
                )

    logger.info(f"Built {len(pairs)} preference pairs from {len(rollouts)} rollouts")

    if max_pairs and len(pairs) > max_pairs:
        import random

        pairs = random.sample(pairs, max_pairs)
        logger.info(f"Subsampled to {len(pairs)} pairs")

    return pairs


def _parsed_to_verifier_output(
    parsed: dict[str, Any] | None, mode: str = "paper"
) -> VerifierOutput | None:
    """Reconstruct a ``VerifierOutput`` (or ``MathVerifierOutput``) from a
    serialised ``parsed`` dict.

    The dict has the form ``{"claims": [{"step_index": ..., "quoted_text": ...}, ...]}``
    as produced by ``collect_verifier_rollouts`` in the self-play loop.

    Args:
        parsed: The serialised dict, or ``None`` if parsing failed.
        mode: ``"paper"`` or ``"math"``.

    Returns:
        A ``VerifierOutput`` / ``MathVerifierOutput``, or ``None`` if
        reconstruction fails.
    """
    if parsed is None:
        return None

    claims_list = parsed.get("claims", [])

    if mode == "math":
        from arappav.errors.schema_math import MathVerifierClaim, MathVerifierOutput

        try:
            return MathVerifierOutput(
                claims=[MathVerifierClaim(**c) for c in claims_list]
            )
        except Exception:
            logger.warning("Failed to reconstruct MathVerifierOutput from parsed dict.")
            return None
    else:
        try:
            return VerifierOutput(
                claims=[VerifierClaim(**c) for c in claims_list]
            )
        except Exception:
            logger.warning("Failed to reconstruct VerifierOutput from parsed dict.")
            return None


def pairs_to_dataset(pairs: list[PreferencePair]) -> Dataset:
    """Convert preference pairs to a Hugging Face Dataset for DPOTrainer.

    Args:
        pairs: List of PreferencePair objects.

    Returns:
        HF Dataset with columns: ``prompt``, ``chosen``, ``rejected``.
    """
    return Dataset.from_dict(
        {
            "prompt": [p.prompt for p in pairs],
            "chosen": [p.chosen for p in pairs],
            "rejected": [p.rejected for p in pairs],
        }
    )
