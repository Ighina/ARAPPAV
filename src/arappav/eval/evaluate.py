"""Held-out evaluation harness.

Runs evaluation on a fixed validation/test split of papers (or math problems)
not used in self-play training. This is the **key signal** for detecting
training collapse or reward hacking.

Tracks over rounds:
- Verifier recall/precision/F1.
- Perturber trick rate.
- Per-error-category breakdown.
- Format-compliance rate.

Optionally generates a qualitative report with N sample episodes.

Supports both ``paper`` and ``math`` modes, which differ in dataset schema
and model.generate() signatures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from arappav.eval.metrics import (
    EvalMetrics,
    aggregate_metrics,
    compute_episode_metrics,
    format_metrics_report,
)

logger = logging.getLogger(__name__)


def run_evaluation(
    perturber_model,
    verifier_model,
    eval_dataset,
    reward_config: dict | None = None,
    num_episodes: int | None = None,
    k_sampler: callable | None = None,
    qualitative_samples: int = 3,
    output_dir: str | Path | None = None,
    mode: str = "paper",
) -> EvalMetrics:
    """Run held-out evaluation.

    Args:
        perturber_model: Perturber model wrapper (``PerturberModel``).
        verifier_model: Verifier model wrapper (``VerifierModel``).
        eval_dataset: HF Dataset (val or test split).
        reward_config: Reward configuration dict.
        num_episodes: Number of evaluation episodes. If None, uses the full dataset.
        k_sampler: Callable ``() -> int`` for sampling k.
        qualitative_samples: Number of episodes to dump as markdown for review.
        output_dir: If set, save metrics and qualitative report to this directory.
        mode: ``"paper"`` or ``"math"`` — controls dataset column access and
            model.generate() argument wiring.

    Returns:
        Aggregated ``EvalMetrics``.
    """
    import random

    if k_sampler is None:
        def k_sampler():
            return random.randint(2, 6)

    episodes = []
    qualitative_episodes = []

    max_ep = num_episodes if num_episodes else len(eval_dataset)

    for i in range(min(max_ep, len(eval_dataset))):
        row = eval_dataset[i]

        # --- Dataset column access (mode-dependent) ---
        if mode == "math":
            problem = row["problem"]
            solution = row["solution"]
            item_id = f"{row.get('topic', 'math')}_{row.get('type', 'unknown')}_{i}"
            # For perturber: pass problem as "text" + solution separately
            perturber_input = problem
            perturber_solution = solution
            verifier_input = None  # will be set after perturber generation
        else:
            paper_text = row["text"]
            paper_id = row["id"]
            item_id = paper_id
            perturber_input = paper_text
            perturber_solution = None
            verifier_input = None

        k = k_sampler()

        # --- Perturber generation ---
        perturber_valid = True
        perturber_reason = None
        try:
            perturber_out, p_err = perturber_model.generate(
                perturber_input, k, solution=perturber_solution,
            )
            if perturber_out is None:
                perturber_valid = False
                perturber_reason = p_err
        except Exception as e:
            perturber_valid = False
            perturber_reason = str(e)
            perturber_out = None

        # --- Normalise perturbed text (mode-dependent attribute name) ---
        if perturber_valid and perturber_out is not None:
            if mode == "math":
                perturbed_text = perturber_out.perturbed_solution
                verifier_input = perturber_out.perturbed_solution
                verifier_problem = problem
            else:
                perturbed_text = perturber_out.perturbed_text
                verifier_input = perturber_out.perturbed_text
                verifier_problem = None
        else:
            perturbed_text = perturber_input if mode == "paper" else solution
            verifier_problem = problem if mode == "math" else None

        # --- Verifier generation ---
        verifier_valid = True
        verifier_claims = []
        if perturber_valid and perturber_out is not None:
            try:
                results = verifier_model.generate(
                    verifier_input, problem=verifier_problem, n_completions=1,
                )
                _, v_out, v_err = results[0] if results else ("", None, "no output")
                if v_out is None:
                    verifier_valid = False
                else:
                    verifier_claims = v_out.claims
            except Exception as e:
                verifier_valid = False
        else:
            verifier_valid = False

        ground_truth = perturber_out.errors if perturber_out else []

        ep_metrics = compute_episode_metrics(
            ground_truth=ground_truth,
            verifier_claims=verifier_claims,
            perturbed_text=perturbed_text,
            k=k,
            paper_id=item_id,
            chunk_id=f"{item_id}_eval_{i}",
            perturber_format_valid=perturber_valid,
            verifier_format_valid=verifier_valid,
            reward_config=reward_config,
        )
        episodes.append(ep_metrics)

        # Collect qualitative samples
        if i < qualitative_samples:
            qualitative_episodes.append(
                _build_qualitative_sample(
                    item_id=item_id,
                    original_text=perturber_input,
                    perturbed_text=perturbed_text,
                    k=k,
                    ground_truth=ground_truth,
                    verifier_claims=verifier_claims,
                    ep_metrics=ep_metrics,
                    mode=mode,
                )
            )

    metrics = aggregate_metrics(episodes)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_path = output_dir / "eval_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "num_episodes": metrics.num_episodes,
                    "verifier_recall_mean": metrics.verifier_recall_mean,
                    "verifier_precision_mean": metrics.verifier_precision_mean,
                    "verifier_f1_mean": metrics.verifier_f1_mean,
                    "perturber_reward_mean": metrics.perturber_reward_mean,
                    "perturber_trick_rate_mean": metrics.perturber_trick_rate_mean,
                    "format_rate_perturber": metrics.num_format_valid_perturber / max(1, metrics.num_episodes),
                    "format_rate_verifier": metrics.num_format_valid_verifier / max(1, metrics.num_episodes),
                    "per_category": {
                        k: {"recall": v.recall, "total": v.total_errors}
                        for k, v in metrics.per_category.items()
                    },
                },
                indent=2,
            )
        )

        # Generate qualitative markdown report
        report = _generate_qualitative_report(qualitative_episodes, metrics)
        report_path = output_dir / "qualitative_report.md"
        report_path.write_text(report)
        logger.info(f"Evaluation report saved to {output_dir}")

    logger.info(format_metrics_report(metrics))
    return metrics


def _build_qualitative_sample(
    item_id: str,
    original_text: str,
    perturbed_text: str,
    k: int,
    ground_truth: list,
    verifier_claims: list,
    ep_metrics,
    mode: str = "paper",
) -> dict:
    """Build a dict for one qualitative evaluation sample."""
    sample = {
        "paper_id": item_id,
        "original_text": original_text,
        "perturbed_text": perturbed_text,
        "k": k,
        "ground_truth": [
            {
                "error_id": e.error_id,
                "type": e.error_type.value,
                "injected": e.injected_text,
                "rationale": e.rationale,
            }
            for e in ground_truth
        ],
        "verifier_claims": [
            {
                "quoted_text": c.quoted_text,
                "explanation": c.explanation,
            }
            for c in verifier_claims
        ],
        "recall": ep_metrics.verifier_recall,
        "precision": ep_metrics.verifier_precision,
        "mode": mode,
    }
    return sample


def _generate_qualitative_report(
    samples: list[dict], metrics: EvalMetrics
) -> str:
    """Generate a human-readable markdown report with sample episodes.

    Args:
        samples: List of per-episode dicts with original/perturbed/claims.
        metrics: Aggregated metrics.

    Returns:
        Markdown string.
    """
    lines = [
        "# ARAPPAV Qualitative Evaluation Report",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Verifier Recall | {metrics.verifier_recall_mean:.4f} |",
        f"| Verifier Precision | {metrics.verifier_precision_mean:.4f} |",
        f"| Verifier F1 | {metrics.verifier_f1_mean:.4f} |",
        f"| Perturber Trick Rate | {metrics.perturber_trick_rate_mean:.4f} |",
        f"| Perturber Format Rate | {metrics.num_format_valid_perturber / max(1, metrics.num_episodes):.2%} |",
        f"| Verifier Format Rate | {metrics.num_format_valid_verifier / max(1, metrics.num_episodes):.2%} |",
        "",
        "---",
        "",
    ]

    for i, sample in enumerate(samples):
        lines.append(f"## Sample {i + 1}: `{sample['paper_id']}` (k={sample['k']})")
        lines.append("")
        lines.append(f"**Verifier Recall:** {sample['recall']:.4f} | **Precision:** {sample['precision']:.4f}")
        lines.append("")

        lines.append("### Ground Truth Errors")
        lines.append("")
        for err in sample["ground_truth"]:
            lines.append(f"- **[{err['type']}]** `{err['error_id']}`: \"{err['injected'][:200]}\"")
            lines.append(f"  - *Rationale:* {err['rationale'][:300]}")
        lines.append("")

        lines.append("### Verifier Claims")
        lines.append("")
        if sample["verifier_claims"]:
            for claim in sample["verifier_claims"]:
                lines.append(f"- \"{claim['quoted_text'][:200]}\"")
                lines.append(f"  - *Explanation:* {claim['explanation'][:300]}")
        else:
            lines.append("*(No claims made)*")
        lines.append("")

        lines.append("### Perturbed Text (first 500 chars)")
        lines.append("")
        lines.append("```")
        lines.append(sample["perturbed_text"][:500])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
