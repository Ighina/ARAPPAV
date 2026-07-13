#!/usr/bin/env python
"""Debug utility: run a single P→V episode and print everything to stdout.

Supports both paper mode and math mode (via --mode math).

Usage (paper mode):
    python scripts/run_single_rollout.py --perturber_model Qwen/Qwen2.5-3B-Instruct \\
                                         --verifier_model Qwen/Qwen2.5-7B-Instruct \\
                                         --text "The model achieves 95.3% accuracy."

Usage (math mode):
    python scripts/run_single_rollout.py --mode math \\
                                         --perturber_model Qwen/Qwen2.5-3B-Instruct \\
                                         --verifier_model Qwen/Qwen2.5-7B-Instruct \\
                                         --k 2
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
    parser = argparse.ArgumentParser(description="Run a single P→V rollout for debugging.")
    parser.add_argument("--mode", type=str, default="paper", choices=["paper", "math"],
                        help="Operating mode.")
    parser.add_argument("--perturber_model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Perturber model ID or path.")
    parser.add_argument("--verifier_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Verifier model ID or path.")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to perturb. In math mode, this is the problem statement.")
    parser.add_argument("--solution", type=str, default=None,
                        help="Correct solution (math mode only).")
    parser.add_argument("--k", type=int, default=3, help="Number of errors to inject.")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for generation.")
    parser.add_argument("--topic", type=str, default="algebra",
                        help="Math topic to sample from (math mode, if --text not provided).")
    parser.add_argument("--level", type=str, default=None,
                        help="Math difficulty level filter (math mode).")
    args = parser.parse_args()

    from arappav.models.perturber import PerturberModel, build_perturber_prompt, build_math_perturber_prompt
    from arappav.models.verifier import VerifierModel, build_verifier_prompt, build_math_verifier_prompt
    from arappav.reward.reward_fns import compute_rewards, compute_math_episode_rewards
    from arappav.utils.logging import setup_logging

    setup_logging()

    # --- Math mode: sample a problem if not provided ---
    if args.mode == "math" and args.text is None:
        from arappav.data.ingest_math import load_math_dataset
        ds = load_math_dataset(topics=[args.topic], split="train", max_examples_per_topic=10)
        import random
        idx = random.randint(0, len(ds) - 1)
        row = ds[idx]
        args.text = row["problem"]
        args.solution = row["solution"]
        print(f"\n  Sampled math problem (topic={row['topic']}, level={row['level']}):")
        print(f"  Problem: {args.text[:300]}...")
        print(f"  Solution: {args.solution[:300]}...\n")

    if args.mode == "paper" and args.text is None:
        args.text = (
            "We train a transformer-based model on the WikiText-103 dataset. "
            "The model achieves 95.3% accuracy on the test set, outperforming "
            "the previous state-of-the-art by 2.1 percentage points. We use the "
            "Adam optimizer with a learning rate of 0.001 and a batch size of 32. "
            "As shown by Smith et al. (2019), this architecture generalizes well "
            "to out-of-distribution data."
        )

    print("=" * 70)
    print(f"  ARAPPAV — Single Rollout Debug  [mode={args.mode}]")
    print("=" * 70)
    print(f"\n  k = {args.k}")

    if args.mode == "math":
        print(f"\n  Problem:\n  {args.text[:500]}...")
        print(f"\n  Correct solution:\n  {args.solution[:500]}...\n")
    else:
        print(f"\n  Original text:\n  {args.text}\n")

    # --- Phase 1: Perturber ---
    print("-" * 70)
    print("  PERTURBER PROMPT (first 800 chars):")
    print("-" * 70)
    if args.mode == "math":
        prompt = build_math_perturber_prompt(args.text, args.solution, args.k)
    else:
        prompt = build_perturber_prompt(args.text, args.k)
    print(prompt[:800] + "...\n" if len(prompt) > 800 else prompt + "\n")

    print("  Generating perturbation...")
    perturber = PerturberModel(
        model_name_or_path=args.perturber_model,
        mode=args.mode,
        use_vllm=args.use_vllm,
    )
    perturber_out, p_err = perturber.generate(
        args.text, args.k, solution=args.solution,
    )

    if perturber_out is None:
        print(f"\n  ❌ Perturber FAILED: {p_err}")
        return

    n_errors = len(perturber_out.errors)
    print(f"\n  ✅ Perturber generated {n_errors} errors:\n")
    for err in perturber_out.errors:
        step_info = f"step {err.step_index}" if hasattr(err, 'step_index') else err.location
        print(f"    [{err.error_type.value}] {err.error_id} @ {step_info}")
        print(f"      Original:  {err.original_text[:120]}")
        print(f"      Injected:  {err.injected_text[:120]}")
        print(f"      Rationale: {err.rationale[:200]}")
        print()

    result_text = getattr(perturber_out, 'perturbed_solution', None) or perturber_out.perturbed_text
    print(f"  Perturbed output (first 500 chars):\n  {result_text[:500]}...\n")

    # --- Phase 2: Verifier ---
    print("-" * 70)
    print("  VERIFIER PROMPT (first 800 chars):")
    print("-" * 70)
    if args.mode == "math":
        v_prompt = build_math_verifier_prompt(args.text, result_text)
    else:
        v_prompt = build_verifier_prompt(result_text)
    print(v_prompt[:800] + "...\n" if len(v_prompt) > 800 else v_prompt + "\n")

    print("  Generating verifier response...")
    verifier = VerifierModel(
        model_name_or_path=args.verifier_model,
        mode=args.mode,
        use_vllm=args.use_vllm,
    )
    verifier_results = verifier.generate(
        result_text, problem=args.text if args.mode == "math" else None, n_completions=1,
    )
    raw_verifier, verifier_out, v_err = verifier_results[0]

    if verifier_out is None:
        print(f"\n  ❌ Verifier FAILED: {v_err}")
    else:
        print(f"\n  ✅ Verifier made {len(verifier_out.claims)} claims:\n")
        for i, claim in enumerate(verifier_out.claims):
            print(f"    Claim {i + 1}:")
            print(f"      Quoted:     {claim.quoted_text[:150]}")
            print(f"      Explanation: {claim.explanation[:200]}")
            print()

    # --- Phase 3: Reward ---
    print("-" * 70)
    print("  REWARD COMPUTATION:")
    print("-" * 70)
    reward_out = compute_rewards(
        ground_truth=perturber_out.errors,
        verifier_claims=verifier_out.claims if verifier_out else [],
        perturbed_text=result_text,
        k=args.k,
        verifier_raw_output=raw_verifier,
    )

    print(f"  Verifier recall:    {reward_out.verifier_recall:.4f}")
    print(f"  Verifier precision: {reward_out.verifier_precision:.4f}")
    print(f"  Verifier F1:        {reward_out.verifier_f_beta:.4f}")
    print(f"  Verifier reward:    {reward_out.verifier_reward:.4f}")
    print(f"  Perturber reward:   {reward_out.perturber_reward:.4f}")
    print(f"  Format penalty:     {reward_out.format_penalty_applied}")
    print(f"  Spam penalty:       {reward_out.spam_penalty:.4f}")
    print(f"  Repetition penalty: {reward_out.repetition_penalty:.4f}")
    print(f"  k effective:        {reward_out.k_effective}/{reward_out.k}")
    print()

    # Per-error match details — distinguishes matcher failures (claim exists
    # but low overlap) from verifier failures (no plausible claim at all).
    print("  MATCH DETAILS:")
    for detail in reward_out.match_details:
        matched = detail["best_claim_idx"] is not None
        status = f"claim {detail['best_claim_idx']}" if matched else "UNMATCHED"
        print(f"    {detail['error_id']} [{detail['error_type']}] -> {status} "
              f"(overlap={detail['best_overlap']:.3f})")
        if not matched and detail["all_overlaps"]:
            for quoted, score in detail["all_overlaps"].items():
                print(f"      candidate ({score:.3f}): {quoted}")
    print()

    # --- Summary ---
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    summary = {
        "mode": args.mode,
        "k_requested": args.k,
        "k_actual": n_errors,
        "perturber_valid": perturber_out is not None,
        "verifier_valid": verifier_out is not None,
        "verifier_recall": reward_out.verifier_recall,
        "verifier_precision": reward_out.verifier_precision,
        "verifier_f1": reward_out.verifier_f_beta,
        "perturber_reward": reward_out.perturber_reward,
        "verifier_reward": reward_out.verifier_reward,
    }
    print(json.dumps(summary, indent=2))
    print()


if __name__ == "__main__":
    main()
