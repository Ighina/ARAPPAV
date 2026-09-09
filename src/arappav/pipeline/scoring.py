"""Reward computation and findings, owned by the orchestrator (spec 10, 13, 14).

The *semantics* are unchanged from the pre-refactor pipeline: parsing goes
through `parse_and_backoff`, scoring through `compute_rewards`, and error units
through the same matcher. What moved is *ownership* — this runs in Python, in
process, rather than inside a skill, so the numbers a round reports are
computed from persisted artefacts and can be recomputed from them later.

Findings (spec 14) are derived mechanically here. No agent summarises a round,
which keeps the policy-update inputs reproducible and keeps a summariser from
quietly widening what the next round's agents get to see.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from arappav.errors.schema_math import (
    MathPerturberOutput,
    validate_math_verifier_output,
)
from arappav.models.perturber import parse_and_backoff
from arappav.reward.matcher import group_errors_into_units
from arappav.reward.reward_fns import compute_rewards
from arappav.utils.parsing import extract_first_json_object, strip_json_fences

REWARD_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "reward" / "reward.yaml"


def load_reward_config(path: Path = REWARD_CONFIG) -> dict:
    try:
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)["reward"]
    except Exception:
        from arappav.reward.reward_fns import _default_config
        return _default_config()


def parse_perturbation(raw: str, k: int, solution: str):
    """Parse a Perturber reply exactly as the pre-refactor pipeline did.

    `original_text=solution` is what confines the mechanical backoff to the
    solution: it can only locate and rewrite spans of the solution, never of
    the problem.
    """
    return parse_and_backoff(raw, k, mode="math", original_text=solution)


def parse_verification(raw: str):
    data, err = extract_first_json_object(strip_json_fences(raw))
    if data is None:
        return None, err or "no JSON object found"
    return validate_math_verifier_output(data)


def score_episode(*, episode_id: str, k: int, perturbed: MathPerturberOutput | None,
                  failure_stage: str | None, failure_reason: str | None,
                  verifier_raw: str, config: dict,
                  history: list | None) -> dict:
    """Score one episode. Preserves the graded format-penalty behaviour."""
    if perturbed is None:
        penalty = (config.get("format_penalty", -10.0) if failure_stage == "json"
                   else config.get("format_penalty_soft", -5.0))
        return {
            "episode_id": episode_id, "k": k,
            "perturber_format_valid": False,
            "failure_stage": failure_stage,
            "format_violation_reason": failure_reason,
            "perturber_reward": penalty,
            "verifier_reward": None, "verifier_recall": None,
            "verifier_precision": None, "scored": False,
        }

    vout, verr = parse_verification(verifier_raw)
    claims = vout.claims if vout is not None else []
    reward = compute_rewards(
        ground_truth=perturbed.errors,
        verifier_claims=claims,
        perturbed_text=perturbed.perturbed_solution,
        k=k,
        config=config,
        perturber_format_valid=True,
        historical_perturbations=history or None,
        verifier_raw_output=verifier_raw,
    )
    rec = dataclasses.asdict(reward)
    rec.update({
        "episode_id": episode_id,
        "perturber_format_valid": True,
        "verifier_parse_error": verr,
        "scored": True,
        "num_error_units": len(group_errors_into_units(
            perturbed.errors, perturbed.perturbed_solution)),
        "errors": [
            {"error_id": e.error_id, "error_type": e.error_type.value,
             "original_text": e.original_text, "injected_text": e.injected_text,
             "rationale": e.rationale}
            for e in perturbed.errors
        ],
        "claims": [
            {"step_index": c.step_index, "quoted_text": c.quoted_text,
             "explanation": c.explanation,
             "error_type": c.error_type.value if c.error_type else None}
            for c in claims
        ],
    })
    return rec


def aggregate(scores: list[dict]) -> dict:
    """Round-level metrics. Same quantities the pre-refactor summary reported."""
    n = len(scores)
    valid = [s for s in scores if s.get("perturber_format_valid")]
    scored = [s for s in valid if s.get("scored")]

    def mean(key, rows):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "num_episodes": n,
        "num_format_valid": len(valid),
        "format_valid_rate": round(len(valid) / n, 4) if n else 0.0,
        "mean_perturber_reward": mean("perturber_reward", scores),
        "mean_perturber_reward_valid_only": mean("perturber_reward", scored),
        "mean_verifier_reward": mean("verifier_reward", scored),
        "mean_verifier_recall": mean("verifier_recall", scored),
        "mean_verifier_precision": mean("verifier_precision", scored),
        "mean_units_per_episode": mean("num_error_units", scored),
        "total_duplicate_penalty": round(sum(s.get("duplicate_penalty", 0) or 0 for s in scored), 4),
        "total_spam_penalty": round(sum(s.get("spam_penalty", 0) or 0 for s in scored), 4),
        "total_repetition_penalty": round(sum(s.get("repetition_penalty", 0) or 0 for s in scored), 4),
        "total_phantom_penalty": round(sum(s.get("phantom_penalty", 0) or 0 for s in scored), 4),
    }


def build_findings(round_index: int, scores: list[dict], metrics: dict) -> dict:
    """The explicit, bounded input to the next round's policy updates (spec 14).

    Deliberately *derived*, not raw: the update skills receive this object and
    nothing else, so what they can learn from is a fixed, reviewable shape
    rather than whatever they might find by reading the run directory.
    """
    undetected, detected, false_positives, format_failures = [], [], [], []
    for s in scores:
        if not s.get("perturber_format_valid"):
            format_failures.append({
                "episode_id": s["episode_id"],
                "failure_stage": s.get("failure_stage"),
                "reason": (s.get("format_violation_reason") or "")[:300],
            })
            continue
        for m in s.get("match_details", []) or []:
            entry = {"episode_id": s["episode_id"], "error_id": m.get("error_id"),
                     "error_type": m.get("error_type"),
                     "best_overlap": round(m.get("best_overlap", 0.0), 3)}
            (detected if m.get("best_claim_idx") is not None else undetected).append(entry)
        matched = {m.get("best_claim_idx") for m in (s.get("match_details") or [])}
        for i, c in enumerate(s.get("claims", [])):
            if i not in matched:
                false_positives.append({
                    "episode_id": s["episode_id"],
                    "quoted_text": c["quoted_text"][:200],
                    "explanation": (c["explanation"] or "")[:300],
                })

    by_type: dict[str, dict] = {}
    for e in detected + undetected:
        t = e.get("error_type") or "unknown"
        row = by_type.setdefault(t, {"injected": 0, "detected": 0})
        row["injected"] += 1
    for e in detected:
        by_type[e.get("error_type") or "unknown"]["detected"] += 1
    for t, row in by_type.items():
        row["detection_rate"] = round(row["detected"] / row["injected"], 3) if row["injected"] else 0.0

    return {
        "round": round_index,
        "metrics": metrics,
        "error_type_detection": by_type,
        "format_failures": format_failures,
        "undetected_errors": undetected,
        "detected_errors": detected,
        "verifier_false_positives": false_positives,
        "unit_collapse": [
            {"episode_id": s["episode_id"], "k": s["k"],
             "num_error_units": s.get("num_error_units")}
            for s in scores
            if s.get("scored") and s.get("num_error_units", s["k"]) < s["k"]
        ],
    }


# ---------------------------------------------------------------------------
# Text-grounded evidence for the policy updaters
# ---------------------------------------------------------------------------
#
# The findings above are counts and identifiers. An updater reading
# `{"error_id": "err_002", "error_type": "whole_number_bias",
#   "best_overlap": 0.0}` is told an error was missed but not what the error
# *was*, and is then asked to write a better verification procedure. The
# functions here carry the actual text.

#: What each number means and which direction is good. Shipped with the
#: evidence because a bare scalar is not interpretable: an updater seeing
#: "0.42" cannot know whether that is good, for whom, or against what.
METRIC_LEGEND = {
    "mean_verifier_reward": "verifier F1 in [0,1]; higher is better for the verifier",
    "mean_verifier_recall": "share of distinct error units the verifier found; higher is better for the verifier",
    "mean_verifier_precision": "share of verifier claims that matched a real error; higher is better for the verifier",
    "mean_perturber_reward_valid_only": "1 - verifier recall, plus penalties; higher is better for the perturber",
    "format_valid_rate": "share of episodes whose perturbation parsed; below 1.0 means output was malformed and scored -5 or -10",
    "mean_units_per_episode": "distinct error units after causally-linked errors were merged; below k means declarations collapsed into one mistake",
    "closest_overlap": "how nearly the closest verifier claim aligned with the error span; 0 means nothing was claimed near it, and >=0.5 would have matched",
}


def _clip(s: str | None, n: int = 400) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " …"


def episode_evidence(scores: list[dict], role: str) -> list[dict]:
    """Per-episode evidence *with text*, sliced for one role.

    The perturber sees its own rationales; the verifier does not. A verifier
    tuned on the adversary's stated intent learns this adversary rather than
    verification, which is the overfitting its own updater is warned against.
    """
    out = []
    for s in scores:
        if not s.get("perturber_format_valid"):
            out.append({"episode_id": s["episode_id"], "format_valid": False,
                        "failure_stage": s.get("failure_stage"),
                        "reason": _clip(s.get("format_violation_reason"), 300)})
            continue
        claims = s.get("claims") or []
        errors = s.get("errors") or []
        matched_claim_idx = {m.get("best_claim_idx") for m in (s.get("match_details") or [])}
        items = []
        for m, e in zip(s.get("match_details") or [], errors):
            rec = {
                "error_id": e.get("error_id"),
                "original_text": _clip(e.get("original_text")),
                "injected_text": _clip(e.get("injected_text")),
                "detected": m.get("best_claim_idx") is not None,
                "closest_overlap": m.get("closest_overlap", 0.0),
            }
            idx = m.get("best_claim_idx")
            if idx is None:
                idx = m.get("closest_claim_idx")
            if idx is not None and idx < len(claims):
                rec["closest_claim"] = _clip(claims[idx].get("quoted_text"))
                rec["closest_claim_explanation"] = _clip(
                    claims[idx].get("explanation"), 300)
            if role == "perturb":
                rec["rationale"] = _clip(e.get("rationale"), 300)
            items.append(rec)
        out.append({
            "episode_id": s["episode_id"],
            "format_valid": True,
            "k": s.get("k"),
            "num_error_units": s.get("num_error_units"),
            "verifier_recall": s.get("verifier_recall"),
            "verifier_precision": s.get("verifier_precision"),
            "errors": items,
            "unmatched_claims": [
                {"quoted_text": _clip(c.get("quoted_text")),
                 "explanation": _clip(c.get("explanation"), 300)}
                for i, c in enumerate(claims) if i not in matched_claim_idx
            ],
        })
    return out
