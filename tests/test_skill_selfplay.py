"""Smoke tests for the skill-tuning self-play harness (``scripts/skill_selfplay.py``).

The loop in ``.claude/skills/selfplay`` delegates every mechanical step to this
script — problem sampling, Perturber validation, the leak-free Verifier input,
scoring, and the invariant check on a new skill version. These tests pin that
contract on a tiny synthetic round.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "skill_selfplay", REPO_ROOT / "scripts" / "skill_selfplay.py"
)
skill_selfplay = importlib.util.module_from_spec(_spec)
sys.modules["skill_selfplay"] = skill_selfplay
_spec.loader.exec_module(skill_selfplay)


PROBLEM = "If the ratio of $b$ to $a$ is 3, then what is the value of $a$ when $b=12-5a$?"
SOLUTION = (
    "The given ratio tells us that $\\frac{b}{a}=3$ or that $b=3a$. We find "
    "\\begin{align*}\n3a&=12-5a \\\\\n\\Rightarrow \\quad 8a&=12 \\\\\n"
    "\\Rightarrow \\quad a &= 12/8.\n\\end{align*}"
)


def _run(*argv: str) -> int:
    old = sys.argv
    sys.argv = ["skill_selfplay.py", *argv]
    try:
        return skill_selfplay.main()
    finally:
        sys.argv = old


@pytest.fixture()
def round_dir(tmp_path: Path) -> Path:
    """A scaffolded round 1 with two episodes drawn from a local problems file."""
    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        "\n".join(
            json.dumps({"source_id": f"p{i}", "problem": PROBLEM, "solution": SOLUTION})
            for i in range(2)
        )
    )
    assert _run(
        "--root", str(tmp_path / "rollouts"), "init", "--round", "1",
        "--episodes", "2", "--k", "2", "--problems-file", str(problems),
    ) == 0
    return tmp_path / "rollouts" / "round_01"


def _perturbation() -> str:
    """A valid k=2 perturbation with two independent errors."""
    perturbed = SOLUTION.replace("8a&=12", "-2a&=12").replace("a &= 12/8", "a &= 8/12")
    return json.dumps(
        {
            "perturbed_solution": perturbed,
            "errors": [
                {
                    "error_id": "err_001", "step_index": 1,
                    "original_text": "8a&=12", "injected_text": "-2a&=12",
                    "error_type": "wrong_operation",
                    "rationale": "3a + 5a = 8a, not -2a.",
                },
                {
                    "error_id": "err_002", "step_index": 2,
                    "original_text": "a &= 12/8", "injected_text": "a &= 8/12",
                    "error_type": "operand_swap",
                    "rationale": "Dividend and divisor swapped; should be 12/8.",
                },
            ],
        }
    )


def test_prepare_verify_hides_ground_truth_from_the_verifier(round_dir: Path):
    """The verifier inbox carries the problem and perturbed text — nothing else."""
    for edir in skill_selfplay.episode_dirs(round_dir):
        (edir / "perturb.json").write_text(_perturbation())

    assert _run("--root", str(round_dir.parent), "prepare-verify", "--round", "1") == 0

    inbox = sorted((round_dir / "verify_inbox").glob("*.json"))
    assert len(inbox) == 2
    payload = json.loads(inbox[0].read_text())
    assert set(payload) == {"episode_id", "problem", "solution_to_review"}
    assert "-2a&=12" in payload["solution_to_review"]
    blob = inbox[0].read_text()
    assert "err_001" not in blob and "rationale" not in blob and "original_text" not in blob


def test_format_failures_are_graded_and_yield_no_verifier_input(round_dir: Path):
    """Unparseable JSON scores -10, schema-invalid scores -5, neither reaches the verifier."""
    episodes = skill_selfplay.episode_dirs(round_dir)
    (episodes[0] / "perturb.json").write_text('{"perturbed_solution": "x", "errors": [ {"error_id"')
    phantom = json.loads(_perturbation())
    for err in phantom["errors"]:
        err["injected_text"] = err["original_text"]  # phantom: nothing changed
    (episodes[1] / "perturb.json").write_text(json.dumps(phantom))

    assert _run("--root", str(round_dir.parent), "prepare-verify", "--round", "1") == 0
    assert not (round_dir / "verify_inbox").exists() or not list(
        (round_dir / "verify_inbox").glob("*.json")
    )

    assert _run("--root", str(round_dir.parent), "score", "--round", "1") == 0
    rewards = {
        json.loads((e / "score.json").read_text())["failure_stage"]:
        json.loads((e / "score.json").read_text())["perturber_reward"]
        for e in episodes
    }
    assert rewards == {"json": -10.0, "schema": -5.0}


def test_scoring_and_summary_reflect_verifier_quality(round_dir: Path):
    """A verifier that quotes both errors exactly drives r_V to 1 and r_P to 0."""
    for edir in skill_selfplay.episode_dirs(round_dir):
        (edir / "perturb.json").write_text(_perturbation())
    assert _run("--root", str(round_dir.parent), "prepare-verify", "--round", "1") == 0

    outbox = round_dir / "verify_outbox"
    outbox.mkdir(exist_ok=True)
    claims = {
        "claims": [
            {"step_index": 1, "quoted_text": "-2a&=12",
             "explanation": "Combining 3a and 5a gives 8a, not -2a.",
             "error_type": "wrong_operation"},
            {"step_index": 2, "quoted_text": "a &= 8/12",
             "explanation": "Dividend and divisor swapped; a = 12/8.",
             "error_type": "operand_swap"},
        ]
    }
    for path in (round_dir / "verify_inbox").glob("*.json"):
        (outbox / path.name).write_text(json.dumps(claims))

    assert _run("--root", str(round_dir.parent), "score", "--round", "1") == 0
    assert _run("--root", str(round_dir.parent), "summarize", "--round", "1") == 0

    summary = json.loads((round_dir / "round_summary.json").read_text())
    assert summary["format_valid_rate"] == 1.0
    assert summary["metrics"]["mean_verifier_recall"] == 1.0
    assert summary["metrics"]["mean_perturber_reward"] == 0.0
    assert summary["learning_signal"]["undetected_errors"] == []
    # Both declarations survive as distinct units — they are independent mistakes.
    assert summary["metrics"]["mean_units_per_episode"] == 2.0


def test_check_skill_rejects_edits_to_the_invariant_contract(tmp_path: Path, monkeypatch):
    """A new version may rewrite the policy, never the contract it is scored against."""
    skills = tmp_path / "skills"
    (skills / "perturb-v1").mkdir(parents=True)
    (skills / "perturb-v2").mkdir(parents=True)
    v1 = (
        "---\nname: perturb-v1\n---\n\n## Output contract\nExactly k errors.\n\n"
        "## Policy\nP1 — be subtle.\n\n## Changelog\n- v1 — seed.\n"
    )
    (skills / "perturb-v1" / "SKILL.md").write_text(v1)
    monkeypatch.setattr(skill_selfplay, "SKILLS_DIR", skills)

    tuned = v1.replace("name: perturb-v1", "name: perturb-v2")
    tuned = tuned.replace("P1 — be subtle.", "P1 — be subtle and independent.")
    tuned += "- v2 — sharpened P1.\n"
    (skills / "perturb-v2" / "SKILL.md").write_text(tuned)
    assert _run("check-skill", "--role", "perturb", "--from-version", "1", "--to-version", "2") == 0

    (skills / "perturb-v2" / "SKILL.md").write_text(tuned.replace("Exactly k errors.", "Any number of errors."))
    assert _run("check-skill", "--role", "perturb", "--from-version", "1", "--to-version", "2") == 1
