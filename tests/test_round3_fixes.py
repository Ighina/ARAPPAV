"""Regression tests for the Round 3 rollout-analysis fixes.

Round 3 found the Perturber exploiting the reward by **stacking** errors:
declaring a root mistake plus its downstream consequences (propagated boxed
answer, overlapping rewrites, near-duplicates, textual redundancy) as
separate errors, capping Verifier recall at 1/k. Covers:

- ``group_errors_into_units`` (unit-level recall collapse),
- the intra-episode anti-duplicate penalty,
- schema rejection of redundant restatements (double ``\\boxed``, "X and X"),
- JSON control-character repair in matcher normalization (``\\b``/``\\f``),
- GRPO reward-diagnostics persistence,

pinned to the real Round 3 episode data in ``tests/fixtures``.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from arappav.errors.schema_math import MathInjectedError, MathVerifierClaim
from arappav.reward.matcher import (
    _normalize_for_matching,
    group_errors_into_units,
)
from arappav.reward.reward_fns import compute_rewards
from arappav.training.grpo_trainer import make_perturber_reward_fn

FIXTURES = Path(__file__).parent / "fixtures"


def _make_error(
    original,
    injected,
    error_id="err_001",
    error_type="wrong_operation",
    step_index=0,
):
    return MathInjectedError(
        error_id=error_id,
        step_index=step_index,
        original_text=original,
        injected_text=injected,
        error_type=error_type,
        rationale="test rationale",
    )


def _make_claim(quoted, explanation="test explanation"):
    return MathVerifierClaim(quoted_text=quoted, explanation=explanation)


# ---------------------------------------------------------------------------
# JSON control-character repair (\b → \boxed, \f → \frac)
# ---------------------------------------------------------------------------


class TestControlCharRepair:
    def test_backspace_from_boxed_repaired(self):
        # A verifier quoting "\boxed{2}" with a single backslash inside JSON
        # emits the VALID escape \b → a literal backspace after parsing.
        # This killed the resp2 match in the Round 1 3_4 episode.
        corrupted = "k is their product, namely (-4) + (-2) = \x08oxed{2}."
        clean = "k is their product, namely (-4) + (-2) = \\boxed{2}."
        assert _normalize_for_matching(corrupted) == _normalize_for_matching(clean)

    def test_formfeed_from_frac_repaired(self):
        corrupted = "so \x0crac{b}{a} = 3"
        clean = "so \\frac{b}{a} = 3"
        assert _normalize_for_matching(corrupted) == _normalize_for_matching(clean)


# ---------------------------------------------------------------------------
# Error-unit grouping
# ---------------------------------------------------------------------------


class TestErrorUnits:
    def test_independent_errors_stay_separate(self):
        text = "The sum is 4 + 5 = 10. The product is 4 * 5 = 21. So x = 31."
        errors = [
            _make_error("4 + 5 = 9", "4 + 5 = 10", "err_001", step_index=0),
            _make_error("4 * 5 = 20", "4 * 5 = 21", "err_002", step_index=1),
        ]
        units = group_errors_into_units(errors, text)
        assert units == [[0], [1]]

    def test_changed_boxed_merges_with_upstream_error(self):
        text = "The pairs give 2(100) + 100. The total is \\boxed{300}."
        errors = [
            _make_error("2(100) + 2(1)", "2(100) + 100", "err_001", step_index=1),
            _make_error(
                "The total is \\boxed{202}.",
                "The total is \\boxed{300}.",
                "err_002",
                step_index=2,
            ),
        ]
        units = group_errors_into_units(errors, text)
        assert units == [[0, 1]]

    def test_unchanged_boxed_does_not_merge(self):
        # An error that merely rewrites text AROUND an unchanged boxed answer
        # is not answer propagation.
        text = "We find 8a = 13. Thus a = \\boxed{3/2} (since b is 3 times a)."
        errors = [
            _make_error("3a = 12 - 5a", "8a = 13", "err_001", step_index=0),
            _make_error(
                "a = \\boxed{3/2}.",
                "a = \\boxed{3/2} (since b is 3 times a).",
                "err_002",
                step_index=1,
            ),
        ]
        units = group_errors_into_units(errors, text)
        assert units == [[0], [1]]

    def test_overlapping_spans_merge(self):
        text = "We expand (13x+15)*2x = 26x^2+30x+15 and simplify."
        errors = [
            _make_error("26x^2+30x", "26x^2+30x+15", "err_001", step_index=0),
            _make_error(
                "(13x+15)*2x = 26x^2+30x",
                "(13x+15)*2x = 26x^2+30x+15",
                "err_002",
                step_index=0,
            ),
        ]
        units = group_errors_into_units(errors, text)
        assert units == [[0, 1]]

    def test_shared_change_fragment_merges_propagated_term(self):
        # The exact Round 1 2_7 pattern: the same phantom "+ 2(1)" term
        # appended to consecutive derivation lines, declared as 3 errors.
        text = (
            "2(19 + 17) + 2(15 + 13) + 2(1) then "
            "2(19 + 17 + 15 + 13) + 2(1) gives 2(64) + 2(1)."
        )
        errors = [
            _make_error(
                "2(19 + 17) + 2(15 + 13)",
                "2(19 + 17) + 2(15 + 13) + 2(1)",
                "err_001",
                step_index=0,
            ),
            _make_error(
                "2(19 + 17 + 15 + 13)",
                "2(19 + 17 + 15 + 13) + 2(1)",
                "err_002",
                step_index=1,
            ),
        ]
        units = group_errors_into_units(errors, text)
        assert units == [[0, 1]]

    def test_merge_rules_can_be_disabled(self):
        text = "The pairs give 2(100) + 100. The total is \\boxed{300}."
        errors = [
            _make_error("2(100) + 2(1)", "2(100) + 100", "err_001", step_index=1),
            _make_error(
                "The total is \\boxed{202}.",
                "The total is \\boxed{300}.",
                "err_002",
                step_index=2,
            ),
        ]
        units = group_errors_into_units(
            errors, text, merge_propagated_boxed=False
        )
        assert units == [[0], [1]]


# ---------------------------------------------------------------------------
# Unit-level recall in compute_rewards
# ---------------------------------------------------------------------------


class TestUnitLevelRecall:
    def _stacked_episode(self):
        """Root error + propagated boxed answer, verifier flags the root once."""
        text = "The pairs give 2(100) + 100. The total is \\boxed{300}."
        errors = [
            _make_error("2(100) + 2(1)", "2(100) + 100", "err_001", step_index=1),
            _make_error(
                "The total is \\boxed{202}.",
                "The total is \\boxed{300}.",
                "err_002",
                step_index=2,
            ),
        ]
        claims = [_make_claim("2(100) + 100")]
        return text, errors, claims

    def test_stacking_no_longer_caps_recall(self):
        text, errors, claims = self._stacked_episode()
        out = compute_rewards(errors, claims, text, k=2)
        assert out.num_error_units == 1
        assert out.num_matched_units == 1
        assert out.verifier_recall == 1.0
        assert out.perturber_reward == 0.0

    def test_per_error_recall_when_units_disabled(self):
        text, errors, claims = self._stacked_episode()
        out = compute_rewards(
            errors, claims, text, k=2, config={"error_units": {"enabled": False}}
        )
        assert out.verifier_recall == pytest.approx(0.5)
        assert out.perturber_reward == pytest.approx(0.5)

    def test_verifier_flagging_whole_chain_keeps_full_precision(self):
        # Claims on both the root and the propagation are both true positives.
        text, errors, _ = self._stacked_episode()
        claims = [
            _make_claim("2(100) + 100"),
            _make_claim("The total is \\boxed{300}."),
        ]
        out = compute_rewards(errors, claims, text, k=2)
        assert out.verifier_recall == 1.0
        assert out.verifier_precision == 1.0

    def test_genuinely_missed_independent_error_still_rewards_perturber(self):
        text = "The sum is 4 + 5 = 10. The product is 4 * 5 = 21."
        errors = [
            _make_error("4 + 5 = 9", "4 + 5 = 10", "err_001", step_index=0),
            _make_error("4 * 5 = 20", "4 * 5 = 21", "err_002", step_index=1),
        ]
        claims = [_make_claim("4 + 5 = 10")]
        out = compute_rewards(errors, claims, text, k=2)
        assert out.num_error_units == 2
        assert out.verifier_recall == pytest.approx(0.5)
        assert out.perturber_reward == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Intra-episode anti-duplicate penalty
# ---------------------------------------------------------------------------


class TestIntraEpisodeDuplicates:
    def test_near_verbatim_reinjection_penalized(self):
        text = "The sum is 4 + 5 = 11. Later, the sum is 4 + 5 = 11 again."
        errors = [
            _make_error("the sum is 4 + 5 = 9", "the sum is 4 + 5 = 11", "err_001"),
            _make_error(
                "the sum is 4 + 5 = 9 again",
                "the sum is 4 + 5 = 11 again",
                "err_002",
                step_index=1,
            ),
        ]
        out = compute_rewards(errors, [], text, k=2)
        assert out.duplicate_penalty <= -1.0

    def test_distinct_errors_not_penalized(self):
        text = "The sum is 4 + 5 = 10. The product is 4 * 5 = 21."
        errors = [
            _make_error("4 + 5 = 9", "4 + 5 = 10", "err_001", step_index=0),
            _make_error("4 * 5 = 20", "4 * 5 = 21", "err_002", step_index=1),
        ]
        out = compute_rewards(errors, [], text, k=2)
        assert out.duplicate_penalty == 0.0


# ---------------------------------------------------------------------------
# Schema: redundant restatements are not errors
# ---------------------------------------------------------------------------


class TestRedundantRestatementRejection:
    def test_double_boxed_rejected(self):
        # The exact Round 3 4_3 err_003 pattern.
        with pytest.raises(ValidationError, match="Redundant restatement"):
            _make_error(
                "leading to $b=251+8=\\boxed{259}$",
                "leading to $b=251+8=\\boxed{259}$ and $b=8+251=\\boxed{259}$",
            )

    def test_and_joined_verbatim_restatement_rejected(self):
        with pytest.raises(ValidationError, match="Redundant restatement"):
            _make_error(
                "so $r=251$ and $s=8$",
                "so $r=251$ and $s=8$ and $r=251$ and $s=8$",
            )

    def test_genuine_term_duplication_accepted(self):
        # A duplicated term inside an expansion changes the math — legit error.
        err = _make_error(
            "13x\\cdot 2x+15\\cdot 2x",
            "13x\\cdot 2x+15\\cdot 2x+15\\cdot 2x",
            error_type="duplication_error",
        )
        assert err.injected_text.endswith("+15\\cdot 2x")

    def test_single_boxed_change_accepted(self):
        err = _make_error(
            "the total is $\\boxed{202}$",
            "the total is $\\boxed{300}$",
        )
        assert "300" in err.injected_text

    def test_distinct_and_clauses_accepted(self):
        # "and"-joined but different statements must not be rejected.
        err = _make_error(
            "so $x = 2$",
            "so $x = 2$ and $x = -3$",
        )
        assert "and" in err.injected_text


# ---------------------------------------------------------------------------
# GRPO reward diagnostics persistence
# ---------------------------------------------------------------------------


class _FailingVerifierStub:
    def generate(self, text, n_completions=1, **kwargs):
        return [("", None, "stub: no output")] * n_completions


class TestRewardDiagnostics:
    def test_diagnostics_written_per_batch(self, tmp_path):
        diag = tmp_path / "diag.jsonl"
        reward_fn = make_perturber_reward_fn(
            verifier_model=_FailingVerifierStub(),
            reward_config={"format_penalty": -10.0, "format_penalty_soft": -5.0},
            mode="math",
            diagnostics_path=diag,
        )
        solution = "The sum is 4 + 5 = 9."
        valid = json.dumps(
            {
                "perturbed_solution": "The sum is 4 + 5 = 10.",
                "errors": [
                    {
                        "error_id": "err_001",
                        "step_index": 0,
                        "original_text": "4 + 5 = 9",
                        "injected_text": "4 + 5 = 10",
                        "error_type": "wrong_operation",
                        "rationale": "off by one",
                    }
                ],
            }
        )
        rewards = reward_fn(
            ["p1", "p1"],
            ["not json at all", valid],
            original_solution=[solution] * 2,
            k=[1] * 2,
        )
        assert len(rewards) == 2

        records = [json.loads(line) for line in diag.read_text().splitlines()]
        assert len(records) == 1
        rec = records[0]
        assert rec["num_completions"] == 2
        assert rec["num_groups"] == 1
        # -10 vs positive reward → nonzero within-group variance
        assert rec["num_uniform_groups"] == 0
        assert rec["group_reward_ranges"][0] > 0
        assert rec["rewards"] == [round(r, 4) for r in rewards]

    def test_no_diagnostics_file_when_not_requested(self, tmp_path):
        reward_fn = make_perturber_reward_fn(
            verifier_model=_FailingVerifierStub(),
            reward_config={"format_penalty": -10.0},
            mode="math",
        )
        reward_fn(["p1"], ["garbage"], original_solution=["x"], k=[1])
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Pinned to real Round 3 episode data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def round3_episodes():
    path = FIXTURES / "round3_math_episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(episodes) == 4
    return {ep["paper_id"]: ep for ep in episodes}


def _score_response(episode, response, ground_truth=None):
    if ground_truth is None:
        ground_truth = [MathInjectedError.model_validate(e) for e in episode["ground_truth"]]
    claims = [MathVerifierClaim.model_validate(c) for c in response["parsed"]["claims"]]
    return compute_rewards(
        ground_truth,
        claims,
        episode["perturbed_text"],
        k=episode["k"],
        verifier_raw_output=response["raw_text"],
    )


class TestRound3Episodes:
    """The exact stacking/propagation episodes from the Round 3 analysis."""

    def test_ep2_7_propagated_boxed_collapsed(self, round3_episodes):
        # err_003 (= \boxed{300}) is the propagated consequence of err_002
        # (2(100) + 100). Previously recall capped at 1/3 → r_P 0.67.
        episode = round3_episodes["algebra_Level 2_7"]
        for response in episode["responses"]:
            out = _score_response(episode, response)
            assert out.num_error_units == 2
            assert out.verifier_recall == pytest.approx(0.5)
            assert out.perturber_reward == pytest.approx(0.5)

    def test_ep1_6_overlapping_rewrites_collapsed(self, round3_episodes):
        # err_003's injected block quotes the region err_001 modified —
        # overlapping declarations over one derivation. err_002 (26x^2+30x^2)
        # is an independent, genuinely-missed error the Perturber keeps
        # credit for.
        episode = round3_episodes["algebra_Level 1_6"]
        for response in episode["responses"]:
            out = _score_response(episode, response)
            assert out.num_error_units == 2
            assert out.verifier_recall == pytest.approx(0.5)
            assert out.perturber_reward == pytest.approx(0.5)

    def test_ep4_3_redundant_restatement_fails_schema(self, round3_episodes):
        # err_003 states the same \boxed{259} twice — with the new validator
        # the whole output fails schema validation (graded -5 in training).
        episode = round3_episodes["algebra_Level 4_3"]
        with pytest.raises(ValidationError, match="Redundant restatement"):
            for e in episode["ground_truth"]:
                MathInjectedError.model_validate(e)

    def test_ep3_0_legitimate_win_keeps_reward(self, round3_episodes):
        # Independent errors, one genuinely missed by resp2 — the Perturber
        # must keep its reward for real (non-stacked) wins.
        episode = round3_episodes["algebra_Level 3_0"]
        out = _score_response(episode, episode["responses"][2])
        assert out.num_error_units == 3
        assert out.verifier_recall == pytest.approx(1 / 3)
        assert out.perturber_reward == pytest.approx(2 / 3)
