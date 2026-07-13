"""Regression tests for the Round 1 rollout-analysis fixes.

Covers the matcher (LaTeX escaping, diff-based change coverage, substring
tie-breaking), fuzzy taxonomy matching, and reward-side fixes (k_effective,
anti-spam floor, repetition penalty, partial configs) — including tests
pinned to the real Round 1 episode data in ``tests/fixtures``.
"""

import json
from pathlib import Path

import pytest

from arappav.errors.fuzzy import fuzzy_match_enum, levenshtein_distance
from arappav.errors.schema_math import (
    MathInjectedError,
    MathVerifierClaim,
    _fuzzy_match_math_error_type,
)
from arappav.errors.taxonomy_math import MathErrorType
from arappav.reward.matcher import (
    _changed_fragments,
    _claim_covers_change,
    _normalize_for_matching,
    _normalize_latex_escapes,
    _substring_match_score,
    error_present_in_text,
    match_claims_to_errors,
)
from arappav.reward.reward_fns import compute_rewards

FIXTURES = Path(__file__).parent / "fixtures"


def _make_error(original, injected, error_id="err_001", error_type="wrong_operation"):
    return MathInjectedError(
        error_id=error_id,
        step_index=0,
        original_text=original,
        injected_text=injected,
        error_type=error_type,
        rationale="test rationale",
    )


def _make_claim(quoted, explanation="test explanation"):
    return MathVerifierClaim(quoted_text=quoted, explanation=explanation)


# ---------------------------------------------------------------------------
# LaTeX escape normalization
# ---------------------------------------------------------------------------


class TestLatexEscapeNormalization:
    def test_collapses_double_escaped_latex(self):
        # A model that double-escapes in JSON yields 2 literal backslashes
        # after parsing; the reference text has 1. Both must normalize equal.
        double = "(4 + 11)/2 \\\\cdot 8 = \\\\boxed{60}"
        single = "(4 + 11)/2 \\cdot 8 = \\boxed{60}"
        assert _normalize_latex_escapes(double) == single

    def test_idempotent_on_single_escaped(self):
        single = "\\frac{1}{2} + \\sqrt{x}"
        assert _normalize_latex_escapes(single) == single

    def test_normalized_forms_compare_equal(self):
        quoted = "the sum is (4 + 11)/2 \\\\cdot 8"  # JSON double-escaped
        text_fragment = "the sum is $(4 + 11)/2 \\cdot 8$"  # as in solution
        assert _normalize_for_matching(quoted) == _normalize_for_matching(text_fragment)

    def test_all_commands_normalized_not_just_last(self):
        # Regression: the old implementation replaced on the input string in
        # a loop, discarding every replacement but the last.
        double = "\\\\cdot x \\\\boxed{1} \\\\frac{a}{b}"
        assert _normalize_latex_escapes(double) == "\\cdot x \\boxed{1} \\frac{a}{b}"

    def test_double_escaped_claim_matches(self):
        text = "The sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8$."
        error = _make_error(
            "The sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60}$.",
            "The sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8$.",
        )
        # Verifier double-escaped its quoted text
        claim = _make_claim("The sum is (4 + 11)/2 \\\\cdot 8 = \\\\boxed{60} + 8")
        result = match_claims_to_errors([error], [claim], text)
        assert result.num_matched_errors == 1


# ---------------------------------------------------------------------------
# Diff-based change coverage
# ---------------------------------------------------------------------------


class TestChangeCoverage:
    def test_changed_fragments_isolate_edit(self):
        orig = "so the sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60}$."
        inj = "so the sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8$."
        frags = _changed_fragments(orig, inj)
        assert any("+ 8" in f for f in frags)

    def test_short_edits_expanded_with_context(self):
        frags = _changed_fragments("x = 2 + 3 = 5", "x = 2 - 3 = 5")
        # The single-char edit must carry context so it can't match anywhere
        assert all(len(f) >= 6 for f in frags)

    def test_claim_quoting_changed_region_covers(self):
        error = _make_error(
            "The sum of the series is computed from the average.",
            "The sum of the series is computed from the average. "
            "The average is just the sum of the first and last term.",
        )
        claim = _make_claim("The average is just the sum of the first and last term.")
        assert _claim_covers_change(error, claim)

    def test_claim_quoting_unchanged_clause_does_not_cover(self):
        error = _make_error(
            "The sum of the series is computed from the average.",
            "The sum of the series is computed from the average. "
            "The average is just the sum of the first and last term.",
        )
        # Quotes only the untouched first sentence
        claim = _make_claim("The sum of the series is computed from the average.")
        assert not _claim_covers_change(error, claim)

    def test_change_coverage_outranks_plain_substring(self):
        # Two claims are both substrings of injected_text; only one covers
        # the changed region — greedy assignment must pick that one.
        text = (
            "We compute the total. The count is 8, so the sum is "
            "$(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8$."
        )
        error = _make_error(
            "The count is 8, so the sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60}$.",
            "The count is 8, so the sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8$.",
        )
        wrong_claim = _make_claim("The count is 8, so the sum is")
        right_claim = _make_claim("the sum is $(4 + 11)/2 \\cdot 8 = \\boxed{60} + 8$")
        result = match_claims_to_errors([error], [wrong_claim, right_claim], text)
        assert result.matched_claim_indices["err_001"] == 1


# ---------------------------------------------------------------------------
# Substring scoring
# ---------------------------------------------------------------------------


class TestSubstringScore:
    def test_meaningful_substring_scores_by_ratio(self):
        long = "The average is just the sum of the first and last term of the series."
        short = "The average is just the sum of the first and last term"
        score = _substring_match_score(long, short)
        assert score == pytest.approx(len(short) / len(long), abs=0.05)

    def test_trivial_fragment_scores_zero(self):
        assert _substring_match_score("a very long sentence about arithmetic series", "ry lo") == 0.0

    def test_non_substring_scores_zero(self):
        assert _substring_match_score("completely different", "unrelated text here") == 0.0


# ---------------------------------------------------------------------------
# Fuzzy taxonomy matching
# ---------------------------------------------------------------------------


class TestFuzzyEnum:
    def test_typo_matches_strict_tier(self):
        assert _fuzzy_match_math_error_type("wrong_opration") == MathErrorType("wrong_operation")

    def test_alternative_wording_matches_relaxed_tier(self):
        assert _fuzzy_match_math_error_type("addition_across") == MathErrorType("adding_across")

    def test_duplicated_operation_maps_to_duplication_error(self):
        # The actual ep2 format failure from Round 1
        assert _fuzzy_match_math_error_type("duplicated_operation") == MathErrorType(
            "duplication_error"
        )

    def test_generic_word_alone_is_not_evidence(self):
        # Regression: names sharing only the generic word "error" with a
        # taxonomy member must not fuzzy-match to it ('sign_error' used to
        # silently map to 'inversion_error'). 'sign_error' itself is now
        # resolved by an explicit curated alias instead — see
        # MATH_ERROR_TYPE_ALIASES and tests/test_round2_fixes.py.
        assert _fuzzy_match_math_error_type("logic_error") is None
        assert _fuzzy_match_math_error_type("misc_error") is None

    def test_garbage_rejected(self):
        assert _fuzzy_match_math_error_type("completely_wrong_name_xyz") is None

    def test_exact_match_passthrough(self):
        assert _fuzzy_match_math_error_type("operand_swap") == MathErrorType("operand_swap")

    def test_levenshtein_basics(self):
        assert levenshtein_distance("abc", "abc") == 0
        assert levenshtein_distance("abc", "abd") == 1
        assert levenshtein_distance("", "abc") == 3

    def test_schema_accepts_near_miss_and_rejects_garbage(self):
        err = _make_error("a correct step", "a wrong step", error_type="addition_across")
        assert err.error_type == MathErrorType("adding_across")
        with pytest.raises(ValueError):
            _make_error("a correct step", "a wrong step", error_type="totally_bogus_xyz")


# ---------------------------------------------------------------------------
# Reward-side fixes
# ---------------------------------------------------------------------------


class TestRewardFixes:
    def _episode(self):
        text = "Step one is fine. The answer is $2 \\cdot 3 = 7$."
        error = _make_error(
            "The answer is $2 \\cdot 3 = 6$.",
            "The answer is $2 \\cdot 3 = 7$.",
        )
        claim = _make_claim("The answer is $2 \\cdot 3 = 7$.")
        return text, error, claim

    def test_partial_config_does_not_crash(self):
        # Regression: config without anti_spam/anti_duplicate raised KeyError
        text, error, claim = self._episode()
        out = compute_rewards([error], [claim], text, k=1, config={"format_penalty": -5.0})
        assert out.verifier_recall == 1.0

    def test_k_effective_tolerates_escaping_drift(self):
        # injected_text double-escaped relative to the perturbed text must
        # still count as present (not penalize P, not shrink V's denominator)
        text = "The answer is $2 \\cdot 3 = 7$."
        error = _make_error(
            "The answer is $2 \\\\cdot 3 = 6$.",
            "The answer is $2 \\\\cdot 3 = 7$.",
        )
        assert error_present_in_text(error.injected_text, text)
        out = compute_rewards([error], [], text, k=1)
        assert out.k_effective == 1

    def test_anti_spam_floor_when_no_errors_present(self):
        # k_effective == 0: the verifier cannot know, so a couple of claims
        # must not trigger the spam penalty.
        error = _make_error("original step text", "injected wrong step text")
        text = "completely unrelated text without the injection"
        claims = [_make_claim("some claim"), _make_claim("another claim")]
        out = compute_rewards([error], claims, text, k=1)
        assert out.k_effective == 0
        assert out.spam_penalty == 0.0

    def test_repetition_penalty_applied_via_raw_output(self):
        text, error, claim = self._episode()
        collapsed = '{"claims": []}\n' * 30
        out = compute_rewards([error], [claim], text, k=1, verifier_raw_output=collapsed)
        assert out.repetition_penalty < 0
        # Perfect-but-repetitive must still beat a total miss (reward > 0)
        assert out.verifier_reward > 0.0

    def test_no_repetition_penalty_for_clean_output(self):
        text, error, claim = self._episode()
        out = compute_rewards([error], [claim], text, k=1, verifier_raw_output='{"claims": []}')
        assert out.repetition_penalty == 0.0

    def test_match_details_exposed(self):
        text, error, claim = self._episode()
        out = compute_rewards([error], [claim], text, k=1)
        assert len(out.match_details) == 1
        assert out.match_details[0]["error_id"] == "err_001"


# ---------------------------------------------------------------------------
# Pinned to real Round 1 episode data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def round1_episodes():
    path = FIXTURES / "round1_math_episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(episodes) == 2
    return {ep["paper_id"]: ep for ep in episodes}


def _score_response(episode, response):
    ground_truth = [MathInjectedError.model_validate(e) for e in episode["ground_truth"]]
    claims = [MathVerifierClaim.model_validate(c) for c in response["parsed"]["claims"]]
    return compute_rewards(
        ground_truth,
        claims,
        episode["perturbed_text"],
        k=episode["k"],
        verifier_raw_output=response["raw_text"],
    )


class TestRound1Episodes:
    """The exact false-zero episodes from the Round 1 analysis."""

    def test_ep1_verifier_gets_full_credit(self, round1_episodes):
        # ep1: verifier found both detectable errors in every response but
        # originally got zero reward (matcher failure).
        episode = round1_episodes["algebra_Level 2_1"]
        for response in episode["responses"]:
            out = _score_response(episode, response)
            assert out.k_effective == 2  # err_003 overlapped and was dropped
            assert out.verifier_recall == 1.0
            assert out.verifier_precision == 1.0
            # Perturber must NOT get the free pass it originally got (1.0)
            assert out.perturber_reward < 0.0

    def test_ep1_collapsed_response_penalized_but_still_positive(self, round1_episodes):
        episode = round1_episodes["algebra_Level 2_1"]
        collapsed = episode["responses"][2]  # 164 repeated JSON blocks
        out = _score_response(episode, collapsed)
        assert out.repetition_penalty < 0
        assert 0.0 < out.verifier_reward < 1.0

    def test_ep6_partial_detection(self, round1_episodes):
        episode = round1_episodes["algebra_Level 1_6"]
        for response in episode["responses"]:
            out = _score_response(episode, response)
            assert out.k_effective == 2
            assert out.verifier_recall == pytest.approx(0.5)
            assert out.verifier_precision == 1.0
            assert out.verifier_reward == pytest.approx(2 / 3, abs=1e-3)
