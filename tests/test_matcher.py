"""Tests for the claim-to-error matcher."""

import pytest

from arappav.errors.schema import InjectedError, VerifierClaim
from arappav.errors.taxonomy import ErrorType
from arappav.reward.matcher import (
    _compute_char_span_overlap,
    exact_text_match,
    match_claims_to_errors,
)


class TestCharSpanOverlap:
    """Tests for _compute_char_span_overlap."""

    def test_perfect_match(self):
        text = "The cat sat on the mat."
        assert _compute_char_span_overlap("cat", "cat", text) == pytest.approx(1.0)

    def test_no_overlap(self):
        text = "The cat sat on the mat."
        assert _compute_char_span_overlap("cat", "dog", text) == pytest.approx(0.0)

    def test_partial_overlap(self):
        text = "The black cat sat on the mat."
        # "black cat" and "cat sat" share "cat"
        iou = _compute_char_span_overlap("black cat", "cat sat", text)
        assert 0.0 < iou < 1.0

    def test_substring_not_found(self):
        text = "The cat sat."
        assert _compute_char_span_overlap("dog", "cat", text) == pytest.approx(0.0)


class TestMatchClaimsToErrors:
    """Tests for match_claims_to_errors."""

    @staticmethod
    def _make_error(
        error_id: str, injected: str, error_type: ErrorType = ErrorType.NUMERICAL
    ) -> InjectedError:
        return InjectedError(
            error_id=error_id,
            location="test",
            original_text="orig",
            injected_text=injected,
            error_type=error_type,
            rationale="test rationale",
        )

    @staticmethod
    def _make_claim(quoted: str) -> VerifierClaim:
        return VerifierClaim(location="test", quoted_text=quoted, explanation="test explanation")

    def test_empty_both(self):
        result = match_claims_to_errors([], [], "some text")
        assert result.num_matched_errors == 0
        assert result.num_false_positives == 0

    def test_no_errors_some_claims(self):
        claims = [self._make_claim("error here")]
        result = match_claims_to_errors([], claims, "error here")
        assert result.num_matched_errors == 0
        assert result.num_false_positives == 1
        assert result.claim_is_true_positive == [False]

    def test_no_claims_some_errors(self):
        errors = [self._make_error("e1", "bad stuff")]
        result = match_claims_to_errors(errors, [], "text with bad stuff")
        assert result.num_matched_errors == 0
        assert result.num_unmatched_errors == 1
        assert result.matched_claim_indices["e1"] is None

    def test_perfect_match_one_to_one(self):
        text = "The accuracy is 95.3% on the test set."
        errors = [self._make_error("e1", "95.3%")]
        claims = [self._make_claim("95.3%")]
        result = match_claims_to_errors(errors, claims, text)
        assert result.num_matched_errors == 1
        assert result.claim_is_true_positive == [True]
        assert result.num_false_positives == 0

    def test_multiple_errors_multiple_claims(self):
        text = (
            "We used Adam with lr=0.001. The accuracy was 95.3%. "
            "As shown by Smith et al. (2019), this is state-of-the-art."
        )
        errors = [
            self._make_error("e1", "lr=0.001"),
            self._make_error("e2", "95.3%"),
            self._make_error("e3", "Smith et al. (2019)"),
        ]
        claims = [
            self._make_claim("lr=0.001"),
            self._make_claim("95.3%"),
            self._make_claim("Smith et al. (2019)"),
        ]
        result = match_claims_to_errors(errors, claims, text)
        assert result.num_matched_errors == 3
        assert result.num_false_positives == 0
        assert all(result.claim_is_true_positive)

    def test_verifier_misses_some(self):
        text = "lr=0.001, accuracy=95.3%, batch_size=32."
        errors = [
            self._make_error("e1", "lr=0.001"),
            self._make_error("e2", "accuracy=95.3%"),
            self._make_error("e3", "batch_size=32"),
        ]
        # Verifier only catches 2 of 3
        claims = [self._make_claim("lr=0.001"), self._make_claim("batch_size=32")]
        result = match_claims_to_errors(errors, claims, text)
        assert result.num_matched_errors == 2
        assert result.num_unmatched_errors == 1
        assert result.num_false_positives == 0

    def test_verifier_hallucinates(self):
        text = "lr=0.001, accuracy=95.3%."
        errors = [self._make_error("e1", "lr=0.001")]
        claims = [
            self._make_claim("lr=0.001"),  # real
            self._make_claim("batch_size=32"),  # hallucination — not in errors
        ]
        result = match_claims_to_errors(errors, claims, text)
        assert result.num_matched_errors == 1
        assert result.num_false_positives == 1
        assert result.claim_is_true_positive == [True, False]

    def test_overlap_threshold_filters_noisy_claims(self):
        text = "The learning rate was 0.001."
        errors = [self._make_error("e1", "learning rate was 0.001")]
        claims = [self._make_claim("rate was")]  # too short to get good IoU
        result = match_claims_to_errors(
            errors, claims, text, span_overlap_threshold=0.5
        )
        # Short substring has low IoU with the longer injected text
        assert result.num_matched_errors == 0


class TestExactTextMatch:
    """Tests for exact_text_match."""

    def test_exact_substring_match(self):
        errors = [
            InjectedError(
                error_id="e1",
                location="test",
                original_text="orig",
                injected_text="the model achieves 99% accuracy",
                error_type=ErrorType.NUMERICAL,
                rationale="wrong number",
            )
        ]
        claims = [VerifierClaim(location="test", quoted_text="99% accuracy", explanation="too high")]
        result = exact_text_match(errors, claims)
        assert result.num_matched_errors == 1

    def test_no_match_different_text(self):
        errors = [
            InjectedError(
                error_id="e1",
                location="test",
                original_text="orig",
                injected_text="the model achieves 99% accuracy",
                error_type=ErrorType.NUMERICAL,
                rationale="wrong number",
            )
        ]
        claims = [VerifierClaim(location="test", quoted_text="the sky is blue", explanation="nonsense")]
        result = exact_text_match(errors, claims)
        assert result.num_matched_errors == 0
