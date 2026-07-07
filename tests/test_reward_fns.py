"""Tests for reward functions."""

import pytest

from arappav.errors.schema import InjectedError, VerifierClaim
from arappav.errors.taxonomy import ErrorType
from arappav.reward.reward_fns import (
    RewardOutput,
    _compute_f_beta,
    _jaccard_similarity,
    compute_rewards,
)


class TestComputeFBeta:
    def test_perfect(self):
        assert _compute_f_beta(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_zero(self):
        assert _compute_f_beta(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_f1(self):
        # precision=0.5, recall=1.0 -> F1 = 2*0.5*1/(0.5+1) = 1/1.5 = 0.666...
        f1 = _compute_f_beta(0.5, 1.0, 1.0)
        assert f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)

    def test_f2_weights_recall_more(self):
        f2 = _compute_f_beta(0.5, 1.0, 2.0)
        f1 = _compute_f_beta(0.5, 1.0, 1.0)
        assert f2 > f1


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _jaccard_similarity("hello", "world") == pytest.approx(0.0)

    def test_partial(self):
        sim = _jaccard_similarity("the cat sat", "the dog sat")
        # tokens: {the, cat, sat} vs {the, dog, sat} → intersection=2, union=4
        assert sim == pytest.approx(0.5)


class TestComputeRewards:
    """End-to-end reward computation tests."""

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
        return VerifierClaim(location="test", quoted_text=quoted, explanation="test")

    def test_format_penalty_applied(self):
        """When Perturber output is invalid, format penalty dominates."""
        result = compute_rewards(
            ground_truth=[],
            verifier_claims=[],
            perturbed_text="text",
            k=3,
            perturber_format_valid=False,
            perturber_format_reason="wrong error count",
        )
        assert result.format_penalty_applied
        assert result.perturber_reward <= -10.0
        assert result.verifier_reward == 0.0

    def test_perfect_verifier(self):
        """Verifier catches all errors with no false positives → r_V ≈ 1.0, r_P ≈ 0.0."""
        text = "lr=0.001, acc=95.3%, batch=32."
        errors = [
            self._make_error("e1", "lr=0.001"),
            self._make_error("e2", "acc=95.3%"),
            self._make_error("e3", "batch=32"),
        ]
        claims = [
            self._make_claim("lr=0.001"),
            self._make_claim("acc=95.3%"),
            self._make_claim("batch=32"),
        ]
        result = compute_rewards(errors, claims, text, k=3)
        assert result.verifier_recall == pytest.approx(1.0)
        assert result.verifier_precision == pytest.approx(1.0)
        assert result.verifier_f_beta == pytest.approx(1.0)
        assert result.verifier_reward > 0.9
        assert result.perturber_reward < 0.1

    def test_blind_verifier(self):
        """Verifier catches nothing → r_V ≈ 0.0, r_P ≈ 1.0."""
        text = "lr=0.001, acc=95.3%, batch=32."
        errors = [
            self._make_error("e1", "lr=0.001"),
            self._make_error("e2", "acc=95.3%"),
            self._make_error("e3", "batch=32"),
        ]
        claims: list[VerifierClaim] = []
        result = compute_rewards(errors, claims, text, k=3)
        assert result.verifier_recall == pytest.approx(0.0)
        assert result.verifier_reward == pytest.approx(0.0)
        assert result.perturber_reward == pytest.approx(1.0)

    def test_verifier_with_false_positives(self):
        """Verifier catches some but also hallucinates → precision < 1.0."""
        text = "lr=0.001, acc=95.3%, batch=32."
        errors = [self._make_error("e1", "lr=0.001")]
        claims = [
            self._make_claim("lr=0.001"),  # correct
            self._make_claim("nonexistent error"),  # hallucination
        ]
        result = compute_rewards(errors, claims, text, k=1)
        assert result.verifier_recall == pytest.approx(1.0)
        assert result.verifier_precision == pytest.approx(0.5)
        assert result.verifier_f_beta < 1.0
        # 2 claims for 1 error — spam penalty may apply
        # 2 claims, 1 matched → 1 unmatched = false positive
        false_positives = result.num_verifier_claims - result.num_matched
        assert false_positives >= 1

    def test_empty_ground_truth(self):
        """Verifier sees unperturbed text — any claims are false positives."""
        text = "All correct text here."
        errors: list[InjectedError] = []
        claims = [self._make_claim("something wrong")]
        result = compute_rewards(errors, claims, text, k=0)
        assert result.verifier_recall == pytest.approx(0.0)
        assert result.verifier_precision == pytest.approx(0.0)
        assert result.verifier_f_beta == pytest.approx(0.0)

    def test_anti_spam_penalty(self):
        """Verifier claims many errors on a single-error text → spam penalty."""
        text = "lr=0.001"
        errors = [self._make_error("e1", "lr=0.001")]
        # 10 claims for 1 error — should trigger spam penalty
        claims = [self._make_claim(f"claim {i}") for i in range(10)]
        config = {
            "anti_spam": {"enabled": True, "max_claims_ratio": 3.0, "penalty_per_excess": -0.5},
            "span_overlap_threshold": 0.5,
            "precision_recall_beta": 1.0,
            "verifier_reward_formula": "f_beta",
            "perturber_reward_formula": "one_minus_recall",
        }
        result = compute_rewards(errors, claims, text, k=1, config=config)
        assert result.spam_penalty < 0.0
        assert result.verifier_reward < result.verifier_base_reward

    def test_reward_output_structure(self):
        """Smoke test that RewardOutput has all expected fields."""
        result = compute_rewards([], [], "text", k=0)
        assert isinstance(result, RewardOutput)
        assert hasattr(result, "perturber_reward")
        assert hasattr(result, "verifier_reward")
        assert hasattr(result, "verifier_recall")
        assert hasattr(result, "verifier_precision")
        assert hasattr(result, "verifier_f_beta")
