"""End-to-end smoke test: runs a single Perturber→Verifier→Reward episode
with stub/mock models (no real LLM loaded) to verify the pipeline wiring.

This test should pass without any GPU or model downloads.
"""

import pytest

from arappav.errors.schema import (
    InjectedError,
    PerturberOutput,
    VerifierClaim,
    VerifierOutput,
    validate_perturber_output,
    validate_verifier_output,
)
from arappav.errors.taxonomy import ErrorType
from arappav.reward.reward_fns import compute_rewards


# Injected texts must be genuinely distinct: near-identical injections are
# (correctly) collapsed into one error unit by the anti-stacking merge rules,
# which is not what these wiring tests exercise.
_FAKE_ORIGINALS = [
    "the boiling point of water is 100 degrees",
    "the arithmetic series sums to 60",
    "gravitational acceleration equals 9.8 meters per second squared",
]
_FAKE_INJECTED = [
    "the boiling point of water is 50 degrees",
    "the arithmetic series sums to 75",
    "gravitational acceleration equals 12.3 meters per second squared",
]


def _make_fake_perturber_output(k: int) -> PerturberOutput:
    """Build a well-formed Perturber output for testing."""
    errors = []
    error_types = list(ErrorType)
    for i in range(k):
        errors.append(
            InjectedError(
                error_id=f"err_{i:03d}",
                location=f"paragraph {i + 1}",
                original_text=_FAKE_ORIGINALS[i % len(_FAKE_ORIGINALS)],
                injected_text=_FAKE_INJECTED[i % len(_FAKE_INJECTED)],
                error_type=error_types[i % len(error_types)],
                rationale=f"This is wrong because {i}.",
            )
        )
    return PerturberOutput(
        perturbed_text="This is a perturbed paper claiming "
        + ", and ".join(_FAKE_INJECTED)
        + ".",
        errors=errors,
    )


def _make_fake_verifier_output(
    catch_indices: list[int], total_k: int
) -> VerifierOutput:
    """Build a Verifier output that catches specific error indices."""
    claims = []
    for i in catch_indices:
        claims.append(
            VerifierClaim(
                location=f"paragraph {i + 1}",
                quoted_text=_FAKE_INJECTED[i % len(_FAKE_INJECTED)],
                explanation=f"Found error {i}.",
            )
        )
    return VerifierOutput(claims=claims)


class TestSmokeEndToEnd:
    """Minimal end-to-end test without real models."""

    def test_full_pipeline_no_errors_caught(self):
        """Perturber injects 3 errors, Verifier catches none → r_P ≈ 1.0, r_V ≈ 0.0."""
        k = 3
        p_out = _make_fake_perturber_output(k)
        v_out = _make_fake_verifier_output([], k)

        reward = compute_rewards(
            ground_truth=p_out.errors,
            verifier_claims=v_out.claims,
            perturbed_text=p_out.perturbed_text,
            k=k,
        )

        assert reward.verifier_recall == pytest.approx(0.0)
        assert reward.verifier_precision == pytest.approx(0.0)  # no claims → no precision
        assert reward.perturber_reward == pytest.approx(1.0)
        assert not reward.format_penalty_applied

    def test_full_pipeline_all_errors_caught(self):
        """Perturber injects 3 errors, Verifier catches all → r_P ≈ 0.0, r_V ≈ 1.0."""
        k = 3
        p_out = _make_fake_perturber_output(k)
        v_out = _make_fake_verifier_output(list(range(k)), k)

        reward = compute_rewards(
            ground_truth=p_out.errors,
            verifier_claims=v_out.claims,
            perturbed_text=p_out.perturbed_text,
            k=k,
        )

        assert reward.verifier_recall == pytest.approx(1.0)
        assert reward.verifier_precision == pytest.approx(1.0)
        assert reward.verifier_f_beta == pytest.approx(1.0)
        assert reward.perturber_reward == pytest.approx(0.0)

    def test_full_pipeline_partial_catch(self):
        """Perturber injects 3 errors, Verifier catches 2 of 3."""
        k = 3
        p_out = _make_fake_perturber_output(k)
        v_out = _make_fake_verifier_output([0, 2], k)  # misses error 1

        reward = compute_rewards(
            ground_truth=p_out.errors,
            verifier_claims=v_out.claims,
            perturbed_text=p_out.perturbed_text,
            k=k,
        )

        assert reward.verifier_recall == pytest.approx(2 / 3)
        assert 0.0 < reward.perturber_reward < 1.0
        assert 0.0 < reward.verifier_reward < 1.0

    def test_format_penalty_triggers(self):
        """Invalid Perturber output (wrong k) triggers format penalty."""
        k = 3
        # Only 1 error when 3 were requested — simulate via validation
        from arappav.errors.schema import validate_perturber_output

        data = {
            "perturbed_text": "text",
            "errors": [
                {
                    "error_id": "err_001",
                    "location": "p1",
                    "original_text": "orig",
                    "injected_text": "inj",
                    "error_type": "numerical",
                    "rationale": "wrong",
                }
            ],
        }
        p_out, p_err = validate_perturber_output(data, expected_k=3)
        assert p_out is None

        reward = compute_rewards(
            ground_truth=[],
            verifier_claims=[],
            perturbed_text="text",
            k=3,
            perturber_format_valid=False,
            perturber_format_reason=p_err,
        )

        assert reward.format_penalty_applied
        assert reward.perturber_reward <= -10.0

    def test_schema_validation_roundtrip(self):
        """PerturberOutput → dict → validate → PerturberOutput round-trips."""
        p_out = _make_fake_perturber_output(2)
        data = p_out.model_dump()
        p_out2, err = validate_perturber_output(data, expected_k=2)
        assert p_out2 is not None
        assert err is None
        assert len(p_out2.errors) == 2

    def test_verifier_output_schema_roundtrip(self):
        """VerifierOutput → dict → validate → VerifierOutput round-trips."""
        v_out = _make_fake_verifier_output([0, 1], 3)
        data = v_out.model_dump()
        v_out2, err = validate_verifier_output(data)
        assert v_out2 is not None
        assert err is None
        assert len(v_out2.claims) == 2

    def test_reward_is_symmetric(self):
        """r_P + r_V_base ≈ 1.0 when no penalties (zero-sum property)."""
        k = 3
        p_out = _make_fake_perturber_output(k)
        v_out = _make_fake_verifier_output([0, 1], k)

        # Disable spam and duplicate penalties for this test
        config = {
            "span_overlap_threshold": 0.5,
            "precision_recall_beta": 1.0,
            "verifier_reward_formula": "f_beta",
            "perturber_reward_formula": "one_minus_recall",
            "anti_spam": {"enabled": False},
            "anti_duplicate": {"enabled": False},
        }

        reward = compute_rewards(
            ground_truth=p_out.errors,
            verifier_claims=v_out.claims,
            perturbed_text=p_out.perturbed_text,
            k=k,
            config=config,
        )

        # With no penalties: r_P_base = 1 - recall, r_V_base = f_beta
        # These should sum to approximately 1.0 (with f_beta approximating recall)
        combined = reward.perturber_base_reward + reward.verifier_base_reward
        assert 0.8 <= combined <= 1.2, f"Combined base rewards: {combined}"
