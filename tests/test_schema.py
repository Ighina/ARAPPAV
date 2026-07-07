"""Tests for structured I/O schema validation."""

import pytest
from pydantic import ValidationError

from arappav.errors.schema import (
    InjectedError,
    PerturberOutput,
    VerifierClaim,
    VerifierOutput,
    validate_perturber_output,
    validate_verifier_output,
)
from arappav.errors.taxonomy import ErrorType


class TestInjectedError:
    def test_valid_error(self):
        err = InjectedError(
            error_id="err_001",
            location="para 1",
            original_text="The accuracy was 95%.",
            injected_text="The accuracy was 99%.",
            error_type=ErrorType.NUMERICAL,
            rationale="Changed statistic from 95% to 99%.",
        )
        assert err.error_id == "err_001"
        assert err.error_type == ErrorType.NUMERICAL

    def test_empty_error_id_fails(self):
        with pytest.raises(ValidationError):
            InjectedError(
                error_id="   ",
                location="test",
                original_text="orig",
                injected_text="inj",
                error_type=ErrorType.LOGICAL,
                rationale="reason",
            )

    def test_empty_text_fails(self):
        with pytest.raises(ValidationError):
            InjectedError(
                error_id="e1",
                location="test",
                original_text="",
                injected_text="inj",
                error_type=ErrorType.LOGICAL,
                rationale="reason",
            )


class TestPerturberOutput:
    def test_valid_output(self):
        output = PerturberOutput(
            perturbed_text="Modified text here.",
            errors=[
                InjectedError(
                    error_id="err_001",
                    location="para 1",
                    original_text="orig",
                    injected_text="inj",
                    error_type=ErrorType.TERMINOLOGY,
                    rationale="Wrong term used.",
                )
            ],
        )
        assert len(output.errors) == 1

    def test_empty_perturbed_text_fails(self):
        with pytest.raises(ValidationError):
            PerturberOutput(perturbed_text="   ", errors=[])


class TestVerifierClaim:
    def test_valid_claim(self):
        claim = VerifierClaim(
            location="para 2",
            quoted_text="99% accuracy",
            explanation="Implausibly high.",
        )
        assert claim.quoted_text == "99% accuracy"

    def test_empty_fields_fail(self):
        with pytest.raises(ValidationError):
            VerifierClaim(location="test", quoted_text="", explanation="reason")


class TestValidatePerturberOutput:
    def test_correct_k_passes(self):
        data = {
            "perturbed_text": "text with errors",
            "errors": [
                {
                    "error_id": "err_001",
                    "location": "p1",
                    "original_text": "orig",
                    "injected_text": "inj",
                    "error_type": "numerical",
                    "rationale": "wrong number",
                },
                {
                    "error_id": "err_002",
                    "location": "p2",
                    "original_text": "orig2",
                    "injected_text": "inj2",
                    "error_type": "logical",
                    "rationale": "wrong logic",
                },
            ],
        }
        output, err = validate_perturber_output(data, expected_k=2)
        assert output is not None
        assert err is None
        assert len(output.errors) == 2

    def test_wrong_k_fails(self):
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
        output, err = validate_perturber_output(data, expected_k=3)
        assert output is None
        assert err is not None
        assert "Expected exactly 3" in err

    def test_duplicate_ids_fail(self):
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
                },
                {
                    "error_id": "err_001",  # duplicate
                    "location": "p2",
                    "original_text": "orig2",
                    "injected_text": "inj2",
                    "error_type": "logical",
                    "rationale": "wrong",
                },
            ],
        }
        output, err = validate_perturber_output(data, expected_k=2)
        assert output is None
        assert "Duplicate" in err

    def test_invalid_error_type_fails(self):
        data = {
            "perturbed_text": "text",
            "errors": [
                {
                    "error_id": "err_001",
                    "location": "p1",
                    "original_text": "orig",
                    "injected_text": "inj",
                    "error_type": "not_a_real_type",
                    "rationale": "wrong",
                }
            ],
        }
        output, err = validate_perturber_output(data, expected_k=1)
        assert output is None
        assert err is not None


class TestValidateVerifierOutput:
    def test_valid_output(self):
        data = {
            "claims": [
                {
                    "location": "p1",
                    "quoted_text": "99%",
                    "explanation": "too high",
                }
            ]
        }
        output, err = validate_verifier_output(data)
        assert output is not None
        assert len(output.claims) == 1

    def test_empty_claims_ok(self):
        data = {"claims": []}
        output, err = validate_verifier_output(data)
        assert output is not None
        assert len(output.claims) == 0

    def test_missing_claims_key(self):
        # VerifierOutput has default_factory=list for claims, so missing key → empty claims
        output, err = validate_verifier_output({})
        assert output is not None
        assert err is None
        assert len(output.claims) == 0
