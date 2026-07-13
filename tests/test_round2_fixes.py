"""Tests for the Round 2 rollout-analysis fixes.

Covers: invalid-JSON-escape repair, the parse→backoff→unchanged-check
ordering in the Perturber pipeline, graded format penalties in the GRPO
reward function, the ``sign_error`` taxonomy alias, and generation-kwargs
translation for the vLLM/HF backends.
"""

import json

import pytest

from arappav.errors.schema_math import MathInjectedError, _fuzzy_match_math_error_type
from arappav.errors.taxonomy_math import MathErrorType
from arappav.models.generation_utils import prepare_generation_kwargs
from arappav.models.perturber import parse_and_backoff
from arappav.training.grpo_trainer import make_perturber_reward_fn
from arappav.utils.parsing import extract_first_json_object


# ---------------------------------------------------------------------------
# Invalid JSON escape repair
# ---------------------------------------------------------------------------


class TestJsonEscapeRepair:
    def test_raw_latex_escape_recovered(self):
        # \c (from \cdot) is an invalid JSON escape — the exact failure that
        # killed episode 2_7 in both rounds.
        raw = '{"perturbed_solution": "the sum is $2 \\cdot 3$", "note": "x"}'
        data, err = extract_first_json_object(raw)
        assert err is None
        assert data["perturbed_solution"] == "the sum is $2 \\cdot 3$"

    def test_valid_escapes_untouched(self):
        raw = '{"a": "line1\\nline2", "b": "quote: \\" end", "c": "back\\\\slash"}'
        data, err = extract_first_json_object(raw)
        assert err is None
        assert data["a"] == "line1\nline2"
        assert data["c"] == "back\\slash"

    def test_genuinely_broken_json_still_fails(self):
        data, err = extract_first_json_object('{"a": 1, "b": }')
        assert data is None
        assert "JSON parse error" in err


# ---------------------------------------------------------------------------
# parse_and_backoff ordering
# ---------------------------------------------------------------------------


SOLUTION = "The number of terms is 8. The sum is (4 + 11)/2 * 8 = 60."


def _perturber_json(perturbed_solution, errors):
    return json.dumps({"perturbed_solution": perturbed_solution, "errors": errors})


def _error(err_id, original, injected, error_type="wrong_operation"):
    return {
        "error_id": err_id,
        "step_index": 0,
        "original_text": original,
        "injected_text": injected,
        "error_type": error_type,
        "rationale": "test",
    }


class TestParseAndBackoff:
    def test_unmodified_solution_salvaged_by_backoff(self):
        # The exact episode-1_2 failure: valid errors defined in JSON, but
        # the model returned the solution unmodified. Previously rejected
        # before the backoff could run.
        errors = [
            _error("err_001", "The number of terms is 8.", "The number of terms is 9."),
            _error("err_002", "= 60.", "= 75."),
        ]
        raw = _perturber_json(SOLUTION, errors)
        parsed, err, stage = parse_and_backoff(raw, k=2, mode="math", original_text=SOLUTION)
        assert err is None and stage is None
        assert "The number of terms is 9." in parsed.perturbed_solution
        assert "= 75." in parsed.perturbed_solution
        assert parsed.perturbed_solution != SOLUTION

    def test_unsalvageable_unchanged_output_rejected(self):
        # original_text not findable in the solution → backoff can't apply
        # anything → still identical → rejected at the schema stage.
        errors = [
            _error("err_001", "text that does not exist anywhere", "replacement one"),
            _error("err_002", "also not present in the solution", "replacement two"),
        ]
        raw = _perturber_json(SOLUTION, errors)
        parsed, err, stage = parse_and_backoff(raw, k=2, mode="math", original_text=SOLUTION)
        assert parsed is None
        assert stage == "schema"
        assert "identical" in err

    def test_unparseable_output_is_json_stage(self):
        parsed, err, stage = parse_and_backoff(
            "no json here at all", k=2, mode="math", original_text=SOLUTION
        )
        assert parsed is None
        assert stage == "json"

    def test_invalid_enum_is_schema_stage(self):
        errors = [
            _error("err_001", "is 8.", "is 9.", error_type="totally_bogus_type_xyz"),
        ]
        raw = _perturber_json(SOLUTION.replace("is 8.", "is 9."), errors)
        parsed, err, stage = parse_and_backoff(raw, k=1, mode="math", original_text=SOLUTION)
        assert parsed is None
        assert stage == "schema"

    def test_model_modified_output_passes_through(self):
        perturbed = SOLUTION.replace("= 60.", "= 75.")
        errors = [_error("err_001", "= 60.", "= 75.")]
        raw = _perturber_json(perturbed, errors)
        parsed, err, stage = parse_and_backoff(raw, k=1, mode="math", original_text=SOLUTION)
        assert err is None and stage is None
        assert parsed.perturbed_solution == perturbed


# ---------------------------------------------------------------------------
# Graded format penalties + reward-fn integration
# ---------------------------------------------------------------------------


class _FailingVerifierStub:
    """Verifier stub whose parse always fails → Perturber wins by default."""

    def generate(self, text, n_completions=1, **kwargs):
        return [("", None, "stub: no output")] * n_completions


class TestGradedPenalties:
    def _reward_fn(self):
        return make_perturber_reward_fn(
            verifier_model=_FailingVerifierStub(),
            reward_config={"format_penalty": -10.0, "format_penalty_soft": -5.0},
            mode="math",
        )

    def test_penalties_graded_by_failure_stage(self):
        reward_fn = self._reward_fn()
        valid = _perturber_json(
            SOLUTION.replace("= 60.", "= 75."),
            [_error("err_001", "= 60.", "= 75.")],
        )
        schema_bad = _perturber_json(SOLUTION.replace("= 60.", "= 75."), [])  # 0 errors, k=1
        json_bad = "utter garbage, not json"

        prompts = ["p1"] * 3
        completions = [json_bad, schema_bad, valid]
        rewards = reward_fn(
            prompts, completions,
            original_solution=[SOLUTION] * 3, k=[1] * 3,
        )
        assert rewards[0] == -10.0   # unparseable
        assert rewards[1] == -5.0    # parses but fails schema
        assert rewards[2] > 0.0      # valid, verifier failed → perturber wins
        # The graded penalties give this all-same-prompt group nonzero variance
        assert max(rewards) - min(rewards) > 0

    def test_backoff_applies_inside_reward_fn(self):
        # Unmodified solution + valid errors must be salvaged (scored), not
        # penalized — mirrors the rollout pipeline.
        reward_fn = self._reward_fn()
        unmodified = _perturber_json(SOLUTION, [_error("err_001", "= 60.", "= 75.")])
        rewards = reward_fn(
            ["p1"], [unmodified], original_solution=[SOLUTION], k=[1],
        )
        assert rewards[0] > 0.0


# ---------------------------------------------------------------------------
# sign_error alias
# ---------------------------------------------------------------------------


class TestSignErrorAlias:
    def test_alias_resolves(self):
        assert _fuzzy_match_math_error_type("sign_error") == MathErrorType(
            "negative_number_error"
        )

    def test_alias_works_through_schema(self):
        err = MathInjectedError(
            error_id="err_001",
            step_index=0,
            original_text="x = 2",
            injected_text="x = -2",
            error_type="sign_error",
            rationale="flipped the sign",
        )
        assert err.error_type == MathErrorType("negative_number_error")


# ---------------------------------------------------------------------------
# Generation-kwargs translation
# ---------------------------------------------------------------------------


class TestPrepareGenerationKwargs:
    CFG = {
        "max_new_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "repetition_penalty": 1.05,
        "n_completions": 3,
    }

    def test_vllm_translation(self):
        kw = prepare_generation_kwargs(self.CFG, "vllm", stop_after_json=True)
        assert kw["max_tokens"] == 2048
        assert "max_new_tokens" not in kw
        assert "do_sample" not in kw
        assert "n_completions" not in kw
        assert kw["stop"] == ["}\n```"]
        assert kw["include_stop_str_in_output"] is True
        assert kw["repetition_penalty"] == 1.05

    def test_vllm_do_sample_false_forces_greedy(self):
        kw = prepare_generation_kwargs({"do_sample": False, "temperature": 0.6}, "vllm")
        assert kw["temperature"] == 0.0

    def test_hf_translation(self):
        kw = prepare_generation_kwargs(self.CFG, "hf")
        assert kw["max_new_tokens"] == 2048
        assert kw["do_sample"] is True
        assert "n_completions" not in kw
        assert "stop" not in kw

    def test_default_token_cap_applied(self):
        assert prepare_generation_kwargs({}, "vllm")["max_tokens"] == 2048
        assert prepare_generation_kwargs({}, "hf")["max_new_tokens"] == 2048
