"""Validation for the deterministic pipeline (spec section 18).

Covers the data contract, the leakage boundary, cold/warm start, freezing,
reward/finding computation and reporting. Everything here runs offline: no
`claude` invocation, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arappav.pipeline import policies, scoring
from arappav.pipeline.agents import ContextLedger, render_context_block
from arappav.pipeline.contracts import (
    GROUND_TRUTH_MARKERS, Episode, LeakageError, assert_problem_unchanged,
    perturb_payload, render_perturb_prompt, render_verify_prompt, verify_payload,
)
from arappav.pipeline.orchestrator import PipelineConfig, Pipeline, _summary_table

EP = Episode(episode_id="ep00", problem="What is 2+2?", solution="2+2 = 4. So $\\boxed{4}$.",
             k=3, source_id="algebra_1_0", topic="algebra", level="Level 1")


# --------------------------------------------------------------------- data
class TestDataContract:
    def test_data_json_is_exactly_the_dataset_pair(self):
        assert EP.data() == {"problem": EP.problem, "solution": EP.solution}

    def test_meta_is_separate_from_data(self):
        assert "problem" not in EP.meta() and "solution" not in EP.meta()
        assert EP.meta()["k"] == 3 and EP.meta()["episode_id"] == "ep00"

    def test_roundtrip(self):
        assert Episode.load(EP.data(), EP.meta()) == EP


# ----------------------------------------------------------------- leakage
class TestPerturberBoundary:
    def test_receives_problem_solution_k_only(self):
        assert set(perturb_payload(EP)) == {"problem", "solution", "k"}

    def test_prompt_marks_problem_read_only_and_solution_mutable(self):
        p = render_perturb_prompt(EP, "/perturb-v1")
        assert "read-only" in p and "ONLY field you may perturb" in p
        assert EP.problem in p and EP.solution in p

    def test_modified_problem_is_rejected(self):
        assert_problem_unchanged(EP, EP.problem)          # verbatim echo is fine
        assert_problem_unchanged(EP, None)                # no echo is fine
        with pytest.raises(LeakageError, match="MODIFIED problem"):
            assert_problem_unchanged(EP, "What is 3+3?")


class TestVerifierBoundary:
    def test_receives_problem_and_perturbed_solution_only(self):
        v = verify_payload(EP, "2+2 = 5.")
        assert set(v) == {"problem", "solution_to_review"}

    def test_original_solution_and_k_never_reach_the_verifier(self):
        p = render_verify_prompt(EP, "2+2 = 5. So $\\boxed{5}$.", "/verify-v1")
        assert EP.solution not in p          # the unperturbed original
        assert "## k" not in p               # the error budget

    def test_ground_truth_structures_are_refused(self):
        leaked = '2+2 = 5. {"errors": [{"rationale": "x"}]}'
        with pytest.raises(LeakageError, match="REFUSING TO SEND"):
            render_verify_prompt(EP, leaked, "/verify-v1")

    def test_reward_state_is_refused(self):
        with pytest.raises(LeakageError):
            render_verify_prompt(EP, "ok perturber_reward 0.5", "/verify-v1")

    def test_markers_cover_the_ground_truth_keys(self):
        for key in ("injected_text", "original_text", "rationale", "error_id"):
            assert key in GROUND_TRUTH_MARKERS


class TestContextControl:
    def test_no_context_suppresses_history(self):
        assert render_context_block({"round": 0}, no_context=True) == ""

    def test_context_block_rendered_when_allowed(self):
        assert "round" in render_context_block({"round": 0}, no_context=False)

    def test_ledger_records_every_step(self, tmp_path):
        led = ContextLedger(tmp_path, no_context=False)
        led.record("perturb", "ep00", {"a": 1}, "prompt text", None)
        out = json.loads(led.flush().read_text())
        assert out["steps"][0]["step"] == "perturb"
        assert out["steps"][0]["allowed_context"] == {"a": 1}
        assert out["steps"][0]["prompt_sha256"]

    def test_findings_slices_differ_by_role(self):
        f = scoring.build_findings(0, [], {"n": 1})
        pf = Pipeline._findings_for("perturb", f)
        vf = Pipeline._findings_for("verify", f)
        # the verifier is not shown format failures or unit collapse: those are
        # facts about the perturber's own output, not evidence about verifying.
        assert "unit_collapse" in pf and "unit_collapse" not in vf
        assert "verifier_false_positives" in vf


# ---------------------------------------------------------------- policies
class TestPolicies:
    def test_cold_start_policy_is_empty(self, tmp_path):
        p = policies.write_version(tmp_path, "perturb", "perturb", 1, "", "cold start")
        assert "No policy" in policies.extract_policy(p.read_text())

    def test_warm_start_policy_is_spliced_in(self, tmp_path):
        body = "1. Place errors on independent branches."
        p = policies.write_version(tmp_path, "perturb", "perturb", 1, body, "warm")
        assert body in policies.extract_policy(p.read_text())

    def test_invariant_is_identical_across_versions(self, tmp_path):
        a = policies.write_version(tmp_path, "verify", "verify", 1, "1. first", "x")
        b = policies.write_version(tmp_path, "verify", "verify", 2, "1. second", "y")
        assert policies.invariant_of(a.read_text()) == policies.invariant_of(b.read_text())
        assert policies.check_invariant(tmp_path, "verify", "verify", 1, 2) == []

    def test_tampered_invariant_is_detected(self, tmp_path):
        policies.write_version(tmp_path, "verify", "verify", 1, "1. first", "x")
        b = policies.write_version(tmp_path, "verify", "verify", 2, "1. second", "y")
        txt = b.read_text()
        marker = "recall    = detected error units"
        assert marker in txt, "invariant anchor moved; update this test"
        b.write_text(txt.replace(marker, "recall    = TAMPERED"))
        assert any("invariant" in p for p in
                   policies.check_invariant(tmp_path, "verify", "verify", 1, 2))

    def test_unchanged_policy_is_flagged(self, tmp_path):
        policies.write_version(tmp_path, "verify", "verify", 1, "1. same", "x")
        policies.write_version(tmp_path, "verify", "verify", 2, "1. same", "y")
        assert any("unchanged" in p for p in
                   policies.check_invariant(tmp_path, "verify", "verify", 1, 2))

    def test_refuses_to_clobber_without_overwrite(self, tmp_path):
        policies.write_version(tmp_path, "verify", "verify", 1, "1. a", "x")
        with pytest.raises(policies.PolicyError, match="already exists"):
            policies.write_version(tmp_path, "verify", "verify", 1, "1. b", "y")

    def test_perturb_template_states_the_problem_is_read_only(self):
        t = (policies.TEMPLATES / "perturb_base.md").read_text()
        assert "read-only context" in t and "only field you may perturb" in t

    def test_verify_template_withholds_original_and_k(self):
        t = (policies.TEMPLATES / "verify_base.md").read_text()
        assert "given the original unperturbed solution" in t
        assert "the number of injected errors" in t


class TestFreeze:
    @pytest.mark.parametrize("freeze,pf,vf", [
        ("none", False, False), ("perturber", True, False),
        ("verifier", False, True), ("both", True, True)])
    def test_freeze_flags_are_independent(self, freeze, pf, vf):
        c = PipelineConfig(freeze=freeze)
        assert c.freeze_perturber() is pf and c.freeze_verifier() is vf

    def test_frozen_role_carries_its_policy_forward(self, tmp_path):
        cfg = PipelineConfig(freeze="perturber", skills_root=str(tmp_path),
                             root=str(tmp_path / "run"), dry_run=True)
        pipe = Pipeline(cfg)
        policies.write_version(tmp_path, "perturb", "perturb", 1, "1. original rule", "x")
        policies.write_version(tmp_path, "verify", "verify", 1, "1. verify rule", "x")
        pipe.rounds.append({"findings": scoring.build_findings(0, [], {})})
        led = ContextLedger(tmp_path / "run" / "round_1", no_context=True)
        pipe.update_policies(1, pipe.rounds[-1]["findings"], led)
        frozen = policies.extract_policy(
            policies.skill_file(tmp_path, "perturb", 2).read_text())
        assert "1. original rule" in frozen


# ----------------------------------------------------------------- scoring
class TestScoring:
    def test_format_invalid_gets_graded_penalty(self):
        cfg = scoring.load_reward_config()
        hard = scoring.score_episode(episode_id="e", k=3, perturbed=None,
                                     failure_stage="json", failure_reason="bad",
                                     verifier_raw="", config=cfg, history=None)
        soft = scoring.score_episode(episode_id="e", k=3, perturbed=None,
                                     failure_stage="schema", failure_reason="bad",
                                     verifier_raw="", config=cfg, history=None)
        assert hard["perturber_reward"] == cfg.get("format_penalty", -10.0)
        assert soft["perturber_reward"] == cfg.get("format_penalty_soft", -5.0)
        assert hard["scored"] is False

    def test_aggregate_handles_all_invalid_round(self):
        rows = [{"perturber_format_valid": False, "perturber_reward": -10.0, "k": 3}]
        m = scoring.aggregate(rows)
        assert m["format_valid_rate"] == 0.0 and m["num_episodes"] == 1

    def test_findings_shape_is_stable(self):
        f = scoring.build_findings(2, [], {"x": 1})
        for key in ("round", "metrics", "error_type_detection", "format_failures",
                    "undetected_errors", "detected_errors",
                    "verifier_false_positives", "unit_collapse"):
            assert key in f


class TestReporting:
    def test_summary_table_has_one_row_per_round(self):
        rounds = [{
            "round": i, "policies": {"perturb": f"perturb-v{i+1}", "verify": f"verify-v{i+1}"},
            "metrics": {"format_valid_rate": 1.0, "mean_perturber_reward": 0.1,
                        "mean_verifier_reward": 0.9, "mean_verifier_recall": 0.9,
                        "mean_verifier_precision": 0.9, "mean_units_per_episode": 3.0},
            "processbench": {"enabled": False},
        } for i in range(3)]
        t = _summary_table(rounds, PipelineConfig())
        assert t.count("\n") == 5                      # header + separator + 3 rows
        assert "disabled" in t

    def test_processbench_f1_appears_when_enabled(self):
        rounds = [{
            "round": 0, "policies": {"perturb": "p-v1", "verify": "v-v1"},
            "metrics": {"format_valid_rate": 1.0, "mean_perturber_reward": 0.0,
                        "mean_verifier_reward": 1.0, "mean_verifier_recall": 1.0,
                        "mean_verifier_precision": 1.0, "mean_units_per_episode": 3.0},
            "processbench": {"enabled": True, "summary": {"processbench_f1": 0.87}},
        }]
        assert "F1 0.87" in _summary_table(rounds, PipelineConfig())


# ------------------------------------------------- infrastructure failures
class TestInfrastructureFailure:
    """A call that never reached a model must never become a reward.

    Regression cover for the 10-round haiku run, where a session limit was
    scored as 171 format failures and produced a fabricated table.
    """

    def _res(self, text="", rc=0, stderr="", dry=False):
        from arappav.pipeline.agents import AgentResult
        return AgentResult("perturb", text, rc, 0.1, 10, "sha", stderr, dry)

    @pytest.mark.parametrize("text", [
        "You've hit your session limit · resets 5:50pm (Europe/London)",
        "Usage limit reached",
        "Rate limit exceeded, try again later",
        "Insufficient credit balance",
    ])
    def test_quota_messages_are_infrastructure_not_data(self, text):
        assert self._res(text, rc=1).infra_failure() is not None

    def test_timeout_is_infrastructure(self):
        assert self._res("", rc=124, stderr="timeout after 900s").infra_failure() == "timeout"

    def test_empty_response_is_infrastructure(self):
        assert self._res("   ", rc=0).infra_failure() == "empty response"

    def test_nonzero_exit_is_infrastructure(self):
        assert "exited 1" in self._res("something", rc=1).infra_failure()

    def test_a_genuinely_malformed_reply_is_data_not_infrastructure(self):
        # This one MUST be scored: the model answered, just badly.
        bad = '{"perturbed_solution": "x", "errors": ['      # truncated JSON
        assert self._res(bad, rc=0).infra_failure() is None

    def test_dry_run_is_never_infrastructure_failure(self):
        assert self._res("", rc=0, dry=True).infra_failure() is None

    def test_guard_raises_and_names_the_step(self):
        from arappav.pipeline.agents import InfrastructureError
        from arappav.pipeline.orchestrator import _guard
        with pytest.raises(InfrastructureError, match="session limit"):
            _guard(self._res("You've hit your session limit", rc=1), "round 3 ep05 perturb")

    def test_guard_passes_a_real_reply_through(self):
        from arappav.pipeline.orchestrator import _guard
        _guard(self._res('{"claims": []}', rc=0), "round 0 ep00 verify")   # no raise
