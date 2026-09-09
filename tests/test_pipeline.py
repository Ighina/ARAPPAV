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
        # With no stderr there is nothing better to say than the return code.
        assert "rc=1" in self._res("something", rc=1).infra_failure()

    def test_failure_reports_the_cause_not_the_transport(self):
        # An API backend has no `claude` process, so the old "claude exited N"
        # wording was misleading; the provider's own message is what helps.
        r = self._res("", rc=2, stderr="APIStatusError: Error code: 402 - "
                                       "Insufficient Balance")
        assert "402" in r.infra_failure() and "claude" not in r.infra_failure()

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


class TestUpdaterModel:
    """Policy authoring can use a different model from episode play."""

    def test_defaults_to_the_player_model(self):
        assert PipelineConfig(model="claude-haiku-4-5").policy_model() == "claude-haiku-4-5"

    def test_override_separates_the_two_roles(self):
        c = PipelineConfig(model="claude-haiku-4-5", updater_model="claude-opus-5")
        assert c.policy_model() == "claude-opus-5"
        assert c.model == "claude-haiku-4-5"

    def test_both_unset_is_none(self):
        assert PipelineConfig().policy_model() is None


class TestBackends:
    """Both transports must deliver the same policy and fail the same way."""

    def _stub(self, monkeypatch, model="claude-haiku-4-5"):
        import types
        import anthropic
        from arappav.pipeline import backends as B

        class Resp:
            content = [types.SimpleNamespace(type="text", text='{"claims": []}')]
            usage = types.SimpleNamespace(input_tokens=500, output_tokens=40,
                                          cache_read_input_tokens=2100,
                                          cache_creation_input_tokens=0)

        class Msgs:
            def __init__(self): self.calls = []
            def create(self, **kw): self.calls.append(kw); return Resp()

        class Client:
            def __init__(self, **kw): self.messages = Msgs()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(anthropic, "Anthropic", Client)
        return B.make_backend("api", model=model,
                              skills_root=Path(".claude/skills"))

    def test_api_requires_a_key(self, monkeypatch):
        from arappav.pipeline import backends as B
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            B.make_backend("api", model="claude-haiku-4-5")

    def test_cli_backend_needs_no_key(self, monkeypatch):
        from arappav.pipeline import backends as B
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert B.make_backend("claude-code").name == "claude-code"

    def test_unknown_backend_rejected(self):
        from arappav.pipeline import backends as B
        with pytest.raises(SystemExit, match="unknown backend"):
            B.make_backend("grpc")

    def test_skill_text_strips_frontmatter_and_keeps_policy(self):
        from arappav.pipeline.backends import skill_text
        t = skill_text(Path(".claude/skills"), "verify-v1")
        assert not t.startswith("---") and "Output contract" in t

    def test_policy_is_sent_as_a_cached_system_prompt(self, monkeypatch, tmp_path):
        b = self._stub(monkeypatch)
        b.run(skill="verify-v1", user="item", step="s", round_dir=tmp_path)
        kw = b._client.messages.calls[0]
        assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "Output contract" in kw["system"][0]["text"]
        # the per-item turn carries only the item, never the policy
        assert kw["messages"][0]["content"] == "item"

    def test_policy_read_once_across_items(self, monkeypatch, tmp_path):
        b = self._stub(monkeypatch)
        for i in range(3):
            b.run(skill="verify-v1", user=f"item{i}", step="s", round_dir=tmp_path)
        assert len(b._cache) == 1 and len(b._client.messages.calls) == 3

    def test_haiku_omits_thinking_but_opus_gets_it(self, monkeypatch, tmp_path):
        h = self._stub(monkeypatch)
        h.run(skill="verify-v1", user="x", step="s", round_dir=tmp_path)
        assert "thinking" not in h._client.messages.calls[0]
        o = self._stub(monkeypatch, model="claude-opus-5")
        o.run(skill="verify-v1", user="x", step="s", round_dir=tmp_path)
        assert o._client.messages.calls[0]["thinking"] == {"type": "adaptive"}

    def test_quota_error_is_infrastructure_not_data(self, monkeypatch, tmp_path):
        b = self._stub(monkeypatch)

        class Boom:
            def create(self, **kw): raise RuntimeError("You've hit your session limit")
        b._client.messages = Boom()
        r = b.run(skill="verify-v1", user="x", step="s", round_dir=tmp_path)
        assert r.returncode == 1 and r.infra_failure() is not None

    def test_usage_is_reported_for_costing(self, monkeypatch, tmp_path):
        b = self._stub(monkeypatch)
        r = b.run(skill="verify-v1", user="x", step="s", round_dir=tmp_path)
        assert r.usage["cache_read"] == 2100 and r.usage["output_tokens"] == 40


class TestBatchBackend:
    """Batch mode: submit all, poll, collect by custom_id."""

    def _stub(self, monkeypatch, fail_one=False):
        import types
        import anthropic
        from arappav.pipeline import backends as B

        def res(cid, ok=True):
            u = types.SimpleNamespace(input_tokens=400, output_tokens=30,
                                      cache_read_input_tokens=2100,
                                      cache_creation_input_tokens=0)
            msg = types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text='{"claims": []}')],
                usage=u)
            return types.SimpleNamespace(
                custom_id=cid,
                result=types.SimpleNamespace(
                    type="succeeded" if ok else "expired", message=msg))

        class Batches:
            def __init__(self): self.submitted = None; self.polls = 0
            def create(self, requests):
                self.submitted = requests
                return types.SimpleNamespace(id="msgbatch_test")
            def retrieve(self, bid):
                self.polls += 1
                return types.SimpleNamespace(
                    processing_status="ended" if self.polls > 1 else "in_progress",
                    request_counts=types.SimpleNamespace(
                        processing=0, succeeded=2, errored=0, canceled=0, expired=0))
            def results(self, bid):
                ids = [r["custom_id"] for r in self.submitted]
                out = [res(i) for i in ids]
                if fail_one:
                    out[-1] = res(ids[-1], ok=False)
                return out

        class Client:
            def __init__(self, **kw):
                self.messages = types.SimpleNamespace(batches=Batches())

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(anthropic, "Anthropic", Client)
        return B.make_backend("batch", model="claude-haiku-4-5",
                              skills_root=Path(".claude/skills"),
                              poll_interval=0, max_wait=10)

    def test_custom_ids_are_legal_and_bounded(self):
        from arappav.pipeline.backends import _custom_id
        cid = _custom_id("hverify-v10", "gsm8k-89")
        assert cid == "hverify-v10__gsm8k-89"
        assert len(_custom_id("x" * 60, "y" * 60)) == 64
        assert _custom_id("a/b", "c d").replace("-", "").replace("_", "").isalnum()

    def test_one_request_per_item_sharing_a_cached_system_prompt(self, monkeypatch):
        b = self._stub(monkeypatch)
        b.submit("verify-v1", [("a", "A"), ("b", "B")])
        reqs = b._client.messages.batches.submitted
        assert len(reqs) == 2
        assert reqs[0]["params"]["system"][0]["cache_control"] == {"type": "ephemeral"}
        # the same system text object is reused, so the cache actually hits
        assert reqs[0]["params"]["system"][0]["text"] is reqs[1]["params"]["system"][0]["text"]
        assert reqs[0]["params"]["messages"][0]["content"] == "A"

    def test_results_are_keyed_by_id_not_position(self, monkeypatch):
        b = self._stub(monkeypatch)
        b.submit("verify-v1", [("gsm8k-89", "A"), ("math-159", "B")])
        texts, failed, usage = b.collect("verify-v1", "msgbatch_test")
        assert set(texts) == {"gsm8k-89", "math-159"} and not failed
        assert usage["cache_read"] == 4200

    def test_failed_items_are_not_returned_as_text(self, monkeypatch):
        # An expired/errored request must yield no output file, so a re-run
        # retries it rather than scoring an empty answer.
        b = self._stub(monkeypatch, fail_one=True)
        b.submit("verify-v1", [("a", "A"), ("b", "B")])
        texts, failed, _ = b.collect("verify-v1", "msgbatch_test")
        assert set(texts) == {"a"} and failed == {"b": "expired"}

    def test_status_reports_terminal_state(self, monkeypatch):
        b = self._stub(monkeypatch)
        b.submit("verify-v1", [("a", "A")])
        assert b.status("msgbatch_test")[0] == "in_progress"
        assert b.status("msgbatch_test")[0] == "ended"


class TestApiProviders:
    """The api backend covers Anthropic, OpenAI and DeepSeek."""

    def _openai_stub(self, monkeypatch, model, env):
        import types
        import openai
        from arappav.pipeline import backends as B

        class Cli:
            def __init__(self, **kw):
                self.chat = types.SimpleNamespace(completions=self)
                self.seen = None
            def create(self, **kw):
                self.seen = kw
                raise RuntimeError("stop-after-capture")

        monkeypatch.setenv(env, "sk-test")
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: Cli())
        b = B.make_backend("api", model=model, skills_root=Path(".claude/skills"))
        b.run(skill="verify-v1", user="item", step="s", round_dir=Path("/tmp"))
        return b

    @pytest.mark.parametrize("model,expected", [
        ("claude-haiku-4-5", "anthropic"), ("claude-opus-5", "anthropic"),
        ("gpt-5", "openai"), ("o3", "openai"), ("chatgpt-4o-latest", "openai"),
        ("deepseek-chat", "deepseek"), ("deepseek-reasoner", "deepseek")])
    def test_provider_inferred_from_model_id(self, model, expected):
        from arappav.pipeline.backends import infer_provider
        assert infer_provider(model) == expected

    def test_unknown_model_must_be_named_explicitly(self):
        from arappav.pipeline.backends import infer_provider
        with pytest.raises(SystemExit, match="cannot infer a provider"):
            infer_provider("llama-3")

    def test_openai_reasoning_model_gets_effort_and_completion_tokens(self, monkeypatch):
        kw = self._openai_stub(monkeypatch, "gpt-5", "OPENAI_API_KEY")._client.seen
        assert kw["max_completion_tokens"] == 8000
        assert kw["reasoning_effort"] == "high"

    def test_openai_non_reasoning_model_gets_no_effort(self, monkeypatch):
        # reasoning_effort on a non-reasoning model is an API error.
        kw = self._openai_stub(monkeypatch, "gpt-4o", "OPENAI_API_KEY")._client.seen
        assert "reasoning_effort" not in kw

    def test_deepseek_uses_max_tokens_and_its_own_key(self, monkeypatch):
        b = self._openai_stub(monkeypatch, "deepseek-chat", "DEEPSEEK_API_KEY")
        assert b.provider == "deepseek"
        assert b._client.seen["max_tokens"] == 8000
        assert "max_completion_tokens" not in b._client.seen

    def test_policy_travels_as_a_system_message(self, monkeypatch):
        kw = self._openai_stub(monkeypatch, "gpt-4o", "OPENAI_API_KEY")._client.seen
        assert [m["role"] for m in kw["messages"]] == ["system", "user"]
        assert "Output contract" in kw["messages"][0]["content"]
        assert kw["messages"][1]["content"] == "item"

    def test_missing_key_names_the_right_variable(self, monkeypatch):
        from arappav.pipeline import backends as B
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
            B.make_backend("api", model="deepseek-chat")

    def test_batch_is_anthropic_only(self, monkeypatch):
        from arappav.pipeline import backends as B
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(SystemExit, match="Anthropic-only"):
            B.make_backend("batch", model="gpt-5")


class TestValidationProtocol:
    """MATH-500 selects; ProcessBench reports. They must stay separate."""

    def _mk(self, tmp_path, versions):
        """A validation root with hand-written answers and verifier outputs."""
        import json
        (tmp_path / "answers").mkdir(parents=True)
        (tmp_path / "inbox").mkdir(parents=True)
        # one perturbed item, one clean item
        pert = {"episode_id": "p1", "clean": False, "k": 1,
                "problem": "2+2?", "original_solution": "2+2 = 4.",
                "solution_to_review": "2+2 = 5.",
                "errors": [{"error_id": "err_001", "step_index": 0,
                            "original_text": "2+2 = 4", "injected_text": "2+2 = 5",
                            "error_type": "wrong_operation", "rationale": "4 not 5"}]}
        clean = {"episode_id": "c1", "clean": True, "k": 0, "problem": "1+1?",
                 "original_solution": "1+1 = 2.", "solution_to_review": "1+1 = 2.",
                 "errors": []}
        for r in (pert, clean):
            (tmp_path / "answers" / f"{r['episode_id']}.json").write_text(json.dumps(r))
        for skill, (pclaim, cclaim) in versions.items():
            d = tmp_path / "outbox" / skill
            d.mkdir(parents=True)
            (d / "p1.json").write_text(pclaim)
            (d / "c1.json").write_text(cclaim)
        return tmp_path

    def test_catching_the_error_beats_missing_it(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v500", Path("scripts/validate_math500.py"))
        v500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v500)
        root = self._mk(tmp_path, {
            "hv-v1": ('{"claims":[{"quoted_text":"2+2 = 5","explanation":"should be 4"}]}',
                      '{"claims":[]}'),
            "hv-v2": ('{"claims":[]}', '{"claims":[]}'),
        })
        cfg = scoring.load_reward_config()
        good = v500._score_version(root, "hv-v1", cfg)
        bad = v500._score_version(root, "hv-v2", cfg)
        assert good["mean_verifier_f1"] > bad["mean_verifier_f1"]

    def test_false_alarms_on_clean_items_are_counted_separately(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v500b", Path("scripts/validate_math500.py"))
        v500 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v500)
        root = self._mk(tmp_path, {
            "hv-v1": ('{"claims":[]}', '{"claims":[]}'),                    # silent
            "hv-v2": ('{"claims":[]}',
                      '{"claims":[{"quoted_text":"1+1 = 2","explanation":"x"}]}'),
        })
        cfg = scoring.load_reward_config()
        assert v500._score_version(root, "hv-v1", cfg)["false_alarm_rate"] == 0.0
        assert v500._score_version(root, "hv-v2", cfg)["false_alarm_rate"] == 1.0

    def test_validation_inbox_carries_no_ground_truth(self, tmp_path):
        # The verifier's view must be problem + text only: the original solution
        # would give the diff away, and the error list is the answer.
        from arappav.pipeline.contracts import VERIFY_INPUT_FIELDS
        assert set(VERIFY_INPUT_FIELDS) == {"problem", "solution_to_review"}
        for forbidden in ("original_solution", "errors", "k", "clean"):
            assert forbidden not in VERIFY_INPUT_FIELDS
