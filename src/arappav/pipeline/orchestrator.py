"""The deterministic pipeline orchestrator (spec 1, 3-6, 10-15).

Python owns the experiment. Skills are invoked as pure functions over inputs
this module constructs; none of them decides what happens next.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from arappav.pipeline import agents, policies, scoring
from arappav.pipeline.agents import (
    ContextLedger, InfrastructureError, render_context_block,
)
from arappav.pipeline.backends import make_backend
from arappav.pipeline.contracts import (
    Episode, LeakageError, assert_problem_unchanged,
    render_perturb_prompt, render_verify_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Configuration — every experiment knob is an explicit field (spec 1)
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    rounds: int = 3
    episodes: int = 8
    k: int = 3
    start: str = "cold"                 # "cold" | "warm"  (spec 3)
    freeze: str = "none"                # none|perturber|verifier|both (spec 5)
    source: str = "hendrycks"           # hendrycks|local|file
    problems_file: str | None = None
    topics: tuple[str, ...] = ("algebra",)
    # Restrict the whole experiment to one MATH subject. Training draws only
    # that topic and evaluation filters MATH-500 to the matching subject, so
    # "restricted" means the same thing on both sides.
    category: str | None = None
    per_topic: int = 50
    seed: int = 42
    root: str = "data/skill_rollouts"
    skills_root: str = ".claude/skills"
    perturb_prefix: str = "perturb"
    verify_prefix: str = "verify"
    model: str | None = None
    # Transport for every agent call. `claude-code` runs `claude -p`, which only
    # reaches Claude models; `api` reaches Anthropic, OpenAI and DeepSeek, and is
    # the only way to run this pipeline on a non-Claude model.
    backend: str = "claude-code"
    provider: str | None = None          # players; inferred from model when None
    updater_provider: str | None = None  # policy author; falls back to provider
    # Model for policy authoring (create-policy-*, update-*) as opposed to
    # playing episodes. Separating them isolates *policy quality* from *player
    # capability*: the 10-round haiku run degraded because the updater fitted
    # per-round noise while the players were competent (see HAIKU_FAILURE.md).
    updater_model: str | None = None
    processbench_enabled: bool = False  # spec 11
    processbench_per_subset: int = 5
    processbench_root: str = "data/skill_evals"
    processbench_seed: int = 0
    no_context: bool = False            # spec 21
    overwrite_policies: bool = False
    dry_run: bool = False
    timeout: int = 900
    max_tokens: int = 16000
    taxonomy_free: bool = False
    infra_retries: int = 2
    # Insert a summarise-context-* step before each policy update, so the
    # updater receives the text of the episodes rather than counts and ids.
    rich_context: bool = False
    # How a policy is revised between rounds. This is the ablation axis:
    #   rewrite   — one updater rewrites the whole policy body (the default)
    #   summarise — a summariser briefs the updater first (--rich-context)
    #   evolve    — parallel per-episode analysts propose typed patches, a merge
    #               step reconciles them, and Python applies the result
    update_mode: str = "rewrite"
    #: Keep a new policy only if it does not regress on a held-out check.
    accept_on_validation: bool = False
    validation_n: int = 12
    retry_format: int = 0   # extra attempts when a reply fails to parse
    resume: bool = False    # skip rounds that already have a summary

    def policy_model(self) -> str | None:
        """Model used to author policies; falls back to the player model."""
        return self.updater_model or self.model

    def policy_provider(self) -> str | None:
        return self.updater_provider or (
            None if self.updater_model else self.provider)

    def freeze_perturber(self) -> bool:
        return self.freeze in ("perturber", "both")

    def freeze_verifier(self) -> bool:
        return self.freeze in ("verifier", "both")


# ---------------------------------------------------------------------------
# Dataset sampling (spec 1, 2)
# ---------------------------------------------------------------------------


def _pool(cfg: PipelineConfig) -> list[dict]:
    if cfg.source == "file":
        if not cfg.problems_file:
            sys.exit("[pipeline] --problems-file is required with --source file")
        rows = []
        for i, line in enumerate(Path(cfg.problems_file).read_text().splitlines()):
            if line.strip():
                r = json.loads(line)
                rows.append({"source_id": r.get("source_id", f"item_{i}"),
                             "problem": r["problem"], "solution": r["solution"],
                             "topic": r.get("topic"), "level": r.get("level")})
        return rows
    if cfg.source == "local":
        pool: dict[str, dict] = {}
        for pat in ("data/rollouts_math/*.jsonl", "new_rollouts/*.jsonl"):
            for path in sorted(REPO_ROOT.glob(pat)):
                if "verifier" in path.name:
                    continue
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = rec.get("paper_id") or rec.get("chunk_id")
                    if not (rec.get("original_text") and rec.get("original_solution") and pid):
                        continue
                    pool.setdefault(pid, {
                        "source_id": pid, "problem": rec["original_text"],
                        "solution": rec["original_solution"],
                        "topic": str(pid).partition("_")[0], "level": None})
        return list(pool.values())

    from arappav.data.categories import train_topics
    from arappav.data.ingest_math import load_math_dataset
    topics = train_topics(cfg.category) if cfg.category else list(cfg.topics)
    ds = load_math_dataset(topics=topics, split="train",
                           max_examples_per_topic=cfg.per_topic, seed=cfg.seed)
    return [{"source_id": f"{r['topic']}_{r['level']}_{i}", "problem": r["problem"],
             "solution": r["solution"], "topic": r["topic"], "level": r["level"]}
            for i, r in enumerate(ds)]


def sample_episodes(cfg: PipelineConfig, round_index: int, seen: set[str]) -> list[Episode]:
    pool = [p for p in _pool(cfg) if p["source_id"] not in seen]
    if not pool:
        sys.exit("[pipeline] problem pool exhausted.")
    rng = random.Random(cfg.seed + round_index)
    chosen = rng.sample(pool, min(cfg.episodes, len(pool)))
    return [
        Episode(episode_id=f"ep{i:02d}", problem=c["problem"], solution=c["solution"],
                k=cfg.k, source_id=c["source_id"], topic=c.get("topic"),
                level=c.get("level"), round_index=round_index)
        for i, c in enumerate(chosen)
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.skills = Path(cfg.skills_root)
        self.reward_cfg = scoring.load_reward_config()
        self.history: list = []          # cross-round anti-duplicate corpus
        self.rounds: list[dict] = []
        self._players = None
        self._author = None
        self._retrying = False
        self._briefings: dict[str, dict] = {}
        self._last_scores: list[dict] = []

    def _backend(self, which: str):
        """Lazily build the play / policy-authoring backend.

        Built on demand so a --dry-run needs no credentials at all.
        """
        cfg = self.cfg
        if which == "players":
            if self._players is None:
                self._players = make_backend(
                    cfg.backend, model=cfg.model, timeout=cfg.timeout,
                    skills_root=self.skills, max_tokens=cfg.max_tokens,
                    **({"provider": cfg.provider} if cfg.backend != "claude-code" else {}))
            return self._players
        if self._author is None:
            self._author = make_backend(
                cfg.backend, model=cfg.policy_model(), timeout=cfg.timeout,
                skills_root=self.skills, max_tokens=cfg.max_tokens,
                **({"provider": cfg.policy_provider()} if cfg.backend != "claude-code" else {}))
        return self._author

    def _call(self, which: str, *, skill: str | None, user: str, step: str,
              round_dir: Path, episode_id: str | None = None):
        """One agent call through the configured transport.

        `skill` is the policy to apply. Under `claude-code` it becomes a
        `/slash` command inside the prompt; under `api` it is loaded from disk
        and sent as a cached system prompt. Steps with no versioned policy
        (create-policy-*, update-*, final_summary) pass skill=None and carry
        their instruction in the prompt body instead.
        """
        cfg = self.cfg
        if cfg.dry_run:
            from arappav.pipeline.agents import run_claude
            text = f"/{skill}\n\n{user}" if skill else user
            return run_claude(text, step=step, round_dir=round_dir,
                              episode_id=episode_id, model=cfg.model,
                              timeout=cfg.timeout, dry_run=True)
        backend = self._backend(which)
        if skill is None:
            # No policy file to send: the whole instruction is the prompt. The
            # CLI backend takes it as-is; the API backend needs a system prompt,
            # so it gets a minimal one.
            return backend.run_raw(user=user, step=step, round_dir=round_dir,
                                   episode_id=episode_id)
        return backend.run(skill=skill, user=user, step=step,
                           round_dir=round_dir, episode_id=episode_id)

    # -- paths ------------------------------------------------------------
    def round_dir(self, i: int) -> Path:
        return self.root / f"round_{i}"

    def ep_dir(self, i: int, eid: str) -> Path:
        return self.round_dir(i) / "episodes" / eid

    # -- policies (spec 3, 4, 5) -----------------------------------------
    def _policy_version(self, i: int) -> int:
        return i + 1                      # round_0 -> v1

    def init_policies(self, ledger: ContextLedger) -> None:
        """Round 0: cold (empty policy) or warm (create-policy-* skills)."""
        cfg, v = self.cfg, self._policy_version(0)
        for role, prefix, skill in (
            ("perturb", cfg.perturb_prefix, "create-policy-perturber"),
            ("verify", cfg.verify_prefix, "create-policy-verifier"),
        ):
            if cfg.start == "cold":
                body, note = "", "cold start — empty policy"
            else:
                prompt = (f"/{skill}\n\n"
                          "Return ONLY the markdown body of the policy section: a short list "
                          "of numbered strategy rules. No headings, no preamble, no fences.\n")
                res = self._call("author", skill=None, user=prompt,
                                 step=f"create-policy-{role}",
                                 round_dir=self.round_dir(0))
                ledger.record(f"create-policy-{role}", None, {}, prompt, res)
                _guard(res, f"create-policy-{role}")
                body = res.text if res.ok() else ""
                note = (f"warm start — policy authored by {skill} "
                        f"({cfg.policy_model() or 'default model'})")
                if not body:
                    note += " (empty: the skill returned nothing)"
            policies.write_version(self.skills, role, prefix, v, body, note,
                                   parent="—", tuned_from="—",
                                   overwrite=cfg.overwrite_policies
                                   or getattr(self, "_retrying", False))
            print(f"[policy] {prefix}-v{v}: {note}")

    def update_policies(self, i: int, findings: dict, ledger: ContextLedger) -> None:
        """Rounds >= 1: bump each policy unless frozen (spec 4, 5)."""
        cfg = self.cfg
        prev, new_v = self._policy_version(i - 1), self._policy_version(i)
        for role, prefix, skill, frozen in (
            ("perturb", cfg.perturb_prefix, "update-perturb", cfg.freeze_perturber()),
            ("verify", cfg.verify_prefix, "update-verify", cfg.freeze_verifier()),
        ):
            old_file = policies.skill_file(self.skills, prefix, prev)
            old_body = policies.extract_policy(old_file.read_text())
            if frozen:
                # Retain the previous policy unchanged, re-published under the
                # new version so every round has a version of its own.
                policies.write_version(self.skills, role, prefix, new_v, old_body,
                                       f"FROZEN — identical to {prefix}-v{prev}",
                                       parent=f"{prefix}-v{prev}", tuned_from="—",
                                       overwrite=cfg.overwrite_policies
                                       or getattr(self, "_retrying", False))
                print(f"[policy] {prefix}-v{new_v}: FROZEN (unchanged from v{prev})")
                continue

            # evolve mode replaces the rewrite entirely: analysts propose typed
            # edits and Python applies them, so no updater call happens here.
            if cfg.update_mode == "evolve":
                got = self.evolve_policy(i, role, self._last_scores, old_body, ledger)
                if got is None:
                    body, note = old_body, f"evolve produced no admissible patch; carried v{prev}"
                else:
                    body, rec = got
                    note = (f"evolved from round {i-1}: {rec['applied']['num_edits']} "
                            f"edits {rec['applied']['by_op']}")
                body = self._gate_on_validation(i, role, old_body, body, prefix) \
                    if cfg.accept_on_validation else body
                policies.write_version(self.skills, role, prefix, new_v, body, note,
                                       parent=f"{prefix}-v{prev}", tuned_from=str(i - 1),
                                       overwrite=cfg.overwrite_policies
                                       or getattr(self, "_retrying", False))
                print(f"[policy] {prefix}-v{new_v}: {note}")
                continue

            brief = self._briefings.get(role) if cfg.rich_context else None
            allowed = brief if brief else self._findings_for(role, findings)
            prompt = (
                f"/{skill}\n\n"
                f"{render_context_block(allowed, cfg.no_context)}"
                "## CURRENT POLICY\n"
                f"{old_body}\n\n"
                "Return ONLY the markdown body of the NEXT policy section — the revised "
                "numbered rules. No headings, no preamble, no fences, no changelog.\n"
            )
            res = self._call("author", skill=None, user=prompt, step=f"{skill}",
                             round_dir=self.round_dir(i))
            ledger.record(skill, None, allowed, prompt, res)
            _guard(res, f"round {i} {skill}")
            body = res.text if res.ok() else old_body
            if res.ok():
                note = f"tuned from round {i-1}"
            elif res.dry_run:
                note = f"dry run — carried v{prev} forward unchanged"
            else:
                note = (f"update produced no policy (rc={res.returncode}); "
                        f"carried v{prev} forward")
            policies.write_version(self.skills, role, prefix, new_v, body, note,
                                   parent=f"{prefix}-v{prev}", tuned_from=str(i - 1),
                                   overwrite=cfg.overwrite_policies
                                   or getattr(self, "_retrying", False))
            print(f"[policy] {prefix}-v{new_v}: {note}")

    def _gate_on_validation(self, i: int, role: str, old_body: str,
                            new_body: str, prefix: str) -> str:
        """Keep a revision only if a held-out check does not get worse.

        Self-play reward is not a usable acceptance signal — measured over ten
        rounds it correlates with held-out F1 at r = +0.275, and selecting on it
        returns a policy worse than the untuned baseline. So acceptance is
        judged on the same MATH-500 validation set used to pick the final round,
        and a revision that regresses is discarded rather than carried forward.
        """
        cfg = self.cfg
        if role != "verify":
            # Only the verifier has a cheap held-out score. The perturber's
            # would need a full perturb+verify pass per candidate, which costs
            # more than the round that produced it.
            return new_body
        try:
            before = self._validation_score(i, prefix, old_body)
            after = self._validation_score(i, prefix, new_body)
        except Exception as e:
            print(f"[accept] validation unavailable ({e}); keeping the revision")
            return new_body
        keep = after >= before
        print(f"[accept] {role}: validation {before:.3f} -> {after:.3f} — "
              f"{'kept' if keep else 'REVERTED'}")
        _write(self.round_dir(i) / f"acceptance_{role}.json",
               {"before": before, "after": after, "kept": keep})
        return new_body if keep else old_body

    def _validation_score(self, i: int, prefix: str, body: str) -> float:
        """Mean verifier F1 for a candidate policy on a small fixed sample."""
        import tempfile
        from arappav.pipeline.backends import make_backend
        cfg = self.cfg
        vdir = Path(tempfile.mkdtemp())
        # Render the candidate as a throwaway policy so the backend can send it
        # exactly as it would in play.
        policies.write_version(vdir, "verify", "cand", 1, body, "validation candidate")
        items = self._validation_items(cfg.validation_n)
        bk = make_backend(cfg.backend, model=cfg.model, timeout=cfg.timeout,
                          skills_root=vdir, max_tokens=cfg.max_tokens,
                          **({"provider": cfg.provider} if cfg.backend != "claude-code" else {}))
        from arappav.errors.schema_math import MathPerturberOutput
        f1s = []
        for it in items:
            ep = Episode(episode_id=it["episode_id"], problem=it["problem"],
                         solution="", k=it["k"])
            res = bk.run(skill="cand-v1",
                         user=render_verify_prompt(ep, it["solution_to_review"], "", "").lstrip(),
                         step="accept-check", round_dir=self.round_dir(i),
                         episode_id=it["episode_id"])
            po = MathPerturberOutput.model_validate(
                {"perturbed_solution": it["solution_to_review"], "errors": it["errors"]})
            s = scoring.score_episode(episode_id=it["episode_id"], k=it["k"],
                                      perturbed=po, failure_stage=None,
                                      failure_reason=None, verifier_raw=res.text,
                                      config=self.reward_cfg, history=None)
            if s.get("verifier_reward") is not None:
                f1s.append(s["verifier_reward"])
        return sum(f1s) / len(f1s) if f1s else 0.0

    def _validation_items(self, n: int) -> list[dict]:
        """A fixed held-out set, built once and cached for the run."""
        cached = self.root / "acceptance_validation.json"
        if cached.exists():
            return json.loads(cached.read_text())[:n]
        # Reuse the MATH-500 validation set when one has been prepared.
        from arappav.data.categories import filter_math500
        from datasets import load_dataset
        import random
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        pool = filter_math500(ds, self.cfg.category)
        rng = random.Random(0)
        picked = rng.sample(pool, min(n, len(pool)))
        out = []
        for idx in picked:
            row = ds[idx]
            out.append({"episode_id": row["unique_id"].replace("/", "_"),
                        "problem": row["problem"],
                        "solution_to_review": row["solution"], "k": 0, "errors": []})
        _write(cached, out)
        return out

    # -- evolve mode -------------------------------------------------------
    def evolve_policy(self, i: int, role: str, scores: list[dict],
                      old_body: str, ledger: ContextLedger) -> tuple[str, dict] | None:
        """Parallel analysts -> merged patch -> deterministic application.

        One analyst per episode, none of which sees the others, so a change
        proposed by several is corroborated rather than merely repeated. The
        merge step reconciles them and Python applies the result, which is what
        makes `delete_rule` reachable at all: a rewrite can only drop a rule by
        forgetting it, whereas here removal is an operation someone has to name.
        """
        from arappav.pipeline import patches as P

        cfg = self.cfg
        analyst = f"evolve-analyse-{'perturber' if role == 'perturb' else 'verifier'}"
        evidence = scoring.episode_evidence(scores, role)

        proposals = []
        for ev in evidence:
            prompt = (f"/{analyst}\n\n## EPISODE\n"
                      + json.dumps(ev, indent=2, ensure_ascii=False)[:20000]
                      + f"\n\n## CURRENT POLICY\n{old_body}\n\n"
                      "Return only the JSON object.\n")
            res = self._call("author", skill=None, user=prompt,
                             step=f"evolve-analyse-{role}", round_dir=self.round_dir(i),
                             episode_id=ev["episode_id"])
            if not res.ok():
                continue
            from arappav.utils.parsing import extract_first_json_object, strip_json_fences
            obj, _ = extract_first_json_object(strip_json_fences(res.text))
            if isinstance(obj, dict) and obj.get("edits"):
                proposals.append({"episode_id": ev["episode_id"], **obj})
        ledger.record(analyst, None, {"episodes": len(evidence)},
                      f"{len(proposals)} proposals", None)

        if not proposals:
            print(f"[evolve] {role}: no analyst proposed a change — policy unchanged")
            return None

        merge_prompt = ("/evolve-merge-patches\n\n## PATCHES\n"
                        + json.dumps(proposals, indent=2, ensure_ascii=False)[:40000]
                        + f"\n\n## CURRENT POLICY\n{old_body}\n\n"
                        "Return only the JSON object.\n")
        mres = self._call("author", skill=None, user=merge_prompt,
                          step=f"evolve-merge-{role}", round_dir=self.round_dir(i))
        ledger.record("evolve-merge-patches", None, {"patches": len(proposals)},
                      merge_prompt, mres)
        if not mres.ok():
            print(f"[evolve] {role}: merge produced nothing — policy unchanged")
            return None
        from arappav.utils.parsing import extract_first_json_object, strip_json_fences
        obj, err = extract_first_json_object(strip_json_fences(mres.text))
        try:
            patch = P.Patch.parse(obj)
            new_body, log = P.apply_patch(old_body, patch)
        except (P.PatchError, TypeError) as e:
            print(f"[evolve] {role}: patch rejected ({e}) — policy unchanged")
            return None

        record = {"role": role, "num_proposals": len(proposals),
                  "applied": P.summarise_patch(patch), "log": log}
        _write(self.round_dir(i) / f"evolve_patch_{role}.json", record)
        print(f"[evolve] {role}: {len(proposals)} proposals -> "
              f"{record['applied']['num_edits']} edits {record['applied']['by_op']}")
        return new_body, record

    def summarise_context(self, i: int, role: str, scores: list[dict],
                          metrics: dict, ledger: ContextLedger) -> dict | None:
        """Compress the round into a text-grounded briefing for the updater.

        Without this the updater receives counts, identifiers and taxonomy
        labels — never the text of the errors it is being asked to reason
        about. The briefing is written to disk so the next round can carry it
        forward and so it is auditable after the fact.
        """
        cfg = self.cfg
        skill = f"summarise-context-{'perturber' if role == 'perturb' else 'verifier'}"
        prev_path = self.round_dir(i - 1) / f"summarised_context_{role}.json" if i else None
        prev = (json.loads(prev_path.read_text())
                if prev_path and prev_path.exists() else None)
        policy = policies.extract_policy(policies.skill_file(
            self.skills,
            cfg.perturb_prefix if role == "perturb" else cfg.verify_prefix,
            self._policy_version(i)).read_text())

        payload = {
            "round": i,
            "metric_legend": scoring.METRIC_LEGEND,
            "metrics": metrics,
            "episodes": scoring.episode_evidence(scores, role),
        }
        prompt = (
            f"/{skill}\n\n"
            "## METRICS (with a legend for what each number means)\n"
            + json.dumps({"metric_legend": payload["metric_legend"],
                          "metrics": payload["metrics"]}, indent=2, ensure_ascii=False)
            + "\n\n## EPISODES\n"
            + json.dumps(payload["episodes"], indent=2, ensure_ascii=False)[:60000]
            + "\n\n## PREVIOUS CONTEXT\n"
            + json.dumps(prev, indent=2, ensure_ascii=False)
            + "\n\n## CURRENT POLICY\n" + policy
            + "\n\nReturn only the JSON object.\n"
        )
        res = self._call("author", skill=None, user=prompt,
                         step=f"summarise-context-{role}", round_dir=self.round_dir(i))
        ledger.record(skill, None, {"round": i, "role": role}, prompt, res)
        if not res.ok():
            print(f"[context] {skill}: no briefing produced "
                  f"({res.infra_failure() or 'empty'}); the updater falls back to findings")
            return None
        from arappav.utils.parsing import extract_first_json_object, strip_json_fences
        obj, err = extract_first_json_object(strip_json_fences(res.text))
        if not isinstance(obj, dict):
            print(f"[context] {skill}: unparseable briefing ({err}); falling back")
            return None
        _write(self.round_dir(i) / f"summarised_context_{role}.json", obj)
        return obj

    @staticmethod
    def _findings_for(role: str, f: dict) -> dict:
        """Slice the findings each updater is entitled to (spec 9.7, 14)."""
        common = {"round": f["round"], "metrics": f["metrics"],
                  "error_type_detection": f["error_type_detection"]}
        if role == "perturb":
            return {**common, "format_failures": f["format_failures"],
                    "undetected_errors": f["undetected_errors"],
                    "detected_errors": f["detected_errors"],
                    "unit_collapse": f["unit_collapse"]}
        return {**common, "undetected_errors": f["undetected_errors"],
                "verifier_false_positives": f["verifier_false_positives"]}

    # -- the two agent passes --------------------------------------------
    def run_round(self, i: int) -> dict:
        cfg = self.cfg
        rdir = self.round_dir(i)
        # Redoing a round that previously aborted: its policy versions exist
        # but the round does not, so they are stale and may be replaced.
        self._retrying = rdir.exists() and not (rdir / "round_summary.json").exists()
        if self._retrying:
            print(f"[round {i}] previous attempt did not finish — replacing its "
                  f"policy versions and redoing the round")
        rdir.mkdir(parents=True, exist_ok=True)
        ledger = ContextLedger(rdir, cfg.no_context)

        if i == 0:
            self.init_policies(ledger)
        else:
            self.update_policies(i, self.rounds[-1]["findings"], ledger)

        pv, vv = self._policy_version(i), self._policy_version(i)
        seen = {e["source_id"] for r in self.rounds for e in r["episodes"]}
        episodes = sample_episodes(cfg, i, seen)

        scores = []
        round_errors: list = []
        for ep in episodes:
            ed = self.ep_dir(i, ep.episode_id)
            ed.mkdir(parents=True, exist_ok=True)
            _write(ed / "data.json", ep.data())
            _write(ed / "meta.json", ep.meta())

            # ---- perturbation -------------------------------------------
            p_body = render_perturb_prompt(
                ep, "", render_context_block({}, cfg.no_context)).lstrip()
            # `retry_format` is an explicit orchestration knob, default 0, so the
            # reward statistics match the pre-refactor pipeline unless asked
            # otherwise. Every attempt is recorded.
            attempts = []
            for attempt in range(cfg.retry_format + 1):
                pres = self._call(
                    "players", skill=f"{cfg.perturb_prefix}-v{pv}", user=p_body,
                    step="perturb", round_dir=rdir,
                    episode_id=f"{ep.episode_id}_a{attempt}" if attempt
                    else ep.episode_id)
                ledger.record("perturb", ep.episode_id, {}, p_body, pres)
                _guard(pres, f"round {i} {ep.episode_id} perturb",
                       attempt, cfg.retry_format)
                parsed, err, stage = scoring.parse_perturbation(pres.text, ep.k, ep.solution)
                attempts.append({"attempt": attempt, "format_valid": parsed is not None,
                                 "failure_stage": stage})
                if parsed is not None or cfg.dry_run:
                    break
                print(f"[round {i}] {ep.episode_id}: unparseable perturbation "
                      f"({stage}); retry {attempt + 1} of {cfg.retry_format}")
            (ed / "perturb_raw.txt").write_text(pres.text)
            _write(ed / "perturb_attempts.json", attempts)
            if parsed is not None:
                try:
                    assert_problem_unchanged(ep, _maybe_problem(pres.text))
                except LeakageError as e:
                    parsed, err, stage = None, str(e), "problem_modified"
            if parsed is not None:
                _write(ed / "perturb.json", json.loads(parsed.model_dump_json()))
            _write(ed / "perturb_status.json",
                   {"format_valid": parsed is not None, "failure_stage": stage,
                    "reason": err, "k": ep.k, "attempts": len(attempts)})

            # ---- verification -------------------------------------------
            vres_text = ""
            if parsed is not None:
                v_input = {"problem": ep.problem,
                           "solution_to_review": parsed.perturbed_solution}
                _write(ed / "verify_input.json", v_input)
                v_body = render_verify_prompt(
                    ep, parsed.perturbed_solution, "",
                    render_context_block({}, cfg.no_context)).lstrip()
                vres = self._call("players", skill=f"{cfg.verify_prefix}-v{vv}",
                                  user=v_body, step="verify", round_dir=rdir,
                                  episode_id=ep.episode_id)
                ledger.record("verify", ep.episode_id, {}, v_body, vres)
                _guard(vres, f"round {i} {ep.episode_id} verify")
                vres_text = vres.text
                (ed / "verify_raw.txt").write_text(vres_text)

            # ---- reward (spec 10, 13) -----------------------------------
            s = scoring.score_episode(
                episode_id=ep.episode_id, k=ep.k, perturbed=parsed,
                failure_stage=stage, failure_reason=err, verifier_raw=vres_text,
                config=self.reward_cfg, history=self.history)
            s["source_id"] = ep.source_id
            _write(ed / "score.json", s)
            scores.append(s)
            if parsed is not None:
                # Buffered, not appended: the anti-duplicate corpus covers
                # *previous* rounds only, matching the pre-refactor
                # `_historical_errors(root, up_to_round)`. Folding this round's
                # own errors in as we go would make episode i duplicate-checked
                # against episodes 0..i-1, which is not the original semantics.
                round_errors.extend(parsed.errors)
            print(f"[round {i}] {ep.episode_id}: r_P={s.get('perturber_reward')} "
                  f"r_V={s.get('verifier_reward')}")

        self.history.extend(round_errors)   # visible from the next round on
        self._last_scores = scores
        metrics = scoring.aggregate(scores)
        findings = scoring.build_findings(i, scores, metrics)
        if cfg.rich_context:
            self._briefings = {}
            for role, frozen in (("perturb", cfg.freeze_perturber()),
                                 ("verify", cfg.freeze_verifier())):
                if frozen:
                    continue          # a frozen side gets no update, so no briefing
                b = self.summarise_context(i, role, scores, metrics, ledger)
                if b:
                    self._briefings[role] = b
        pb = self.run_processbench(i, vv) if cfg.processbench_enabled else None

        summary = {
            "round": i,
            "config": {"start": cfg.start, "freeze": cfg.freeze, "k": cfg.k,
                       "episodes": cfg.episodes, "no_context": cfg.no_context},
            "policies": {"perturb": f"{cfg.perturb_prefix}-v{pv}",
                         "verify": f"{cfg.verify_prefix}-v{vv}"},
            "metrics": metrics,
            "processbench": pb if pb else {"enabled": False,
                                           "note": "ProcessBench was not run for this round."},
            "episodes": [{"episode_id": e.episode_id, "source_id": e.source_id} for e in episodes],
        }
        _write(rdir / "round_summary.json", summary)
        _write(rdir / "findings.json", findings)
        ledger.flush()
        summary["findings"] = findings
        return summary

    # -- ProcessBench (spec 11) ------------------------------------------
    def run_processbench(self, i: int, vv: int) -> dict:
        cfg = self.cfg
        root = Path(cfg.processbench_root)
        rd = root / f"round_{i:02d}"
        skill = f"{cfg.verify_prefix}-v{vv}"
        prep = subprocess.run(
            [sys.executable, "scripts/processbench_eval.py", "--root", str(root),
             "prepare", "--round", str(i), "--verify-skill", skill,
             "--per-subset", str(cfg.processbench_per_subset),
             "--seed", str(cfg.processbench_seed), "--force"],
            capture_output=True, text=True)
        if prep.returncode != 0:
            return {"enabled": True, "error": prep.stderr[-600:]}

        inbox, outbox = rd / "inbox", rd / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        for f in sorted(inbox.glob("*.json")):
            item = json.loads(f.read_text())
            ep = Episode(episode_id=f.stem, problem=item["problem"],
                         solution="", k=0)
            body = render_verify_prompt(ep, item["solution_to_review"], "", "").lstrip()
            r = self._call("players", skill=skill, user=body, step="processbench",
                           round_dir=self.round_dir(i), episode_id=f.stem)
            (outbox / f.name).write_text(r.text or '{"claims": []}')
        sc = subprocess.run(
            [sys.executable, "scripts/processbench_eval.py", "--root", str(root),
             "score", "--round", str(i)], capture_output=True, text=True)
        summ = rd / "eval_summary.json"
        out = {"enabled": True, "verify_skill": skill,
               "stdout": sc.stdout[-800:]}
        if summ.exists():
            out["summary"] = json.loads(summ.read_text()).get("overall")
        return out

    # -- driver -----------------------------------------------------------
    def run(self) -> dict:
        if self.cfg.taxonomy_free:
            # Set before any policy is rendered or any reply parsed, since both
            # the template choice and schema validation read it.
            import os
            os.environ["ARAPPAV_TAXONOMY_FREE"] = "1"
            print("[pipeline] taxonomy-free: the perturber names its own error "
                  "types and free-text labels no longer void an episode")
        self.root.mkdir(parents=True, exist_ok=True)
        _write(self.root / "run_config.json", asdict(self.cfg))
        for i in range(self.cfg.rounds):
            done = self.round_dir(i) / "round_summary.json"
            if self.cfg.resume and done.exists():
                summary = json.loads(done.read_text())
                fpath = self.round_dir(i) / "findings.json"
                summary["findings"] = (json.loads(fpath.read_text())
                                       if fpath.exists() else {})
                # Rebuild the anti-duplicate corpus from disk so a resumed run
                # penalises repeats exactly as an uninterrupted one would.
                self.history.extend(self._errors_on_disk(i))
                self.rounds.append(summary)
                print(f"=== round {i}: already complete, skipping (--resume)")
                continue
            print(f"\n=== round {i} " + "=" * 50)
            self.rounds.append(self.run_round(i))
        report = self.final_summary()
        return report

    def _errors_on_disk(self, i: int) -> list:
        """Ground-truth errors persisted for a completed round (for --resume)."""
        from arappav.errors.schema_math import MathInjectedError
        out = []
        eps = self.round_dir(i) / "episodes"
        for ed in sorted(eps.iterdir()) if eps.is_dir() else []:
            f = ed / "perturb.json"
            if not f.exists():
                continue
            for e in json.loads(f.read_text()).get("errors", []):
                try:
                    out.append(MathInjectedError.model_validate(e))
                except Exception:
                    pass
        return out

    # -- final summary (spec 15) -----------------------------------------
    def final_summary(self) -> dict:
        cfg = self.cfg
        payload = {
            "config": asdict(cfg),
            "rounds": [{k: v for k, v in r.items() if k != "findings"} for r in self.rounds],
            "findings_by_round": [r["findings"] for r in self.rounds],
        }
        _write(self.root / "final_summary_input.json", payload)

        table = _summary_table(self.rounds, cfg)
        (self.root / "summary_table.md").write_text(table)

        prompt = (
            "/final_summary\n\n"
            "Write the final experiment report from the JSON below. It is the only "
            "source; do not read any files.\n\n"
            "## RUN DATA\n" + json.dumps(payload, indent=2, ensure_ascii=False)[:120000] +
            "\n\n## PRECOMPUTED ROUND TABLE\n" + table + "\n"
        )
        res = self._call("author", skill=None, user=prompt, step="final_summary",
                         round_dir=self.root)
        if res.infra_failure():
            print(f"[final] narrative summary unavailable ({res.infra_failure()}); "
                  "writing the deterministic table instead.")
        md = res.text if res.ok() else (
            "# Final summary\n\n_The `/final_summary` skill did not return a report; "
            "the deterministic table below is generated from persisted round data._\n\n" + table)
        (self.root / "final_summary.md").write_text(md + "\n")
        print(f"\n[final] {self.root / 'final_summary.md'}")
        return {"table": table, "report_path": str(self.root / "final_summary.md")}


def _summary_table(rounds: list[dict], cfg: PipelineConfig) -> str:
    """Deterministic round-by-round table (spec 15), built without an agent."""
    head = ("| round | perturb policy | verify policy | format-valid | mean r_P | mean r_V "
            "| recall | precision | units/ep | penalties | ProcessBench |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in rounds:
        m = r["metrics"]
        pen = round(sum(m.get(k) or 0 for k in
                        ("total_duplicate_penalty", "total_spam_penalty",
                         "total_repetition_penalty", "total_phantom_penalty")), 3)
        pb = r.get("processbench") or {}
        if not pb.get("enabled"):
            pbs = "disabled"
        elif pb.get("error"):
            pbs = "error: " + pb["error"][:24]
        elif pb.get("summary"):
            f1 = pb["summary"].get("processbench_f1")
            # The headline metric is a harmonic mean over both classes; it is
            # undefined when the sample contained no erroneous (or no correct)
            # chain, which happens at very small --processbench-per-subset.
            pbs = f"F1 {f1}" if f1 is not None else "F1 n/a (one class absent)"
        else:
            pbs = "n/a"
        rows.append(
            f"| {r['round']} | {r['policies']['perturb']} | {r['policies']['verify']} "
            f"| {m['format_valid_rate']} | {m['mean_perturber_reward']} "
            f"| {m['mean_verifier_reward']} | {m['mean_verifier_recall']} "
            f"| {m['mean_verifier_precision']} | {m['mean_units_per_episode']} "
            f"| {pen} | {pbs} |")
    return head + "\n".join(rows) + "\n"


def _guard(res, what: str, attempt: int = 0, retries: int = 0) -> None:
    """Abort the run if a call never reached a model.

    Deliberately fail-fast. The alternative — recording a penalty — silently
    turns an outage into 80 episodes of fabricated data, which is exactly the
    failure this guard exists to prevent. Completed rounds are already on disk,
    so `--resume` continues from the last good round once the cause is fixed.
    """
    reason = res.infra_failure()
    if reason and attempt < retries:
        # Transient blips (a dropped connection, a momentary 5xx) should cost
        # one call, not the run. A real outage persists and still aborts below.
        print(f"[retry] {what}: {reason} — attempt {attempt + 1}/{retries}")
        return
    if reason:
        raise InfrastructureError(
            f"{what}: the model was never reached ({reason}). "
            f"No reward is recorded for this call. Fix the cause and re-run "
            f"with --resume to continue from the last completed round."
        )


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _maybe_problem(raw: str) -> str | None:
    """If the perturber echoed a `problem` field, return it so it can be checked."""
    try:
        from arappav.utils.parsing import extract_first_json_object, strip_json_fences
        obj, _ = extract_first_json_object(strip_json_fences(raw))
        return obj.get("problem") if isinstance(obj, dict) else None
    except Exception:
        return None
