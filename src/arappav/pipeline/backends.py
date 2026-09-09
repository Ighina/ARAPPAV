"""Two ways to run a policy: the Claude Code CLI, or the Messages API directly.

Both send the same policy and the same episode; they differ only in transport,
and the difference is expensive.

`claude -p` boots a full agent session per call. Measured on a real ProcessBench
item with claude-haiku-4-5: 17,209 input tokens, of which roughly 13,150 is the
harness system prompt and tool definitions re-sent every single time — 76%
overhead — plus 6,263 output tokens (mostly thinking) to produce a 30-byte
answer, at $0.048 and ~45s per call.

The API backend sends only the policy and the item: ~2,600 input tokens, and the
policy half is identical across every item of a run, so it is marked for prompt
caching and billed at roughly a tenth after the first call. Same work, about 4%
of the cost.

Keep the CLI backend for parity checks and for running without an API key; use
the API backend for volume.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from arappav.pipeline.agents import AgentResult, run_claude

#: Signatures meaning the request never reached a model. Mirrors
#: AgentResult.infra_failure so both backends fail the same way.
FATAL_PATTERNS = (
    "session limit", "usage limit", "rate limit", "quota",
    "insufficient credit", "credit balance", "authentication",
    "invalid api key", "not logged in", "overloaded",
)

#: Models that reject `thinking` / `output_config.effort` on the Messages API.
NO_THINKING = {"claude-haiku-4-5"}


def skill_text(skills_root: Path, skill: str) -> str:
    """The policy file, minus its YAML frontmatter, as a system prompt.

    Under `claude -p` the model receives this by invoking `/skill`; the API has
    no such mechanism, so the file is sent verbatim instead. The two backends
    therefore give the model the same instructions by different routes.
    """
    p = Path(skills_root) / skill / "SKILL.md"
    if not p.exists():
        raise FileNotFoundError(f"no such policy: {p}")
    return re.sub(r"\A---\n.*?\n---\n", "", p.read_text(), flags=re.S).strip()


@dataclass
class Backend:
    """Common surface: give it a policy and an item, get text back."""

    name: str
    model: str | None = None
    timeout: int = 600
    max_tokens: int = 8000
    effort: str = "high"
    skills_root: Path = Path(".claude/skills")

    def run(self, *, skill: str, user: str, step: str, round_dir: Path,
            episode_id: str | None = None) -> AgentResult:
        raise NotImplementedError


@dataclass
class ClaudeCodeBackend(Backend):
    """`claude -p` — full harness per call. Slow and expensive, but needs no key."""

    def run(self, *, skill, user, step, round_dir, episode_id=None) -> AgentResult:
        # The slash command loads the policy; the harness supplies everything else.
        return run_claude(f"/{skill}\n\n{user}", step=step, round_dir=round_dir,
                          episode_id=episode_id, model=self.model,
                          timeout=self.timeout)


@dataclass
class ApiBackend(Backend):
    """Anthropic Messages API — policy as a cached system prompt, item as the turn."""

    def __post_init__(self):
        try:
            import anthropic
        except ImportError:
            raise SystemExit("[api] `pip install anthropic` first.")
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit(
                "[api] ANTHROPIC_API_KEY is not set. Export a key, or use "
                "--backend claude-code to go through the CLI instead.")
        self._client = anthropic.Anthropic(max_retries=4)
        self._cache: dict[str, str] = {}

    def run(self, *, skill, user, step, round_dir, episode_id=None) -> AgentResult:
        system = self._cache.setdefault(skill, skill_text(self.skills_root, skill))
        tag = f"{step}__{episode_id}" if episode_id else step
        pdir = round_dir / "prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{tag}.txt").write_text(f"[system: {skill}]\n\n{user}")

        # The policy is identical for every item of a run, so it is cached; only
        # the item is charged at the full input rate after the first call.
        kwargs = dict(
            model=self.model or "claude-haiku-4-5",
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if (self.model or "") not in NO_THINKING:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.effort}

        t0 = time.time()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            rc = 1 if any(p in msg.lower() for p in FATAL_PATTERNS) else 2
            return AgentResult(step, "", rc, round(time.time() - t0, 2),
                               len(user), "", stderr=msg)

        text = "".join(b.text for b in resp.content if b.type == "text")
        (pdir / f"{tag}.stdout.txt").write_text(text)
        r = AgentResult(step, text.strip(), 0, round(time.time() - t0, 2),
                        len(system) + len(user), "")
        u = resp.usage
        r.usage = {                                    # type: ignore[attr-defined]
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read": getattr(u, "cache_read_input_tokens", 0),
            "cache_write": getattr(u, "cache_creation_input_tokens", 0),
        }
        return r


def make_backend(kind: str, **kw) -> Backend:
    if kind == "api":
        return ApiBackend(name="api", **kw)
    if kind == "claude-code":
        return ClaudeCodeBackend(name="claude-code", **kw)
    raise SystemExit(f"[backend] unknown backend {kind!r} (api | claude-code)")
