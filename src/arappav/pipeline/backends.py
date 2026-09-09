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

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
#
# Claude goes through the official Anthropic SDK. OpenAI and DeepSeek both go
# through the official OpenAI SDK — DeepSeek's own documented integration path
# is that SDK pointed at their base URL — so they share one code path and
# differ only in credentials, base URL and capabilities.

PROVIDERS = {
    "anthropic": {"sdk": "anthropic", "env": "ANTHROPIC_API_KEY", "base_url": None},
    "openai":    {"sdk": "openai",    "env": "OPENAI_API_KEY",    "base_url": None},
    "deepseek":  {"sdk": "openai",    "env": "DEEPSEEK_API_KEY",
                  "base_url": "https://api.deepseek.com"},
}

#: Model-id prefixes -> provider. Override with an explicit `provider`.
_PREFIXES = (
    ("claude-", "anthropic"),
    ("deepseek", "deepseek"),
    ("gpt-", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("chatgpt", "openai"),
)

#: OpenAI reasoning families that accept `reasoning_effort`. Sending it to a
#: non-reasoning model is an error, so it is opt-in by prefix.
_OPENAI_REASONING = ("gpt-5", "o1", "o3", "o4")


def infer_provider(model: str | None) -> str:
    """Map a model id to its provider; unknown ids must be named explicitly."""
    low = (model or "").lower()
    for prefix, prov in _PREFIXES:
        if low.startswith(prefix):
            return prov
    raise SystemExit(
        f"[backend] cannot infer a provider for {model!r} — pass --provider "
        f"({'/'.join(PROVIDERS)}).")


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

    def run_raw(self, *, user: str, step: str, round_dir: Path,
                episode_id: str | None = None) -> AgentResult:
        """A call with no versioned policy file — the prompt is the instruction.

        Used by create-policy-*, update-* and final_summary, which reference a
        static skill by name inside the prompt rather than being governed by a
        versioned policy.
        """
        raise NotImplementedError


@dataclass
class ClaudeCodeBackend(Backend):
    """`claude -p` — full harness per call. Slow and expensive, but needs no key."""

    def run(self, *, skill, user, step, round_dir, episode_id=None) -> AgentResult:
        # The slash command loads the policy; the harness supplies everything else.
        return run_claude(f"/{skill}\n\n{user}", step=step, round_dir=round_dir,
                          episode_id=episode_id, model=self.model,
                          timeout=self.timeout)

    def run_raw(self, *, user, step, round_dir, episode_id=None) -> AgentResult:
        return run_claude(user, step=step, round_dir=round_dir,
                          episode_id=episode_id, model=self.model,
                          timeout=self.timeout)


@dataclass
class ApiBackend(Backend):
    """Direct API — policy as a (cached) system prompt, item as the turn.

    Anthropic, OpenAI and DeepSeek are all supported. On Anthropic the policy is
    marked with `cache_control` explicitly; OpenAI and DeepSeek cache long
    prompt prefixes automatically, so the same saving applies without a flag.
    """

    provider: str | None = None

    def __post_init__(self):
        self.provider = self.provider or infer_provider(self.model)
        spec = PROVIDERS.get(self.provider)
        if spec is None:
            raise SystemExit(f"[api] unknown provider {self.provider!r} "
                             f"({'/'.join(PROVIDERS)})")
        self._cache: dict[str, str] = {}
        self._cache_checked = False

        if spec["sdk"] == "anthropic":
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
            return

        try:
            import openai
        except ImportError:
            raise SystemExit(
                f"[api] the {self.provider} backend needs the OpenAI SDK: "
                f"`pip install openai`.")
        key = os.environ.get(spec["env"])
        if not key:
            raise SystemExit(
                f"[api] {spec['env']} is not set — required for provider "
                f"{self.provider!r}.")
        self._client = openai.OpenAI(api_key=key, base_url=spec["base_url"],
                                     max_retries=4)

    def anthropic_kwargs(self, system: str, user: str) -> dict:
        kwargs = dict(
            model=self.model or "claude-haiku-4-5",
            max_tokens=self.max_tokens,
            # The policy is identical for every item of a run, so it is cached;
            # only the item is charged at the full input rate after the first.
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if (self.model or "") not in NO_THINKING:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def openai_kwargs(self, system: str, user: str) -> dict:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        # OpenAI reasoning models reject max_tokens and take
        # max_completion_tokens; DeepSeek takes max_tokens.
        if self.provider == "openai":
            kwargs["max_completion_tokens"] = self.max_tokens
            if any((self.model or "").lower().startswith(f) for f in _OPENAI_REASONING):
                # OpenAI's effort scale has no xhigh/max; clamp onto its top.
                kwargs["reasoning_effort"] = {"xhigh": "high",
                                              "max": "high"}.get(self.effort, self.effort)
        else:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def run_raw(self, *, user, step, round_dir, episode_id=None) -> AgentResult:
        """No versioned policy: resolve a leading `/skill` line off disk instead.

        The API has no slash-command mechanism, so a prompt that opens with
        `/update-verify` would otherwise reach the model as literal text. The
        named skill file is loaded and sent as the system prompt, and the line
        is stripped from the body.
        """
        system, body = "You are a careful assistant. Follow the instructions exactly.", user
        m = re.match(r"\s*/([A-Za-z0-9_-]+)\s*\n", user)
        if m:
            try:
                system = skill_text(self.skills_root, m.group(1))
                body = user[m.end():].lstrip()
            except FileNotFoundError:
                pass
        return self._send(system, body, step, round_dir, episode_id)

    def run(self, *, skill, user, step, round_dir, episode_id=None) -> AgentResult:
        system = self._cache.setdefault(skill, skill_text(self.skills_root, skill))
        return self._send(system, user, step, round_dir, episode_id)

    def _send(self, system: str, user: str, step: str, round_dir: Path,
              episode_id: str | None) -> AgentResult:
        tag = f"{step}__{episode_id}" if episode_id else step
        pdir = round_dir / "prompts"
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{tag}.txt").write_text(f"[system]\n{system}\n\n[user]\n{user}")
        anthropic_path = PROVIDERS[self.provider]["sdk"] == "anthropic"

        t0 = time.time()
        try:
            if anthropic_path:
                resp = self._client.messages.create(
                    **self.anthropic_kwargs(system, user))
            else:
                resp = self._client.chat.completions.create(
                    **self.openai_kwargs(system, user))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            rc = 1 if any(p in msg.lower() for p in FATAL_PATTERNS) else 2
            return AgentResult(step, "", rc, round(time.time() - t0, 2),
                               len(user), "", stderr=msg)

        if anthropic_path:
            text = "".join(b.text for b in resp.content if b.type == "text")
            u = resp.usage
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_read": getattr(u, "cache_read_input_tokens", 0),
                     "cache_write": getattr(u, "cache_creation_input_tokens", 0)}
        else:
            text = resp.choices[0].message.content or ""
            u = resp.usage
            # OpenAI and DeepSeek cache long prefixes automatically and report
            # the hit under different names; normalise onto ours.
            cached = 0
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            cached = cached or getattr(u, "prompt_cache_hit_tokens", 0) or 0
            usage = {"input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                     "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                     "cache_read": cached, "cache_write": 0}

        (pdir / f"{tag}.stdout.txt").write_text(text)
        self._warn_if_not_caching(usage)
        r = AgentResult(step, text.strip(), 0, round(time.time() - t0, 2),
                        len(system) + len(user), "")
        r.usage = usage                                # type: ignore[attr-defined]
        return r

    def _warn_if_not_caching(self, usage: dict) -> None:
        """Say so, once, when the policy prefix is not actually being cached.

        Every provider has a minimum cacheable prefix (2048 tokens on Claude
        Haiku 4.5, lower on larger models) and silently declines to cache
        anything shorter. A short policy therefore bills at the full input rate
        on every call with no error — measured here at ~2,030 tokens for a seed
        policy, just under the threshold. Worth knowing, since it is the
        difference between ~$0.0026 and ~$0.0008 per item.
        """
        if self._cache_checked:
            return
        self._cache_checked = True
        if usage.get("cache_read", 0) or usage.get("cache_write", 0):
            return
        print(f"[api] note: the policy prefix is not being cached "
              f"({usage.get('input_tokens', 0):,} input tokens/call at the full "
              f"rate). Providers decline to cache prefixes below a minimum "
              f"(2048 tokens on claude-haiku-4-5). Cost estimates that assume "
              f"caching do not apply.", flush=True)


# ---------------------------------------------------------------------------
# Batch submission (Message Batches API)
# ---------------------------------------------------------------------------
#
# Batching is a different shape from the per-item backends above: you submit
# every request at once, the server works through them asynchronously, and you
# collect results later. It bills at 50% of the standard rate, which stacks with
# the prompt caching the API backend already does.
#
# The trade is latency — a batch can take up to 24h, though small ones usually
# finish in minutes — so it suits a full-benchmark sweep and not an interactive
# check. The batch id is persisted, so a killed poll resumes instead of
# resubmitting and paying twice.

#: A batch is terminal when the server stops working on it.
_BATCH_DONE = "ended"


def _custom_id(prefix: str, item_id: str) -> str:
    """Batch custom_ids allow [a-zA-Z0-9_-] and at most 64 chars."""
    raw = f"{prefix}__{item_id}"
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", raw)
    return safe[:64]


@dataclass
class BatchBackend(ApiBackend):
    """Messages API in batch mode: 50% cheaper, asynchronous."""

    poll_interval: int = 20
    max_wait: int = 24 * 3600

    def __post_init__(self):
        super().__post_init__()
        if PROVIDERS[self.provider]["sdk"] != "anthropic":
            # OpenAI's batch API is a different, file-upload shape and DeepSeek
            # has none, so this backend does not pretend to cover them.
            raise SystemExit(
                f"[batch] the batch backend is Anthropic-only; {self.provider!r} "
                f"has no compatible Message Batches API. Use --backend api.")

    def build_requests(self, skill: str, items: list[tuple[str, str]]) -> list[dict]:
        """One request per (item_id, user_prompt), sharing a cached system prompt."""
        system = self._cache.setdefault(skill, skill_text(self.skills_root, skill))
        params_common = {
            "model": self.model or "claude-haiku-4-5",
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
        }
        if (self.model or "") not in NO_THINKING:
            params_common["thinking"] = {"type": "adaptive"}
            params_common["output_config"] = {"effort": self.effort}
        return [
            {"custom_id": _custom_id(skill, iid),
             "params": {**params_common,
                        "messages": [{"role": "user", "content": user}]}}
            for iid, user in items
        ]

    def submit(self, skill: str, items: list[tuple[str, str]]) -> str:
        batch = self._client.messages.batches.create(
            requests=self.build_requests(skill, items))
        return batch.id

    def status(self, batch_id: str) -> tuple[str, dict]:
        b = self._client.messages.batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None)
        return b.processing_status, {
            k: getattr(counts, k, 0) for k in
            ("processing", "succeeded", "errored", "canceled", "expired")
        } if counts else (b.processing_status, {})

    def collect(self, skill: str, batch_id: str) -> tuple[dict[str, str], dict[str, str], dict]:
        """Return (texts by item id, failures by item id, summed usage).

        Results arrive in arbitrary order, so everything is keyed by custom_id
        and never by position.
        """
        texts: dict[str, str] = {}
        failed: dict[str, str] = {}
        usage: dict[str, int] = {}
        prefix = _custom_id(skill, "")
        for r in self._client.messages.batches.results(batch_id):
            iid = r.custom_id[len(prefix):] if r.custom_id.startswith(prefix) else r.custom_id
            kind = r.result.type
            if kind != "succeeded":
                failed[iid] = kind        # errored | canceled | expired
                continue
            msg = r.result.message
            texts[iid] = "".join(b.text for b in msg.content if b.type == "text").strip()
            u = msg.usage
            for key, val in (("input_tokens", u.input_tokens),
                             ("output_tokens", u.output_tokens),
                             ("cache_read", getattr(u, "cache_read_input_tokens", 0)),
                             ("cache_write", getattr(u, "cache_creation_input_tokens", 0))):
                usage[key] = usage.get(key, 0) + (val or 0)
        return texts, failed, usage


def make_backend(kind: str, **kw) -> Backend:
    if kind == "batch":
        return BatchBackend(name="batch", **kw)
    if kind == "api":
        return ApiBackend(name="api", **kw)
    if kind == "claude-code":
        return ClaudeCodeBackend(name="claude-code", **kw)
    raise SystemExit(f"[backend] unknown backend {kind!r} (api | claude-code)")