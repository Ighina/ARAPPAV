"""Invoking Claude as a pure function, and the context ledger (spec 21).

Every agent call is `claude -p "<prompt>"` in a **fresh** subprocess. There is
no `--continue` and no `--resume` anywhere in this module, so no harness
conversation, no prior round, and no other agent's transcript can reach the
callee: the prompt is the entire input.

Two further hardening measures:

* **Tools are disabled by default.** The agents in this pipeline receive their
  inputs inline and answer on stdout, so they have no legitimate reason to
  touch the filesystem. Denying the tools turns "the verifier must not read
  `episodes/`" from an instruction into an impossibility.
* **Every prompt is persisted verbatim** next to a context ledger entry, so the
  leakage audit in spec 9/18 can be run against what was actually sent rather
  than against what the code was supposed to send.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Tools denied to every pipeline agent. Inputs arrive inline; answers leave on
#: stdout. Anything else is a leakage path.
DENIED_TOOLS = (
    "Bash", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
    "WebFetch", "WebSearch", "Task", "Agent", "TodoWrite",
)


class AgentError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    """The CLI never ran the model — quota, auth, network, timeout.

    This is emphatically NOT an experimental result. A model that returns a
    malformed answer is data; a call that never reached a model is the absence
    of data, and scoring it as a format failure fabricates measurements. The
    orchestrator aborts on this rather than recording a reward.
    """


#: stdout/stderr signatures that mean the request never reached a model.
#: `claude -p` exits non-zero AND prints these to stdout, so a naive reader
#: mistakes the message for the model's reply.
FATAL_PATTERNS = (
    "session limit", "usage limit", "rate limit", "quota",
    "insufficient credit", "credit balance", "authentication",
    "invalid api key", "not logged in", "please run /login",
)


@dataclass
class AgentResult:
    step: str
    text: str
    returncode: int
    duration_s: float
    prompt_chars: int
    prompt_sha256: str
    stderr: str = ""
    dry_run: bool = False

    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.text.strip())

    def infra_failure(self) -> str | None:
        """Return a reason if the call never reached a model, else None."""
        if self.dry_run:
            return None
        blob = f"{self.text}\n{self.stderr}".lower()
        for pat in FATAL_PATTERNS:
            if pat in blob:
                return pat
        if self.returncode == 124:
            return "timeout"
        if self.returncode != 0:
            return f"claude exited {self.returncode}"
        if not self.text.strip():
            return "empty response"
        return None


@dataclass
class ContextLedger:
    """Per-round record of exactly what context each step was given (spec 21)."""

    round_dir: Path
    no_context: bool
    entries: list[dict] = field(default_factory=list)

    def record(self, step: str, episode_id: str | None, allowed: dict,
               prompt: str, result: "AgentResult | None") -> None:
        self.entries.append({
            "step": step,
            "episode_id": episode_id,
            "no_context": self.no_context,
            "allowed_context_keys": sorted(allowed),
            "allowed_context": allowed,
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "returncode": None if result is None else result.returncode,
            "duration_s": None if result is None else result.duration_s,
        })

    def flush(self) -> Path:
        out = self.round_dir / "context_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"no_context": self.no_context, "steps": self.entries},
            indent=2, ensure_ascii=False) + "\n")
        return out


def render_context_block(allowed: dict, no_context: bool) -> str:
    """Render the allowed history for a step, or nothing under --no-context."""
    if no_context or not allowed:
        return ""
    return ("## CONTEXT FROM EARLIER ROUNDS (the only history you are given)\n"
            + json.dumps(allowed, indent=2, ensure_ascii=False) + "\n\n")


def claude_available() -> bool:
    return shutil.which("claude") is not None


def run_claude(prompt: str, *, step: str, round_dir: Path,
               episode_id: str | None = None, model: str | None = None,
               timeout: int = 900, dry_run: bool = False,
               allow_tools: bool = False) -> AgentResult:
    """One `claude -p` call. Persists the prompt, returns stdout."""
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    tag = f"{step}__{episode_id}" if episode_id else step
    pdir = round_dir / "prompts"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{tag}.txt").write_text(prompt)

    if dry_run:
        return AgentResult(step, "", 0, 0.0, len(prompt), sha, dry_run=True)

    if not claude_available():
        raise AgentError("`claude` CLI not found on PATH.")

    cmd = ["claude", "-p", prompt, "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    if not allow_tools:
        cmd += ["--disallowed-tools", *DENIED_TOOLS]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return AgentResult(step, "", 124, round(time.time() - t0, 2), len(prompt), sha,
                           stderr=f"timeout after {timeout}s")
    out = AgentResult(step, proc.stdout.strip(), proc.returncode,
                      round(time.time() - t0, 2), len(prompt), sha,
                      stderr=proc.stderr[-2000:])
    (pdir / f"{tag}.stdout.txt").write_text(proc.stdout)
    return out
