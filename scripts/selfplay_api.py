#!/usr/bin/env python
"""API-driven self-play runner — the *lesson-tuning* loop.

`scripts/skill_selfplay.py` runs the game inside the Claude Code harness: the
agent plays both roles, and the policies live in `.claude/skills/<role>-vN/`.
That has two problems this script exists to remove:

1. **Leakage.** The Perturber and the Verifier are the same agent in the same
   process. Isolation is enforced by instructions and a subagent boundary,
   which is a convention, not a guarantee.
2. **Reproducibility.** A harness turn carries a system prompt, tool results,
   memory and skill files we neither control nor can replay. What produced a
   given rollout is not fully knowable.

Here each role is a **single stateless API call**: one system prompt (the
versioned `prompt.txt`), one user message (the episode), one JSON reply. The
two roles share no context by construction — there is nothing to leak *through*
— and a round is replayable from the prompt files plus the episode inputs.

Three providers are supported, inferred from the model id or forced with
`--provider`: **anthropic** (official `anthropic` SDK), **openai** and
**deepseek** (both the official `openai` SDK, DeepSeek against its own base
URL, which is DeepSeek's documented integration path). Each model tunes its own
prompt tree and writes its own rollout root, so several can be compared on
identical episodes without touching each other.

The tuned artefact is therefore a **prompt**, not a skill::

    prompts/<role>/<model_slug>/v<N>/prompt.txt

Everything else — sampling problems, parsing, error units, the reward
function, cross-round anti-duplicate history, the round summary — is reused
verbatim from `scripts/skill_selfplay.py` by pointing it at a separate
rollout root, so both loops are scored by exactly the same code.

Typical round::

    R=data/api_rollouts/opus-5

    python scripts/selfplay_api.py seed --model claude-opus-5
    python scripts/skill_selfplay.py --root $R init --round 1 --episodes 8 --k 3 \
        --source hendrycks
    python scripts/selfplay_api.py perturb --model claude-opus-5 --version 1 --round 1
    python scripts/skill_selfplay.py --root $R prepare-verify --round 1
    python scripts/selfplay_api.py verify  --model claude-opus-5 --version 1 --round 1
    python scripts/skill_selfplay.py --root $R score --round 1
    python scripts/skill_selfplay.py --root $R summarize --round 1

Note that `skill_selfplay.py` takes `--root` *before* its subcommand.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PROMPTS_ROOT = REPO_ROOT / "prompts"
SEED_DIR = PROMPTS_ROOT / "_seed"
DEFAULT_ROLLOUT_ROOT = REPO_ROOT / "data" / "api_rollouts"

INVARIANT_BEGIN = "<<<INVARIANT>>>"
INVARIANT_END = "<<<END INVARIANT>>>"
POLICY_BEGIN = "<<<POLICY>>>"
POLICY_END = "<<<END POLICY>>>"

# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
#
# Three backends. Claude goes through the official Anthropic SDK. OpenAI and
# DeepSeek both go through the official OpenAI SDK — DeepSeek's own documented
# integration path is that SDK pointed at their base URL — so they share one
# code path and differ only in credentials, base URL and capabilities.

PROVIDERS = {
    "anthropic": {"sdk": "anthropic", "env": "ANTHROPIC_API_KEY", "base_url": None},
    "openai":    {"sdk": "openai",    "env": "OPENAI_API_KEY",    "base_url": None},
    "deepseek":  {"sdk": "openai",    "env": "DEEPSEEK_API_KEY",
                  "base_url": "https://api.deepseek.com"},
}

#: Model-id prefixes → provider, for `infer_provider`. Override with --provider.
_PREFIXES = (
    ("claude-", "anthropic"),
    ("deepseek", "deepseek"),
    ("gpt-", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("chatgpt", "openai"),
)

#: OpenAI reasoning families that accept `reasoning_effort`. Sending it to a
#: non-reasoning model is an error, so it is opt-in by prefix.
_OPENAI_REASONING = ("gpt-5", "o1", "o3", "o4")

#: DeepSeek reasoning families (v4-*): thinking is on by default and accepts
#: `reasoning_effort`. Legacy ids (deepseek-chat / deepseek-reasoner) do not.
_DEEPSEEK_REASONING = ("deepseek-v4",)

#: Anthropic models that reject `thinking` / `output_config.effort`.
NO_THINKING = {"claude-haiku-4-5"}

#: USD per 1M tokens (input, output), for the cost line in run logs. Only
#: Anthropic rates are hard-coded; for every other model pass --price-in /
#: --price-out if you want costed logs, rather than trusting a stale table.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}


def infer_provider(model: str) -> str:
    """Map a model id to its provider. Explicit --provider always wins."""
    low = model.lower()
    for prefix, prov in _PREFIXES:
        if low.startswith(prefix):
            return prov
    sys.exit(f"[api] cannot infer a provider for {model!r} — pass --provider "
             f"({'/'.join(PROVIDERS)}).")


def resolve_provider(args) -> str:
    prov = getattr(args, "provider", None) or infer_provider(args.model)
    if prov not in PROVIDERS:
        sys.exit(f"[api] unknown provider {prov!r} — choose from {'/'.join(PROVIDERS)}.")
    return prov


# ---------------------------------------------------------------------------
# Paths and prompt files
# ---------------------------------------------------------------------------


def model_slug(model: str) -> str:
    """Directory name for a model: `claude-sonnet-5` -> `sonnet-5`, `gpt-5` -> `gpt-5`.

    Only the `claude-` prefix is stripped (it is pure noise across an
    all-Anthropic tree); other ids already carry their family. Anything that
    would be awkward in a path is flattened to a dash.
    """
    s = model[len("claude-"):] if model.startswith("claude-") else model
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def prompt_dir(role: str, model: str, version: int) -> Path:
    return PROMPTS_ROOT / role / model_slug(model) / f"v{version}"


def prompt_path(role: str, model: str, version: int) -> Path:
    return prompt_dir(role, model, version) / "prompt.txt"


def rollout_root(model: str, root: str | None) -> Path:
    return Path(root) if root else DEFAULT_ROLLOUT_ROOT / model_slug(model)


def latest_version(role: str, model: str) -> int:
    base = PROMPTS_ROOT / role / model_slug(model)
    if not base.is_dir():
        return 0
    vs = [int(m.group(1)) for p in base.iterdir()
          if (m := re.fullmatch(r"v(\d+)", p.name)) and (p / "prompt.txt").exists()]
    return max(vs, default=0)


def section(text: str, begin: str, end: str) -> str | None:
    """Return the text strictly between two markers, or None if malformed."""
    i, j = text.find(begin), text.find(end)
    if i == -1 or j == -1 or j < i:
        return None
    return text[i + len(begin):j]


def read_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Message construction — the isolation boundary lives here
# ---------------------------------------------------------------------------

#: The ONLY episode fields that may reach the Perturber.
PERTURB_FIELDS = ("problem", "solution", "k")
#: The ONLY episode fields that may reach the Verifier. Note the absence of
#: `k`: the Verifier is not told how many errors to look for.
VERIFY_FIELDS = ("problem", "solution_to_review")

#: Keys that must never appear in a rendered Verifier message. A hit means the
#: ground truth has leaked into the prompt.
GT_MARKERS = ("injected_text", "original_text", "rationale", "error_id",
              "error_type", "perturbed_solution", "\"errors\"")


def render_perturb_user(problem: dict) -> str:
    """Build the Perturber's user message from an explicit field allowlist."""
    missing = [f for f in PERTURB_FIELDS if f not in problem]
    if missing:
        raise KeyError(f"problem.json is missing {missing}")
    return (
        f"PROBLEM:\n{problem['problem']}\n\n"
        f"CORRECT SOLUTION:\n{problem['solution']}\n\n"
        f"k = {problem['k']}\n\n"
        "Return only the JSON object."
    )


def render_verify_user(inbox: dict) -> str:
    """Build the Verifier's user message from an explicit field allowlist.

    Only `problem` and `solution_to_review` are read. Nothing else in the inbox
    record — and nothing at all from `episodes/` — can reach the model, because
    no other field is ever referenced.
    """
    missing = [f for f in VERIFY_FIELDS if f not in inbox]
    if missing:
        raise KeyError(f"verify_inbox record is missing {missing}")
    msg = (
        f"PROBLEM:\n{inbox['problem']}\n\n"
        f"SOLUTION TO REVIEW:\n{inbox['solution_to_review']}\n\n"
        "Return only the JSON object."
    )
    leaked = [m for m in GT_MARKERS if m in msg]
    if leaked:
        raise RuntimeError(
            f"REFUSING TO SEND: ground-truth markers {leaked} appear in the verifier "
            "message. The round is compromised; investigate before re-running."
        )
    return msg


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _price(model: str, args) -> tuple[float, float] | None:
    """Per-1M-token (input, output) rates: CLI override, else the table, else none."""
    if getattr(args, "price_in", None) is not None or getattr(args, "price_out", None) is not None:
        return (args.price_in or 0.0, args.price_out or 0.0)
    return PRICES.get(model)


def _cost(model: str, args, tin: int, tout: int):
    rates = _price(model, args)
    return None if rates is None else round((tin * rates[0] + tout * rates[1]) / 1e6, 5)


def _call_anthropic(client, model, system, user, max_tokens, effort, args) -> dict:
    kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
    if model not in NO_THINKING:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": effort}

    t0 = time.time()
    resp = client.messages.create(**kwargs)
    elapsed = time.time() - t0

    tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
    return {
        "text": "".join(b.text for b in resp.content if b.type == "text"),
        "stop_reason": resp.stop_reason,
        "input_tokens": tin, "output_tokens": tout,
        "cost_usd": _cost(model, args, tin, tout),
        "latency_s": round(elapsed, 2),
        "request_id": getattr(resp, "_request_id", None),
        "model": resp.model,
    }


def _call_openai_compatible(client, model, system, user, max_tokens, effort,
                            args, provider: str) -> dict:
    """OpenAI and DeepSeek, both via the OpenAI SDK's chat-completions surface."""
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    # Reasoning models on OpenAI reject `max_tokens` and take
    # `max_completion_tokens`; DeepSeek takes `max_tokens`.
    if provider == "openai":
        kwargs["max_completion_tokens"] = max_tokens
        if any(model.lower().startswith(f) for f in _OPENAI_REASONING):
            # OpenAI's scale has no xhigh/max — clamp onto its top level.
            kwargs["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(effort, effort)
    else:
        kwargs["max_tokens"] = max_tokens
        # DeepSeek v4-* models reason by default; their scale is low/high/max
        # and `medium`/`xhigh` are accepted as compat aliases for high.
        if any(model.lower().startswith(f) for f in _DEEPSEEK_REASONING):
            kwargs["reasoning_effort"] = {"medium": "high", "xhigh": "high"}.get(effort, effort)
    if getattr(args, "json_mode", False):
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0

    choice = resp.choices[0]
    usage = resp.usage
    tin = getattr(usage, "prompt_tokens", 0) or 0
    tout = getattr(usage, "completion_tokens", 0) or 0
    return {
        "text": choice.message.content or "",
        "stop_reason": choice.finish_reason,
        "input_tokens": tin, "output_tokens": tout,
        "cost_usd": _cost(model, args, tin, tout),
        "latency_s": round(elapsed, 2),
        "request_id": getattr(resp, "id", None),
        "model": resp.model,
    }


def call_model(client, provider: str, model: str, system: str, user: str,
               max_tokens: int, effort: str, args) -> dict:
    """One stateless request against any supported provider."""
    if provider == "anthropic":
        return _call_anthropic(client, model, system, user, max_tokens, effort, args)
    return _call_openai_compatible(client, model, system, user, max_tokens,
                                   effort, args, provider)


def make_client(provider: str, max_retries: int):
    spec = PROVIDERS[provider]
    import os

    if spec["sdk"] == "anthropic":
        try:
            import anthropic
        except ImportError:
            sys.exit("[api] `pip install anthropic` first.")
        try:
            return anthropic.Anthropic(max_retries=max_retries)
        except Exception as e:
            sys.exit(f"[api] could not construct the Anthropic client: {e}\n"
                     f"      Set {spec['env']}, or run `ant auth login`.")

    try:
        import openai
    except ImportError:
        sys.exit(f"[api] the {provider} backend needs the OpenAI SDK: `pip install openai`.")
    key = os.environ.get(spec["env"])
    if not key:
        sys.exit(f"[api] {spec['env']} is not set — required for provider {provider!r}.")
    try:
        return openai.OpenAI(api_key=key, base_url=spec["base_url"],
                             max_retries=max_retries)
    except Exception as e:
        sys.exit(f"[api] could not construct the {provider} client: {e}")


def run_pass(items, fn, concurrency: int):
    """Map `fn` over `items` with bounded concurrency, preserving input order."""
    results = [None] * len(items)
    if concurrency <= 1:
        for i, it in enumerate(items):
            results[i] = fn(it)
        return results
    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        fut = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for f in futures.as_completed(fut):
            results[fut[f]] = f.result()
    return results


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def cmd_seed(args) -> int:
    roles = ["perturb", "verify"] if args.role == "both" else [args.role]
    for role in roles:
        dest = prompt_path(role, args.model, 1)
        if dest.exists() and not args.force:
            print(f"[seed] {dest} exists — pass --force to overwrite.")
            continue

        if args.from_skill:
            skill = REPO_ROOT / ".claude" / "skills" / f"{role}-v{args.from_skill}" / "SKILL.md"
            if not skill.exists():
                print(f"[seed] no such skill: {skill}")
                return 1
            body = skill.read_text()
            i, j = body.find("## Policy"), body.find("## Changelog")
            if i == -1 or j == -1:
                print(f"[seed] could not locate the policy section in {skill}")
                return 1
            imported = body[body.find("\n", body.find("\n", i) + 1):j].strip()
            base = (SEED_DIR / f"{role}.txt").read_text()
            pol = section(base, POLICY_BEGIN, POLICY_END)
            base = base.replace(pol, f"\n{imported}\n")
            text = base
            provenance = f"imported from .claude/skills/{role}-v{args.from_skill}"
        else:
            text = (SEED_DIR / f"{role}.txt").read_text()
            provenance = "seed template (first principles)"

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        write_json(dest.parent / "meta.json", {
            "role": role, "model": args.model, "version": 1,
            "parent": None, "tuned_from_round": None, "provenance": provenance,
        })
        print(f"[seed] {dest}  ({provenance})")
    return 0


# ---------------------------------------------------------------------------
# perturb / verify passes
# ---------------------------------------------------------------------------


def _load_prompt(role: str, args) -> tuple[str, int]:
    version = args.version or latest_version(role, args.model)
    if version == 0:
        sys.exit(f"[{role}] no prompt found under {PROMPTS_ROOT / role / model_slug(args.model)} "
                 f"— run `seed --model {args.model}` first.")
    p = prompt_path(role, args.model, version)
    if not p.exists():
        sys.exit(f"[{role}] missing {p}")
    return p.read_text(), version


def cmd_perturb(args) -> int:
    root = rollout_root(args.model, args.root)
    rdir = root / f"round_{args.round:02d}"
    eps = sorted(p for p in (rdir / "episodes").iterdir() if p.is_dir()) if (rdir / "episodes").is_dir() else []
    if not eps:
        sys.exit(f"[perturb] no episodes in {rdir} — run skill_selfplay.py --root {root} init first.")

    system, version = _load_prompt("perturb", args)
    jobs = [(e, render_perturb_user(read_json(e / "problem.json"))) for e in eps]

    if args.dry_run:
        for edir, user in jobs[: args.dry_run]:
            print(f"\n{'='*70}\n### {edir.name}  (perturb v{version}, {args.model} @ {infer_provider(args.model) if not args.provider else args.provider})\n{'='*70}")
            print(f"--- SYSTEM ({len(system)} chars) ---\n{system[:400]}\n   …[{len(system)-400} more chars]…")
            print(f"--- USER ---\n{user}")
        return 0

    provider = resolve_provider(args)
    client = make_client(provider, args.max_retries)

    def one(job):
        edir, user = job
        try:
            r = call_model(client, provider, args.model, system, user,
                           args.max_tokens, args.effort, args)
        except Exception as e:
            return edir, {"error": f"{type(e).__name__}: {e}"}
        if r["stop_reason"] == "length":
            out = edir / "perturb.json"
            if out.exists():
                out.unlink()
            return edir, {"error": "truncated at max_tokens "
                                   "(finish_reason=length) — raise --max-tokens"}
        (edir / "perturb.json").write_text(r.pop("text"))
        return edir, r

    out = run_pass(jobs, one, args.concurrency)
    _report(out, rdir, "perturb", version, args)
    return 0


def cmd_verify(args) -> int:
    # Two modes. Normally the pass reads a self-play round's verify_inbox and
    # writes its verify_outbox. With --inbox/--outbox it reads any directory of
    # {problem, solution_to_review} records — which is exactly the shape
    # processbench_eval.py's `prepare` emits, so the held-out evaluation runs
    # through the same prompt, the same renderer and the same leak guard.
    if args.inbox:
        inbox = Path(args.inbox)
        rdir = Path(args.outbox).parent if args.outbox else inbox.parent
        outbox_dir = Path(args.outbox) if args.outbox else inbox.parent / "outbox"
    else:
        if args.round is None:
            sys.exit("[verify] pass --round (self-play) or --inbox (arbitrary directory).")
        root = rollout_root(args.model, args.root)
        rdir = root / f"round_{args.round:02d}"
        inbox = rdir / "verify_inbox"
        outbox_dir = rdir / "verify_outbox"
    if not inbox.is_dir():
        sys.exit(f"[verify] no {inbox} — run skill_selfplay.py --root ... prepare-verify first.")
    files = sorted(inbox.glob("*.json"))
    if not files:
        sys.exit(f"[verify] {inbox} is empty (every perturbation may have been format-invalid).")

    system, version = _load_prompt("verify", args)
    jobs = [(f, render_verify_user(read_json(f))) for f in files]

    if args.dry_run:
        for f, user in jobs[: args.dry_run]:
            print(f"\n{'='*70}\n### {f.stem}  (verify v{version}, {args.model} @ {infer_provider(args.model) if not args.provider else args.provider})\n{'='*70}")
            print(f"--- SYSTEM ({len(system)} chars) ---\n{system[:400]}\n   …[{len(system)-400} more chars]…")
            print(f"--- USER ---\n{user}")
        return 0

    provider = resolve_provider(args)
    client = make_client(provider, args.max_retries)
    outbox = outbox_dir
    outbox.mkdir(parents=True, exist_ok=True)

    def one(job):
        f, user = job
        try:
            r = call_model(client, provider, args.model, system, user,
                           args.max_tokens, args.effort, args)
        except Exception as e:
            return f, {"error": f"{type(e).__name__}: {e}"}
        if r["stop_reason"] == "length":
            out = outbox / f.name
            if out.exists():
                out.unlink()
            return f, {"error": "truncated at max_tokens "
                                "(finish_reason=length) — raise --max-tokens"}
        (outbox / f.name).write_text(r.pop("text"))
        return f, r

    out = run_pass(jobs, one, args.concurrency)
    _report(out, rdir, "verify", version, args)
    return 0


def _report(results, rdir: Path, role: str, version: int, args) -> None:
    provider = resolve_provider(args)
    log = {"role": role, "provider": provider, "model": args.model,
           "prompt_version": version, "effort": args.effort,
           "max_tokens": args.max_tokens, "episodes": {}}
    errs = ok = 0
    tin = tout = 0
    cost = 0.0
    priced = True
    for key, meta in results:
        name = key.name if key.is_dir() else key.stem
        log["episodes"][name] = meta
        if "error" in meta:
            errs += 1
            print(f"[{role}] {name}: ERROR {meta['error']}")
            continue
        ok += 1
        tin += meta["input_tokens"]; tout += meta["output_tokens"]
        if meta["cost_usd"] is None:
            priced = False
            shown = "cost n/a"
        else:
            cost += meta["cost_usd"]
            shown = f"${meta['cost_usd']}"
        print(f"[{role}] {name}: ok  {meta['output_tokens']}tok  {meta['latency_s']}s  {shown}")
    total_cost = round(cost, 4) if priced else None
    log["totals"] = {"ok": ok, "errors": errs, "input_tokens": tin,
                     "output_tokens": tout, "cost_usd": total_cost}
    write_json(rdir / "api_logs" / f"{role}_v{version}.json", log)
    money = f"${total_cost}" if total_cost is not None else "cost n/a (pass --price-in/--price-out)"
    print(f"[{role}] {ok} ok, {errs} failed — {tin}+{tout} tok, {money} "
          f"→ {rdir / 'api_logs' / f'{role}_v{version}.json'}")


# ---------------------------------------------------------------------------
# check-prompt
# ---------------------------------------------------------------------------


def cmd_check_prompt(args) -> int:
    old_p = prompt_path(args.role, args.model, args.from_version)
    new_p = prompt_path(args.role, args.model, args.to_version)
    problems = []
    for p in (old_p, new_p):
        if not p.exists():
            print(f"[check-prompt] missing {p}")
            return 1
    old, new = old_p.read_text(), new_p.read_text()

    o_inv, n_inv = section(old, INVARIANT_BEGIN, INVARIANT_END), section(new, INVARIANT_BEGIN, INVARIANT_END)
    o_pol, n_pol = section(old, POLICY_BEGIN, POLICY_END), section(new, POLICY_BEGIN, POLICY_END)
    if n_inv is None or n_pol is None:
        problems.append("new prompt is missing the INVARIANT and/or POLICY markers")
    else:
        if o_inv != n_inv:
            import difflib
            d = list(difflib.unified_diff(o_inv.splitlines(), n_inv.splitlines(),
                                          "old", "new", lineterm="", n=1))
            problems.append("invariant region was modified:\n  " + "\n  ".join(d[:30]))
        if o_pol == n_pol:
            problems.append("policy section is unchanged — the update produced no learning")
    meta = new_p.parent / "meta.json"
    if not meta.exists():
        problems.append(f"missing {meta}")

    if problems:
        print(f"[check-prompt] {args.role} v{args.to_version}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[check-prompt] {args.role} v{args.to_version} ({args.model}): OK — invariant preserved, "
          f"policy changed ({len(o_pol.splitlines())} → {len(n_pol.splitlines())} lines).")
    return 0


def cmd_status(args) -> int:
    for role in ("perturb", "verify"):
        base = PROMPTS_ROOT / role / model_slug(args.model)
        latest = latest_version(role, args.model)
        print(f"{role:8s} {model_slug(args.model):12s} latest=v{latest or '-'}  {base}")
    root = rollout_root(args.model, args.root)
    if root.is_dir():
        for rd in sorted(root.glob("round_*")):
            s = rd / "round_summary.json"
            if s.exists():
                m = read_json(s)["metrics"]
                print(f"  {rd.name}: r_P={m['mean_perturber_reward']:.3f} "
                      f"r_V={m['mean_verifier_reward']:.3f} "
                      f"recall={m['mean_verifier_recall']:.3f}")
            else:
                print(f"  {rd.name}: not summarized")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Shared options, attached to every subcommand so they may follow it on the
    # command line (`selfplay_api.py seed --model ...`), which is how everyone
    # actually types them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="claude-opus-5",
                        help="API model id (default: claude-opus-5).")
    common.add_argument("--root", default=None,
                        help="Rollout root (default: data/api_rollouts/<model_slug>).")
    common.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                        help="Override the provider inferred from the model id.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", parents=[common], help="Create v1 prompt(s) for a model.")
    s.add_argument("--role", choices=["perturb", "verify", "both"], default="both")
    s.add_argument("--from-skill", type=int, default=None, dest="from_skill",
                   help="Import the policy section from .claude/skills/<role>-v<N> instead "
                        "of the first-principles seed.")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_seed)

    for name, fn, helptxt in (("perturb", cmd_perturb, "Run the Perturber pass over a round."),
                              ("verify", cmd_verify, "Run the Verifier pass over a round.")):
        q = sub.add_parser(name, parents=[common], help=helptxt)
        q.add_argument("--round", type=int, required=(name == "perturb"), default=None)
        if name == "verify":
            q.add_argument("--inbox", default=None,
                           help="Score an arbitrary directory of {problem, solution_to_review} "
                                "records (e.g. a ProcessBench eval inbox) instead of a round.")
            q.add_argument("--outbox", default=None,
                           help="Where to write replies when --inbox is used "
                                "(default: <inbox>/../outbox).")
        q.add_argument("--version", type=int, default=None,
                       help="Prompt version (default: highest present).")
        q.add_argument("--concurrency", type=int, default=4)
        q.add_argument("--max-tokens", type=int, default=16000, dest="max_tokens")
        q.add_argument("--effort", default="high",
                       choices=["low", "medium", "high", "xhigh", "max"])
        q.add_argument("--max-retries", type=int, default=4, dest="max_retries")
        q.add_argument("--dry-run", type=int, nargs="?", const=1, default=0, dest="dry_run",
                       help="Render the first N requests instead of sending them.")
        q.add_argument("--price-in", type=float, default=None, dest="price_in",
                       help="USD per 1M input tokens, for costed logs on models "
                            "with no built-in rate.")
        q.add_argument("--price-out", type=float, default=None, dest="price_out",
                       help="USD per 1M output tokens.")
        q.add_argument("--json-mode", action="store_true", dest="json_mode",
                       help="OpenAI/DeepSeek only: force response_format=json_object. "
                            "Off by default — it has no Anthropic equivalent, so "
                            "enabling it confounds cross-provider comparisons.")
        q.set_defaults(func=fn)

    c = sub.add_parser("check-prompt", parents=[common], help="Verify a new prompt version against its parent.")
    c.add_argument("--role", choices=["perturb", "verify"], required=True)
    c.add_argument("--from-version", type=int, required=True, dest="from_version")
    c.add_argument("--to-version", type=int, required=True, dest="to_version")
    c.set_defaults(func=cmd_check_prompt)

    t = sub.add_parser("status", parents=[common], help="Show prompt versions and scored rounds for a model.")
    t.set_defaults(func=cmd_status)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    raise SystemExit(a.func(a))
