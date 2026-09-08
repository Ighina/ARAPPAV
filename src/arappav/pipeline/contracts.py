"""Data contracts and the leakage boundary.

Every payload that reaches an agent is built **here**, from an explicit field
allowlist, and nowhere else. That is the whole leakage-prevention strategy: a
field that is not named in this module cannot reach a model, because no other
code path constructs agent input.

The episode contract (spec section 2)::

    data.json   {"problem": ..., "solution": ...}   <- the dataset pair
    meta.json   {"episode_id", "source_id", "topic", "level", "k", "round"}

`problem` and `solution` follow the Hendrycks MATH fields. They are kept in a
separate file from the metadata so that the pair a dataset supplies is
distinguishable, on disk, from anything the pipeline adds around it.

**Only `solution` is ever perturbed.** The perturber receives `problem` as
read-only context — it needs it to judge what a plausible error looks like —
and returns a rewritten solution. The orchestrator re-attaches the original
`problem` verbatim afterwards and asserts it is unchanged, so a perturber that
tried to rewrite, simplify or solve the problem cannot affect the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Field allowlists — the leakage boundary
# ---------------------------------------------------------------------------

#: The only episode fields the Perturber may see. `solution` is the field it
#: rewrites; `problem` is read-only context; `k` is the error budget.
PERTURB_INPUT_FIELDS = ("problem", "solution", "k")

#: The only episode fields the Verifier may see. Note what is absent:
#: `solution` (the *original*, which would give the diff away), `k` (how many
#: errors to expect), and every field of the perturber's ground truth.
VERIFY_INPUT_FIELDS = ("problem", "solution_to_review")

#: Substrings that must never appear in a Verifier payload. These are the
#: serialized keys of the perturber's ground truth; their presence means a
#: structure leaked rather than a field being copied.
GROUND_TRUTH_MARKERS = (
    "injected_text", "original_text", "rationale", "error_id",
    "error_type", "perturbed_solution", '"errors"', "step_index",
)

#: Substrings that must never appear in *any* agent payload: reward state and
#: cross-round information that the protocol releases only through the
#: findings -> policy-update channel.
REWARD_MARKERS = (
    "perturber_reward", "verifier_reward", "verifier_recall",
    "verifier_precision", "num_matched", "best_overlap", "score.json",
)


class LeakageError(RuntimeError):
    """Raised before a request is sent, when a payload violates a boundary."""


def _scan(payload: str, markers: tuple[str, ...], who: str, allow: tuple = ()) -> None:
    hits = [m for m in markers if m in payload and m not in allow]
    if hits:
        raise LeakageError(
            f"REFUSING TO SEND to {who}: forbidden markers {hits} appear in the "
            f"payload. This is a leak — fix the caller, do not relax the check."
        )


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """One sampled dataset item. `problem`/`solution` are the dataset pair."""

    episode_id: str
    problem: str
    solution: str
    k: int
    source_id: str = ""
    topic: str | None = None
    level: str | None = None
    round_index: int = 0

    def data(self) -> dict:
        """`data.json` — exactly the dataset pair, nothing else."""
        return {"problem": self.problem, "solution": self.solution}

    def meta(self) -> dict:
        """`meta.json` — everything the pipeline adds around the pair."""
        return {
            "episode_id": self.episode_id, "source_id": self.source_id,
            "topic": self.topic, "level": self.level, "k": self.k,
            "round": self.round_index,
        }

    @classmethod
    def load(cls, data: dict, meta: dict) -> "Episode":
        return cls(
            episode_id=meta["episode_id"], problem=data["problem"],
            solution=data["solution"], k=meta["k"],
            source_id=meta.get("source_id", ""), topic=meta.get("topic"),
            level=meta.get("level"), round_index=meta.get("round", 0),
        )


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def perturb_payload(ep: Episode) -> dict:
    """Build the Perturber's input from the allowlist (spec 7)."""
    return {"problem": ep.problem, "solution": ep.solution, "k": ep.k}


def render_perturb_prompt(ep: Episode, policy_ref: str, context_block: str = "") -> str:
    """Render the Perturber's `claude -p` prompt.

    The problem is presented explicitly as read-only, and the solution as the
    only mutable field, so the contract is visible in the prompt itself and not
    only in the skill file.
    """
    p = perturb_payload(ep)
    body = (
        f"{policy_ref}\n\n"
        f"{context_block}"
        "## PROBLEM (read-only context — never modify, never restate as changed)\n"
        f"{p['problem']}\n\n"
        "## SOLUTION (this is the ONLY field you may perturb)\n"
        f"{p['solution']}\n\n"
        f"## k\n{p['k']}\n\n"
        "Return only the JSON object described in your output contract."
    )
    _scan(body, REWARD_MARKERS, "perturber")
    return body


def verify_payload(ep: Episode, perturbed_solution: str) -> dict:
    """Build the Verifier's input from the allowlist (spec 8).

    Takes the *perturbed* solution only. The original solution, the error list
    and `k` are not parameters of this function, so they cannot be forwarded
    even by mistake.
    """
    return {"problem": ep.problem, "solution_to_review": perturbed_solution}


def render_verify_prompt(ep: Episode, perturbed_solution: str,
                         policy_ref: str, context_block: str = "") -> str:
    """Render the Verifier's `claude -p` prompt, then audit it for leakage."""
    v = verify_payload(ep, perturbed_solution)
    body = (
        f"{policy_ref}\n\n"
        f"{context_block}"
        "## PROBLEM\n"
        f"{v['problem']}\n\n"
        "## SOLUTION TO REVIEW\n"
        f"{v['solution_to_review']}\n\n"
        "Return only the JSON object described in your output contract."
    )
    # `step_index` is part of the verifier's own output contract, so it is
    # allowed to appear in its policy reference — but nowhere in the episode
    # data. Scan only the data half.
    data_half = body[len(policy_ref):]
    _scan(data_half, GROUND_TRUTH_MARKERS, "verifier")
    _scan(body, REWARD_MARKERS, "verifier")
    return body


def assert_problem_unchanged(ep: Episode, returned_problem: str | None) -> None:
    """Hard gate on spec 2/7: the perturber must not touch `problem`.

    A perturber that echoes the problem at all must echo it verbatim; the
    orchestrator then discards its copy and re-attaches the original anyway,
    so this is a tripwire rather than the mechanism.
    """
    if returned_problem is None:
        return
    if returned_problem.strip() != ep.problem.strip():
        raise LeakageError(
            f"{ep.episode_id}: the perturber returned a MODIFIED problem. "
            "Only `solution` may be perturbed."
        )
