"""Structured edits to a policy body — the Trace2Skill-style update path.

The default updater rewrites the whole policy every round. Measured over ten
rounds that produced monotone accumulation: bodies grew from 90 to 2,922
characters while held-out F1 stayed flat, with consecutive word-level diffs of
+253, -13, +31, +1, +24, +68, -11, +12, +51. The updater appends; it almost
never deletes, because deleting requires actively choosing to omit something
while rewriting, and omission is not a salient action.

Naming the operation changes that. Here an analyst emits typed edits —
including `delete_rule` — and Python applies them. Deletion becomes a thing you
*do* rather than a thing you fail to do, and every edit is small, inspectable
and reversible.

Applied to the tuned policy body only. The invariant region is never visible to
this machinery, so no patch can reach the contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: The operations an analyst may propose.
OPS = (
    "append_to_rule",     # add a sentence to the end of an existing rule
    "replace_in_rule",    # swap specific text inside a rule
    "rewrite_rule",       # replace one rule's body entirely
    "add_rule",           # introduce a new rule
    "delete_rule",        # remove a rule outright — the point of this design
)


class PatchError(ValueError):
    pass


@dataclass
class Edit:
    op: str
    target: str = ""          # which rule, by its leading label or first words
    content: str = ""         # new text, for the ops that add any
    old_text: str = ""        # for replace_in_rule
    after: str = ""           # for add_rule placement
    reason: str = ""          # why, for the audit trail

    @classmethod
    def parse(cls, d: dict) -> "Edit":
        if not isinstance(d, dict):
            raise PatchError(f"edit is not an object: {d!r}")
        op = d.get("op")
        if op not in OPS:
            raise PatchError(f"unknown op {op!r}; expected one of {OPS}")
        e = cls(op=op, target=str(d.get("target", "")),
                content=str(d.get("content", "")),
                old_text=str(d.get("old_text", "")),
                after=str(d.get("after", "")),
                reason=str(d.get("reason", "")))
        if op != "add_rule" and not e.target:
            raise PatchError(f"{op} needs a target")
        if op in ("append_to_rule", "rewrite_rule", "add_rule") and not e.content:
            raise PatchError(f"{op} needs content")
        if op == "replace_in_rule" and not e.old_text:
            raise PatchError("replace_in_rule needs old_text")
        return e


@dataclass
class Patch:
    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""

    @classmethod
    def parse(cls, d: dict) -> "Patch":
        if not isinstance(d, dict):
            raise PatchError("patch is not an object")
        raw = d.get("edits")
        if not isinstance(raw, list):
            raise PatchError("patch has no `edits` list")
        return cls(edits=[Edit.parse(e) for e in raw],
                   reasoning=str(d.get("reasoning", "")))


# ---------------------------------------------------------------------------
# Splitting a policy body into addressable rules
# ---------------------------------------------------------------------------

#: A rule starts at a numbered item ("3."), a bold label ("**P2 —"), or a
#: markdown heading. Anything else belongs to the rule above it.
_RULE_START = re.compile(r"^\s*(?:\d+\.\s|\*\*[A-Za-z0-9]|#{1,6}\s)")


def split_rules(body: str) -> list[str]:
    """Split a policy body into rule blocks, preserving text exactly."""
    lines = body.splitlines(keepends=True)
    blocks, cur = [], []
    for ln in lines:
        if _RULE_START.match(ln) and cur and "".join(cur).strip():
            blocks.append("".join(cur))
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        blocks.append("".join(cur))
    return [b for b in blocks if b.strip()]


def _label(block: str) -> str:
    """A short handle for a rule, used to match an edit's `target`."""
    first = block.strip().splitlines()[0] if block.strip() else ""
    return first.strip()


def _find(blocks: list[str], target: str) -> int:
    """Index of the rule an edit addresses. Matching is deliberately lenient:
    an analyst refers to a rule the way a reader would, not by index."""
    t = target.strip().lower()
    if not t:
        raise PatchError("empty target")
    for i, b in enumerate(blocks):                       # exact label prefix
        if _label(b).lower().startswith(t):
            return i
    for i, b in enumerate(blocks):                       # substring anywhere
        if t in b.lower():
            return i
    raise PatchError(f"no rule matching target {target!r}")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_patch(body: str, patch: Patch) -> tuple[str, list[str]]:
    """Apply a patch to a policy body. Returns (new_body, per-edit log).

    Edits are resolved against the *original* block list and applied in one
    pass, so two edits cannot silently interact through a target that a
    previous edit moved or removed.
    """
    blocks = split_rules(body)
    if not blocks:
        blocks = [body] if body.strip() else []
    plan: list[tuple[int, Edit]] = []
    log: list[str] = []

    for e in patch.edits:
        if e.op == "add_rule":
            idx = _find(blocks, e.after) if e.after else len(blocks) - 1
            plan.append((idx, e))
            continue
        plan.append((_find(blocks, e.target), e))

    # A rule may be touched once. Two edits to the same rule is the conflict
    # the merge step is supposed to have resolved; refusing here keeps a
    # botched merge from silently producing mangled text.
    seen: dict[int, str] = {}
    for idx, e in plan:
        if e.op == "add_rule":
            continue
        if idx in seen:
            raise PatchError(
                f"two edits target the same rule ({seen[idx]} and {e.op}) — "
                f"the merge step should have combined them")
        seen[idx] = e.op

    out = list(blocks)
    inserts: list[tuple[int, str]] = []
    for idx, e in plan:
        if e.op == "append_to_rule":
            out[idx] = out[idx].rstrip() + " " + e.content.strip() + "\n"
        elif e.op == "replace_in_rule":
            if e.old_text not in out[idx]:
                raise PatchError(f"replace_in_rule: {e.old_text[:60]!r} not in the rule")
            out[idx] = out[idx].replace(e.old_text, e.content)
        elif e.op == "rewrite_rule":
            out[idx] = e.content.rstrip() + "\n"
        elif e.op == "delete_rule":
            out[idx] = ""
        elif e.op == "add_rule":
            inserts.append((idx + 1, e.content.rstrip() + "\n"))
        log.append(f"{e.op}: {(e.target or e.after or '(end)')[:50]}"
                   + (f" — {e.reason[:80]}" if e.reason else ""))

    for at, text in sorted(inserts, reverse=True):
        out.insert(at, text)
    new = "\n".join(b.rstrip() for b in out if b.strip()) + "\n"
    return new, log


def summarise_patch(patch: Patch) -> dict:
    """Counts by operation, for the round record."""
    counts: dict[str, int] = {}
    for e in patch.edits:
        counts[e.op] = counts.get(e.op, 0) + 1
    return {"num_edits": len(patch.edits), "by_op": counts,
            "reasoning": patch.reasoning[:400]}
