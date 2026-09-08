"""Policy (skill) lifecycle, owned by Python.

A "policy" here is the tuned section of a versioned skill file,
`.claude/skills/<prefix>-v<N>/SKILL.md`. The orchestrator — not an agent —
creates every version, so:

* the invariant region (input contract, output contract, scoring, taxonomy) is
  copied byte-for-byte from the template and can never drift (spec 7, 17);
* an agent that writes a policy returns only *policy text* on stdout; Python
  splices it in. A skill cannot rewrite its own contract even if it tries.

Cold start (spec 3) leaves the policy body empty. Warm start fills it from
`create-policy-perturber` / `create-policy-verifier`. Later rounds fill it from
`update-perturb` / `update-verify`, unless the role is frozen (spec 5).
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
POLICY_START = "## Policy"
CHANGELOG_START = "## Changelog"

EMPTY_POLICY = (
    "_No policy. This is a cold start: act on the contract above and your own "
    "judgement alone._"
)


class PolicyError(RuntimeError):
    pass


def _template(role: str) -> str:
    return (TEMPLATES / f"{role}_base.md").read_text()


def skill_dir(skills_root: Path, prefix: str, version: int) -> Path:
    return skills_root / f"{prefix}-v{version}"


def skill_file(skills_root: Path, prefix: str, version: int) -> Path:
    return skill_dir(skills_root, prefix, version) / "SKILL.md"


def render(role: str, name: str, version: int, parent: str, tuned_from: str,
           policy: str, changelog: str) -> str:
    """Fill the template. The invariant region is untouched by construction."""
    return (
        _template(role)
        .replace("{name}", name)
        .replace("{version}", str(version))
        .replace("{parent}", parent)
        .replace("{tuned_from}", tuned_from)
        .replace("{policy}", policy.strip() or EMPTY_POLICY)
        .replace("{changelog}", changelog)
    )


def extract_policy(text: str) -> str:
    """Return the tuned body between the Policy heading and the changelog."""
    i, j = text.find(POLICY_START), text.find(CHANGELOG_START)
    if i == -1 or j == -1 or j < i:
        raise PolicyError("could not locate the policy section")
    body = text[text.index("\n", i) + 1: j]
    # drop the standing "TUNED SECTION" blockquote and trailing rule
    body = re.sub(r"^\s*>.*?\n\n", "", body, flags=re.S)
    return body.strip().removesuffix("---").strip()


def invariant_of(text: str) -> str:
    i = text.find("## Input")
    j = text.find(POLICY_START)
    if i == -1 or j == -1:
        raise PolicyError("could not locate the invariant region")
    return text[i:j]


def write_version(skills_root: Path, role: str, prefix: str, version: int,
                  policy: str, changelog: str, parent: str = "—",
                  tuned_from: str = "—", overwrite: bool = False) -> Path:
    """Create `<prefix>-v<version>` and return the SKILL.md path."""
    path = skill_file(skills_root, prefix, version)
    if path.exists() and not overwrite:
        raise PolicyError(
            f"{path} already exists. Pass --overwrite-policies to replace it, or "
            f"use --skill-prefix to run in a separate namespace."
        )
    name = f"{prefix}-v{version}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(role, name, version, parent, tuned_from, policy, changelog))
    return path


def check_invariant(skills_root: Path, role: str, prefix: str,
                    from_version: int, to_version: int) -> list[str]:
    """Validate a version bump the way the pre-refactor `check-skill` did."""
    problems: list[str] = []
    old_p = skill_file(skills_root, prefix, from_version)
    new_p = skill_file(skills_root, prefix, to_version)
    for p in (old_p, new_p):
        if not p.exists():
            return [f"missing {p}"]
    old, new = old_p.read_text(), new_p.read_text()

    m = re.search(r"^name:\s*(\S+)\s*$", new, re.MULTILINE)
    if not m:
        problems.append("new skill has no `name:` in its frontmatter")
    elif m.group(1) != f"{prefix}-v{to_version}":
        problems.append(f"frontmatter name {m.group(1)!r} != {prefix}-v{to_version}")

    if invariant_of(old) != invariant_of(new):
        problems.append("invariant region was modified")
    if extract_policy(old) == extract_policy(new):
        problems.append("policy section is unchanged — the update produced no learning")
    return problems


def latest_version(skills_root: Path, prefix: str) -> int:
    if not skills_root.is_dir():
        return 0
    vs = [int(m.group(1)) for p in skills_root.iterdir()
          if (m := re.fullmatch(rf"{re.escape(prefix)}-v(\d+)", p.name))
          and (p / "SKILL.md").exists()]
    return max(vs, default=0)
