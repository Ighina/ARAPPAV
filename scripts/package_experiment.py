#!/usr/bin/env python
"""Package an experiment for transport, and unpack it on another machine.

An experiment is scattered across four trees that do not share a naming scheme:

    data/skill_rollouts/<name>/        the run itself
    data/policy_evals/<name>/          held-out evaluations
    data/validation_math500/<name*>/   validation sets
    .claude/skills/<prefix>-v*/        the policies the run produced

The skill prefix is not the experiment name — `algebra_evolve` writes
`algev_perturb-v1..v10` — so the prefixes are read from the run's own
`run_config.json` rather than guessed. Everything needed to reproduce or
re-score the experiment travels together, with a manifest recording where each
piece came from.

    python scripts/package_experiment.py pack algebra_evolve
    # ... copy algebra_evolve.zip to the other machine ...
    python scripts/package_experiment.py unpack algebra_evolve.zip

Unpacking refuses to overwrite an existing experiment unless --force is given:
silently merging two runs of the same name produces a directory that looks like
one experiment and is two.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = "arappav_experiment.json"

#: (tree, whether the experiment name is a prefix rather than an exact match)
ROOTS = (
    ("data/skill_rollouts", False),
    ("data/policy_evals", False),
    ("data/validation_math500", True),
    ("data/perturber_evals", True),
    ("data/final_test", True),
)


def _prefixes(name: str) -> list[str]:
    """Skill prefixes this experiment wrote, from its own run config."""
    cfg = REPO / "data" / "skill_rollouts" / name / "run_config.json"
    if not cfg.exists():
        return []
    d = json.loads(cfg.read_text())
    return [p for p in (d.get("perturb_prefix"), d.get("verify_prefix")) if p]


def _known_experiments() -> set[str]:
    base = REPO / "data" / "skill_rollouts"
    return {p.name for p in base.iterdir() if p.is_dir()} if base.is_dir() else set()


def _aliases(name: str) -> list[str]:
    """Every name this experiment's directories might be filed under.

    Evaluation and validation roots are conventionally named after the skill
    prefix rather than the experiment — algebra_evolve's validation sets live
    under `algev/` and `algev_indep/` — so matching on the experiment name
    alone silently ships a package with no validation data in it.
    """
    out = {name}
    for pre in _prefixes(name):
        out.add(pre)
        stem = pre.rsplit("_", 1)[0]          # algev_perturb -> algev
        if stem:
            out.add(stem)
    return sorted(out, key=len, reverse=True)


def _matches(child: str, name: str, aliases: list[str], others: set[str]) -> bool:
    """Does a directory belong to this experiment and not to a sibling?

    `startswith` alone is wrong: "algebra_evolve_b" starts with
    "algebra_evolve", so packing one run would swallow the other's data.
    """
    if child in others and child != name:
        return False                          # it is another experiment outright
    for a in aliases:
        if child == a or child.startswith(a + "_"):
            # A longer alias belonging to a different experiment wins, so
            # algev_indep goes to algebra_evolve but algevb_* does not.
            better = [o for o in others if o != name and
                      (child == o or child.startswith(o + "_"))]
            return not better
    return False


def _collect(name: str) -> tuple[list[Path], dict]:
    """Every path belonging to the experiment, plus a summary by component."""
    paths: list[Path] = []
    summary: dict[str, int] = {}
    aliases = _aliases(name)
    others = _known_experiments()
    # A sibling's aliases must not be claimed either: algevb belongs to
    # algebra_evolve_b, and algev is a prefix of it.
    sibling_aliases = {a for o in others if o != name for a in _aliases(o)}

    for tree, by_prefix in ROOTS:
        base = REPO / tree
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if by_prefix:
                hit = _matches(child.name, name, aliases, others)
                if hit and any(child.name == s or child.name.startswith(s + "_")
                               for s in sibling_aliases - set(aliases)):
                    hit = False
            else:
                hit = child.name == name
            if not hit:
                continue
            files = [p for p in child.rglob("*") if p.is_file()]
            paths += files
            summary[f"{tree}/{child.name}"] = len(files)

    for prefix in _prefixes(name):
        for skill in sorted((REPO / ".claude" / "skills").glob(f"{prefix}-v*")):
            files = [p for p in skill.rglob("*") if p.is_file()]
            paths += files
            summary[f".claude/skills/{skill.name}"] = len(files)

    return paths, summary


def cmd_pack(args) -> int:
    name = args.name
    paths, summary = _collect(name)
    if not paths:
        sys.exit(f"[pack] found nothing for experiment {name!r}. Known runs: "
                 f"{sorted(p.name for p in (REPO/'data'/'skill_rollouts').iterdir() if p.is_dir())}")

    out = Path(args.out) if args.out else REPO / f"{name}.zip"
    cfg_path = REPO / "data" / "skill_rollouts" / name / "run_config.json"
    manifest = {
        "experiment": name,
        "packed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skill_prefixes": _prefixes(name),
        "run_config": json.loads(cfg_path.read_text()) if cfg_path.exists() else None,
        "components": summary,
        "num_files": len(paths),
    }

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
        for p in paths:
            z.write(p, p.relative_to(REPO).as_posix())

    mb = out.stat().st_size / 1e6
    print(f"[pack] {name}: {len(paths)} files, {mb:.1f} MB → {out}")
    for comp, n in sorted(summary.items()):
        leaf = comp.rsplit("/", 1)[-1]
        # Flag anything matched by alias rather than by the experiment name, so
        # a stray directory from a deleted run is visible rather than silent.
        mark = "" if leaf == name or comp.startswith(".claude/skills") else "  (by alias)"
        print(f"         {n:5d}  {comp}{mark}")
    if not any(k.startswith(".claude/skills") for k in summary):
        print("[pack] NOTE: no policy skills found — the run's prefixes may have "
              "been deleted, so the package cannot be re-scored as-is.")
    return 0


def cmd_unpack(args) -> int:
    src = Path(args.file)
    if not src.exists():
        sys.exit(f"[unpack] no such file: {src}")
    dest = Path(args.into) if args.into else REPO

    with zipfile.ZipFile(src) as z:
        try:
            manifest = json.loads(z.read(MANIFEST))
        except KeyError:
            sys.exit(f"[unpack] {src} has no {MANIFEST}; it was not written by "
                     f"package_experiment.py")
        name = manifest["experiment"]
        members = [m for m in z.namelist() if m != MANIFEST]

        # Refuse to merge two runs of the same name: the result looks like one
        # experiment and is silently two, which is unrecoverable after the fact.
        clashes = [t for t, byp in ROOTS
                   for c in [dest / t / name]
                   if c.exists()]
        skills = [dest / ".claude" / "skills" / s
                  for pre in manifest.get("skill_prefixes", [])
                  for s in [p.name for p in (dest / ".claude" / "skills").glob(f"{pre}-v*")]]
        if (clashes or skills) and not args.force:
            where = clashes + [str(s) for s in skills]
            sys.exit(f"[unpack] {name!r} already exists here ({where[:3]}...). "
                     f"Pass --force to overwrite, or --into a clean directory.")

        for m in members:
            target = dest / m
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as fh, open(target, "wb") as out:
                out.write(fh.read())

    print(f"[unpack] {name}: {len(members)} files → {dest}")
    for comp, n in sorted(manifest.get("components", {}).items()):
        print(f"           {n:5d}  {comp}")
    cfg = manifest.get("run_config") or {}
    if cfg:
        print(f"[unpack] run config: category={cfg.get('category')} "
              f"update_mode={cfg.get('update_mode')} rounds={cfg.get('rounds')} "
              f"episodes={cfg.get('episodes')} model={cfg.get('model')}")
    print(f"[unpack] policies restored under .claude/skills/: "
          f"{', '.join(manifest.get('skill_prefixes') or ['(none)'])}")
    return 0


def cmd_list(args) -> int:
    with zipfile.ZipFile(args.file) as z:
        manifest = json.loads(z.read(MANIFEST))
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("pack", help="bundle an experiment into a zip")
    a.add_argument("name", help="experiment name, e.g. algebra_evolve")
    a.add_argument("--out", default=None, help="output path (default <name>.zip)")
    a.set_defaults(func=cmd_pack)

    b = sub.add_parser("unpack", help="restore a bundle into this repository")
    b.add_argument("file")
    b.add_argument("--into", default=None, help="repository root (default: this one)")
    b.add_argument("--force", action="store_true",
                   help="overwrite an experiment of the same name")
    b.set_defaults(func=cmd_unpack)

    c = sub.add_parser("list", help="show a bundle's manifest without extracting")
    c.add_argument("file")
    c.set_defaults(func=cmd_list)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    raise SystemExit(a.func(a))
