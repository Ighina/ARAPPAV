#!/usr/bin/env python
"""Post-hoc leakage audit for a completed pipeline round (spec 9, 18).

The pipeline prevents leakage when it *builds* a prompt. This script checks the
other direction: given the artefacts a round actually produced, does any prompt
that was sent contain something the receiving agent should not have had?

It reads the persisted prompts, not the code, so it catches leaks introduced by
a future change to the renderers as well.

    python scripts/audit_leakage.py data/skill_rollouts/round_0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def audit_round(rdir: Path) -> tuple[int, int]:
    prompts = rdir / "prompts"
    eps = sorted((rdir / "episodes").iterdir()) if (rdir / "episodes").is_dir() else []
    checks = failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal checks, failures
        checks += 1
        if not ok:
            failures += 1
            print(f"  FAIL  {name} {detail}")

    for ed in eps:
        eid = ed.name
        data = json.loads((ed / "data.json").read_text())
        gt_path = ed / "perturb.json"
        vprompt = prompts / f"verify__{eid}.txt"
        pprompt = prompts / f"perturb__{eid}.txt"

        if pprompt.exists():
            p = pprompt.read_text()
            check(f"{eid} perturber got the problem", data["problem"] in p)
            check(f"{eid} perturber got the solution", data["solution"] in p)
            check(f"{eid} perturber prompt free of reward state",
                  not any(m in p for m in ("perturber_reward", "verifier_reward")))

        if not (gt_path.exists() and vprompt.exists()):
            continue
        gt = json.loads(gt_path.read_text())
        v = vprompt.read_text()

        check(f"{eid} verifier did NOT get the original solution",
              data["solution"] not in v)
        check(f"{eid} verifier was NOT told k", "## k" not in v)
        check(f"{eid} verifier got the problem verbatim", data["problem"] in v)
        for e in gt.get("errors", []):
            check(f"{eid} no original_text span leaked", e["original_text"] not in v,
                  f"({e['error_id']})")
            check(f"{eid} no rationale leaked", e["rationale"][:40] not in v,
                  f"({e['error_id']})")
            check(f"{eid} no error_type leaked", e["error_type"] not in v,
                  f"({e['error_id']})")
        check(f"{eid} no ground-truth keys leaked",
              not any(m in v for m in ("error_id", '"errors"', "injected_text", "rationale")))
        check(f"{eid} no reward state leaked",
              not any(m in v for m in ("perturber_reward", "verifier_reward", "best_overlap")))

        # spec 2/7: the problem must survive perturbation untouched
        check(f"{eid} perturbed solution differs from the original",
              gt["perturbed_solution"] != data["solution"])
        check(f"{eid} problem text absent from the perturbed solution",
              data["problem"] not in gt["perturbed_solution"])

    return checks, failures


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    total = bad = 0
    for arg in sys.argv[1:]:
        rdir = Path(arg)
        print(f"=== {rdir} ===")
        c, f = audit_round(rdir)
        total += c
        bad += f
        print(f"  {c - f}/{c} checks passed")
    print(f"\n{'LEAKAGE DETECTED' if bad else 'CLEAN'}: {total - bad}/{total} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
