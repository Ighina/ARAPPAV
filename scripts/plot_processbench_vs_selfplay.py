#!/usr/bin/env python
"""Plot held-out ProcessBench F1 against self-play verifier F1, by round.

    python scripts/plot_processbench_vs_selfplay.py \
        data/skill_rollouts/haiku_cold_10x8 data/policy_evals hverify

Self-play F1 is measured against the perturber that co-evolved with the policy,
on fresh problems each round. ProcessBench F1 is measured on one fixed held-out
sample of human-annotated errors, identical for every policy version. Both are
F1 on a 0-1 scale, so they share a y-axis.

The comparison is the point: self-play movement that does not appear on the
held-out line is the policy fitting its opponent, not learning to verify.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from arappav.pipeline.plotting import (  # noqa: E402
    SERIES, SURFACE, TEXT_SECONDARY, load_rounds, plot_series, save,
    style_axes, titles,
)


def main() -> int:
    run = Path(sys.argv[1] if len(sys.argv) > 1
               else "data/skill_rollouts/haiku_cold_10x8")
    evals = Path(sys.argv[2] if len(sys.argv) > 2 else "data/policy_evals")
    prefix = sys.argv[3] if len(sys.argv) > 3 else "hverify"

    rows = load_rounds(run, "mean_verifier_reward")
    xs, sp = [], []
    pb_x, pb = [], []
    for r in rows:
        xs.append(r["round"])
        sp.append(r["mean_verifier_reward"])
        # round i is played by policy v(i+1)
        v = r["round"] + 1
        f = evals / f"{prefix}-v{v}" / f"round_{v:02d}" / "eval_summary.json"
        if f.exists():
            o = json.loads(f.read_text()).get("overall") or {}
            if o.get("processbench_f1") is not None:
                pb_x.append(r["round"])
                pb.append(o["processbench_f1"])

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    plot_series(ax, xs, sp, SERIES[0], "Self-play F1 (vs co-evolved perturber)")
    plot_series(ax, pb_x, pb, SERIES[2], "ProcessBench F1 (held-out, fixed items)",
                linestyle=(0, (4, 3)), markersize=5, zorder=4)

    ax.set_xlabel("Self-play round", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Verifier F1", fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks(xs)
    lo = min(sp + pb) - 0.06
    ax.set_ylim(max(0, lo), 1.03)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5,
              labelcolor=TEXT_SECONDARY, handlelength=2.4)

    n = len(pb)
    delta = pb[-1] - pb[0] if n > 1 else 0.0
    titles(fig, ax, "Self-play gains did not transfer to held-out data",
           f"{run.name} · cold start · {rows[0]['n_episodes']} episodes per round · "
           f"claude-haiku-4-5\n"
           f"ProcessBench: {n} policy versions on one identical 80-item sample · "
           f"v1 {pb[0]:.3f} → v{n} {pb[-1]:.3f} ({delta:+.3f})")
    fig.subplots_adjust(top=0.76, right=0.965, left=0.095, bottom=0.13)

    out = save(fig, run / "processbench_vs_selfplay.pdf")
    print(f"[plot] → {out}")
    print(f"{'round':>6}{'self-play F1':>14}{'ProcessBench F1':>18}")
    for i, r in enumerate(xs):
        p = f"{pb[i]:.3f}" if i < len(pb) else "n/a"
        print(f"{r:>6}{sp[i]:>14.3f}{p:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
