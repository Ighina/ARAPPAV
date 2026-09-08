#!/usr/bin/env python
"""Plot verifier precision and recall across self-play rounds.

    python scripts/plot_verification_evolution.py data/skill_rollouts/haiku_cold_10x8

Both series are rates on 0-1, so they share one y-axis (never a dual axis).
Colours are the validated categorical slots 1 and 2 — CVD Delta E 24.7, well
above the 8 threshold — and each series carries a direct label at its final
point as well as a legend, so identity never rests on colour alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from arappav.pipeline.plotting import (  # noqa: E402
    SERIES, SURFACE, TEXT_SECONDARY, load_rounds, plot_series, save,
    style_axes, titles,
)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else "data/skill_rollouts/haiku_cold_10x8")
    rows = load_rounds(root, "mean_verifier_precision")
    if not rows:
        sys.exit(f"no scored rounds under {root}")

    xs = [r["round"] for r in rows]
    prec = [r["mean_verifier_precision"] for r in rows]
    rec = [r["mean_verifier_recall"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    plot_series(ax, xs, prec, SERIES[0], "Precision")
    plot_series(ax, xs, rec, SERIES[1], "Recall", linestyle=(0, (4, 3)),
                markersize=5, zorder=4)

    for ys, label, off in ((prec, "precision", 10), (rec, "recall", -12)):
        ax.annotate(f"{label} {ys[-1]:.2f}", xy=(xs[-1], ys[-1]),
                    xytext=(-6, off), textcoords="offset points",
                    va="center", ha="right", fontsize=9.5,
                    color=TEXT_SECONDARY)

    ax.set_xlabel("Self-play round", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Rate", fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=9.5,
              labelcolor=TEXT_SECONDARY, ncols=2, handlelength=2.4,
              columnspacing=2.2)

    titles(fig, ax, "Verifier precision and recall across self-play rounds",
           f"{root.name} · cold start · {rows[0]['n_episodes']} episodes per round · "
           f"claude-haiku-4-5")
    fig.subplots_adjust(top=0.78, right=0.965, left=0.095, bottom=0.13)

    out = save(fig, root / "verification_evolution.pdf")
    print(f"[plot] {len(rows)} round(s) → {out}")
    for r, p, q in zip(xs, prec, rec):
        print(f"  round {r}: precision={p:.3f} recall={q:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
