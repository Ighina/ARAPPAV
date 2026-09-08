#!/usr/bin/env python
"""Plot perturber reward across self-play rounds.

    python scripts/plot_perturber_reward.py data/skill_rollouts/haiku_cold_10x8

Two series on one axis: the mean over all episodes, and the mean over
format-valid episodes only. They coincide exactly whenever every episode
parsed, so the **gap between them is the format-penalty drag** — a round where
the lines separate is a round the perturber lost reward to malformed output
rather than to the verifier.

Because that gap is the story, format validity is drawn beneath as a small
multiple. It is a rate and the reward is not, so it gets its own axis in its
own panel rather than a second scale on one chart.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from arappav.pipeline.plotting import (  # noqa: E402
    GRID, SERIES, SURFACE, TEXT_SECONDARY, load_rounds, plot_series, save,
    style_axes, titles,
)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else "data/skill_rollouts/haiku_cold_10x8")
    rows = load_rounds(root, "mean_perturber_reward")
    if not rows:
        sys.exit(f"no scored rounds under {root}")

    xs = [r["round"] for r in rows]
    all_ep = [r["mean_perturber_reward"] for r in rows]
    valid = [r["mean_perturber_reward_valid_only"] for r in rows]
    fmt = [r["format_valid_rate"] for r in rows]
    coincide = all(a == b for a, b in zip(all_ep, valid))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.5, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, ax2):
        style_axes(a)

    # Shade the gap: its area is exactly the reward lost to format penalties.
    ax.fill_between(xs, all_ep, valid, color=SERIES[1], alpha=0.13, zorder=1,
                    linewidth=0)
    # The series coincide wherever nothing failed to parse, so a solid line on
    # top would hide the other entirely. All-episodes goes down solid; valid-only
    # rides on top dashed, which reads as "both here" where they overlap and as
    # two distinct lines where they part.
    plot_series(ax, xs, all_ep, SERIES[0], "All episodes", zorder=3)
    plot_series(ax, xs, valid, SERIES[1], "Valid episodes only",
                linestyle=(0, (4, 3)), markersize=5, zorder=4)

    # Label the last point of each series, inside the axes so nothing clips.
    for ys, label, va in ((valid, "valid only", "bottom"),
                          (all_ep, "all episodes", "top")):
        ax.annotate(f"{label} {ys[-1]:+.2f}", xy=(xs[-1], ys[-1]),
                    xytext=(-6, 9 if va == "bottom" else -9),
                    textcoords="offset points", va=va, ha="right",
                    fontsize=9.5, color=TEXT_SECONDARY)

    ax.axhline(0, color=GRID, linewidth=1, zorder=2)
    ax.set_ylabel("Mean perturber reward", fontsize=10, color=TEXT_SECONDARY)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              fontsize=9.5, labelcolor=TEXT_SECONDARY, ncols=2,
              handlelength=2.4, columnspacing=2.2)
    # Headroom so the round-8 drop and its label do not touch the frame.
    lo, hi = min(all_ep + valid), max(all_ep + valid)
    pad = (hi - lo) * 0.18 or 0.1
    ax.set_ylim(lo - pad, hi + pad)

    plot_series(ax2, xs, fmt, SERIES[2], "Format-valid rate", markersize=5.5)
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_ylabel("Format\nvalid", fontsize=9, color=TEXT_SECONDARY)
    ax2.set_xlabel("Self-play round", fontsize=10, color=TEXT_SECONDARY)
    ax2.set_xticks(xs)

    note = ("the two series coincide — every episode parsed"
            if coincide else
            "where the series separate, the gap is reward lost to format penalties")
    titles(fig, ax, "Perturber reward across self-play rounds",
           f"{root.name} · cold start · {rows[0]['n_episodes']} episodes per round · "
           f"claude-haiku-4-5\n{note}")
    fig.subplots_adjust(top=0.80, right=0.965, left=0.115, bottom=0.10)

    out = save(fig, root / "perturber_reward_evolution.pdf")
    print(f"[plot] {len(rows)} round(s) → {out}")
    for r, a, b, f in zip(xs, all_ep, valid, fmt):
        flag = "" if a == b else "   <- format penalty drag"
        print(f"  round {r}: all={a:+.3f} valid_only={b:+.3f} fmt={f:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
