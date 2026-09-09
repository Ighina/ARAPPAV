#!/usr/bin/env python
"""Plot verifier F1 and perturber reward across self-play rounds.

    python scripts/plot_f1_vs_perturber.py data/skill_rollouts/haiku_cold_10x8

Both series are rewards on the same scale, so they share one y-axis. Verifier
F1 is `mean_verifier_reward` (the macro mean of per-episode F1); perturber
reward is `mean_perturber_reward_valid_only`, i.e. format-valid episodes only.
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
    rows = load_rounds(root, "mean_verifier_reward")
    if not rows:
        sys.exit(f"no scored rounds under {root}")

    xs = [r["round"] for r in rows]
    f1 = [r["mean_verifier_reward"] for r in rows]
    rp = [r["mean_perturber_reward_valid_only"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    plot_series(ax, xs, f1, SERIES[0], "Verifier F1")
    plot_series(ax, xs, rp, SERIES[1], "Perturber reward (valid only)",
                linestyle=(0, (4, 3)), markersize=5, zorder=4)

    ax.set_xlabel("Self-play round", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Reward", fontsize=10, color=TEXT_SECONDARY)
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01),
              fontsize=9.5, labelcolor=TEXT_SECONDARY, handlelength=2.4,
              ncols=2, columnspacing=2.2)

    titles(fig, ax, "Verifier F1 and perturber reward across self-play rounds",
           f"{root.name} · cold start · {rows[0]['n_episodes']} episodes per round · "
           f"claude-haiku-4-5")
    fig.subplots_adjust(top=0.74, right=0.965, left=0.095, bottom=0.13)

    out = save(fig, root / "f1_vs_perturber_reward.pdf")
    print(f"[plot] {len(rows)} round(s) → {out}")
    for r, a, b in zip(xs, f1, rp):
        print(f"  round {r}: verifier_F1={a:.3f}  perturber_reward={b:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
