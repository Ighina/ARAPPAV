#!/usr/bin/env python
"""Plot verifier F1 against perturber reward (format-valid episodes only).

    python scripts/plot_f1_vs_perturber.py data/skill_rollouts/haiku_cold_10x8

**Read this chart knowing the two axes are not independent.** The perturber's
reward is defined as `r_P = (1 - unit_recall) + penalties`, so with no penalties
it is exactly `1 - recall` — verified identical to four decimals in every round
of this run. The verifier's F1 is `2PR/(P+R)`. Fixing r_P therefore fixes recall,
and F1 is then determined by precision alone.

That is why the grey iso-precision curves are drawn: each is F1 as a function of
r_P at a constant precision. A point's position *between* the curves reads off
that round's precision directly, and vertical scatter is precision variation —
not noise in an independent measurement. Left is better for the verifier
(high recall), right is better for the perturber.

F1 here is the macro average of per-episode F1, which is what the reward uses;
it equals `mean_verifier_reward` whenever no penalties fire. The F1 computed
from the round-mean precision and recall differs slightly and is not used.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from arappav.pipeline.plotting import (  # noqa: E402
    GRID, SERIES, SURFACE, TEXT_SECONDARY, load_rounds, save, style_axes, titles,
)

ISO_PRECISIONS = (0.80, 0.85, 0.90, 0.95, 1.00)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1
                else "data/skill_rollouts/haiku_cold_10x8")
    rows = load_rounds(root, "mean_verifier_reward")
    if not rows:
        sys.exit(f"no scored rounds under {root}")

    rp = [r["mean_perturber_reward_valid_only"] for r in rows]
    f1 = [r["mean_verifier_reward"] for r in rows]
    prec = [r["mean_verifier_precision"] for r in rows]
    rnd = [r["round"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)

    xmax = max(rp) * 1.25 + 0.02
    xs = [i * xmax / 200 for i in range(201)]

    # Iso-precision reference curves: F1 at constant precision, as recall (and
    # hence r_P) varies. Recessive grey so the data reads first.
    for p in ISO_PRECISIONS:
        ys = [(2 * p * (1 - x) / (p + (1 - x))) if (p + (1 - x)) else 0 for x in xs]
        ax.plot(xs, ys, color="#b9b8b2", linewidth=1, linestyle=(0, (2, 2)),
                zorder=1)
        # Label near the right end but inside the frame, nudged off the curve.
        xi = int(len(xs) * 0.90)
        ax.annotate(f"P={p:.2f}", xy=(xs[xi], ys[xi]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="#8a8983", zorder=1)

    # Faint trajectory line: reading order is round 0 -> last, not left-to-right.
    ax.plot(rp, f1, color=SERIES[0], linewidth=1, alpha=0.35, zorder=2)
    ax.scatter(rp, f1, s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
               zorder=4)
    # Labels sit beside the marker, not inside it, and are nudged apart when two
    # rounds land on nearly the same coordinates (identical r_P is common, since
    # r_P moves in steps of 1/(total error units)).
    # Labels sit beside the marker, not inside it. Several rounds share an
    # identical r_P (it moves in steps of 1/total-error-units), so a label is
    # tried at a few positions around its point and takes the first that does
    # not land on one already placed. Offsets stay small, so a label is never
    # far from the marker it names.
    span_x, span_y = xmax or 1.0, (max(f1) - min(f1)) or 1.0
    ring = ((9, 5), (9, -11), (-9, 5), (-9, -11), (0, 11), (0, -15))
    taken: list[tuple[float, float]] = []

    def collides(x, y, dx, dy):
        px, py = x + dx * span_x / 640, y + dy * span_y / 380
        return any(abs(px - qx) / span_x < 0.030 and abs(py - qy) / span_y < 0.055
                   for qx, qy in taken)

    for x, y, r in zip(rp, f1, rnd):
        dx, dy = next((o for o in ring if not collides(x, y, *o)), ring[0])
        ax.annotate(str(r), xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color=TEXT_SECONDARY, zorder=5,
                    ha="left" if dx > 0 else "right" if dx < 0 else "center",
                    va="bottom" if dy >= 0 else "top")
        taken.append((x + dx * span_x / 640, y + dy * span_y / 380))

    ax.set_xlabel("Perturber reward, format-valid episodes  "
                  "(= 1 − recall;  → perturber wins)",
                  fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Verifier F1  (↑ verifier wins)", fontsize=10,
                  color=TEXT_SECONDARY)
    ax.set_xlim(0, xmax)
    ax.set_ylim(min(f1) - 0.08, 1.02)

    titles(fig, ax, "Verifier F1 against perturber reward",
           f"{root.name} · cold start · {rows[0]['n_episodes']} episodes per round · "
           f"claude-haiku-4-5\n"
           "labels are round numbers · grey curves are constant precision;\n"
           "a point's height between them is that round's precision")
    fig.subplots_adjust(top=0.76, right=0.965, left=0.10, bottom=0.14)

    out = save(fig, root / "f1_vs_perturber_reward.pdf")
    print(f"[plot] {len(rows)} round(s) → {out}")
    print(f"{'rnd':>4}{'r_P(valid)':>12}{'F1':>9}{'precision':>11}")
    for r, x, y, p in zip(rnd, rp, f1, prec):
        print(f"{r:>4}{x:>12.4f}{y:>9.4f}{p:>11.3f}")

    n = len(rp)
    if n > 2:
        mx, my = sum(rp) / n, sum(f1) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(rp, f1))
        vx = sum((a - mx) ** 2 for a in rp) ** 0.5
        vy = sum((b - my) ** 2 for b in f1) ** 0.5
        if vx and vy:
            print(f"\nPearson r(r_P, F1) = {cov / (vx * vy):+.3f}  "
                  f"(expected strongly negative: r_P is 1 − recall by construction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
