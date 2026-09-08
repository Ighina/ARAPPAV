"""Shared chart style for pipeline plots.

One place for the palette and chrome so every figure in the project reads as
one system. Colours are the validated categorical slots (CVD Delta E 24.7
between slots 1 and 2, well above the 8 threshold), and every chart built on
this module pairs colour with a direct label so identity never rests on hue
alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, light mode.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load_rounds(root: Path, require: str) -> list[dict]:
    """Round summaries in round order, skipping rounds without `require`."""
    rows = []
    for f in sorted(root.glob("round_*/round_summary.json"),
                    key=lambda p: int(p.parent.name.split("_")[1])):
        d = json.loads(f.read_text())
        if d["metrics"].get(require) is None:
            continue
        rows.append({"round": d["round"], **d["metrics"],
                     "n_episodes": len(d.get("episodes", []))})
    return rows


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def plot_series(ax, xs, ys, colour, label, *, marker="o", linestyle="-",
                markersize=6.5, zorder=3) -> None:
    """A 2px line with surface-ringed markers so overlaps stay legible."""
    ax.plot(xs, ys, color=colour, linewidth=2, zorder=zorder, label=label,
            marker=marker, markersize=markersize, linestyle=linestyle,
            markerfacecolor=colour, markeredgecolor=SURFACE, markeredgewidth=1.5)


def titles(fig, ax, title: str, subtitle: str, *, x=0.09, y=0.945,
           gap=0.055) -> None:
    """Title, then subtitle beneath it. Both are figure-level so they align
    with the left edge of the plot area rather than the axes box."""
    fig.text(x, y, title, fontsize=13, color=TEXT_PRIMARY, ha="left", va="top",
             fontweight="medium")
    fig.text(x, y - gap, subtitle, fontsize=9, color=TEXT_SECONDARY,
             ha="left", va="top", linespacing=1.5)


def save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", facecolor=SURFACE)
    fig.savefig(out.with_suffix(".png"), dpi=160, facecolor=SURFACE)
    return out
