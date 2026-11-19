"""Figures for the manuscript, drawn only from paper/numbers_v2.json.

Colours are the first two slots of a validated categorical palette (blue #2a78d6,
orange #eb6834: CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on white). Significance
is carried by marker fill as well as by position, so nothing depends on colour alone.

    python3 probe/figures_v2.py   -> paper/figures/{forest,compile}.{pdf,png}
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N = json.load(open(os.path.join(ROOT, "paper", "numbers_v2.json")))
OUT = os.path.join(ROOT, "paper", "figures")
SMALL = ["qwen2.5-coder:3b", "llama3.2:3b", "phi4-mini:latest", "qwen3:4b",
         "gemma3:4b", "qwen2.5-coder:7b", "qwen3:8b"]
LABEL = {"phi4-mini:latest": "phi4-mini"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0, "ytick.major.size": 0, "xtick.major.size": 2.5, "pdf.fonttype": 42, "savefig.dpi": 600,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def log_axis(ax, lo, hi, ticks):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    for t in ticks:
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    ax.axvline(1, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def point(ax, est, y, color, marker, filled, size=5.5):
    ax.plot([est["lo"], est["hi"]], [y, y], color=color, lw=1.4, solid_capstyle="round",
            zorder=2)
    ax.plot(est["rr"], y, marker, ms=size, mec=color, mew=1.3,
            mfc=color if filled else "white", zorder=3)


def forest():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 3.0),
                               gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.55})

    # (a) pooled effect of each prompt against the skeleton
    rows = [("Reminder", N["H3_generic_vs_scaffold_T"]["pooled"], BLUE),
            ("Mismatched rules", N["placebo_vs_scaffold_T"]["pooled"], BLUE),
            ("Rule identifiers", N["ids_vs_scaffold_T"]["pooled"], BLUE),
            ("Target rules", N["H1_teach_vs_scaffold"]["pooled"], BLUE)]
    for i, (name, est, col) in enumerate(rows):
        y = len(rows) - 1 - i
        point(a, est, y, col, "o", True, size=6)
        a.text(est["hi"] * 1.012, y, f"{est['rr']:.2f}", va="center", ha="left",
               fontsize=8.5, color=INK)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[0] for r in rows][::-1])
    a.set_ylim(-0.6, len(rows) - 0.4)
    log_axis(a, 0.62, 1.05, [0.7, 0.8, 0.9, 1.0])
    a.set_xlabel("rate ratio against skeleton alone")
    a.set_title("(a) Pooled over seven models", fontsize=9, loc="left", color=INK)

    # (b) per model: target rules against skeleton and against mismatched rules
    per = [(LABEL.get(m, m), N["H1_teach_vs_scaffold"]["per_model"][m],
            N["H2_teach_vs_placebo"]["per_model"][m]) for m in SMALL]
    r = N["robustness"]
    per.append(("Random effects", r["H1_hksj"], r["H2_hksj"]))
    for i, (name, h1, h2) in enumerate(per):
        y = len(per) - 1 - i
        last = name == "Random effects"
        for est, dy, col, mk in [(h1, 0.17, BLUE, "o"), (h2, -0.17, ORANGE, "s")]:
            sig = (est.get("p", 1) < 0.05) if last else (est.get("holm") or 1) < 0.05
            point(b, est, y + dy, col, "D" if last else mk, sig, size=5 if not last else 5.5)
    b.axhline(0.5, color=GRID, lw=0.8)
    b.set_yticks(range(len(per)))
    b.set_yticklabels([p[0] for p in per][::-1])
    b.set_ylim(-0.7, len(per) - 0.3)
    log_axis(b, 0.25, 1.6, [0.3, 0.5, 0.7, 1.0, 1.5])
    b.set_xlabel("rate ratio of target-rule violations")
    b.set_title("(b) Target rules, per model", fontsize=9, loc="left", color=INK)
    handles = [
        Line2D([], [], color=BLUE, marker="o", lw=1.4, ms=5, label="against skeleton alone"),
        Line2D([], [], color=ORANGE, marker="s", lw=1.4, ms=5, label="against mismatched rules"),
        Line2D([], [], color=INK2, marker="o", lw=0, ms=5, mfc="white",
               label="open marker: not significant")]
    b.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.95, -0.2), ncol=3,
             frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.0)
    save(fig, "forest")


def compile_panel():
    models = SMALL + ["qwen3.8-27b"]
    c = N["compilation"]
    clean = [100 * c[m]["clean_share"] for m in models]
    given = [100 * (c[m]["compile_given_clean"] or 0) for m in models]
    fig, ax = plt.subplots(figsize=(2.95, 2.45))
    w = 0.38
    x = list(range(len(models)))
    ax.bar([i - w / 2 for i in x], clean, w, color=BLUE, edgecolor="white", linewidth=0.8,
           label="no named rule violation\n(% of all pages)", zorder=2)
    ax.bar([i + w / 2 for i in x], given, w, color=ORANGE, edgecolor="white", linewidth=0.8,
           label="of those, compiled\n(% of blue bar)", zorder=2)
    for i in x:
        for xv, v in ((i - w / 2, clean[i]), (i + w / 2, given[i])):
            ax.text(xv, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(m, m) for m in models], rotation=40, ha="right",
                       fontsize=8)
    ax.set_ylabel("percent (skeleton-alone pages)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.2)
    save(fig, "compile")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    forest()
    compile_panel()
    print("wrote", sorted(os.listdir(OUT)))
