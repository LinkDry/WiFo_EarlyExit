import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = Path(__file__).resolve().parent
R = W / "results"
OUT = W / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "pdf.fonttype": 42,
    "figure.dpi": 160,
})

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#7A828C"


def rows(name):
    with (R / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def costs_d17():
    result = {}
    for row in rows("11_pareto.csv"):
        if row["dataset"] == "D17" and row["task"] == "temporal" and row["kind"] == "h1" and row["backbone"] in {"base", "small"}:
            key = (row["backbone"], int(row["enc_depth"]), int(row["dec_depth"]))
            result[key] = float(row["gflops"])
    return result


def main():
    curves = rows("10_curves.csv")
    selections = rows("10_selection.csv")
    costs = costs_d17()
    selected = {
        (row["backbone"], row["task"], int(row["selected_enc_depth"]), int(row["selected_dec_depth"]))
        for row in selections
    }
    route_depths = {
        "encoder": [(6, 4), (5, 4), (4, 4), (3, 4)],
        "decoder": [(6, 4), (6, 3), (6, 2)],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), sharey=True)

    for panel, (axis, backbone) in enumerate(zip(axes, ("base", "small"))):
        full_cost = costs[(backbone, 6, 4)]
        midpoints = {}
        for route, depths in route_depths.items():
            color = BLUE if route == "encoder" else ORANGE
            for task, linestyle in (("temporal", "-"), ("fre", "--")):
                points = []
                for enc, dec in depths:
                    group = [
                        row for row in curves
                        if row["backbone"] == backbone and row["task"] == task
                        and int(row["enc_depth"]) == enc and int(row["dec_depth"]) == dec
                    ]
                    assert len(group) == 3, (backbone, task, enc, dec, len(group))
                    values = np.array([float(row["depth_cost_db"]) for row in group])
                    x = 100 * (1 - costs[(backbone, enc, dec)] / full_cost)
                    y, sd = values.mean(), values.std()
                    points.append((x, y, sd, enc, dec))
                    midpoints[(route, task, enc, dec)] = (x, y)
                points.sort()
                axis.errorbar(
                    [p[0] for p in points], [p[1] for p in points], yerr=[p[2] for p in points],
                    color=color, linestyle=linestyle, marker="o" if route == "encoder" else "^",
                    markersize=4.2, markerfacecolor="white" if task == "fre" else color,
                    markeredgecolor=color, markeredgewidth=0.9,
                    ecolor=color, elinewidth=0.65, capsize=2,
                )
                for x, y, _, enc, dec in points:
                    if (backbone, task, enc, dec) in selected:
                        axis.scatter(x, y, s=145, facecolors="none", edgecolors=GREEN, linewidths=1.1)

        for route, depths in route_depths.items():
            for enc, dec in depths:
                if (enc, dec) == (6, 4):
                    continue
                x = midpoints[(route, "temporal", enc, dec)][0]
                y = np.mean([midpoints[(route, task, enc, dec)][1] for task in ("temporal", "fre")])
                axis.annotate(f"({enc},{dec})", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.2, color=color)
        axis.annotate("Full", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=6.4, color=GRAY)
        for budget in (0.25, 0.5, 1.0):
            axis.axhline(budget, color="#A8AFB6", linewidth=0.6, linestyle=(0, (2, 3)), zorder=0)
            axis.annotate(f"ε={budget:g}", (29.5, budget), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=6.2, color=GRAY)
        axis.set_xlim(-2.2, 32)
        axis.set_ylim(-0.08, 3.02)
        axis.set_xticks([0, 10, 20, 30])
        axis.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        axis.set_xlabel("GFLOPs reduction vs. full H1 (%)")
        axis.set_title(f"({chr(97 + panel)}) {backbone.capitalize()}", pad=5)
        axis.grid(axis="y", alpha=0.12, linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation NMSE increase (dB)")

    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", label="Encoder", markersize=4),
        plt.Line2D([], [], color=ORANGE, marker="^", label="Decoder", markersize=4),
        plt.Line2D([], [], color=GRAY, marker="o", label="Temporal", markersize=4),
        plt.Line2D([], [], color=GRAY, marker="D", markerfacecolor="white", label="Frequency", markersize=4),
        plt.Line2D([], [], color=GREEN, marker="o", markerfacecolor="none", label="Selected", markersize=6),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), columnspacing=1.3, handletextpad=0.4)
    fig.text(0.5, 0.015, "Points: 3-seed mean; error bars: ±1 SD; dashed guides: accuracy budgets ε", ha="center", fontsize=6.5, color=GRAY)
    fig.tight_layout(rect=(0, 0.055, 1, 0.88), w_pad=1.4)

    for suffix in ("pdf", "svg"):
        fig.savefig(OUT / f"cl_fig2_depth_cost.{suffix}", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "cl_fig2_depth_cost.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT / 'cl_fig2_depth_cost.pdf'}")


if __name__ == "__main__":
    main()
