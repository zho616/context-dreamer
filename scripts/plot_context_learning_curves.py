"""Plot paired learning curves from logged evaluations, without synthetic bands."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np


PANELS = (
    ("episode/eval_score", "Episode Reward", "Reward"),
    ("episode/eval_success", "Success Rate", "Success Rate"),
    ("episode/eval_crash", "Collision Rate", "Collision Rate"),
    ("episode/eval_route_completion", "Route Completion Rate", "Route Completion"),
)


def ema(values, weight):
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = weight * result[i - 1] + (1 - weight) * values[i]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 <= args.weight < 1:
        parser.error("EMA weight must be in [0, 1)")
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CDD2DD", "axes.linewidth": 0.7,
        "axes.facecolor": "#F3F4F8", "figure.facecolor": "white",
        "grid.color": "white", "grid.linewidth": 0.9,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    series = []
    source_data = []
    fingerprints = {}
    specs = (
        ("R2-Dreamer", args.baseline, "#648840", "--"),
        ("R2-Dreamer + Context", args.context, "#8854AC", "-"),
    )
    for label, path, color, style in specs:
        raw = path.read_bytes()
        fingerprints[label] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        rows = [r for r in rows if all(key in r for key, _, _ in PANELS)]
        rows.sort(key=lambda r: r["step"])
        steps = np.array([r["step"] for r in rows], dtype=float)
        if len(rows) < 2 or not np.all(np.diff(steps) > 0):
            raise ValueError("Need at least two strictly increasing evaluation steps")
        if not all(r["episode/eval_episodes"] == 20 for r in rows):
            raise ValueError("This figure expects 20 evaluation episodes per checkpoint")
        values = {}
        for key, _, _ in PANELS:
            y = np.array([r[key] for r in rows], dtype=float)
            if not np.isfinite(y).all():
                raise ValueError(f"Nonfinite observations: {label}, {key}")
            if key != "episode/eval_score" and not ((0 <= y) & (y <= 1)).all():
                raise ValueError(f"Rate outside [0,1]: {label}, {key}")
            values[key] = (y, ema(y, args.weight))
        series.append((label, steps, values, color, style))
        for i, row in enumerate(rows):
            record = {"method": label, "step": row["step"], "eval_episodes": 20}
            for key, _, _ in PANELS:
                record[key] = values[key][0][i]
                record[key + "_ema"] = values[key][1][i]
            source_data.append(record)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.19,
                        wspace=0.27, hspace=0.40)
    xmax = max(item[1][-1] for item in series)
    for index, (ax, (key, title, ylabel)) in enumerate(zip(axes.flat, PANELS)):
        for _, steps, values, color, style in series:
            raw, smooth = values[key]
            ax.plot(steps, raw, color=color, alpha=0.19, lw=0.8,
                    linestyle=style, zorder=2)
            ax.plot(steps, smooth, color=color, lw=1.9,
                    linestyle=style, zorder=3)
        ax.set(title=title, xlabel="Environment Steps", ylabel=ylabel, xlim=(0, xmax))
        ax.set_xticks(np.arange(0, 500001, 100000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{x / 1000:.0f}k"))
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.16, 1.08, chr(ord("a") + index), transform=ax.transAxes,
                weight="bold", fontsize=10, va="top")
        if key != "episode/eval_score":
            ax.set_ylim(-0.025, 1.025)
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        else:
            raw_values = np.concatenate([item[2][key][0] for item in series])
            lo = 20 * np.floor(raw_values.min() / 20)
            hi = 20 * np.ceil(raw_values.max() / 20)
            ax.set_ylim(lo, hi + 5)

    handles = [Line2D([], [], color=color, ls=style, lw=2, label=label)
               for label, _, color, style in specs]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.072),
               ncol=2, frameon=False, handlelength=3.2, columnspacing=2.8, fontsize=9)
    fig.text(0.54, 0.045,
             f"EMA weight = {args.weight:g}; faint lines = raw evaluations",
             ha="center", fontsize=7, color="#555B66")
    fig.text(0.54, 0.019,
             "One training seed per method; 20 episodes per evaluation; no confidence bands",
             ha="center", fontsize=6.6, color="#555B66")

    stem = args.output / "r2_context_learning_curves"
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=350, facecolor="white")
    plt.close(fig)
    with (args.output / "source_data.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_data[0]))
        writer.writeheader()
        writer.writerows(source_data)
    metadata = dict(
        smoothing={"method": "EMA", "weight": args.weight,
                   "formula": "s[0]=y[0]; s[t]=weight*s[t-1]+(1-weight)*y[t]"},
        methods=[x[0] for x in series],
        evaluation_counts={x[0]: len(x[1]) for x in series},
        source_sha256=fingerprints, training_seeds=1, evaluation_episodes=20,
        uncertainty="Not estimated; faint lines are raw observations, not confidence intervals",
        interpolation="Straight line segments between logged points; no spline fitting",
        figure_inches=[7.2, 5.45], raster_dpi=350,
    )
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
