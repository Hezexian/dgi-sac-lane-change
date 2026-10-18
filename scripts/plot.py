"""Plot validation curves from training runs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [("success_rate", "Success rate (%)", 100), ("unsafe_rate", "Unsafe rate (%)", 100),
          ("mean_speed", "Mean speed (m/s)", 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories written by train.py")
    parser.add_argument("--output", default="curves.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for run in map(Path, args.runs):
        rows = list(csv.DictReader((run / "validation.csv").open(encoding="utf-8")))
        steps = [int(r["step"]) / 1000 for r in rows]
        for ax, (key, label, scale) in zip(axes, PANELS, strict=True):
            ax.plot(steps, [float(r[key]) * scale for r in rows], label=run.name)
            ax.set(xlabel="Environment steps (x1000)", ylabel=label)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
