"""The four-region plot -- x = instrument percentile, y = physics percentile -- with
MOMENT shown as colour.

MOMENT produces a single score, so it cannot supply an axis here; it is the colour.
Points ringed in red are MOMENT anomalies (percentile >= 0.95).

    ~/tess-venv/bin/python -m disentangle_attempt.plot_moment_regions \
      --scores .../moment_baseline/moment_anomaly_scores.csv
"""

import argparse
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD = 0.95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    table = pd.read_csv(args.scores)
    if args.split != "all":
        table = table[table["split"] == args.split]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)),
                                   "moment_regions.png")

    colours = {"physics": "tab:red", "instrument": "tab:blue", "both": "tab:purple",
               "typical": "0.7"}
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for name, colour in colours.items():
        pick = table["classification"] == name
        ax.scatter(table.loc[pick, "instrument_percentile"],
                   table.loc[pick, "physics_percentile"], s=8, color=colour,
                   label=f"{name} ({int(pick.sum())})", linewidths=0)
    high = table["moment_percentile"] >= THRESHOLD
    ax.scatter(table.loc[high, "instrument_percentile"],
               table.loc[high, "physics_percentile"], s=55, facecolors="none",
               edgecolors="black", linewidths=0.9,
               label=f"MOMENT anomaly ({int(high.sum())})")
    ax.axvline(THRESHOLD, color="0.3", lw=0.8, ls="--")
    ax.axhline(THRESHOLD, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("instrument anomaly percentile")
    ax.set_ylabel("physics anomaly percentile")
    ax.set_title("four interpretation regions, with MOMENT anomalies circled",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
