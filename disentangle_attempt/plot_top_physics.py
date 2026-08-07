"""The N highest physics-anomaly test stars, raw versus cleaned.

Each candidate is cleaned against a quiet reference built on ITS OWN chip: a reference
group must share the target's absolute cadence grid and detector neighbourhood, so one
global reference cannot clean a multi-chip run. References are built on demand for
only the chips the candidates land on.

    python -m disentangle_attempt.plot_top_physics \
      --checkpoint .../multichip_5sectors_v1/best.pt \
      --scores .../anomaly_analysis_20k_pca90/anomaly_scores.csv \
      --parquet ... --top 10
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disentangle_attempt.dataset import (CrossSectorPatch, infer_require_cross_sector,
                                         target_from_checkpoint)
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import build_model
from disentangle_attempt.reference_context import build_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--tics", default=None,
                        help="comma-separated TICs to plot instead of the top-N")
    parser.add_argument("--max-instrument", type=float, default=None,
                        help="keep only candidates below this instrument percentile")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    args = parser.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)),
                                   "selected_physics_cleaned.png" if args.tics
                                   else f"top{args.top}_physics_cleaned.png")
    table = pd.read_csv(args.scores)
    if args.split != "all":
        table = table[table["split"] == args.split]
    if args.tics:
        wanted = [t.strip() for t in args.tics.split(",") if t.strip()]
        table = table[table["TIC"].astype(str).isin(wanted)]
        top = table.sort_values("physics_nll", ascending=False).reset_index(drop=True)
        missing = set(wanted) - set(top["TIC"].astype(str))
        if missing:
            print(f"  not found in the scores table: {sorted(missing)}", flush=True)
        print(f"plotting {len(top)} requested TICs", flush=True)
    else:
        if args.max_instrument is not None:
            table = table[table["instrument_percentile"] < args.max_instrument]
        top = table.nlargest(args.top, "physics_nll").reset_index(drop=True)
        print(f"top {len(top)} physics anomalies in the {args.split} split"
              + (f" with instrument percentile < {args.max_instrument}"
                 if args.max_instrument is not None else ""), flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    sector, camera, ccd = target_from_checkpoint(state, config)
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(config, args.require_cross_sector),
        verbose=False)

    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    masks = complementary_masks(config["curve_length"], n_masks=4)

    rows = [int(r) for r in top["row"]]
    chips = {int(r): (int(patch.sector[r]), int(patch.camera[r]), int(patch.ccd[r]))
             for r in rows}
    references = {}
    for chip in sorted(set(chips.values())):
        try:
            references[chip] = build_reference_context(patch, "train",
                                                       config["n_peers"], chip=chip)
        except RuntimeError as error:
            print(f"  chip {chip}: no quiet reference ({error})", flush=True)

    cleaned = {}
    for chip, reference in references.items():
        members = [r for r in rows if chips[r] == chip]
        peers = np.stack([patch.peers_for_row(r, args.split)[0] for r in members])
        quiet_flux = torch.from_numpy(patch.X[reference["rows"]]).unsqueeze(0)
        quiet_mask = torch.from_numpy(patch.M[reference["rows"]]).unsqueeze(0)
        actual, ref, _, _, _ = dual_context_prediction(
            model, torch.from_numpy(patch.X[members]), torch.from_numpy(patch.M[members]),
            torch.from_numpy(patch.X[peers]), torch.from_numpy(patch.M[peers]),
            quiet_flux.expand(len(members), -1, -1),
            quiet_mask.expand(len(members), -1, -1), masks, device)
        correction = (actual - ref).numpy()
        for k, row in enumerate(members):
            cleaned[row] = patch.X[row] - correction[k]

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 1.9 * len(rows)))
    axes = np.atleast_1d(axes)
    for ax, (_, record) in zip(axes, top.iterrows()):
        row = int(record["row"])
        valid = patch.M[row]
        x = np.arange(len(valid))
        ax.scatter(x[valid], patch.X[row][valid], s=2, color="0.6", linewidths=0,
                   label="raw")
        if row in cleaned:
            ax.scatter(x[valid], cleaned[row][valid], s=2, color="tab:blue", linewidths=0,
                       label="cleaned")
        ax.set_ylabel(f"TIC {record['TIC']}\n{record['chip']}\n"
                      f"phy {record['physics_percentile']:.3f}", fontsize=6)
    axes[0].legend(fontsize=8, markerscale=4, loc="upper right")
    axes[-1].set_xlabel("cadence index (gaps = removed cadences)")
    fig.suptitle(f"{len(rows)} physics candidates ({args.split} split), "
                 "cleaned against each star's own chip reference", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(top[["TIC", "chip", "physics_percentile", "instrument_percentile",
               "physics_nll"]].to_string(index=False))


if __name__ == "__main__":
    main()
