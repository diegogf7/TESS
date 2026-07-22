"""One-gated test evaluation for matched group-JEPA and scratch fine-tunes."""

from __future__ import annotations

import json
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.group_level_jepa import build_groupmean_jepa
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame


GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
ENCODER_VIEW = os.environ.get("ENCODER_VIEW", "online")
SEEDS = tuple(int(value) for value in os.environ.get("SEEDS", "0,1,2").split(","))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet",
)
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic")
)
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa")
)
ART_DIR = os.environ.get(
    "GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "group_level")
)
CKPT_DIR = os.environ.get(
    "GROUP_CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/group_level",
)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = build_groupmean_jepa().context_encoder
        self.head = nn.Linear(16 * 16, 16)

    def forward(self, flux, mask):
        return self.head(self.encoder(flux.unsqueeze(-1), mask).flatten(1))


def selected_run(arm, seed):
    """Select backbone LR on validation only, before loading any test curve."""
    pattern = os.path.join(
        ART_DIR,
        f"result_ft_{arm}_k{GROUP_SIZE}_{ENCODER_VIEW}_camccd_s{seed}_lr*.json",
    )
    candidates = []
    for path in glob.glob(pattern):
        with open(path) as handle:
            candidates.append(json.load(handle))
    if not candidates:
        raise RuntimeError(f"no fine-tune validation results: {pattern}")
    return max(candidates, key=lambda row: row["best_val_bacc16"])


def main():
    frame = pd.read_parquet(S14_DATA)
    frame = frame[frame["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, _, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, frame, train_tics)
    test_frame = frame[frame["TIC"].astype(str).isin(test_tics)].reset_index(drop=True)
    flux, mask = grid_frame(test_frame, "shared", time_range)
    labels = np.asarray(
        [chip_index(camera, ccd) for camera, ccd in zip(test_frame["camera"], test_frame["ccd"])],
        dtype=np.int64,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(flux), torch.from_numpy(mask), torch.from_numpy(labels)
        ),
        batch_size=256,
        shuffle=False,
    )

    rows = []
    for arm in ("random", "group"):
        for seed in SEEDS:
            selection = selected_run(arm, seed)
            path = selection["checkpoint"]
            if not os.path.exists(path):
                raise RuntimeError(f"missing selected fine-tune checkpoint: {path}")
            model = Classifier().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            predicted, actual = [], []
            with torch.no_grad():
                for batch_flux, batch_mask, target in loader:
                    logits = model(batch_flux.to(DEVICE), batch_mask.to(DEVICE))
                    predicted.append(logits.argmax(dim=1).cpu().numpy())
                    actual.append(target.numpy())
            score = float(
                balanced_accuracy_score(np.concatenate(actual), np.concatenate(predicted))
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "backbone_lr": selection["backbone_lr"],
                    "validation_bacc16": selection["best_val_bacc16"],
                    "test_bacc16": score,
                }
            )
            print(
                f"{arm} seed {seed}: selected lr={selection['backbone_lr']:g} "
                f"val={selection['best_val_bacc16']:.4f} test={score:.4f}"
            )

    summary = {}
    for arm in ("random", "group"):
        values = np.asarray([row["test_bacc16"] for row in rows if row["arm"] == arm])
        summary[arm] = {"mean": float(values.mean()), "std": float(values.std())}
    summary["group_minus_random"] = (
        summary["group"]["mean"] - summary["random"]["mean"]
    )
    output = {"rows": rows, "summary": summary, "test_tics": len(test_frame)}
    os.makedirs(ART_DIR, exist_ok=True)
    out_path = os.path.join(ART_DIR, "group_level_test.json")
    with open(out_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
