from __future__ import annotations
"""Drill-down on the PhyTS A/B: what happens to held-out APERIODIC and ECLIPSE
curves under raw vs instrument-cleaned physics classification? Reuses the exact
data/split/models/probe from eval_phyts_instrument_ab (default = the weak-encoder
config that produced the +0.018 "cleaning helps" result). No retraining.

  python -m src.instrument_v2.inspect_phyts_aperiodic_eclipse
"""

import json
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from src.data.data import CLASSES, CLASS_TO_IDX
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.eval_phyts_instrument_ab import (  # exact reuse -- no new pipeline
    DATA_PATH, PHYS_CKPT, INST_CKPT, DECODER_CKPT, GRID_RANGE, DEVICE, GRID,
    physics_grid, instrument_cleaned_curve, encode_physics, matched_split, _freeze,
)

OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join("artifacts", "instrument_v2", "phyts_aperiodic_eclipse"))
FOCUS = ("APERIODIC", "ECLIPSE")


def predict_arm(latents, y, train_idx, test_idx):
    """StandardScaler(train) + KNN(20) -> predicted class idx for each test row
    (same probe as eval_phyts_instrument_ab.classify, but returns predictions)."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(latents[train_idx])
    x_te = scaler.transform(latents[test_idx])
    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(x_tr, y[train_idx])
    return knn.predict(x_te)


def pick_first_by_tic(test_tics, test_y, cls_idx):
    """Position (within the test arrays) of the first held-out example of cls_idx
    after sorting by TIC -- deterministic."""
    where = np.flatnonzero(test_y == cls_idx)
    if len(where) == 0:
        return None
    return int(where[np.argsort(test_tics[where], kind="stable")[0]])


def _plot_gapped(ax, time, flux, gap=0.5, **kw):
    """Line broken at time gaps > `gap` days so missing cadences stay gaps."""
    time = np.asarray(time, dtype=float); flux = np.asarray(flux, dtype=float)
    order = np.argsort(time); time, flux = time[order], flux[order]
    t, f = [time[0]], [flux[0]]
    for k in range(1, len(time)):
        if time[k] - time[k - 1] > gap:
            t.append(np.nan); f.append(np.nan)
        t.append(time[k]); f.append(flux[k])
    ax.plot(t, f, **kw)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GRID_RANGE) as fh:
        gr = json.load(fh)
    t0, t1 = float(gr["t0"]), float(gr["t1"])

    df = pd.read_parquet(DATA_PATH)
    df = df[df["sector"] == 14].reset_index(drop=True)
    n = len(df)
    tics = df["TIC"].to_numpy().astype(str)
    sectors = df["sector"].to_numpy()
    y = np.array([CLASS_TO_IDX[l] for l in df["label"]], dtype=np.int64)

    # --- frozen models (identical to the A/B) --------------------------------
    phys = _freeze(build_latent_jepa().to(DEVICE))
    phys.load_state_dict(torch.load(PHYS_CKPT, map_location=DEVICE), strict=False)
    _freeze(phys)
    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                      readout="mean", predictor_type="mlp").to(DEVICE)
    inst.load_state_dict(torch.load(INST_CKPT, map_location=DEVICE))
    _freeze(inst)
    decoder = _freeze(build_decoder(1024).to(DEVICE))
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
    _freeze(decoder)
    hashes = {"phys": state_hash(phys), "inst_teacher": state_hash(inst.teacher),
              "inst_student": state_hash(inst.student),
              "inst_predictor": state_hash(inst.predictor), "decoder": state_hash(decoder)}

    # --- both arms on identical rows -----------------------------------------
    A_X = np.zeros((n, GRID), np.float32); A_M = np.zeros((n, GRID), np.float32)
    B_X = np.zeros((n, GRID), np.float32); B_M = np.zeros((n, GRID), np.float32)
    for i in range(n):
        t = df["time"].iloc[i]; f = df["flux"].iloc[i]
        A_X[i], A_M[i] = physics_grid(t, f)
        ct, cf = instrument_cleaned_curve(t, f, inst, decoder, t0, t1)
        B_X[i], B_M[i] = physics_grid(ct, cf)
        if i % 400 == 0:
            print(f"  building arms {i}/{n}", flush=True)
    lat_a = encode_physics(phys, A_X, A_M)
    lat_b = encode_physics(phys, B_X, B_M)

    assert state_hash(phys) == hashes["phys"], "physics changed"
    assert state_hash(inst.teacher) == hashes["inst_teacher"], "instrument changed"
    assert state_hash(decoder) == hashes["decoder"], "decoder changed"

    # --- one shared split; predictions for both arms on the SAME held-out rows
    train_idx, test_idx = matched_split(tics, y)
    raw_pred = predict_arm(lat_a, y, train_idx, test_idx)
    clean_pred = predict_arm(lat_b, y, train_idx, test_idx)
    test_tics, test_y, test_sec = tics[test_idx], y[test_idx], sectors[test_idx]

    # --- per-curve table for held-out APERIODIC + ECLIPSE --------------------
    focus_idx = {c: CLASS_TO_IDX[c] for c in FOCUS}
    keep = np.isin(test_y, list(focus_idx.values()))
    rows = []
    for j in np.flatnonzero(keep):
        rows.append({
            "TIC": test_tics[j], "sector": int(test_sec[j]),
            "true_label": CLASSES[test_y[j]],
            "raw_prediction": CLASSES[raw_pred[j]],
            "cleaned_prediction": CLASSES[clean_pred[j]],
            "raw_correct": int(raw_pred[j] == test_y[j]),
            "cleaned_correct": int(clean_pred[j] == test_y[j])})
    per_curve = pd.DataFrame(rows)
    per_curve.to_csv(os.path.join(OUT_DIR, "per_curve_predictions.csv"), index=False)

    # --- predicted-class counts per true class per arm -----------------------
    count_rows = []
    for c in FOCUS:
        m = test_y == focus_idx[c]
        for arm, pred in (("raw", raw_pred), ("cleaned", clean_pred)):
            uni, cnt = np.unique(pred[m], return_counts=True)
            for u, k in zip(uni, cnt):
                count_rows.append({"true_label": c, "arm": arm,
                                   "predicted_label": CLASSES[u], "count": int(k)})
    counts = pd.DataFrame(count_rows)
    counts.to_csv(os.path.join(OUT_DIR, "prediction_counts.csv"), index=False)

    def confusion(true_c, pred_c):
        m = test_y == CLASS_TO_IDX[true_c]
        return {"raw": int((raw_pred[m] == CLASS_TO_IDX[pred_c]).sum()),
                "cleaned": int((clean_pred[m] == CLASS_TO_IDX[pred_c]).sum())}

    # --- 2x2 figure: [AP raw | AP cleaned] / [EC raw | EC cleaned] -----------
    picks = {c: pick_first_by_tic(test_tics, test_y, focus_idx[c]) for c in FOCUS}
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    selected = {}
    for r, c in enumerate(FOCUS):
        j = picks[c]
        gi = test_idx[j]
        tic = test_tics[j]; rp = CLASSES[raw_pred[j]]; cp = CLASSES[clean_pred[j]]
        selected[c] = {"TIC": tic, "sector": int(test_sec[j]),
                       "raw_prediction": rp, "cleaned_prediction": cp}
        time = np.asarray(df["time"].iloc[gi], float); flux = np.asarray(df["flux"].iloc[gi], float)
        g = np.isfinite(time) & np.isfinite(flux)
        _plot_gapped(axes[r, 0], time[g], flux[g], lw=0.6, color="0.3")
        ct, cf = instrument_cleaned_curve(df["time"].iloc[gi], df["flux"].iloc[gi],
                                          inst, decoder, t0, t1)
        _plot_gapped(axes[r, 1], ct, cf, lw=0.6, color="tab:green")
        for k, tag in ((0, "raw"), (1, "cleaned")):
            axes[r, k].set_title(f"{c} ({tag})  TIC {tic}  raw->{rp}  clean->{cp}", fontsize=9)
    fig.suptitle("PhyTS held-out APERIODIC / ECLIPSE: raw vs instrument-cleaned")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, "examples.png"), dpi=130)
    plt.close(fig)

    summary = {
        "phys_ckpt": PHYS_CKPT, "inst_ckpt": INST_CKPT, "decoder_ckpt": DECODER_CKPT,
        "n_test": int(len(test_idx)),
        "n_aperiodic_heldout": int((test_y == focus_idx["APERIODIC"]).sum()),
        "n_eclipse_heldout": int((test_y == focus_idx["ECLIPSE"]).sum()),
        "aperiodic_predicted_eclipse": confusion("APERIODIC", "ECLIPSE"),
        "eclipse_predicted_aperiodic": confusion("ECLIPSE", "APERIODIC"),
        "selected_examples": selected,
        "model_hashes_unchanged": True,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ("aperiodic_predicted_eclipse", "eclipse_predicted_aperiodic",
                       "n_aperiodic_heldout", "n_eclipse_heldout")}, indent=2), flush=True)
    print(f"wrote {OUT_DIR}/{{examples.png, per_curve_predictions.csv, "
          f"prediction_counts.csv, summary.json}}", flush=True)


if __name__ == "__main__":
    main()
