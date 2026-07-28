from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import(
    Sector14GroupStatDataset,
    ensure_area_column,
    group_statistics


)

from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.regional_cbv import build_or_load_area_bases, ridge_reconstruct
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit


#these variables are all from CLAUDE bc I kept messing it up

SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))            # stars per group target
CBV_RANK = int(os.environ.get("CBV_RANK", "8"))                # CBV components = weight-decoder output dim
MIN_VALID_STARS = int(os.environ.get("MIN_VALID_STARS", "16"))  # min observed stars for a valid cadence
assert 1 <= MIN_VALID_STARS <= GROUP_SIZE, (MIN_VALID_STARS, GROUP_SIZE)
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
GROUP_ART_DIR = os.environ.get(
    "GROUP_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_qclean_v1"))
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_qclean_v1")
STAGE_B_CKPT = os.environ.get(
    "STAGE_B_CKPT", os.path.join(
        CKPT_DIR, f"group_cbv_mlp_g{GROUP_SIZE}_r{CBV_RANK}_mv{MIN_VALID_STARS}_s{SEED}_best.pth"))
MODE = os.environ.get("DECODER_MODE", "direct")           # "direct" (1024 flux) | "weights" (8 CBV)
assert MODE in ("direct", "weights"), MODE
_OUT_SUB = "single_star_decode" if MODE == "direct" else "single_star_weight_decode"
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GROUP_ART_DIR, _OUT_SUB))


def build_decoder(out_dim=1024):
    final_model = nn.Sequential(
        nn.Flatten(), nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, out_dim))

    return final_model


def masked_smooth_l1(prediction, target, mask):

    per_loss = F.smooth_l1_loss(prediction, target, reduction = "none")

    final = (per_loss * mask).sum() / mask.sum().clamp(min = 1.0)

    return final

def masked_metrics(pred, target, mask):

    observed = mask > 0

    prediction, tar = pred[observed], target[observed]

    if prediction.size < 2:
        return np.nan, np.nan, np.nan
    

    correlation = float(np.corrcoef(prediction, tar)[0,1]) if prediction.std() > 0 and tar.std() > 0 else np.nan

    root_mean = float(np.sqrt(np.mean((prediction - tar) ** 2)))
    ss_total = float(np.sum((tar - tar.mean()) ** 2))

    r_squared = float(1.0 - np.sum((tar - prediction) ** 2) / ss_total) if ss_total >0 else np.nan

    return correlation, root_mean, r_squared


def teacher_latent(model, reconstruction, log_mad, valid):

    stats = torch.tensor(np.stack([reconstruction, log_mad], -1), dtype = torch.float32)[None].to(DEVICE)
    v = torch.tensor(valid, dtype = torch.float32)[None].to(DEVICE)

    tokens = model.teacher(stats, v)

    return F.layer_norm(tokens, (tokens.shape[-1],))


def deterministic_area_rows(ds):

    order = np.argsort(np.asarray(ds.tics, dtype = str))

    rank = np.empty(len(ds.tics), dtype = np.int64)

    rank[order] = np.arange(len(ds.tics))
    out = {}

    for i, a in enumerate(ds.areas):

        out.setdefault(int(a), []).append(i)
    
    return {a: list(np.asarray(rows)[np.argsort(rank[rows])]) for a, rows in out.items()}

def build_decoder_trainset(model, train_ds, bases):

    area_rows = deterministic_area_rows(train_ds)

    latitude = []
    target = []
    mask = []
    basis = []

    with torch.no_grad():

        for area, B in bases.items():
            rows = area_rows.get(area, [])
            for g in range(len(rows) // GROUP_SIZE):

                group = rows[g * GROUP_SIZE: (g+1) * GROUP_SIZE]
                median, log_mad, valid, _ = group_statistics(train_ds.X[group], train_ds.M[group], train_ds.min_valid)

                reconstruction = ridge_reconstruct(median, valid, B, RIDGE_LAMBDA)

                latitude.append(teacher_latent(model, reconstruction, log_mad, valid).squeeze(0).cpu().numpy())
                target.append(reconstruction)
                mask.append(valid)
                basis.append(B)                       # (1024, 8) area basis, used only in weights mode

    return (np.stack(latitude).astype(np.float32), np.stack(target).astype(np.float32),
            np.stack(mask).astype(np.float32), np.stack(basis).astype(np.float32))


def reference_target(ds, area_rows, i, bases):

    a = int(ds.areas[i])

    if a not in bases:
        return None, None
    
    neight = [r for r in area_rows.get(a, []) if r != i]

    if len(neight) < GROUP_SIZE:
        return None, None

    group = neight[:GROUP_SIZE]

    median, log_mad, valid, _ = group_statistics(ds.X[group], ds.M[group], ds.min_valid)

    final = ridge_reconstruct(median, valid, bases[a], RIDGE_LAMBDA), valid

    return final

def decode(model, decoder, raw, mask, basis=None):

    r = torch.tensor(raw, dtype = torch.float32)[None].to(DEVICE)
    #then for the masked part
    m = torch.tensor(mask, dtype = torch.float32)[None].to(DEVICE)
    predicted_latent = model.encode(r, m, view = "predicted")
    out = decoder(predicted_latent)
    if basis is not None:                              # weights mode: 8 weights -> 1024 curve
        Bt = torch.tensor(basis, dtype=torch.float32).to(DEVICE)     # (1024, 8)
        out = out @ Bt.T                               # (1,8) @ (8,1024) = (1,1024)
    return out.squeeze(0).detach().cpu().numpy()


def predict_weights(model, decoder, raw, mask):
    r = torch.tensor(raw, dtype=torch.float32)[None].to(DEVICE)
    m = torch.tensor(mask, dtype=torch.float32)[None].to(DEVICE)
    w = decoder(model.encode(r, m, view="predicted"))
    return w.squeeze(0).detach().cpu().numpy()          # (8,)

#the rest of this code is from Claude code 

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE,
                                        min_valid=MIN_VALID_STARS)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE,
                                      min_valid=MIN_VALID_STARS)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, "test TIC leaked"

    bases = build_or_load_area_bases(train_ds.X, train_ds.M, train_ds.areas,
                                     sorted(train_ds.tics), CBV_RANK, GROUP_ART_DIR,
                                     GROUP_SIZE, MIN_VALID_STARS)

    # --- frozen Stage-B model (student + teacher + predictor all inside) ------
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean",
                                       predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hashes = {"teacher": state_hash(model.teacher), "student": state_hash(model.student),
              "predictor": state_hash(model.predictor)}
    print(f"frozen Stage-B model {STAGE_B_CKPT}", flush=True)
    print(f"hashes teacher {hashes['teacher'][:12]} student {hashes['student'][:12]} "
          f"predictor {hashes['predictor'][:12]}", flush=True)

    # --- train ONLY the decoder ----------------------------------------------
    Z, T, M, A = build_decoder_trainset(model, train_ds, bases)
    print(f"decoder trainset: {len(Z)} teacher-target pairs (mode={MODE})", flush=True)
    decoder = build_decoder(CBV_RANK if MODE == "weights" else 1024).to(DEVICE)
    opt = torch.optim.AdamW(decoder.parameters(), lr=LR)
    Zt, Tt, Mt, At = (torch.from_numpy(Z), torch.from_numpy(T),
                      torch.from_numpy(M), torch.from_numpy(A))
    grad_checked = False
    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        idx = torch.randperm(len(Zt))
        total, n = 0.0, 0
        for s in range(0, len(Zt), BATCH):
            j = idx[s:s + BATCH]
            z, t, m = Zt[j].to(DEVICE), Tt[j].to(DEVICE), Mt[j].to(DEVICE)
            out = decoder(z)
            if MODE == "weights":                        # 8 weights -> curve via area basis
                a = At[j].to(DEVICE)                      # (B, 1024, 8)
                curve = torch.einsum("bk,blk->bl", out, a)
            else:
                curve = out
            loss = masked_smooth_l1(curve, t, m)
            opt.zero_grad()
            loss.backward()
            if not grad_checked:      # only decoder params receive gradients
                if MODE == "weights":
                    assert out.shape[1] == CBV_RANK, out.shape   # decoder output [batch, 8]
                    assert curve.shape[1] == 1024, curve.shape   # reconstructed curve [batch, 1024]
                assert all(p.grad is None for p in model.parameters())
                assert any(p.grad is not None for p in decoder.parameters())
                grad_checked = True
            opt.step()
            total += float(loss.detach())
            n += 1
        print(f"decoder epoch {epoch:02d}: train_smoothl1={total / max(1, n):.6f}", flush=True)

    torch.save(decoder.state_dict(), os.path.join(OUT_DIR, "decoder.pth"))

    # frozen models must be bit-identical
    assert state_hash(model.teacher) == hashes["teacher"], "teacher changed"
    assert state_hash(model.student) == hashes["student"], "student changed"
    assert state_hash(model.predictor) == hashes["predictor"], "predictor changed"

    # --- validation inference over the WHOLE val split ------------------------
    decoder.eval()
    val_area_rows = deterministic_area_rows(val_ds)
    corrs, rmses, r2s = [], [], []
    for i in range(len(val_ds.X)):
        ref, valid = reference_target(val_ds, val_area_rows, i, bases)
        if ref is None:
            continue
        basis = bases[int(val_ds.areas[i])] if MODE == "weights" else None
        pred = decode(model, decoder, val_ds.X[i], val_ds.M[i], basis)
        assert pred.shape == (1024,), f"reconstructed curve length {pred.shape}"
        c, rm, r2 = masked_metrics(pred, ref, valid)
        if np.isfinite(c) and np.isfinite(rm) and np.isfinite(r2):
            corrs.append(c); rmses.append(rm); r2s.append(r2)

    def iqr_summary(name, xs):
        a = np.asarray(xs)
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        print(f"{name}: median={med:.4f} IQR=[{q1:.4f}, {q3:.4f}] (n={len(a)})", flush=True)
        return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": int(len(a))}

    metrics = {"masked_pearson": iqr_summary("corr", corrs),
               "masked_rmse": iqr_summary("rmse", rmses),
               "masked_r2": iqr_summary("R2", r2s)}

    # --- fixed validation-star figure (direct: 2 rows x 5; weights: 6 rows x 6)
    n_show = 6 if MODE == "weights" else 2
    n_col = 6 if MODE == "weights" else 5
    eligible = [i for i in range(len(val_ds.X))
                if reference_target(val_ds, val_area_rows, i, bases)[0] is not None]
    seen, picks = set(), []
    for i in eligible:                       # first eligible star per distinct area
        a = int(val_ds.areas[i])
        if a not in seen:
            seen.add(a); picks.append(i)
        if len(picks) == n_show:
            break
    x = np.arange(1024)
    fig, axes = plt.subplots(len(picks), n_col, figsize=(4.4 * n_col, 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(picks):
        ref, valid = reference_target(val_ds, val_area_rows, i, bases)
        a = int(val_ds.areas[i])
        basis = bases[a] if MODE == "weights" else None
        pred = decode(model, decoder, val_ds.X[i], val_ds.M[i], basis)
        raw_m = np.where(val_ds.M[i] > 0, val_ds.X[i], np.nan)
        obs = valid > 0
        refm = np.where(obs, ref, np.nan)
        predm = np.where(obs, pred, np.nan)
        resid = np.where(val_ds.M[i] > 0, val_ds.X[i] - pred, np.nan)
        axes[row, 0].plot(x, raw_m, lw=0.6, color="0.4")
        axes[row, 1].plot(x, refm, lw=0.7, color="tab:blue")
        axes[row, 2].plot(x, predm, lw=0.7, color="tab:orange")
        axes[row, 3].plot(x, refm, lw=0.7, color="tab:blue", label="target")
        axes[row, 3].plot(x, predm, lw=0.7, color="tab:orange",
                          label="prediction" if MODE == "weights" else "decoded")
        axes[row, 4].plot(x, resid, lw=0.6, color="tab:green")
        if MODE == "weights":                # column 6: the 8 predicted CBV weights
            w = predict_weights(model, decoder, val_ds.X[i], val_ds.M[i])
            axes[row, 5].bar(np.arange(CBV_RANK), w, color="tab:purple")
        axes[row, 0].set_ylabel(f"area {a}", fontsize=8)
    titles = (["raw star", f"{GROUP_SIZE}-star r{CBV_RANK} target", "MLP-decoded",
               "target vs decoded", "predicted physics (raw - instrument)"]
              if MODE == "direct" else
              ["raw star", f"{GROUP_SIZE}-star r{CBV_RANK} target", f"{CBV_RANK}-weight recon",
               "target vs prediction", "raw - prediction", f"{CBV_RANK} CBV weights"])
    for c, ttl in enumerate(titles):
        axes[0, c].set_title(ttl, fontsize=9)
    axes[0, 3].legend(fontsize=6, loc="upper right")
    fig.suptitle("single-star instrument decode" if MODE == "direct"
                 else f"single-star weight decode ({CBV_RANK} CBV weights, frozen Stage-B)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_name = "single_star_decode.png" if MODE == "direct" else "single_star_weight_decode.png"
    fig.savefig(os.path.join(OUT_DIR, fig_name), dpi=130)
    plt.close(fig)

    # --- combined qclean panels: context / group median / CBV target / decoder (gaps at flags)
    if MODE == "direct":
        cpick = picks[:4]
        figq, axq = plt.subplots(len(cpick), 4, figsize=(18, 3 * len(cpick)))
        axq = np.atleast_2d(axq)
        for row, i in enumerate(cpick):
            a = int(val_ds.areas[i])
            neigh = [r for r in val_area_rows.get(a, []) if r != i][:GROUP_SIZE]
            median, log_mad, valid, _ = group_statistics(val_ds.X[neigh], val_ds.M[neigh],
                                                         val_ds.min_valid)
            recon = ridge_reconstruct(median, valid, bases[a], RIDGE_LAMBDA)
            pred = decode(model, decoder, val_ds.X[i], val_ds.M[i].astype(np.float32))
            cmask = val_ds.M[i] > 0
            obs = valid > 0

            def g(arr, m):
                out = np.full(1024, np.nan); out[m] = arr[m]; return out
            axq[row, 0].plot(x, g(val_ds.X[i], cmask), lw=0.6, color="0.4")
            axq[row, 1].plot(x, g(median, obs), lw=0.7, color="tab:blue")
            axq[row, 2].plot(x, g(recon, obs), lw=0.7, color="tab:green")
            axq[row, 3].plot(x, g(pred, obs), lw=0.7, color="tab:orange")
            axq[row, 0].set_ylabel(f"area {a}", fontsize=8)
        for c, ttl in enumerate(["filtered context star", f"filtered {GROUP_SIZE}-star median",
                                 "CBV target", "direct decoder output"]):
            axq[0, c].set_title(ttl, fontsize=9)
        figq.suptitle("qclean pipeline (flagged cadences appear as gaps in every panel)")
        figq.tight_layout(rect=[0, 0, 1, 0.98])
        figq.savefig(os.path.join(OUT_DIR, "qclean_pipeline_panels.png"), dpi=130)
        plt.close(figq)

    vs_direct = None
    if MODE == "weights":                    # median delta vs the existing direct 1024-output decoder
        direct_json = os.path.join(os.path.dirname(OUT_DIR), "single_star_decode",
                                   "final_summary.json")
        if os.path.exists(direct_json):
            d = json.load(open(direct_json)).get("metrics", {})
            vs_direct = {k: round(metrics[k]["median"] - d[k]["median"], 6)
                         for k in metrics if k in d}

    report = {"git_commit": git_commit(), "stage_b_ckpt": STAGE_B_CKPT,
              "decoder_mode": MODE, "group_size": GROUP_SIZE, "cbv_rank": CBV_RANK,
              "min_valid": MIN_VALID_STARS, "ridge_lambda": RIDGE_LAMBDA,
              "decoder_epochs": EPOCHS, "n_decoder_train_groups": int(len(Z)),
              "frozen_hashes_unchanged": True,
              "decoder_raw_output_dim": CBV_RANK if MODE == "weights" else 1024,
              "reconstructed_curve_length": 1024,
              "prediction_and_target_normalization": "identical (both are the "
              "median/MAD-normalized CBV reconstruction space)",
              "test_split_loaded": False, "metrics": metrics,
              "vs_direct_decoder_median_delta": vs_direct}
    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {OUT_DIR}/{fig_name} + final_summary.json", flush=True)


if __name__ == "__main__":
    main()

