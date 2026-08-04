from __future__ import annotations
"""Self-supervised instrument correction: shared S4D -> 32 latent -> MLP -> 1024
correction c; residual r=x-c; minimize squared pairwise RESIDUAL correlation +
lambda_size * normalized correction energy. ONE backward + ONE step per 32-group.
    LAMBDA_SIZE=0.01 python -m src.shared_s4d.train_correction
"""

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit, seed_worker, effective_rank
from src.shared_s4d.ae_dataset import AreaGroupAEDataset
from src.shared_s4d.model import (build_model, preprocessing_config, experiment_tag,
                                  GRID, LATENT_DIM, N_TOKENS, TOKEN_DIM)
from src.shared_s4d.correction_losses import (
    masked_pairwise_residual_correlation, normalized_correction_energy, mean_abs_pairwise_corr,
    topk_fixed_cov_loss, relative_correction_size,
    windowed_group_cov_loss, pairwise_window_cov_loss, soft_cap_size)


SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))
GROUPING_MODE = os.environ.get("GROUPING_MODE", "random")       # random | nearest (RA/Dec anchor groups)
N_STARS = int(os.environ.get("N_STARS", "1000"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LR = float(os.environ.get("LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0"))
GROUPS_PER_AREA = int(os.environ.get("GROUPS_PER_AREA", "100"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "3"))
LAMBDA_SIZE = float(os.environ.get("LAMBDA_SIZE", "0.01"))
LOSS_MODE = os.environ.get("LOSS_MODE", "topk_fixed_cov")     # topk_fixed_cov | legacy_corr
TOPK_PEERS = int(os.environ.get("TOPK_PEERS", "8"))
assert LOSS_MODE in ("topk_fixed_cov", "legacy_corr", "windowed_group_cov", "pairwise_window_cov"), LOSS_MODE
MIN_OVERLAP = int(os.environ.get("MIN_OVERLAP", "64"))          # min shared observed cadences per pair
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
REQUIRE_FULL = os.environ.get("REQUIRE_FULL", "1").lower() not in ("0", "false", "no")
COLLAPSE_STD = float(os.environ.get("COLLAPSE_STD", "1e-3"))
USE_AMP = os.environ.get("USE_AMP", "0") == "1"                  # S4D complex kernels -> default off
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))          # >0 = smoke
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "shared_s4d", "correction_v1"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/shared_s4d_correction_v1")

def group_losses(residuals, corrections, curves, masks):
    """(shared, size) for the active LOSS_MODE."""
    if LOSS_MODE == "topk_fixed_cov":
        shared = topk_fixed_cov_loss(residuals, curves, masks, TOPK_PEERS, MIN_OVERLAP)
        size = relative_correction_size(corrections, curves, masks)
    elif LOSS_MODE == "pairwise_window_cov":
        shared = pairwise_window_cov_loss(residuals, curves, masks)   # square-before-average, all windows
        size = soft_cap_size(corrections, curves, masks)      # run this mode with LAMBDA_SIZE=0.1
    elif LOSS_MODE == "windowed_group_cov":
        shared = windowed_group_cov_loss(residuals, curves, masks)
        size = soft_cap_size(corrections, curves, masks)      # run this mode with LAMBDA_SIZE=0.1
    else:
        shared = masked_pairwise_residual_correlation(residuals, masks, MIN_OVERLAP)
        size = normalized_correction_energy(corrections, curves, masks)
    return shared, size


def run_train_epoch(model, loader, optimizer, scaler):
    model.train()
    tot = shl = szl = 0.0; nb = 0
    for bi, (Xg, Mg) in enumerate(loader):
        if MAX_BATCHES and bi >= MAX_BATCHES:
            break
        Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)   # ONE group (32, L)
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
            corrections, latents = model(Xg, Mg)                      # c_i (32, L)
            residuals = Xg - corrections                              # r_i = x_i - c_i
            shared_loss, size_loss = group_losses(residuals, corrections, Xg, Mg)

            group_loss = shared_loss + LAMBDA_SIZE * size_loss        # combine, THEN backward
        if scaler.is_enabled():
            scaler.scale(group_loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            group_loss.backward(); optimizer.step()                   # one backward + one step / group
        tot += float(group_loss.detach()); shl += float(shared_loss.detach()); szl += float(size_loss.detach()); nb += 1
    return tot / max(1, nb), shl / max(1, nb), szl / max(1, nb)


def val_metrics(model, loader):
    model.eval()
    befs, afts, ratios, caps, shareds, sizes, lat_chunks = [], [], [], [], [], [], []
    pwbs, pwas, ers = [], [], []
    with torch.no_grad():
        for bi, (Xg, Mg) in enumerate(loader):
            if MAX_BATCHES and bi >= MAX_BATCHES:
                break
            Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)
            corrections, lat = model(Xg, Mg)
            residuals = Xg - corrections
            # PRIMARY metric: the exact pairwise-window loss before (c=0 -> r=x) and after
            pwbs.append(float(pairwise_window_cov_loss(Xg, Xg, Mg)))
            pwas.append(float(pairwise_window_cov_loss(residuals, Xg, Mg)))
            befs.append(mean_abs_pairwise_corr(Xg, Mg, MIN_OVERLAP))    # secondary: full-curve corr
            afts.append(mean_abs_pairwise_corr(residuals, Mg, MIN_OVERLAP))
            sh, sz = group_losses(residuals, corrections, Xg, Mg)
            shareds.append(float(sh)); sizes.append(float(sz))
            ers.append(float(relative_correction_size(corrections, Xg, Mg)))   # raw energy ratio

            m = (Mg > 0).float()
            crms = torch.sqrt((m * corrections ** 2).sum(1) / m.sum(1).clamp(min=1.0))
            xrms = torch.sqrt((m * Xg ** 2).sum(1) / m.sum(1).clamp(min=1.0))
            ratios.append(float((crms / xrms.clamp(min=1e-8)).mean()))
            xmean = (Xg * m).sum(1) / m.sum(1).clamp(min=1.0)     # soft-cap ratio = mean(c^2)/var(x)
            xvar = (((Xg - xmean[:, None]) * m) ** 2).sum(1) / m.sum(1).clamp(min=1.0) + 1e-6
            energy_ratio = (m * corrections ** 2).sum(1) / m.sum(1).clamp(min=1.0) / xvar
            caps.append(float((energy_ratio > 0.5).float().mean()))   # fraction over the 0.5 soft cap
            lat_chunks.append(lat.cpu().numpy())
    bef = float(np.nanmean(befs)); aft = float(np.nanmean(afts))
    pwb = float(np.mean(pwbs)); pwa = float(np.mean(pwas))
    Z = np.concatenate(lat_chunks, 0); shared = float(np.mean(shareds)); size = float(np.mean(sizes))
    return {"corr_before": bef, "corr_after": aft,
            "pct_reduction": float(100.0 * (bef - aft) / bef) if bef > 1e-8 else float("nan"),
            "pw_before": pwb, "pw_after": pwa,
            "pw_reduction": float(100.0 * (pwb - pwa) / pwb) if pwb > 1e-12 else float("nan"),
            "energy_ratio": float(np.mean(ers)),
            "corr_rms_over_input_rms": float(np.mean(ratios)), "frac_over_cap": float(np.mean(caps)),
            "shared_loss": shared, "size_loss": size, "total_loss": shared + LAMBDA_SIZE * size,
            "latent_std": float(Z.std(0).mean()), "effective_rank": float(effective_rank(Z))}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = experiment_tag(LOSS_MODE, GROUPING_MODE, GROUP_SIZE, N_TOKENS, TOKEN_DIM, LAMBDA_SIZE, SEED,
                         lr=LR, weight_decay=WEIGHT_DECAY, groups_per_area=GROUPS_PER_AREA)
    ckpt_base = os.path.join(CKPT_DIR, tag)
    print(f"git {git_commit()}  tag {tag}  device {DEVICE}  loss {LOSS_MODE} grouping {GROUPING_MODE} "
          f"g{GROUP_SIZE} tokens {N_TOKENS}x{TOKEN_DIM}=z{LATENT_DIM} topk {TOPK_PEERS}  "
          f"lambda_size {LAMBDA_SIZE}  min_overlap {MIN_OVERLAP}  "
          f"require_full {REQUIRE_FULL}  amp {USE_AMP}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    if GROUPING_MODE == "detector_nearest" and {"DETECTOR_X", "DETECTOR_Y"} <= set(df.columns):
        n0 = len(df)                                          # drop the rare boundary star with NaN coords
        df = df[np.isfinite(df["DETECTOR_X"]) & np.isfinite(df["DETECTOR_Y"])].reset_index(drop=True)
        if len(df) < n0:
            print(f"dropped {n0 - len(df)} stars with non-finite DETECTOR_X/Y", flush=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    # base loader only supplies X/M/areas/tics; its own group_size is unrelated to the
    # experiment's GROUP_SIZE -> fix it at 32 so min_valid=16 stays valid for GROUP_SIZE<16.
    base_tr = Sector14GroupStatDataset(df, train_tics, t_range, "area", 32, min_valid=16)
    base_va = Sector14GroupStatDataset(df, val_tics, t_range, "area", 32, min_valid=16)
    assert not (set(base_tr.tics) | set(base_va.tics)) & test_tics, "test TIC leaked"

    def radec_for(tics):                                         # RA/Dec aligned to the dataset's stars
        if GROUPING_MODE != "nearest":
            return None
        if not {"ra", "dec"} <= set(df.columns):
            raise RuntimeError("GROUPING_MODE=nearest but the parquet has no ra/dec columns "
                               "-- cannot group spatially (fix the data, do not fall back to random)")
        sub = df.set_index(df["TIC"].astype(str))
        return sub.loc[[str(t) for t in tics], ["ra", "dec"]].to_numpy(dtype=float)

    def detxy_for(tics):                                         # physical DETECTOR_X/DETECTOR_Y per star
        if GROUPING_MODE != "detector_nearest":
            return None
        if not {"DETECTOR_X", "DETECTOR_Y"} <= set(df.columns):
            raise RuntimeError(
                "GROUPING_MODE=detector_nearest but the parquet has no DETECTOR_X/DETECTOR_Y columns. "
                "Regenerate the parquet with src/tglc/merge_detector_positions.py (RA/Dec -> TESS "
                "detector col/row via tess-point). Refusing to fall back to RA/Dec or random.")
        sub = df.set_index(df["TIC"].astype(str))
        xy = sub.loc[[str(t) for t in tics], ["DETECTOR_X", "DETECTOR_Y"]].to_numpy(dtype=float)
        if not np.isfinite(xy).all():
            raise RuntimeError("DETECTOR_X/DETECTOR_Y has non-finite values -- bad coordinate merge")
        return xy

    train_ds = AreaGroupAEDataset(base_tr.X, base_tr.M, base_tr.areas, base_tr.tics,
                                  n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                  require_full=REQUIRE_FULL, resample=True, grouping_mode=GROUPING_MODE,
                                  radec=radec_for(base_tr.tics), detxy=detxy_for(base_tr.tics),
                                  groups_per_area=GROUPS_PER_AREA)
    val_ds = AreaGroupAEDataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                require_full=False, resample=False, grouping_mode=GROUPING_MODE,
                                radec=radec_for(base_va.tics), detxy=detxy_for(base_va.tics),
                                groups_per_area=GROUPS_PER_AREA)
    print(f"train: {len(train_ds.eligible)} areas -> {len(train_ds)} groups/epoch | "
          f"val: {len(val_ds.eligible)} areas -> {len(val_ds)} groups", flush=True)
    if getattr(train_ds, "group_stats", None):                   # per-area: pool, groups, neighbor dist
        gs = train_ds.group_stats
        pools = [s["pool"] for s in gs.values()]; grps = [s["groups"] for s in gs.values()]
        meds = [s["med_dist"] for s in gs.values()]; maxs = [s["max_dist"] for s in gs.values()]
        print(f"group_stats over {len(gs)} areas: pool[min {min(pools)} max {max(pools)}] "
              f"groups[min {min(grps)} max {max(grps)}] "
              f"neighbor_dist med~{np.nanmedian(meds):.1f} max~{np.nanmax(maxs):.1f}", flush=True)
        for a in sorted(gs)[:8]:                                  # first few areas verbatim
            s = gs[a]; print(f"    area {a}: pool {s['pool']} groups {s['groups']} "
                             f"med_dist {s['med_dist']:.1f} max_dist {s['max_dist']:.1f}", flush=True)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    with open(os.path.join(ART_DIR, "preprocessing.json"), "w") as fh:
        json.dump(preprocessing_config(), fh, indent=2)
    fields = ["epoch", "train_total", "train_shared", "train_size", "val_total", "val_shared",
              "val_size", "pw_before", "pw_after", "pw_reduction", "energy_ratio",
              "corr_before", "corr_after", "pct_reduction", "corr_rms_over_input_rms",
              "frac_over_cap", "latent_std", "effective_rank"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    # Select on the PAIRWISE-WINDOW loss REDUCTION (a near-zero correction gives ~0% reduction,
    # so it cannot win), and reject epochs whose reduction is negligible.
    MIN_PW_REDUCTION = float(os.environ.get("MIN_PW_REDUCTION", "1.0"))   # percent
    best = {"pw_reduction": -float("inf"), "epoch": None}
    best_pwr = -float("inf"); no_improve = 0                              # early-stop bookkeeping
    collapsed = False
    for epoch in range(1, EPOCHS + 1):
        train_ds.set_epoch(epoch)
        train_ds.assert_contracts()
        tl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED + epoch))
        vl = DataLoader(val_ds, batch_size=1, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(SEED))

        tr_tot, tr_sh, tr_sz = run_train_epoch(model, tl, optimizer, scaler)
        scheduler.step()
        vm = val_metrics(model, vl)

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_total": tr_tot, "train_shared": tr_sh, "train_size": tr_sz,
                 "val_total": vm["total_loss"], "val_shared": vm["shared_loss"], "val_size": vm["size_loss"],
                 "pw_before": vm["pw_before"], "pw_after": vm["pw_after"], "pw_reduction": vm["pw_reduction"],
                 "energy_ratio": vm["energy_ratio"],
                 "corr_before": vm["corr_before"], "corr_after": vm["corr_after"],
                 "pct_reduction": vm["pct_reduction"], "corr_rms_over_input_rms": vm["corr_rms_over_input_rms"],
                 "frac_over_cap": vm["frac_over_cap"],
                 "latent_std": vm["latent_std"], "effective_rank": vm["effective_rank"]})
        marker = ""
        pwr = vm["pw_reduction"]
        improved = np.isfinite(pwr) and pwr > best_pwr                    # any new max = improvement
        if improved:
            best_pwr = pwr; no_improve = 0
            if pwr >= MIN_PW_REDUCTION:                                   # save only meaningful reductions
                best = {"pw_reduction": pwr, "epoch": epoch, "pw_before": vm["pw_before"],
                        "pw_after": vm["pw_after"], "corr_before": vm["corr_before"], "corr_after": vm["corr_after"],
                        "energy_ratio": vm["energy_ratio"], "corr_rms_over_input_rms": vm["corr_rms_over_input_rms"]}
                torch.save({"model": model.state_dict(), "config": preprocessing_config(), "epoch": epoch,
                            "lambda_size": LAMBDA_SIZE, "pw_reduction": pwr}, f"{ckpt_base}_best.pth")
                torch.save(model.encoder.state_dict(), f"{ckpt_base}_best_encoder.pth")
                torch.save(model.decoder.state_dict(), f"{ckpt_base}_best_decoder.pth")
                marker = " <- best"
        else:
            no_improve += 1
        print(f"[epoch {epoch:02d}] pw {vm['pw_before']:.4f}->{vm['pw_after']:.4f} "
              f"(-{vm['pw_reduction']:.1f}%) | corr {vm['corr_before']:.3f}->{vm['corr_after']:.3f} "
              f"(-{vm['pct_reduction']:.0f}%) energy={vm['energy_ratio']:.3f} "
              f"c/x_rms={vm['corr_rms_over_input_rms']:.3f} over_cap={vm['frac_over_cap']:.3f} "
              f"lstd={vm['latent_std']:.3f} erank={vm['effective_rank']:.1f}{marker}", flush=True)

        if vm["latent_std"] < COLLAPSE_STD:
            print(f"!! LATENT COLLAPSE: std {vm['latent_std']:.2e} < {COLLAPSE_STD:.1e} -- stopping", flush=True)
            collapsed = True
            break
        if no_improve >= EARLY_STOP_PATIENCE:                    # stop after N epochs w/o improved pw reduction
            print(f"!! EARLY STOP: {no_improve} epochs without improved pairwise-window reduction "
                  f"(best {best_pwr:.1f}% @ epoch {best['epoch']}); best checkpoint preserved", flush=True)
            break

    meaningful = best["epoch"] is not None                       # did any epoch clear MIN_PW_REDUCTION?
    if not meaningful:
        print(f"!! NO checkpoint reached >= {MIN_PW_REDUCTION}% pairwise-window reduction -- "
              f"NOT a success (near-zero / ineffective correction)", flush=True)
    selection = {"tag": tag, "seed": SEED, "group_size": GROUP_SIZE, "latent_dim": LATENT_DIM,
                 "n_tokens": N_TOKENS, "token_dim": TOKEN_DIM,
                 "loss_mode": LOSS_MODE, "topk_peers": TOPK_PEERS, "grouping_mode": GROUPING_MODE,
                 "n_stars": N_STARS, "epochs": EPOCHS, "lambda_size": LAMBDA_SIZE, "min_overlap": MIN_OVERLAP,
                 "min_pw_reduction": MIN_PW_REDUCTION, "meaningful_reduction": meaningful,
                 "require_full": REQUIRE_FULL, "collapsed": collapsed, "best": best,
                 "checkpoint": f"{ckpt_base}_best.pth", "encoder_checkpoint": f"{ckpt_base}_best_encoder.pth",
                 "decoder_checkpoint": f"{ckpt_base}_best_decoder.pth",
                 "preprocessing": os.path.join(ART_DIR, "preprocessing.json"), "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
