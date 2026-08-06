"""Does the correction actually capture residual scattered light, or just smooth curves?

Read-only evaluation of a trained masked same-sector checkpoint. Nothing is retrained
and no architecture is touched.

IMPORTANT SCOPE. Preprocessing removes every cadence with a nonzero TESS or TGLC flag,
including the ones TESS itself marks as stray light. So this measures RESIDUAL,
UNFLAGGED scattered light -- not the flagged excursions, which the model never sees.

Definitions used throughout (per the evaluation spec, which differs from infer.py):

    correction = pred_actual - pred_reference
    cleaned    = raw_anchor - correction

Evidence is only called strong when all four hold: the correction tracks the nearby
peer common mode at ~zero lag, cleaning materially reduces the anchor's common-mode
projection, correct nearest peers beat random and time-shuffled controls, and injected
astrophysical signals survive.

    python -m disentangle_attempt.eval_scattered_light \
      --checkpoint disentangle_attempt/outputs/fast_strict/best.pt
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from disentangle_attempt.dataset import (CrossSectorPatch,
                                        infer_require_cross_sector)
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

MIN_VALID_PEERS = 4
MAX_LAG = 50


# ------------------------------------------------------------------ small helpers
def robust_std(values):
    values = np.asarray(values, float)
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def safe_pearson(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pearsonr(a, b)[0])


def safe_spearman(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def best_lag(a, b, max_lag=MAX_LAG):
    """Lag (in cadences) maximizing |normalized cross-correlation| of a against b."""
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    lags = np.arange(-max_lag, max_lag + 1)
    scores = []
    for lag in lags:
        if lag < 0:
            scores.append(np.dot(a[-lag:], b[:len(b) + lag]) / denom)
        elif lag > 0:
            scores.append(np.dot(a[:len(a) - lag], b[lag:]) / denom)
        else:
            scores.append(np.dot(a, b) / denom)
    scores = np.asarray(scores)
    peak = int(np.argmax(np.abs(scores)))
    return float(lags[peak]), float(scores[peak])


def projection(curve, basis):
    """beta = <curve, basis> / <basis, basis>."""
    denom = float(np.dot(basis, basis))
    return float(np.dot(curve, basis) / denom) if denom > 0 else np.nan


def peer_common_mode(patch, peer_rows):
    """Median over the 8 nearest peers, valid only where >= MIN_VALID_PEERS observe."""
    flux, mask = patch.X[peer_rows], patch.M[peer_rows]
    counts = mask.sum(axis=0)
    stacked = np.where(mask, flux, np.nan)
    common = np.zeros(flux.shape[1], dtype=np.float64)
    seen = counts > 0
    if seen.any():                      # nanmedian warns on all-NaN columns
        common[seen] = np.nanmedian(stacked[:, seen], axis=0)
    valid = (counts >= MIN_VALID_PEERS) & np.isfinite(common)
    return np.nan_to_num(common), valid


# --------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, patch, rows, peer_rows, quiet, masks, device):
    """Stitched complementary-mask inference -> (pred_actual, pred_reference)."""
    raw = torch.from_numpy(patch.X[rows])
    valid = torch.from_numpy(patch.M[rows])
    actual, reference, _, _, _ = dual_context_prediction(
        model, raw, valid,
        torch.from_numpy(patch.X[peer_rows]), torch.from_numpy(patch.M[peer_rows]),
        quiet["peer_raw"].unsqueeze(0).expand(len(rows), -1, -1),
        quiet["peer_mask"].unsqueeze(0).expand(len(rows), -1, -1), masks, device)
    return actual.numpy(), reference.numpy()


@torch.no_grad()
def predict_curves(model, curves, valid, peer_flux, peer_mask, quiet, masks, device):
    """Same, but on explicitly supplied anchor curves (for the injection test)."""
    actual, reference, _, _, _ = dual_context_prediction(
        model, torch.from_numpy(curves), torch.from_numpy(valid),
        torch.from_numpy(peer_flux), torch.from_numpy(peer_mask),
        quiet["peer_raw"].unsqueeze(0).expand(len(curves), -1, -1),
        quiet["peer_mask"].unsqueeze(0).expand(len(curves), -1, -1), masks, device)
    return actual.numpy(), reference.numpy()


# ----------------------------------------------------------------- injections
def make_transit(length, centre, duration, depth):
    x = np.arange(length)
    edge = np.clip((duration / 2 - np.abs(x - centre)) / max(duration * 0.2, 1), 0, 1)
    return (-depth * edge).astype(np.float32)


def make_sinusoid(length, period, amplitude):
    return (amplitude * np.sin(2 * np.pi * np.arange(length) / period)).astype(np.float32)


def make_flare(length, centre, decay, amplitude):
    x = np.arange(length).astype(float)
    rise = np.clip((x - (centre - decay * 0.3)) / max(decay * 0.3, 1), 0, 1)
    fall = np.exp(-np.clip(x - centre, 0, None) / decay)
    return (amplitude * rise * np.where(x >= centre, fall, 1.0)).astype(np.float32)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--n-stars", type=int, default=100)
    parser.add_argument("--n-control", type=int, default=50)
    parser.add_argument("--n-inject", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = os.path.join(run_dir, "scattered_light_test")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    target = state.get("target")
    sector, camera, ccd = target if target else ("auto", "auto", "auto")

    # require_cross_sector reproduces the eligibility rule this checkpoint trained
    # under, so the held-out TICs really are the ones it never saw.
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(
            config, args.require_cross_sector), verbose=False)
    anchors_all = patch.split_anchors["test"]
    assert len(anchors_all), "no test anchors"

    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    quiet = load_reference_context(run_dir, expected_cadence_ids=patch.grids[patch.target[0]])
    masks = complementary_masks(config["curve_length"], n_masks=4)

    anchors = [int(a) for a in anchors_all[:args.n_stars]]
    peer_rows = np.stack([patch.peers_for_row(a, "test")[0] for a in anchors])
    print(f"checkpoint {args.checkpoint}\nevaluating {len(anchors)} held-out test stars "
          f"on sector {patch.target[0]} cam{patch.target[1]}-ccd{patch.target[2]}", flush=True)

    background_available = bool(np.any(patch.BG[anchors] != 0))
    print(f"local background column: "
          f"{'TGLC background (available)' if background_available else 'UNAVAILABLE'}",
          flush=True)

    pred_actual, pred_reference = predict(model, patch, anchors, peer_rows, quiet, masks, device)
    correction_all = pred_actual - pred_reference
    cleaned_all = patch.X[anchors] - correction_all

    # ------------------------------------------------- tests 1-3, per star
    records = []
    for k, anchor in enumerate(anchors):
        common, common_valid = peer_common_mode(patch, peer_rows[k])
        use = patch.M[anchor] & common_valid
        if use.sum() < 100:
            continue
        raw = patch.X[anchor][use]
        corr = correction_all[k][use]
        clean = cleaned_all[k][use]
        cm = common[use]
        cm_centred = cm - np.mean(cm)

        lag, lag_score = best_lag(corr, cm)
        beta_raw = projection(raw - np.mean(raw), cm_centred)
        beta_clean = projection(clean - np.mean(clean), cm_centred)
        removal = (1 - abs(beta_clean) / abs(beta_raw)) if abs(beta_raw) > 1e-9 else np.nan

        # test 3: high-systematics vs quiet cadences, ranked by |peer_common|
        magnitude = np.abs(cm_centred)
        high = magnitude >= np.quantile(magnitude, 0.90)
        low = magnitude <= np.quantile(magnitude, 0.50)
        response = (np.median(np.abs(corr[high])) / np.median(np.abs(corr[low]))
                    if np.median(np.abs(corr[low])) > 0 else np.nan)

        record = {
            "tic": patch.tic[anchor], "row": anchor, "n_cadences": int(use.sum()),
            "pearson_correction_peercommon": safe_pearson(corr, cm),
            "spearman_correction_peercommon": safe_spearman(corr, cm),
            "xcorr_peak_lag": lag, "xcorr_peak_value": lag_score,
            "beta_raw": beta_raw, "beta_clean": beta_clean, "removal_fraction": removal,
            "event_response_ratio": response,
            "raw_vs_peercommon_high": safe_pearson(raw[high], cm[high]),
            "cleaned_vs_peercommon_high": safe_pearson(clean[high], cm[high]),
            "correction_rms": float(np.sqrt(np.mean(corr ** 2))),
        }
        if background_available:
            bg = patch.BG[anchor][use]
            if np.std(bg) > 0:
                record["pearson_correction_background"] = safe_pearson(corr, bg)
                record["spearman_correction_background"] = safe_spearman(corr, bg)
                record["xcorr_zero_lag_background"] = best_lag(corr, bg)[1]
                record["background_lag"] = best_lag(corr, bg)[0]
        records.append(record)

    per_star = pd.DataFrame(records)
    per_star.to_csv(os.path.join(out_dir, "per_star_metrics.csv"), index=False)

    def stat(column):
        if column not in per_star or per_star[column].dropna().empty:
            return None
        values = per_star[column].dropna()
        return {"median": float(values.median()),
                "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75)),
                "n": int(len(values))}

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "n_stars_evaluated": int(len(per_star)),
        "sector_camera_ccd": list(patch.target),
        "background_field": ("TGLC background column" if background_available
                             else "UNAVAILABLE"),
        "scope_note": ("All nonzero TESS/TGLC flags were removed in preprocessing, so "
                       "this measures RESIDUAL UNFLAGGED scattered light, not the "
                       "flagged stray-light cadences."),
        "test1_peer_common_mode": {k: stat(k) for k in [
            "pearson_correction_peercommon", "spearman_correction_peercommon",
            "xcorr_peak_lag", "xcorr_peak_value", "beta_raw", "beta_clean",
            "removal_fraction"]},
        "test2_local_background": ({k: stat(k) for k in [
            "pearson_correction_background", "spearman_correction_background",
            "xcorr_zero_lag_background", "background_lag"]}
            if background_available else "UNAVAILABLE"),
        "test3_high_systematics": {k: stat(k) for k in [
            "event_response_ratio", "raw_vs_peercommon_high", "cleaned_vs_peercommon_high"]},
    }
    if not per_star.empty:
        summary["test1_peer_common_mode"]["fraction_lag_within_2_cadences"] = float(
            (per_star["xcorr_peak_lag"].abs() <= 2).mean())

    # ---------------------------------------------------- test 4: peer controls
    control_rows = anchors[:args.n_control]
    control_peers = peer_rows[:args.n_control]
    conditions = {}
    for name in ("nearest", "random", "time_shuffled"):
        if name == "nearest":
            flux, mask = patch.X[control_peers], patch.M[control_peers]
        elif name == "random":
            picked = patch.random_peer_rows(np.asarray(control_rows), "test", rng)
            flux, mask = patch.X[picked], patch.M[picked]
        else:
            perm = rng.permutation(config["curve_length"])   # ONE permutation, all peers
            flux = patch.X[control_peers][:, :, perm]
            mask = patch.M[control_peers][:, :, perm]
        actual, reference = predict_curves(
            model, patch.X[control_rows].copy(), patch.M[control_rows], flux, mask,
            quiet, masks, device)
        corrections = actual - reference
        cleaned = patch.X[control_rows] - corrections

        losses, correlations, lags, removals = [], [], [], []
        for k, anchor in enumerate(control_rows):
            common, common_valid = peer_common_mode(patch, control_peers[k])
            use = patch.M[anchor] & common_valid
            if use.sum() < 100:
                continue
            cm = common[use] - np.mean(common[use])
            corr = corrections[k][use]
            losses.append(float(np.mean(np.abs(actual[k][use] - patch.X[anchor][use]))))
            correlations.append(safe_pearson(corr, cm))
            lags.append(best_lag(corr, cm)[0])
            b_raw = projection(patch.X[anchor][use] - np.mean(patch.X[anchor][use]), cm)
            b_clean = projection(cleaned[k][use] - np.mean(cleaned[k][use]), cm)
            removals.append(1 - abs(b_clean) / abs(b_raw) if abs(b_raw) > 1e-9 else np.nan)
        conditions[name] = {
            "reconstruction_l1": float(np.nanmedian(losses)),
            "pearson_correction_peercommon": float(np.nanmedian(correlations)),
            "median_abs_lag": float(np.nanmedian(np.abs(lags))),
            "fraction_lag_within_2": float(np.mean(np.abs(np.asarray(lags)) <= 2)),
            "removal_fraction": float(np.nanmedian(removals)),
            "n": int(len(correlations)),
        }
    summary["test4_peer_controls"] = conditions

    # ------------------------------------------- test 5: physics preservation
    inject_rows = anchors[:args.n_inject]
    length = config["curve_length"]
    inject_peers_flux = patch.X[peer_rows[:args.n_inject]]
    inject_peers_mask = patch.M[peer_rows[:args.n_inject]]
    base_actual, base_reference = predict_curves(
        model, patch.X[inject_rows].copy(), patch.M[inject_rows],
        inject_peers_flux, inject_peers_mask, quiet, masks, device)
    cleaned_original = patch.X[inject_rows] - (base_actual - base_reference)

    scales = np.array([robust_std(patch.X[r][patch.M[r]]) for r in inject_rows])
    signals = {}
    for duration in (5, 10, 20):
        signals[f"transit_{duration}"] = ("transit", duration,
                                          [make_transit(length, 0.45 * length, duration, 1.0)] )
    for period in (50, 100, 200):
        signals[f"sinusoid_{period}"] = ("sinusoid", period,
                                         [make_sinusoid(length, period, 1.0)])
    signals["flare_10"] = ("flare", 10, [make_flare(length, 0.45 * length, 10, 1.0)])

    injection_rows, injection_examples = [], {}
    for name, (kind, param, shapes) in signals.items():
        shape = shapes[0]
        injected = (patch.X[inject_rows]
                    + (scales[:, None] * shape[None, :])).astype(np.float32)
        actual, reference = predict_curves(
            model, injected.copy(), patch.M[inject_rows], inject_peers_flux,
            inject_peers_mask, quiet, masks, device)
        cleaned_injected = injected - (actual - reference)
        recovered = cleaned_injected - cleaned_original
        # how much of the injected signal leaked into the instrument correction
        leak = (actual - reference) - (base_actual - base_reference)

        ratios, correlations, leaks = [], [], []
        for k, anchor in enumerate(inject_rows):
            use = patch.M[anchor]
            truth = (scales[k] * shape).astype(np.float32)
            if kind == "transit":
                window = shape < -0.5
                ratios.append(float(np.mean(recovered[k][use & window])
                                    / np.mean(truth[use & window])))
            elif kind == "sinusoid":
                ratios.append(projection(recovered[k][use], truth[use]))
            else:
                window = shape > 0.5
                ratios.append(float(np.max(recovered[k][use & window])
                                    / np.max(truth[use & window])))
            correlations.append(safe_pearson(recovered[k][use], truth[use]))
            leaks.append(float(np.sqrt(np.mean(leak[k][use] ** 2)) / max(scales[k], 1e-9)))
        injection_rows.append({
            "signal": name, "kind": kind, "parameter": param,
            "median_recovery_ratio": float(np.nanmedian(ratios)),
            "q1_recovery": float(np.nanquantile(ratios, 0.25)),
            "q3_recovery": float(np.nanquantile(ratios, 0.75)),
            "median_correlation": float(np.nanmedian(correlations)),
            "median_correction_leak_rel": float(np.nanmedian(leaks)),
        })
        injection_examples[name] = ((scales[0] * shape).astype(np.float32),
                                    recovered[0], patch.M[inject_rows[0]])
    injection_table = pd.DataFrame(injection_rows)
    injection_table.to_csv(os.path.join(out_dir, "injection_recovery.csv"), index=False)
    summary["test5_injection_recovery"] = injection_rows
    summary["test5_note"] = (
        "cleaned = raw - correction, so an injection that does not perturb the "
        "correction is recovered exactly by construction. median_correction_leak_rel "
        "reports how much the injection moved the correction (small = physics untouched).")

    # ------------------------------------------------------------------ verdict
    t1 = summary["test1_peer_common_mode"]
    pear = t1["pearson_correction_peercommon"]
    removal = t1["removal_fraction"]
    lag_ok = t1.get("fraction_lag_within_2_cadences", 0.0)
    controls = summary["test4_peer_controls"]
    beats_controls = (controls["nearest"]["pearson_correction_peercommon"]
                      > max(controls["random"]["pearson_correction_peercommon"],
                            controls["time_shuffled"]["pearson_correction_peercommon"]))
    physics_ok = float(np.nanmedian([r["median_recovery_ratio"] for r in injection_rows]))
    criteria = {
        "correction_tracks_peer_common": bool(pear and pear["median"] > 0.3),
        "peak_lag_near_zero": bool(lag_ok > 0.5),
        "cleaning_reduces_common_mode": bool(removal and removal["median"] > 0.2),
        "nearest_peers_beat_controls": bool(beats_controls),
        "physics_preserved": bool(0.9 <= physics_ok <= 1.1),
    }
    passed = sum(criteria.values())
    verdict = ("strong" if passed == 5 else "partial" if passed >= 3
               else "weak" if passed >= 1 else "none")
    summary["criteria"] = criteria
    summary["verdict"] = verdict

    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    # --------------------------------------------------------------------- plots
    strength = per_star["beta_raw"].abs().fillna(0).to_numpy() if not per_star.empty else np.array([])
    order = np.argsort(-strength)[:6]
    random_pick = rng.choice(len(per_star), size=min(6, len(per_star)), replace=False)
    for tag, picks in (("strongest", order), ("random", random_pick)):
        plot_examples(patch, anchors, peer_rows, correction_all, cleaned_all, per_star,
                      picks, background_available,
                      os.path.join(out_dir, f"examples_{tag}.png"), tag)
    plot_summary(per_star, conditions, injection_table, os.path.join(out_dir, "summary.png"))
    plot_injections(injection_examples, os.path.join(out_dir, "injection_recovery.png"))

    # --------------------------------------------------------------------- report
    print("\n=== TEST 1: correction vs nearby-peer common mode ===")
    for key in ("pearson_correction_peercommon", "spearman_correction_peercommon",
                "xcorr_peak_lag", "removal_fraction"):
        value = t1[key]
        if value:
            print(f"  {key:34s} median {value['median']:+.4f}  IQR "
                  f"[{value['q1']:+.4f}, {value['q3']:+.4f}]")
    print(f"  fraction with |lag| <= 2 cadences: {lag_ok:.2%}")
    print("\n=== TEST 2: local background ===")
    print(f"  {summary['background_field']}")
    if background_available and stat("pearson_correction_background"):
        b = stat("pearson_correction_background")
        print(f"  pearson(correction, background)    median {b['median']:+.4f}  IQR "
              f"[{b['q1']:+.4f}, {b['q3']:+.4f}]")
    print("\n=== TEST 3: high-systematics windows ===")
    for key in ("event_response_ratio", "raw_vs_peercommon_high", "cleaned_vs_peercommon_high"):
        value = summary["test3_high_systematics"][key]
        if value:
            print(f"  {key:34s} median {value['median']:+.4f}")
    print("\n=== TEST 4: peer controls ===")
    print(f"  {'condition':16s} {'recon L1':>9s} {'r(corr,cm)':>11s} {'|lag|':>7s} {'removal':>9s}")
    for name, values in conditions.items():
        print(f"  {name:16s} {values['reconstruction_l1']:9.4f} "
              f"{values['pearson_correction_peercommon']:+11.4f} "
              f"{values['median_abs_lag']:7.1f} {values['removal_fraction']:+9.4f}")
    print("\n=== TEST 5: injection recovery ===")
    print(injection_table.to_string(index=False))
    print(f"\ncriteria: {criteria}")
    print(f"VERDICT: {verdict.upper()} evidence")
    print(f"\noutputs in {out_dir}")


# --------------------------------------------------------------------------- plots
def plot_examples(patch, anchors, peer_rows, corrections, cleaned, per_star, picks,
                  background_available, path, tag):
    picks = [int(p) for p in picks if p < len(per_star)]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 4, figsize=(19, 2.5 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, index in enumerate(picks):
        row = int(per_star.iloc[index]["row"])
        k = anchors.index(row)
        valid = patch.M[row]
        common, common_valid = peer_common_mode(patch, peer_rows[k])
        x = np.arange(len(valid))
        show = lambda a, m: np.where(m, a, np.nan)      # gaps stay gaps, never zero

        axes[r, 0].scatter(x[valid], patch.X[row][valid], s=1.5, color="0.6", linewidths=0,
                           label="raw")
        axes[r, 0].scatter(x[valid], cleaned[k][valid], s=1.5, color="tab:blue",
                           linewidths=0, label="cleaned")
        axes[r, 0].set_ylabel(f"TIC {patch.tic[row]}", fontsize=6)

        axes[r, 1].plot(x, show(corrections[k], valid), lw=0.7, color="tab:red",
                        label="correction")
        axes[r, 1].plot(x, show(common - np.nanmean(common[common_valid]), common_valid),
                        lw=0.7, color="tab:green", label="peer common mode")

        if background_available and np.std(patch.BG[row][valid]) > 0:
            axes[r, 2].plot(x, show(patch.BG[row], valid), lw=0.7, color="tab:purple",
                            label="TGLC background")
        else:
            axes[r, 2].text(0.5, 0.5, "background unavailable", ha="center",
                            transform=axes[r, 2].transAxes, fontsize=8)

        axes[r, 3].fill_between(x, 0, 1, where=~valid, color="0.85", step="mid",
                                label="removed (flag/gap)")
        axes[r, 3].fill_between(x, 0, 1, where=valid, color="tab:blue", step="mid",
                                alpha=0.6, label="valid")
        axes[r, 3].set_ylim(0, 1)
        axes[r, 3].set_yticks([])
        for c in range(4):
            if r == 0:
                axes[r, c].legend(fontsize=6, loc="upper right")
    for c, title in enumerate(["raw vs cleaned", "correction vs peer common mode",
                               "local background", "valid / removed cadences"]):
        axes[0, c].set_title(title, fontsize=9)
    fig.suptitle(f"scattered-light evaluation - {tag} examples "
                 "(removed cadences shown as gaps, never zero)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(per_star, conditions, injection_table, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].hist(per_star["pearson_correction_peercommon"].dropna(), bins=25,
                 color="tab:blue")
    axes[0].axvline(0, color="0.4", lw=0.8)
    axes[0].set_title("r(correction, peer common mode)", fontsize=10)

    axes[1].hist(per_star["xcorr_peak_lag"].dropna(), bins=np.arange(-20.5, 21.5),
                 color="tab:green")
    axes[1].axvline(0, color="0.4", lw=0.8)
    axes[1].set_title("cross-correlation peak lag (cadences)", fontsize=10)

    axes[2].hist(per_star["removal_fraction"].dropna(), bins=25, color="tab:red")
    axes[2].axvline(0, color="0.4", lw=0.8)
    axes[2].set_title("common-mode removal fraction", fontsize=10)

    names = list(conditions)
    axes[3].bar(names, [conditions[n]["pearson_correction_peercommon"] for n in names],
                color=["tab:blue", "0.6", "0.8"])
    axes[3].axhline(0, color="0.4", lw=0.8)
    axes[3].set_title("peer control: r(correction, peer common)", fontsize=10)
    axes[3].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_injections(examples, path):
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 1.9 * len(examples)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (truth, recovered, valid)) in zip(axes, examples.items()):
        x = np.arange(len(truth))
        ax.plot(x, np.where(valid, truth, np.nan), lw=1.1, color="0.4", label="injected")
        ax.plot(x, np.where(valid, recovered, np.nan), lw=0.9, color="tab:orange",
                label="recovered (cleaned_injected - cleaned_original)")
        ax.set_ylabel(name, fontsize=7)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("cadence index")
    fig.suptitle("injection recovery through the cleaning pipeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
