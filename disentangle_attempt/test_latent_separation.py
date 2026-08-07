"""Does the trained model actually separate physics from instrument information?

Eight read-only tests on a frozen checkpoint. Nothing is retrained, no cleaned curve is
produced, and no quiet-reference peers are used. Only the two lightweight probes in
test 5 are fitted, and they never touch the encoders.

The tests are chosen so that each one can fail independently:

  3  physics latents stable across masks        (is the physics latent about the star?)
  5  which latent predicts the peer common mode (where does instrument info live?)
  6  anchor-only injections                     (does physics respond, instrument not?)
  7  peer timing controls                       (is the instrument branch time-aligned?)
  8  latent swaps                               (does each latent move the prediction
                                                 in its own direction?)

    python -m disentangle_attempt.test_latent_separation \
      --checkpoint .../local_s1_c4_ccd2_12px/base_model/best.pt
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from disentangle_attempt.additive_heads import state_hash
from disentangle_attempt.dataset import CrossSectorPatch
from disentangle_attempt.losses import masked_smooth_l1
from disentangle_attempt.masking import complementary_masks, contiguous_hidden_mask
from disentangle_attempt.model import build_model
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

MIN_VALID_PEERS = 4
PCA_DIM = 64
PROBE_EPOCHS = 200
PROBE_PATIENCE = 20


# ------------------------------------------------------------------ small helpers
def safe_pearson(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def safe_spearman(a, b):
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def explained_variance(truth, prediction):
    denom = float(np.var(truth))
    return float(1.0 - np.var(truth - prediction) / denom) if denom > 0 else float("nan")


def cosine_matrix(latents):
    normed = latents / np.clip(np.linalg.norm(latents, axis=1, keepdims=True), 1e-9, None)
    return normed @ normed.T


def peer_common_mode(patch, peer_rows):
    """Common mode of the eight peers. The anchor is never used."""
    flux = patch.X[peer_rows].astype(np.float64)
    mask = patch.M[peer_rows]
    centred = np.where(mask, flux, np.nan) - np.nanmedian(
        np.where(mask, flux, np.nan), axis=1, keepdims=True)
    counts = mask.sum(axis=0)
    valid = counts >= MIN_VALID_PEERS
    common = np.zeros(flux.shape[1])
    if valid.any():
        common[valid] = np.nanmedian(centred[:, valid], axis=0)
        common[valid] -= np.median(common[valid])
    return np.nan_to_num(common).astype(np.float32), valid


# ------------------------------------------------------------------ latent tables
@torch.no_grad()
def extract(model, patch, rows, peer_rows, mask, device, batch=32):
    """Physics latent under ONE fixed mask, plus the concatenated instrument context."""
    physics = np.zeros((len(rows), model.physics_out_dim), dtype=np.float32)
    instrument = np.zeros((len(rows), model.instrument_out_dim), dtype=np.float32)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        peers = peer_rows[start:start + batch]
        raw = torch.from_numpy(patch.X[chunk]).to(device)
        valid = torch.from_numpy(patch.M[chunk]).to(device)
        hidden = mask.to(device).unsqueeze(0).expand(len(chunk), -1)
        physics[start:start + len(chunk)] = model.physics_vector(
            raw.masked_fill(hidden, 0.0), valid & ~hidden).cpu().numpy()
        _, context = model.encode_peers(torch.from_numpy(patch.X[peers]).to(device),
                                        torch.from_numpy(patch.M[peers]).to(device))
        instrument[start:start + len(chunk)] = context.cpu().numpy()
    return physics, instrument


# --------------------------------------------------------------------- test 3
@torch.no_grad()
def mask_invariance(model, patch, rows, device, n_masks=8, hidden_fraction=0.25, seed=0):
    """Same star under 8 random masks vs different stars: cosine similarity."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.from_numpy(patch.X[rows]).to(device)
    valid = torch.from_numpy(patch.M[rows]).to(device)
    stack = []
    for _ in range(n_masks):
        hidden = contiguous_hidden_mask(valid.cpu(), hidden_fraction,
                                        generator=generator).to(device)
        stack.append(model.physics_vector(raw.masked_fill(hidden, 0.0),
                                          valid & ~hidden).cpu().numpy())
    stack = np.stack(stack)                                   # [n_masks, N, 512]

    same, records = [], []
    for k in range(len(rows)):
        similarity = cosine_matrix(stack[:, k, :])
        upper = similarity[np.triu_indices(n_masks, k=1)]
        same.extend(upper.tolist())
        records.append({"tic": patch.tic[rows[k]], "same_star_cosine_median": float(np.median(upper))})

    reference = stack[0]                                      # one mask per star
    across = cosine_matrix(reference)
    different = across[np.triu_indices(len(rows), k=1)]

    # Retrieval: for each (star, mask) latent, is the nearest OTHER latent the same star?
    flat = stack.reshape(-1, stack.shape[-1])
    labels = np.tile(np.arange(len(rows)), n_masks)
    similarity = cosine_matrix(flat)
    np.fill_diagonal(similarity, -np.inf)
    nearest = similarity.argmax(axis=1)
    retrieval = float((labels[nearest] == labels).mean())
    return np.asarray(same), np.asarray(different), retrieval, pd.DataFrame(records)


# --------------------------------------------------------------------- test 5
def fit_probe(features, targets, target_valid, splits, seed=0):
    """Linear(64, 1024) on PCA features; identical settings for both latent types."""
    torch.manual_seed(seed)
    train, val, test = (splits == "train"), (splits == "val"), (splits == "test")
    scaler = StandardScaler().fit(features[train])
    pca = PCA(n_components=PCA_DIM, random_state=seed).fit(scaler.transform(features[train]))
    reduced = torch.from_numpy(pca.transform(scaler.transform(features)).astype(np.float32))
    y = torch.from_numpy(targets)
    m = torch.from_numpy(target_valid)

    probe = torch.nn.Linear(PCA_DIM, targets.shape[1])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    best = {"val": float("inf"), "state": None}
    since = 0
    for _ in range(PROBE_EPOCHS):
        probe.train()
        loss = masked_smooth_l1(probe(reduced[train]), y[train], m[train])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_loss = float(masked_smooth_l1(probe(reduced[val]), y[val], m[val]))
        if val_loss < best["val"]:
            best = {"val": val_loss, "state": {k: v.clone() for k, v in probe.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= PROBE_PATIENCE:
                break
    probe.load_state_dict(best["state"])
    probe.eval()
    with torch.no_grad():
        prediction = probe(reduced).numpy()
    return prediction, best["val"], pca.explained_variance_ratio_.sum()


def probe_metrics(prediction, targets, target_valid, mask, label):
    rows = []
    for k in np.flatnonzero(mask):
        pick = target_valid[k]
        if pick.sum() < 32:
            continue
        rows.append({
            "probe": label,
            "pearson": safe_pearson(prediction[k][pick], targets[k][pick]),
            "spearman": safe_spearman(prediction[k][pick], targets[k][pick]),
            "explained_variance": explained_variance(targets[k][pick], prediction[k][pick]),
        })
    frame = pd.DataFrame(rows)
    smooth = float(masked_smooth_l1(torch.from_numpy(prediction[mask]),
                                    torch.from_numpy(targets[mask]),
                                    torch.from_numpy(target_valid[mask])))
    return frame, smooth


# --------------------------------------------------------------------- test 6
def box_transit(length, centre, duration, depth):
    x = np.arange(length)
    return (-depth * (np.abs(x - centre) <= duration / 2)).astype(np.float32)


def gaussian_flare(length, centre, width, amplitude):
    x = np.arange(length)
    return (amplitude * np.exp(-0.5 * ((x - centre) / width) ** 2)).astype(np.float32)


def sinusoid(length, period, amplitude):
    return (amplitude * np.sin(2 * np.pi * np.arange(length) / period)).astype(np.float32)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-inject", type=int, default=25)
    parser.add_argument("--n-pairs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output_dir or os.path.join(run_dir, "latent_separation_tests")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    assert (config.get("sector"), config.get("camera"), config.get("ccd")) == (1, 4, 2), \
        "not the local Sector 1 / camera 4 / CCD 2 model"
    assert float(config.get("peer_min_distance_px", -1)) == 12.0, \
        "checkpoint was not trained with the 12 px minimum peer distance"

    model = build_model(config).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    hashes_before = {"physics_s4d": state_hash(model.physics_encoder),
                     "instrument_s4d": state_hash(model.instrument_encoder),
                     "decoder": state_hash(model.decoder)}

    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=config["sector"], camera=config["camera"], ccd=config["ccd"],
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        peer_min_distance=config["peer_min_distance_px"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        verbose=False)
    masks = complementary_masks(config["curve_length"], n_masks=4)

    rows, splits = [], []
    for split in ("train", "val", "test"):
        for anchor in patch.split_anchors[split]:
            rows.append(int(anchor))
            splits.append(split)
    rows = np.asarray(rows)
    splits = np.asarray(splits)
    peer_rows = np.stack([patch.peers_for_row(int(r), s)[0] for r, s in zip(rows, splits)])
    peer_distances = np.stack([patch.peers_for_row(int(r), s)[1] for r, s in zip(rows, splits)])

    print("=" * 72)
    print(f"checkpoint            {os.path.abspath(args.checkpoint)}")
    print(f"sector/camera/CCD     {config['sector']} / {config['camera']} / {config['ccd']}")
    print(f"peer_min_distance_px  {config['peer_min_distance_px']}  |  n_peers "
          f"{config['n_peers']}")
    print(f"anchors               train {int((splits=='train').sum())}  "
          f"val {int((splits=='val').sum())}  test {int((splits=='test').sum())}")
    print(f"physics latent        [{model.physics_out_dim}]")
    print(f"instrument latent     [{model.instrument_out_dim}] "
          f"({model.n_peers} x {model.peer_out_dim})")
    print(f"checkpoint epoch      {state.get('epoch')}")
    print("=" * 72, flush=True)

    with torch.no_grad():
        physics, instrument = extract(model, patch, rows, peer_rows, masks[0], device)
    common = np.zeros((len(rows), patch.curve_length), dtype=np.float32)
    common_valid = np.zeros((len(rows), patch.curve_length), dtype=bool)
    for k in range(len(rows)):
        common[k], common_valid[k] = peer_common_mode(patch, peer_rows[k])
    np.savez(os.path.join(out_dir, "latents.npz"), physics=physics, instrument=instrument,
             rows=rows, splits=splits, tics=patch.tic[rows], peer_rows=peer_rows,
             peer_distances=peer_distances, common=common, common_valid=common_valid,
             detector_x=patch.det_x[rows], detector_y=patch.det_y[rows])

    test_mask = splits == "test"
    test_rows = rows[test_mask]
    summary = {"checkpoint": os.path.abspath(args.checkpoint),
               "config": {k: config.get(k) for k in
                          ("sector", "camera", "ccd", "peer_min_distance_px", "n_peers",
                           "curve_length", "d_model", "seed")},
               "epoch": state.get("epoch"),
               "anchors": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
               "hashes_before": hashes_before}

    # ---------------------------------------------------------------- test 3
    with torch.no_grad():
        same, different, retrieval, invariance_table = mask_invariance(
            model, patch, test_rows, device, seed=args.seed)
    invariance_table.to_csv(os.path.join(out_dir, "per_star_mask_invariance.csv"), index=False)
    quart = lambda a: (float(np.median(a)), float(np.quantile(a, .25)), float(np.quantile(a, .75)))
    same_stats, different_stats = quart(same), quart(different)
    summary["test3_mask_invariance"] = {
        "same_star_cosine": {"median": same_stats[0], "q1": same_stats[1], "q3": same_stats[2]},
        "different_star_cosine": {"median": different_stats[0], "q1": different_stats[1],
                                  "q3": different_stats[2]},
        "difference": same_stats[0] - different_stats[0],
        "same_tic_retrieval_accuracy": retrieval}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(same, bins=40, alpha=0.65, color="tab:blue", density=True, label="same star, 8 masks")
    ax.hist(different, bins=40, alpha=0.65, color="0.6", density=True, label="different stars")
    ax.set_xlabel("cosine similarity between physics latents")
    ax.set_title(f"physics mask invariance: same {same_stats[0]:.3f} vs different "
                 f"{different_stats[0]:.3f}, retrieval {retrieval:.1%}", fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, "physics_mask_invariance.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] mask invariance: same {same_stats[0]:.4f} | different {different_stats[0]:.4f} "
          f"| diff {same_stats[0]-different_stats[0]:+.4f} | retrieval {retrieval:.1%}", flush=True)

    # ---------------------------------------------------------------- test 5
    probe_rows, predictions = [], {}
    for label, features in (("physics", physics), ("instrument", instrument)):
        prediction, val_loss, retained = fit_probe(features, common, common_valid, splits,
                                                   seed=args.seed)
        frame, smooth = probe_metrics(prediction, common, common_valid, test_mask, label)
        predictions[label] = prediction
        probe_rows.append({"probe": label, "test_smooth_l1": smooth,
                           "val_smooth_l1": val_loss, "pca_retained_variance": float(retained),
                           "median_pearson": float(frame["pearson"].median()),
                           "median_spearman": float(frame["spearman"].median()),
                           "median_explained_variance": float(frame["explained_variance"].median())})
    mean_curve = common[splits == "train"].mean(axis=0)
    baseline = np.tile(mean_curve, (len(rows), 1))
    frame, smooth = probe_metrics(baseline, common, common_valid, test_mask, "mean_baseline")
    predictions["mean_baseline"] = baseline
    probe_rows.append({"probe": "mean_baseline", "test_smooth_l1": smooth, "val_smooth_l1": None,
                       "pca_retained_variance": None,
                       "median_pearson": float(frame["pearson"].median()),
                       "median_spearman": float(frame["spearman"].median()),
                       "median_explained_variance": float(frame["explained_variance"].median())})
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(os.path.join(out_dir, "probe_metrics.csv"), index=False)
    summary["test5_common_mode_probes"] = probes.to_dict("records")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, column, title in ((axes[0], "test_smooth_l1", "test masked Smooth-L1 (lower better)"),
                              (axes[1], "median_pearson", "median per-star Pearson"),
                              (axes[2], "median_explained_variance", "median explained variance")):
        ax.bar(probes["probe"], probes[column],
               color=["tab:blue", "tab:red", "0.6"][:len(probes)])
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "common_mode_probe_comparison.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    picks = np.flatnonzero(test_mask)[:4]
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 2.3 * len(picks)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, picks):
        v = common_valid[k]
        x = np.arange(patch.curve_length)
        ax.plot(x, np.where(v, common[k], np.nan), lw=0.9, color="0.4", label="peer common mode")
        ax.plot(x, np.where(v, predictions["instrument"][k], np.nan), lw=0.8,
                color="tab:blue", label="instrument probe")
        ax.plot(x, np.where(v, predictions["physics"][k], np.nan), lw=0.8,
                color="tab:red", label="physics probe")
        ax.set_ylabel(f"TIC {patch.tic[rows[k]]}", fontsize=7)
    axes[0].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("cadence index")
    fig.suptitle("predicting the peer common mode from each frozen latent", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "common_mode_probe_examples.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[5] probes: {probes[['probe','test_smooth_l1','median_pearson']].to_dict('records')}",
          flush=True)

    # ---------------------------------------------------------------- test 6
    inject_rows = test_rows[:args.n_inject]
    inject_peers = np.stack([patch.peers_for_row(int(r), "test")[0] for r in inject_rows])
    length = patch.curve_length
    signals = {}
    for amplitude in (0.5, 1.0):
        signals[f"transit_a{amplitude}"] = box_transit(length, 0.45 * length, 24, amplitude)
        signals[f"flare_a{amplitude}"] = gaussian_flare(length, 0.30 * length, 6, amplitude)
        signals[f"sinusoid_a{amplitude}"] = sinusoid(length, 120, amplitude)

    injection_rows = []
    with torch.no_grad():
        raw = torch.from_numpy(patch.X[inject_rows]).to(device)
        valid = torch.from_numpy(patch.M[inject_rows]).to(device)
        hidden = masks[0].to(device).unsqueeze(0).expand(len(inject_rows), -1)
        base_physics = model.physics_vector(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        _, base_context = model.encode_peers(
            torch.from_numpy(patch.X[inject_peers]).to(device),
            torch.from_numpy(patch.M[inject_peers]).to(device))
        base_prediction = model.decoder(torch.cat([base_physics, base_context], dim=-1))

        for name, signal in signals.items():
            signal_t = torch.from_numpy(signal).to(device)
            injected = raw + signal_t
            physics_injected = model.physics_vector(injected.masked_fill(hidden, 0.0),
                                                    valid & ~hidden)
            # peers untouched -> the instrument context must be bit-identical
            _, context_injected = model.encode_peers(
                torch.from_numpy(patch.X[inject_peers]).to(device),
                torch.from_numpy(patch.M[inject_peers]).to(device))
            instrument_change = float(torch.norm(context_injected - base_context, dim=1).max())
            assert instrument_change == 0.0, \
                f"instrument latent moved by {instrument_change} with peers unchanged"
            prediction = model.decoder(torch.cat([physics_injected, base_context], dim=-1))
            delta = (prediction - base_prediction).cpu().numpy()
            cosine = torch.nn.functional.cosine_similarity(base_physics, physics_injected,
                                                           dim=-1).cpu().numpy()
            l2 = torch.norm(physics_injected - base_physics, dim=1).cpu().numpy()
            for k in range(len(inject_rows)):
                v = patch.M[inject_rows[k]]
                truth = signal[v]
                injection_rows.append({
                    "signal": name, "tic": patch.tic[inject_rows[k]],
                    "physics_cosine_distance": float(1 - cosine[k]),
                    "physics_l2_change": float(l2[k]),
                    "instrument_l2_change": instrument_change,
                    "corr_delta_vs_injection": safe_pearson(delta[k][v], truth),
                    "recovered_amplitude": float(np.dot(delta[k][v], truth) / max(np.dot(truth, truth), 1e-9)),
                })
    injections = pd.DataFrame(injection_rows)
    injections.to_csv(os.path.join(out_dir, "injection_metrics.csv"), index=False)
    grouped = injections.groupby("signal").median(numeric_only=True)
    summary["test6_injection"] = grouped.to_dict("index")
    summary["test6_instrument_latent_unchanged"] = bool(
        (injections["instrument_l2_change"] == 0).all())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].bar(grouped.index, grouped["physics_l2_change"], color="tab:blue",
                label="physics latent L2 change")
    axes[0].bar(grouped.index, grouped["instrument_l2_change"], color="tab:red",
                label="instrument latent L2 change")
    axes[0].set_title("latent response to anchor-only injections", fontsize=9)
    axes[0].tick_params(axis="x", rotation=60, labelsize=7)
    axes[0].legend(fontsize=8)
    axes[1].bar(grouped.index, grouped["corr_delta_vs_injection"], color="tab:green")
    axes[1].set_title("corr(decoder prediction change, injected signal)", fontsize=9)
    axes[1].tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "injection_latent_response.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(grouped.index, grouped["recovered_amplitude"], color="tab:purple")
    ax.axhline(1.0, color="0.3", ls="--", lw=0.8, label="full recovery")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_ylabel("recovered amplitude (projection onto the injected signal)")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "injection_recovery.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[6] injection: physics L2 median "
          f"{injections['physics_l2_change'].median():.4f}, instrument L2 "
          f"{injections['instrument_l2_change'].max():.1e}, corr "
          f"{injections['corr_delta_vs_injection'].median():+.4f}", flush=True)

    # ---------------------------------------------------------------- test 7
    control_rows = []
    with torch.no_grad():
        raw = torch.from_numpy(patch.X[test_rows]).to(device)
        valid = torch.from_numpy(patch.M[test_rows]).to(device)
        hidden = masks[0].to(device).unsqueeze(0).expand(len(test_rows), -1)
        physics_latent = model.physics_vector(raw.masked_fill(hidden, 0.0), valid & ~hidden)
        test_peer_rows = np.stack([patch.peers_for_row(int(r), "test")[0] for r in test_rows])
        shift = int(rng.integers(200, patch.curve_length - 200))
        pool = patch.split_pool["test"][patch.chips[0]]
        random_peers = np.stack([
            rng.choice(pool[patch.tic[pool] != patch.tic[r]], size=model.n_peers,
                       replace=False) for r in test_rows])
        conditions = {
            "actual": (patch.X[test_peer_rows], patch.M[test_peer_rows]),
            "time_shifted": (np.roll(patch.X[test_peer_rows], shift, axis=2),
                             np.roll(patch.M[test_peer_rows], shift, axis=2)),
            "random_same_chip": (patch.X[random_peers], patch.M[random_peers]),
        }
        for name, (flux, mask_array) in conditions.items():
            _, context = model.encode_peers(torch.from_numpy(flux).to(device),
                                            torch.from_numpy(mask_array).to(device))
            prediction = model.decoder(torch.cat([physics_latent, context], dim=-1))
            loss = float(masked_smooth_l1(prediction, raw, hidden & valid))
            control_rows.append({"condition": name, "masked_smooth_l1": loss})
    controls = pd.DataFrame(control_rows)
    controls.to_csv(os.path.join(out_dir, "peer_control_metrics.csv"), index=False)
    summary["test7_peer_controls"] = controls.set_index("condition")["masked_smooth_l1"].to_dict()
    summary["test7_time_shift_cadences"] = shift
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(controls["condition"], controls["masked_smooth_l1"],
           color=["tab:blue", "0.6", "0.75"])
    ax.set_ylabel("masked Smooth-L1 (lower better)")
    ax.set_title(f"peer controls on held-out test anchors (shift {shift} cadences)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "peer_control_losses.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[7] peer controls: {summary['test7_peer_controls']}", flush=True)

    # ---------------------------------------------------------------- test 8
    test_indices = np.flatnonzero(test_mask)
    pairs = [(int(a), int(b)) for a, b in
             zip(rng.choice(test_indices, args.n_pairs), rng.choice(test_indices, args.n_pairs))
             if a != b]
    swap_rows = []
    with torch.no_grad():
        physics_t = torch.from_numpy(physics).to(device)
        instrument_t = torch.from_numpy(instrument).to(device)
        for a, b in pairs:
            za, zb = physics_t[a:a + 1], physics_t[b:b + 1]
            ia, ib = instrument_t[a:a + 1], instrument_t[b:b + 1]
            base = model.decoder(torch.cat([za, ia], dim=-1))[0].cpu().numpy()
            instrument_swap = model.decoder(torch.cat([za, ib], dim=-1))[0].cpu().numpy()
            physics_swap = model.decoder(torch.cat([zb, ia], dim=-1))[0].cpu().numpy()
            v = patch.M[rows[a]] & patch.M[rows[b]] & common_valid[a] & common_valid[b]
            if v.sum() < 64:
                continue
            common_delta = (common[b] - common[a])[v]
            instrument_delta = (instrument_swap - base)[v]
            proxy_delta = ((patch.X[rows[b]] - common[b]) - (patch.X[rows[a]] - common[a]))[v]
            physics_delta = (physics_swap - base)[v]
            swap_rows.append({
                "a": patch.tic[rows[a]], "b": patch.tic[rows[b]],
                "corr_instrument_swap_vs_common_delta": safe_pearson(instrument_delta, common_delta),
                "corr_physics_swap_vs_proxy_delta": safe_pearson(physics_delta, proxy_delta),
                "instrument_swap_rms": float(np.sqrt((instrument_delta ** 2).mean())),
                "physics_swap_rms": float(np.sqrt((physics_delta ** 2).mean())),
            })
    swaps = pd.DataFrame(swap_rows)
    swaps.to_csv(os.path.join(out_dir, "latent_swap_metrics.csv"), index=False)
    summary["test8_latent_swap"] = {
        "n_pairs": int(len(swaps)),
        "median_corr_instrument_swap_vs_common_delta":
            float(swaps["corr_instrument_swap_vs_common_delta"].median()),
        "median_corr_physics_swap_vs_proxy_delta":
            float(swaps["corr_physics_swap_vs_proxy_delta"].median()),
        "median_instrument_swap_rms": float(swaps["instrument_swap_rms"].median()),
        "median_physics_swap_rms": float(swaps["physics_swap_rms"].median())}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(swaps["corr_instrument_swap_vs_common_delta"].dropna(), bins=30,
                 color="tab:blue")
    axes[0].axvline(0, color="0.4", lw=0.8)
    axes[0].set_title("instrument swap vs Δ peer common mode", fontsize=9)
    axes[1].hist(swaps["corr_physics_swap_vs_proxy_delta"].dropna(), bins=30, color="tab:red")
    axes[1].axvline(0, color="0.4", lw=0.8)
    axes[1].set_title("physics swap vs Δ (raw − common mode)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latent_swap_metrics.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with torch.no_grad():
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        a, b = pairs[0]
        za, zb = physics_t[a:a + 1], physics_t[b:b + 1]
        ia, ib = instrument_t[a:a + 1], instrument_t[b:b + 1]
        base = model.decoder(torch.cat([za, ia], dim=-1))[0].cpu().numpy()
        swap_i = model.decoder(torch.cat([za, ib], dim=-1))[0].cpu().numpy()
        swap_p = model.decoder(torch.cat([zb, ia], dim=-1))[0].cpu().numpy()
        v = patch.M[rows[a]]
        x = np.arange(patch.curve_length)
        axes[0].plot(x, np.where(v, base, np.nan), lw=0.8, color="0.4", label="base(A,A)")
        axes[0].plot(x, np.where(v, swap_i, np.nan), lw=0.8, color="tab:blue",
                     label="instrument swap (A physics, B peers)")
        axes[0].legend(fontsize=8)
        axes[1].plot(x, np.where(v, swap_i - base, np.nan), lw=0.8, color="tab:blue",
                     label="Δ prediction")
        axes[1].plot(x, np.where(v, common[b] - common[a], np.nan), lw=0.8,
                     color="tab:olive", label="Δ peer common mode")
        axes[1].legend(fontsize=8)
        axes[2].plot(x, np.where(v, swap_p - base, np.nan), lw=0.8, color="tab:red",
                     label="Δ prediction (physics swap)")
        axes[2].plot(x, np.where(v, (patch.X[rows[b]] - common[b]) -
                                 (patch.X[rows[a]] - common[a]), np.nan), lw=0.8,
                     color="tab:purple", label="Δ physics proxy")
        axes[2].legend(fontsize=8)
        axes[2].set_xlabel("cadence index")
        fig.suptitle(f"latent swap example: A=TIC {patch.tic[rows[a]]}, B=TIC {patch.tic[rows[b]]}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "latent_swap_examples.png"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"[8] latent swap: {summary['test8_latent_swap']}", flush=True)

    # ---------------------------------------------------------------- verdict
    hashes_after = {"physics_s4d": state_hash(model.physics_encoder),
                    "instrument_s4d": state_hash(model.instrument_encoder),
                    "decoder": state_hash(model.decoder)}
    assert hashes_before == hashes_after, "model weights changed during evaluation"
    summary["hashes_after"] = hashes_after
    summary["weights_unchanged"] = True

    physics_probe = probes.set_index("probe").loc["physics"]
    instrument_probe = probes.set_index("probe").loc["instrument"]
    baseline_probe = probes.set_index("probe").loc["mean_baseline"]
    criteria = {
        "physics_stable_across_masks": bool(same_stats[0] > different_stats[0]),
        "instrument_probe_beats_physics": bool(
            instrument_probe["median_pearson"] > physics_probe["median_pearson"]
            and instrument_probe["test_smooth_l1"] < physics_probe["test_smooth_l1"]),
        "instrument_probe_beats_baseline": bool(
            instrument_probe["test_smooth_l1"] < baseline_probe["test_smooth_l1"]),
        "physics_responds_to_injection": bool(injections["physics_l2_change"].median() > 1e-3),
        "instrument_unchanged_by_injection": bool((injections["instrument_l2_change"] == 0).all()),
        "actual_peers_beat_controls": bool(
            summary["test7_peer_controls"]["actual"]
            < min(summary["test7_peer_controls"]["time_shifted"],
                  summary["test7_peer_controls"]["random_same_chip"])),
        "instrument_swap_follows_common_mode": bool(
            summary["test8_latent_swap"]["median_corr_instrument_swap_vs_common_delta"] > 0),
        "physics_swap_follows_proxy": bool(
            summary["test8_latent_swap"]["median_corr_physics_swap_vs_proxy_delta"] > 0),
    }
    passed = sum(criteria.values())
    verdict = ("SUPPORTED" if passed == len(criteria)
               else "PARTIAL" if passed >= len(criteria) - 2 else "NOT SUPPORTED")
    summary["criteria"] = criteria
    summary["verdict"] = verdict
    summary["leakage_summary"] = {
        "A_physics_mask_invariance_difference": same_stats[0] - different_stats[0],
        "B_instrument_common_mode_pearson": float(instrument_probe["median_pearson"]),
        "C_physics_common_mode_leakage_pearson": float(physics_probe["median_pearson"]),
        "D_physics_injection_l2": float(injections["physics_l2_change"].median()),
        "D_instrument_injection_l2": float(injections["instrument_l2_change"].max()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print("\n" + "=" * 72)
    print(f"A physics mask invariance (same - different): "
          f"{summary['leakage_summary']['A_physics_mask_invariance_difference']:+.4f}")
    print(f"B instrument -> common mode  Pearson {instrument_probe['median_pearson']:+.4f} "
          f"| Smooth-L1 {instrument_probe['test_smooth_l1']:.4f}")
    print(f"C physics    -> common mode  Pearson {physics_probe['median_pearson']:+.4f} "
          f"| Smooth-L1 {physics_probe['test_smooth_l1']:.4f}   (leakage)")
    print(f"  mean baseline              Pearson {baseline_probe['median_pearson']:+.4f} "
          f"| Smooth-L1 {baseline_probe['test_smooth_l1']:.4f}")
    print(f"D injection: physics L2 {injections['physics_l2_change'].median():.4f}, "
          f"instrument L2 {injections['instrument_l2_change'].max():.1e}")
    for name, value in criteria.items():
        print(f"  {'PASS' if value else 'FAIL'}  {name}")
    print(f"VERDICT: {verdict}")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
