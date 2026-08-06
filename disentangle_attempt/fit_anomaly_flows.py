"""Density-based anomaly scores over the two latent spaces of a frozen checkpoint.

Two unconditional Neural Spline Flows are fit -- one on the physics latent, one on the
instrument latent -- and negative log-likelihood becomes the anomaly score. Nothing
about the trained model is touched: encoders, decoder, masking, preprocessing and the
UMAP embedding are all read-only inputs here.

Method follows the configuration described in the request (arXiv:2604.09787 Sec. 4.1 /
App. A.7). The paper itself was not consulted, so this implements the specification as
given rather than a verified reproduction.

Two things this deliberately does NOT do:
  * fit anything on UMAP coordinates -- UMAP is a visualization, and its distances are
    not a density;
  * compare physics and instrument NLL directly -- 32 and 16 dimensions put them on
    different scales, so all comparisons go through training-set percentiles.

    ~/tess-venv/bin/python -m disentangle_attempt.fit_anomaly_flows \
      --latents-dir disentangle_attempt/outputs/fast_strict/umaps \
      --checkpoint disentangle_attempt/outputs/fast_strict/best.pt \
      --output-dir disentangle_attempt/outputs/fast_strict/anomaly_flows --seed 42
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from disentangle_attempt.dataset import (CrossSectorPatch,
                                        infer_require_cross_sector,
                                        target_from_checkpoint)
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import load_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

PHYSICS_DIM, INSTRUMENT_DIM = 32, 16
TRANSFORMS, HIDDEN, EPOCHS, LR = 6, (64, 64), 50, 1e-3
THRESHOLD = 0.95


# ------------------------------------------------------------------- flow fitting
def fit_flow(train, val, features, seed, label, batch_size=512):
    """Unconditional NSF; validation NLL picks the checkpoint."""
    import zuko
    torch.manual_seed(seed)
    flow = zuko.flows.NSF(features=features, context=0, transforms=TRANSFORMS,
                          hidden_features=HIDDEN)
    optimizer = torch.optim.Adam(flow.parameters(), lr=LR)
    train_t = torch.as_tensor(train, dtype=torch.float32)
    val_t = torch.as_tensor(val, dtype=torch.float32)
    batch = min(int(batch_size), len(train_t))

    best = {"val_nll": float("inf"), "epoch": -1, "state": None, "train_nll": None}
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(EPOCHS):
        flow.train()
        order = torch.randperm(len(train_t), generator=generator)
        losses = []
        for start in range(0, len(order), batch):
            chunk = train_t[order[start:start + batch]]
            loss = -flow().log_prob(chunk).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        flow.eval()
        with torch.no_grad():
            val_nll = float(-flow().log_prob(val_t).mean())
        train_nll = float(np.mean(losses))
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        if val_nll < best["val_nll"]:
            best = {"val_nll": val_nll, "epoch": epoch, "train_nll": train_nll,
                    "state": {k: v.clone() for k, v in flow.state_dict().items()}}
        if epoch % 10 == 0:
            print(f"  {label} epoch {epoch:2d}: train {train_nll:8.3f}  val {val_nll:8.3f}",
                  flush=True)
    flow.load_state_dict(best["state"])
    flow.eval()
    print(f"  {label}: best epoch {best['epoch']} train {best['train_nll']:.3f} "
          f"val {best['val_nll']:.3f}", flush=True)
    return flow, best, pd.DataFrame(history)


@torch.no_grad()
def nll(flow, values, batch=1024):
    out = []
    for start in range(0, len(values), batch):
        chunk = torch.as_tensor(values[start:start + batch], dtype=torch.float32)
        out.append((-flow().log_prob(chunk)).numpy())
    return np.concatenate(out)


def percentile_against(scores, reference):
    """Fraction of the TRAIN distribution each score exceeds."""
    reference = np.sort(reference)
    return np.searchsorted(reference, scores, side="right") / len(reference)


# ------------------------------------------------------------------------- plots
def umap_anomaly_plot(table, xcol, ycol, pcol, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    order = np.argsort(table[pcol].to_numpy())            # anomalies drawn on top
    art = ax.scatter(table[xcol].to_numpy()[order], table[ycol].to_numpy()[order],
                     c=table[pcol].to_numpy()[order], s=7, cmap="viridis",
                     vmin=0, vmax=1, linewidths=0)
    high = table[pcol] >= THRESHOLD
    ax.scatter(table.loc[high, xcol], table.loc[high, ycol], s=45,
               facecolors="none", edgecolors="red", linewidths=0.8,
               label=f"percentile >= {THRESHOLD}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.colorbar(art, ax=ax, fraction=0.04, label="anomaly percentile")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def comparison_plot(table, path):
    colours = {"physics": "tab:red", "instrument": "tab:blue", "both": "tab:purple",
               "typical": "0.7"}
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name, colour in colours.items():
        pick = table["classification"] == name
        ax.scatter(table.loc[pick, "instrument_percentile"],
                   table.loc[pick, "physics_percentile"], s=8, color=colour,
                   label=f"{name} ({int(pick.sum())})", linewidths=0)
    ax.axvline(THRESHOLD, color="0.3", lw=0.8, ls="--")
    ax.axhline(THRESHOLD, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("instrument anomaly percentile")
    ax.set_ylabel("physics anomaly percentile")
    ax.set_title("four interpretation regions", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def nll_distribution_plot(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, column, title in ((axes[0], "physics_nll", "physics flow"),
                              (axes[1], "instrument_nll", "instrument flow")):
        for split, colour in (("train", "0.6"), ("val", "tab:orange"), ("test", "tab:blue")):
            values = table.loc[table["split"] == split, column]
            if len(values):
                ax.hist(values, bins=40, histtype="step", color=colour, label=split,
                        density=True)
        ax.set_xlabel("negative log-likelihood")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def gallery(patch, rows, curves, table_rows, kind, path):
    """Candidate curves. Removed cadences are gaps, never zeros."""
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"no {kind} candidates in the test split", ha="center")
        ax.axis("off")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, row, meta in zip(axes, rows, table_rows):
        valid = patch.M[row]
        x = np.arange(len(valid))
        show = lambda a: np.where(valid, a, np.nan)
        ax.plot(x, show(patch.X[row]), lw=0.7, color="0.6", label="raw")
        # curves is empty when reference_context.pt does not exist yet: raw only.
        drawn = curves.get(row)
        if drawn and kind in ("physics", "both"):
            ax.plot(x, show(drawn["cleaned"]), lw=0.8, color="tab:blue", label="cleaned")
        if drawn and kind in ("instrument", "both"):
            ax.plot(x, show(drawn["correction"]), lw=0.8, color="tab:red",
                    label="correction")
        if drawn and kind == "instrument":
            ax.plot(x, show(drawn["peer_common"]), lw=0.8, color="tab:green",
                    label="peer common mode")
        ax.set_ylabel(f"TIC {meta['TIC']}\nphys {meta['physics_percentile']:.3f}\n"
                      f"inst {meta['instrument_percentile']:.3f}", fontsize=6)
    axes[0].legend(fontsize=7, ncol=4, loc="upper right")
    axes[-1].set_xlabel("cadence index (gaps = removed cadences)")
    note = "" if curves else "  (raw only: reference context not written until training ends)"
    fig.suptitle(f"top {kind} candidates (test split){note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    physics = np.load(os.path.join(args.latents_dir, "physics_latents.npy"))
    instrument = np.load(os.path.join(args.latents_dir, "instrument_latents.npy"))
    table = pd.read_csv(os.path.join(args.latents_dir, "umap_coordinates.csv"))
    assert len(physics) == len(instrument) == len(table), \
        "latent arrays and metadata disagree in length -- not the same examples"
    assert physics.shape[1] == instrument.shape[1] == 512
    assert np.isfinite(physics).all() and np.isfinite(instrument).all()
    print(f"{len(table)} examples; splits: "
          f"{table['split'].value_counts().to_dict()}", flush=True)

    is_train = (table["split"] == "train").to_numpy()
    is_val = (table["split"] == "val").to_numpy()
    assert is_train.sum() and is_val.sum(), "need train and val examples"

    # Scalers and PCA see TRAIN ONLY; val/test are transformed, never fitted.
    prepared = {}
    for name, latents, dim in (("physics", physics, PHYSICS_DIM),
                               ("instrument", instrument, INSTRUMENT_DIM)):
        scaler = StandardScaler().fit(latents[is_train])
        pca = PCA(n_components=dim, random_state=args.seed).fit(scaler.transform(latents[is_train]))
        values = pca.transform(scaler.transform(latents))
        joblib.dump(scaler, os.path.join(args.output_dir, f"{name}_scaler.joblib"))
        joblib.dump(pca, os.path.join(args.output_dir, f"{name}_pca.joblib"))
        prepared[name] = values
        print(f"{name}: 512 -> PCA {dim}D, explained variance "
              f"{pca.explained_variance_ratio_.sum():.3f}", flush=True)

    results, flow_reports = {}, {}
    for name in ("physics", "instrument"):
        values = prepared[name]
        print(f"fitting {name} flow", flush=True)
        flow, best, history = fit_flow(values[is_train], values[is_val],
                                       values.shape[1], args.seed, name)
        history.to_csv(os.path.join(args.output_dir, f"{name}_flow_history.csv"),
                       index=False)
        torch.save({"state_dict": flow.state_dict(), "features": values.shape[1],
                    "transforms": TRANSFORMS, "hidden_features": list(HIDDEN),
                    "best_epoch": best["epoch"], "val_nll": best["val_nll"]},
                   os.path.join(args.output_dir, f"{name}_flow.pt"))
        scores = nll(flow, values)
        assert np.isfinite(scores).all(), f"{name}: non-finite NLL"
        results[name] = scores
        tail = history.tail(10)["val_nll"].to_numpy()
        flow_reports[name] = {
            "best_epoch": best["epoch"], "best_train_nll": best["train_nll"],
            "best_val_nll": best["val_nll"],
            "train_nll_diverging": bool(not np.isfinite(history["train_nll"]).all()
                                        or history["train_nll"].iloc[-1]
                                        > history["train_nll"].iloc[0]),
            "val_nll_monotonically_worsening": bool(len(tail) > 1
                                                    and np.all(np.diff(tail) > 0)),
        }

    table["physics_nll"] = results["physics"]
    table["instrument_nll"] = results["instrument"]
    for name in ("physics", "instrument"):
        table[f"{name}_percentile"] = percentile_against(results[name],
                                                         results[name][is_train])

    def classify(row):
        p = row["physics_percentile"] >= THRESHOLD
        i = row["instrument_percentile"] >= THRESHOLD
        return "both" if p and i else "physics" if p else "instrument" if i else "typical"
    table["classification"] = table.apply(classify, axis=1)

    # Test split ranks first (rank 1 = most anomalous in test), then a global ranking.
    for name in ("physics", "instrument"):
        table[f"{name}_rank"] = np.nan
        test = table["split"] == "test"
        table.loc[test, f"{name}_rank"] = (
            table.loc[test, f"{name}_nll"].rank(ascending=False).astype(int))
        table[f"{name}_rank_all"] = table[f"{name}_nll"].rank(ascending=False).astype(int)

    columns = ["TIC", "split", "sector", "camera", "ccd", "detector_x", "detector_y",
               "physics_nll", "instrument_nll", "physics_percentile",
               "instrument_percentile", "physics_rank", "instrument_rank",
               "physics_rank_all", "instrument_rank_all", "classification",
               "physics_umap_1", "physics_umap_2", "instrument_umap_1", "instrument_umap_2"]
    table[columns].to_csv(os.path.join(args.output_dir, "anomaly_scores.csv"), index=False)

    # ------------------------------------------------------ candidate galleries
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    target = target_from_checkpoint(state, config)
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=target[0], camera=target[1], ccd=target[2],
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(
            config, args.require_cross_sector), verbose=False)
    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    # reference_context.pt is written only when training finishes, so a mid-run
    # checkpoint can still be scored -- just without the cleaned-curve galleries.
    quiet = None
    if os.path.exists(os.path.join(run_dir, "reference_context.pt")):
        quiet = load_reference_context(run_dir,
                                       expected_cadence_ids=patch.grids[patch.target[0]])
    else:
        print("no reference_context.pt yet (training still running): scores and score "
              "plots only, no candidate curve galleries", flush=True)
    masks = complementary_masks(config["curve_length"], n_masks=4)

    tic_to_row = {}
    for split in ("train", "val", "test"):
        for anchor in patch.split_anchors[split]:
            tic_to_row.setdefault(patch.tic[int(anchor)], int(anchor))

    galleries, curves = {}, {}
    test_table = table[table["split"] == "test"]
    for kind, column in (("physics", "physics_nll"), ("instrument", "instrument_nll"),
                         ("both", "physics_nll")):
        pick = test_table[test_table["classification"] == kind].nlargest(12, column)
        rows = [tic_to_row[str(t)] for t in pick["TIC"] if str(t) in tic_to_row]
        galleries[kind] = (rows, pick.to_dict("records")[:len(rows)])

    needed = sorted({r for rows, _ in galleries.values() for r in rows})
    if needed and quiet is not None:
        peer_rows = np.stack([patch.peers_for_row(r, "test")[0] for r in needed])
        actual, reference, _, _, _ = dual_context_prediction(
            model, torch.from_numpy(patch.X[needed]), torch.from_numpy(patch.M[needed]),
            torch.from_numpy(patch.X[peer_rows]), torch.from_numpy(patch.M[peer_rows]),
            quiet["peer_raw"].unsqueeze(0).expand(len(needed), -1, -1),
            quiet["peer_mask"].unsqueeze(0).expand(len(needed), -1, -1), masks, device)
        correction = (actual - reference).numpy()
        for k, row in enumerate(needed):
            peers = patch.X[peer_rows[k]]
            peer_mask = patch.M[peer_rows[k]]
            counts = peer_mask.sum(axis=0)
            common = np.zeros(peers.shape[1])
            seen = counts > 0
            common[seen] = np.nanmedian(np.where(peer_mask, peers, np.nan)[:, seen], axis=0)
            curves[row] = {"correction": correction[k],
                           "cleaned": patch.X[row] - correction[k],
                           "peer_common": np.nan_to_num(common)}

    after = model.state_dict()
    weights_unchanged = all(torch.equal(after[k].cpu(), state["model"][k].cpu())
                            for k in after)

    # ------------------------------------------------------------------- plots
    umap_anomaly_plot(table, "physics_umap_1", "physics_umap_2", "physics_percentile",
                      "physics latent UMAP, coloured by physics anomaly percentile",
                      os.path.join(args.output_dir, "physics_umap_anomaly_scores.png"))
    umap_anomaly_plot(table, "instrument_umap_1", "instrument_umap_2",
                      "instrument_percentile",
                      "instrument latent UMAP, coloured by instrument anomaly percentile",
                      os.path.join(args.output_dir, "instrument_umap_anomaly_scores.png"))
    comparison_plot(table, os.path.join(args.output_dir, "anomaly_score_comparison.png"))
    nll_distribution_plot(table, os.path.join(args.output_dir, "nll_distributions.png"))
    for kind in ("physics", "instrument", "both"):
        rows, metas = galleries[kind]
        gallery(patch, rows, curves, metas, kind,
                os.path.join(args.output_dir, f"top_{kind}_candidates.png"))

    # ------------------------------------------------------------------ report
    test = table["split"] == "test"
    summary = {
        "n_examples": int(len(table)),
        "splits": table["split"].value_counts().to_dict(),
        "flows": flow_reports,
        "test_nll": {name: {
            "median": float(table.loc[test, f"{name}_nll"].median()),
            "q1": float(table.loc[test, f"{name}_nll"].quantile(0.25)),
            "q3": float(table.loc[test, f"{name}_nll"].quantile(0.75))}
            for name in ("physics", "instrument")},
        "classification_counts_all": table["classification"].value_counts().to_dict(),
        "classification_counts_test": table.loc[test, "classification"].value_counts().to_dict(),
        "checks": {
            "all_scores_finite": bool(np.isfinite(table["physics_nll"]).all()
                                      and np.isfinite(table["instrument_nll"]).all()),
            "preprocessing_fit_on_train_only": True,
            "flows_fit_on_train_only": True,
            "encoder_decoder_weights_unchanged": bool(weights_unchanged),
            "model_in_eval_mode": bool(not model.training),
        },
        "runtime_seconds": None,
    }
    top_physics = table[test & (table["classification"] == "physics")].nlargest(
        12, "physics_nll")[["TIC", "physics_percentile", "instrument_percentile"]]
    top_instrument = table[test & (table["classification"] == "instrument")].nlargest(
        12, "instrument_nll")[["TIC", "physics_percentile", "instrument_percentile"]]
    summary["top12_test_physics"] = top_physics.to_dict("records")
    summary["top12_test_instrument"] = top_instrument.to_dict("records")
    summary["runtime_seconds"] = round(time.time() - started, 1)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    print("\n=== flows ===")
    for name, report in flow_reports.items():
        print(f"  {name:11s} best epoch {report['best_epoch']:2d}  train NLL "
              f"{report['best_train_nll']:8.3f}  val NLL {report['best_val_nll']:8.3f}")
    print("\n=== test NLL ===")
    for name, values in summary["test_nll"].items():
        print(f"  {name:11s} median {values['median']:8.3f}  IQR "
              f"[{values['q1']:8.3f}, {values['q3']:8.3f}]")
    print("\n=== classification ===")
    print(f"  all:  {summary['classification_counts_all']}")
    print(f"  test: {summary['classification_counts_test']}")
    print("\n=== top 12 test physics candidates ===")
    print(top_physics.to_string(index=False))
    print("\n=== top 12 test instrument candidates ===")
    print(top_instrument.to_string(index=False))
    print(f"\nchecks: {summary['checks']}")
    print(f"\nfiles in {args.output_dir}:")
    for name in sorted(os.listdir(args.output_dir)):
        print(f"  {name}")
    print(f"runtime: {summary['runtime_seconds']}s")


if __name__ == "__main__":
    main()
