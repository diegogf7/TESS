from __future__ import annotations
"""Validation-only diagnostic: are the K=8 CBV-decoder failures driven by
brightness (Tmag), weak instrument signal (target/raw RMS), or detector
position (camera/CCD/subregion)?

Nothing is trained or rebuilt. It reads the EXISTING oracle per-example metrics
(which already carry the stable TIC id), joins by TIC to:
  * Tmag        from tglc_positions.parquet (GAIADR3+sector -> tessmag)
  * raw-curve RMS from the validation gridded curve (reuse the val dataset)
  * camera / ccd / subregion  from the area code (camera*100 + ccd*10 + ring)
  * n_cbv_groups, target_rms, valid_frac, decoder_vs_target Pearson (from the CSV)
then correlates every feature with Pearson, computes failure rates by detector
location, and fits a standardized 5-fold logistic model with per-feature
permutation importance to rank the likely cause.

    python -m src.instrument_v2.analyze_cbv_failure_correlations
"""

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit
from src.instrument_v2.decode_single_star_k8 import (
    GROUP_SIZE, MIN_VALID_STARS, S14_DATA, SPLIT_DIR, BASE_ART_DIR, GROUP_ART_DIR,
)

ORACLE_CSV = os.environ.get(
    "ORACLE_CSV", os.path.join(GROUP_ART_DIR, "oracle_ceiling", "per_example_metrics.csv"))
POSITIONS_PATH = os.environ.get(
    "POSITIONS_PATH", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_positions.parquet")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GROUP_ART_DIR, "failure_correlation"))
EXPECTED_N = int(os.environ.get("EXPECTED_N", "10498"))
PEARSON_COL = "decoder_vs_target_pearson"

CONTINUOUS = ["Tmag", "target_rms", "raw_rms", "valid_frac", "n_groups"]
CATEGORICAL = ["camera", "ccd", "subregion"]
FEATURES = CONTINUOUS + CATEGORICAL


# ------------------------------------------------------------- feature joining
def derive_detector(area):
    """area = camera*100 + ccd*10 + subregion(ring). Pure arithmetic, no join."""
    a = np.asarray(area, dtype=int)
    return a // 100, (a // 10) % 10, a % 10


def load_tmag_by_tic(tics):
    """TIC -> Tmag via dense_v2 (TIC,GAIADR3,sector) joined to positions
    (GAIADR3,sector,tessmag). Missing values stay NaN and are reported."""
    import pyarrow.parquet as pq
    want = tics.astype(str)
    tmag = pd.Series(np.nan, index=pd.Index(np.unique(want), name="TIC"), dtype=float)
    try:
        cols = set(pq.read_schema(S14_DATA).names)
        if "tessmag" in cols:                                  # dense_v2 already carries it
            m = pd.read_parquet(S14_DATA, columns=["TIC", "tessmag"]).dropna()
            m["TIC"] = m["TIC"].astype(str)
            s = m.drop_duplicates("TIC").set_index("TIC")["tessmag"]
            tmag.loc[tmag.index.intersection(s.index)] = s.reindex(tmag.index).dropna()
            return tmag.to_dict()
        meta = pd.read_parquet(S14_DATA, columns=["TIC", "GAIADR3", "sector"])
        meta["TIC"] = meta["TIC"].astype(str)
        pos = pd.read_parquet(POSITIONS_PATH, columns=["GAIADR3", "sector", "tessmag"])
        j = meta.merge(pos, on=["GAIADR3", "sector"], how="left").dropna(subset=["tessmag"])
        s = j.drop_duplicates("TIC").set_index("TIC")["tessmag"]
        common = tmag.index.intersection(s.index)
        tmag.loc[common] = s.reindex(common)
    except Exception as e:
        print(f"WARNING: Tmag source unavailable ({e}); leaving Tmag NaN", flush=True)
    return tmag.to_dict()


def raw_rms_by_tic():
    """TIC -> std of the median/MAD-normalized validation gridded curve on
    observed cadences (individual-star variability amplitude). Reuses the exact
    validation dataset; no model, CPU only."""
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE,
                                      min_valid=MIN_VALID_STARS)
    assert not (set(val_ds.tics) & test_tics), "test TIC leaked into validation"
    out = {}
    for i, tic in enumerate(val_ds.tics):
        obs = val_ds.M[i] > 0
        x = val_ds.X[i][obs]
        out[str(tic)] = float(x.std()) if x.size > 1 else np.nan
    return out


def build_features():
    pe = pd.read_csv(ORACLE_CSV)
    pe["tic"] = pe["tic"].astype(str)
    cam, ccd, sub = derive_detector(pe["area"].to_numpy())
    feat = pd.DataFrame({
        "tic": pe["tic"], "area": pe["area"].astype(int),
        "camera": cam, "ccd": ccd, "subregion": sub,
        "n_groups": pe["n_cbv_groups"].astype(int),
        "target_rms": pe["target_rms"].astype(float),
        "valid_frac": pe["valid_frac"].astype(float),
        "pearson": pe[PEARSON_COL].astype(float)})
    rr = raw_rms_by_tic()
    feat["raw_rms"] = feat["tic"].map(rr).astype(float)
    tm = load_tmag_by_tic(feat["tic"].to_numpy())
    feat["Tmag"] = feat["tic"].map(tm).astype(float)
    feat["failed"] = (feat["pearson"] < 0).astype(int)
    return feat


# ------------------------------------------------------------------- analysis
def spearman_table(feat):
    rows = []
    for f in CONTINUOUS:
        sub = feat[[f, "pearson"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) >= 3 and sub[f].nunique() > 1:
            rho, p = stats.spearmanr(sub[f], sub["pearson"])
        else:
            rho, p = np.nan, np.nan
        rows.append({"feature": f, "spearman_rho_vs_pearson": float(rho),
                     "p_value": float(p), "n": int(len(sub))})
    return pd.DataFrame(rows)


def failure_rate_table(feat):
    rows = []
    for col in CATEGORICAL:
        for level, g in feat.groupby(col):
            rows.append({"grouping": col, "level": int(level), "n": int(len(g)),
                         "n_failed": int(g["failed"].sum()),
                         "failure_pct": round(100.0 * g["failed"].mean(), 3)})
    return pd.DataFrame(rows)


def logistic_permutation_importance(feat, seed=0):
    """Standardized logistic `failed ~ features`, 5-fold CV ROC-AUC, and
    per-feature permutation importance averaged across folds."""
    X = feat[FEATURES].copy()
    y = feat["failed"].to_numpy()
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), CONTINUOUS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    pipe = Pipeline([("ct", ct), ("lr", LogisticRegression(max_iter=2000))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    imps = {f: [] for f in FEATURES}
    aucs = []
    for tr, te in skf.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], pipe.predict_proba(X.iloc[te])[:, 1])))
        r = permutation_importance(pipe, X.iloc[te], y[te], n_repeats=10,
                                   random_state=seed, scoring="roc_auc")
        for j, f in enumerate(FEATURES):
            imps[f].append(float(r.importances_mean[j]))
    imp_df = pd.DataFrame([{"feature": f, "perm_importance_mean": float(np.mean(v)),
                            "perm_importance_std": float(np.std(v))}
                           for f, v in imps.items()]).sort_values(
                               "perm_importance_mean", ascending=False).reset_index(drop=True)
    return imp_df, {"cv_roc_auc_mean": float(np.mean(aucs)),
                    "cv_roc_auc_std": float(np.std(aucs)), "n_splits": 5}


def rank_causes(imp_df):
    cat_of = {"Tmag": "brightness/SNR (Tmag)", "target_rms": "weak instrument signal (target RMS)",
              "raw_rms": "weak instrument signal (raw RMS)", "valid_frac": "coverage (valid fraction)",
              "n_groups": "data-scaling (CBV groups)", "camera": "detector position (camera)",
              "ccd": "detector position (CCD)", "subregion": "detector position (subregion)"}
    ranked = [{"rank": i + 1, "feature": r.feature,
               "perm_importance_mean": round(r.perm_importance_mean, 5),
               "cause": cat_of.get(r.feature, r.feature)}
              for i, r in enumerate(imp_df.itertuples())]
    top = imp_df.iloc[0]["feature"]
    headline = {"Tmag": "BRIGHTNESS/SNR effect: Tmag dominates after controlling for RMS and location.",
                "target_rms": "WEAK INSTRUMENT SIGNAL: target RMS dominates.",
                "raw_rms": "WEAK INSTRUMENT SIGNAL: raw-curve RMS dominates.",
                "valid_frac": "COVERAGE effect: valid-cadence fraction dominates.",
                "n_groups": "DATA-SCALING effect: number of CBV groups dominates.",
                "camera": "DETECTOR-POSITION effect: camera dominates.",
                "ccd": "DETECTOR-POSITION effect: CCD dominates.",
                "subregion": "DETECTOR-POSITION effect: subregion dominates."}[top]
    return ranked, headline


# --------------------------------------------------------------------- figures
def _binned(x, y, bins=12):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    cx, med, q1, q3 = [], [], [], []
    for b in range(len(edges) - 1):
        yb = y[idx == b]
        if yb.size:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            med.append(np.median(yb)); q1.append(np.percentile(yb, 25)); q3.append(np.percentile(yb, 75))
    return np.array(cx), np.array(med), np.array(q1), np.array(q3)


def make_figures(feat, failure_rates, out_dir):
    # 1) Pearson vs Tmag (binned median + Q1-Q3 band)
    cx, med, q1, q3 = _binned(feat["Tmag"], feat["pearson"])
    fig, ax = plt.subplots(figsize=(7, 5))
    if cx.size:
        ax.fill_between(cx, q1, q3, alpha=0.25, color="tab:blue", label="Q1-Q3")
        ax.plot(cx, med, "-o", color="tab:blue", ms=4, label="median")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("Tmag"); ax.set_ylabel("decoder-vs-target Pearson")
    ax.set_title("Pearson vs Tmag (binned)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pearson_vs_tmag.png"), dpi=130); plt.close(fig)

    # 2) target RMS vs Tmag
    cx2, med2, q1b, q3b = _binned(feat["Tmag"], feat["target_rms"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(feat["Tmag"], feat["target_rms"], s=4, alpha=0.15, color="tab:gray", edgecolor="none")
    if cx2.size:
        ax.plot(cx2, med2, "-o", color="tab:red", ms=4, label="binned median")
    ax.set_xlabel("Tmag"); ax.set_ylabel("target RMS"); ax.set_yscale("log")
    ax.set_title("target RMS vs Tmag"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "target_rms_vs_tmag.png"), dpi=130); plt.close(fig)

    # 3) failure rate by subregion
    sub = failure_rates[failure_rates["grouping"] == "subregion"].sort_values("level")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(sub["level"].astype(str), sub["failure_pct"], color="tab:orange", edgecolor="k")
    for _, r in sub.iterrows():
        ax.text(str(int(r.level)), r.failure_pct, f"{r.failure_pct:.0f}%\n(n={int(r.n)})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("subregion (ring 1-4)"); ax.set_ylabel("failure %  (Pearson < 0)")
    ax.set_title("failure rate by subregion")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "failure_rate_by_subregion.png"), dpi=130); plt.close(fig)

    # 4) camera x CCD failure heatmap per subregion
    subs = sorted(feat["subregion"].unique())
    fig, axes = plt.subplots(1, len(subs), figsize=(4.2 * len(subs), 4), squeeze=False)
    for k, s in enumerate(subs):
        ax = axes[0, k]
        piv = (feat[feat["subregion"] == s]
               .pivot_table(index="camera", columns="ccd", values="failed", aggfunc="mean") * 100.0)
        im = ax.imshow(piv.values, cmap="magma", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                v = piv.values[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                            color="white" if v < 60 else "black", fontsize=7)
        ax.set_title(f"subregion {int(s)}"); ax.set_xlabel("CCD"); ax.set_ylabel("camera")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="failure %")
    fig.suptitle("camera x CCD failure % per subregion")
    fig.savefig(os.path.join(out_dir, "detector_failure_heatmaps.png"), dpi=130); plt.close(fig)


def run_analysis(feat, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    feat.to_csv(os.path.join(out_dir, "per_example_features.csv"), index=False)
    corr = spearman_table(feat)
    corr.to_csv(os.path.join(out_dir, "feature_correlations.csv"), index=False)
    fr = failure_rate_table(feat)
    fr.to_csv(os.path.join(out_dir, "failure_rates.csv"), index=False)
    imp_df, cv = logistic_permutation_importance(feat)
    imp_df.to_csv(os.path.join(out_dir, "model_feature_importance.csv"), index=False)
    ranked, headline = rank_causes(imp_df)
    make_figures(feat, fr, out_dir)
    return corr, fr, imp_df, cv, ranked, headline


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)
    print(f"oracle csv: {ORACLE_CSV}", flush=True)

    feat = build_features()

    # --- retain every scored validation example; report Tmag coverage --------
    n = len(feat)
    n_tmag_missing = int(feat["Tmag"].isna().sum())
    n_rawrms_missing = int(feat["raw_rms"].isna().sum())
    assert n == EXPECTED_N, f"expected {EXPECTED_N} validation examples, got {n}"
    print(f"validation examples retained: {n} (== {EXPECTED_N})", flush=True)
    print(f"Tmag missing: {n_tmag_missing}/{n}   raw_rms missing: {n_rawrms_missing}/{n}", flush=True)
    print(f"failed (Pearson<0): {int(feat['failed'].sum())} ({100*feat['failed'].mean():.1f}%)", flush=True)

    corr, fr, imp_df, cv, ranked, headline = run_analysis(feat, OUT_DIR)

    report = {
        "n_val_examples": n, "expected_n": EXPECTED_N,
        "n_tmag_missing": n_tmag_missing, "n_raw_rms_missing": n_rawrms_missing,
        "failed_definition": f"{PEARSON_COL} < 0",
        "n_failed": int(feat["failed"].sum()), "failure_pct": round(100 * feat["failed"].mean(), 3),
        "spearman_pearson_vs_feature": corr.set_index("feature")[
            ["spearman_rho_vs_pearson", "p_value", "n"]].to_dict(orient="index"),
        "failure_rates": fr.to_dict(orient="records"),
        "logistic_cv": cv,
        "permutation_importance_ranked": ranked,
        "headline_cause": headline,
        "features": FEATURES, "git_commit": git_commit(),
    }
    with open(os.path.join(OUT_DIR, "failure_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("\nSpearman(Pearson, feature):", flush=True)
    for r in corr.itertuples():
        print(f"  {r.feature:<12} rho={r.spearman_rho_vs_pearson:+.3f} p={r.p_value:.2e} (n={r.n})", flush=True)
    print(f"\nlogistic 5-fold ROC-AUC = {cv['cv_roc_auc_mean']:.3f} +/- {cv['cv_roc_auc_std']:.3f}", flush=True)
    print("permutation importance (ranked):", flush=True)
    for r in ranked:
        print(f"  {r['rank']}. {r['feature']:<12} {r['perm_importance_mean']:+.4f}   {r['cause']}", flush=True)
    print(f"\nLIKELY CAUSE: {headline}", flush=True)
    print(f"wrote per_example_features.csv, feature_correlations.csv, failure_rates.csv, "
          f"model_feature_importance.csv, failure_analysis.json + 4 PNGs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
