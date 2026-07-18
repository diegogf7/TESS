# All this code is from Claude
"""Raw region-coherence diagnostic: is there ANY area-specific common-mode
signal in the raw group-statistic curves, BEFORE any encoder?

For each comparison, two K=8 star groups are reduced to their per-cadence
median and log-MAD curves (existing area_commonmode_dataset machinery) and
compared by Pearson correlation over MUTUALLY VALID cadences only:

  same_area   two disjoint groups, same area
  cross_area  two groups, same camera/CCD, different areas
  shuffled    same-area protocol under permuted area labels (null control)

Pass rule (pre-registered): the combined (median+logMAD) same-minus-cross
95% bootstrap CI on the VALIDATION split must be entirely above zero.
Encoders cannot learn area structure that is absent at this level.

Run:  python -m src.instrument_v2.diagnose_raw_area_commonmode
Env:  N_COMPARISONS (default 1000 per condition per split), K, SEED,
      S14_DATA, AREA_SOURCE, SPLIT_DIR, BASE_ART_DIR, ACM2_ART_DIR
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    area_to_chip,
    ensure_area_column,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit

N_COMPARISONS = int(os.environ.get("N_COMPARISONS", "1000"))
K = int(os.environ.get("K", "8"))
SEED = int(os.environ.get("SEED", "0"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "2000"))

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ART_DIR = os.environ.get(
    "ACM2_ART_DIR", os.path.join("artifacts", "instrument_v2", "area_commonmode_v2"))


def masked_correlation(a, b, mutual):
    """Pearson r over mutually valid cadences only; nan if under 8 cadences."""
    rows = np.flatnonzero(mutual)
    if len(rows) < 8:
        return float("nan")
    x, y = a[rows], b[rows]
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom <= 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def pair_similarities(dataset, rows_a, rows_b):
    """(median_r, logmad_r, combined_r) for two explicit star-row groups."""
    med_a, mad_a, valid_a = (t.numpy() for t in dataset.group_target_tensors(rows_a))
    med_b, mad_b, valid_b = (t.numpy() for t in dataset.group_target_tensors(rows_b))
    mutual = (valid_a > 0) & (valid_b > 0)
    med_r = masked_correlation(med_a, med_b, mutual)
    mad_r = masked_correlation(mad_a, mad_b, mutual)
    finite = [r for r in (med_r, mad_r) if np.isfinite(r)]
    combined = float(np.mean(finite)) if finite else float("nan")
    return med_r, mad_r, combined, float(mutual.mean())


class ShuffledAreaView:
    """Same dataset, area labels permuted across stars (seeded) -- the null."""

    def __init__(self, dataset, seed):
        self.base = dataset
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(dataset.group_labels)
        self.group_rows = {}
        for group in np.unique(shuffled):
            rows = np.flatnonzero(shuffled == group)
            if len(rows) >= 2 * dataset.k:
                self.group_rows[int(group)] = rows
        self.group_list = sorted(self.group_rows)
        self.k = dataset.k

    def sample_disjoint_same_group(self):
        if not self.group_list:
            return None
        group = self.group_list[np.random.randint(len(self.group_list))]
        rows = np.random.choice(self.group_rows[group], size=2 * self.k,
                                replace=False)
        return rows[:self.k], rows[self.k:], group


def run_split(dataset, split_name):
    np.random.seed(SEED)
    shuffled = ShuffledAreaView(dataset, SEED)
    records = []
    conditions = (("same_area", dataset.sample_disjoint_same_group),
                  ("cross_area", dataset.sample_cross_group),
                  ("shuffled", shuffled.sample_disjoint_same_group))
    for condition, sampler in conditions:
        drawn = 0
        while drawn < N_COMPARISONS:
            draw = sampler()
            if draw is None:
                raise RuntimeError(f"{split_name}/{condition}: sampler exhausted")
            rows_a, rows_b, group = draw
            med_r, mad_r, combined, coverage = pair_similarities(
                dataset, rows_a, rows_b)
            if not np.isfinite(combined):
                continue
            area = int(group) if condition == "same_area" else (
                int(group[0]) if condition == "cross_area" else -1)
            records.append({"split": split_name, "condition": condition,
                            "area": area,
                            "parent_chip": area_to_chip(area) if area > 0 else -1,
                            "median_r": med_r, "logmad_r": mad_r,
                            "combined_r": combined, "coverage": coverage})
            drawn += 1
    return pd.DataFrame(records)


def bootstrap_diff(same, cross, rng):
    diffs = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        diffs[b] = (np.mean(rng.choice(same, len(same)))
                    - np.mean(rng.choice(cross, len(cross))))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(same) - np.mean(cross)), float(lo), float(hi)


def summarize(frame, split_name, rng):
    part = frame[frame["split"] == split_name]
    out = {"n_per_condition": int((part["condition"] == "same_area").sum()),
           "mean_coverage": float(part["coverage"].mean()),
           "condition_means": {}}
    for condition in ("same_area", "cross_area", "shuffled"):
        rows = part[part["condition"] == condition]
        out["condition_means"][condition] = {
            metric: float(rows[metric].mean())
            for metric in ("median_r", "logmad_r", "combined_r")}
    same = part[part["condition"] == "same_area"]
    cross = part[part["condition"] == "cross_area"]
    shuffled = part[part["condition"] == "shuffled"]
    out["same_minus_cross"] = {}
    for metric in ("median_r", "logmad_r", "combined_r"):
        mean, lo, hi = bootstrap_diff(same[metric].to_numpy(),
                                      cross[metric].to_numpy(), rng)
        out["same_minus_cross"][metric] = {"mean": mean, "ci95": [lo, hi]}
    mean, lo, hi = bootstrap_diff(same["combined_r"].to_numpy(),
                                  shuffled["combined_r"].to_numpy(), rng)
    out["same_minus_shuffled_combined"] = {"mean": mean, "ci95": [lo, hi]}
    # per-area and per-chip breakdown (same-area comparisons only)
    out["per_area_same_combined"] = {
        str(area): float(rows["combined_r"].mean())
        for area, rows in same.groupby("area")}
    out["per_chip_same_minus_cross_combined"] = {}
    for chip in sorted(set(same["parent_chip"]) & set(cross["parent_chip"])):
        s = same[same["parent_chip"] == chip]["combined_r"].to_numpy()
        c = cross[cross["parent_chip"] == chip]["combined_r"].to_numpy()
        if len(s) >= 20 and len(c) >= 20:
            mean, lo, hi = bootstrap_diff(s, c, rng)
            out["per_chip_same_minus_cross_combined"][str(chip)] = {
                "mean": mean, "ci95": [lo, hi]}
    return out


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    results = {"git_commit": git_commit(), "k": K,
               "n_comparisons": N_COMPARISONS, "splits": {},
               "area_star_counts": {}}
    frames = []
    for split_name, tics in (("train", train_tics), ("val", val_tics)):
        dataset = Sector14GroupStatDataset(df, tics, t_range, "area", K)
        assert not set(dataset.tics) & test_tics, "test TIC leaked into diagnostic"
        results["area_star_counts"][split_name] = {
            str(row["group"]): row["n_stars"]
            for row in dataset.group_count_table()}
        frame = run_split(dataset, split_name)
        frames.append(frame)
        results["splits"][split_name] = summarize(
            frame, split_name, np.random.default_rng(SEED))
        print(f"{split_name}: same={results['splits'][split_name]['condition_means']['same_area']['combined_r']:.4f} "
              f"cross={results['splits'][split_name]['condition_means']['cross_area']['combined_r']:.4f} "
              f"shuffled={results['splits'][split_name]['condition_means']['shuffled']['combined_r']:.4f}",
              flush=True)

    val_ci = results["splits"]["val"]["same_minus_cross"]["combined_r"]["ci95"]
    results["passes"] = bool(val_ci[0] > 0.0)
    results["pass_rule"] = ("combined same-minus-cross 95% CI entirely above "
                            "zero on the validation split")
    results["test_untouched"] = "Test TICs were never loaded in this diagnostic."

    json_path = os.path.join(ART_DIR, "raw_region_diagnostic.json")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)

    lines = ["# Raw region-coherence diagnostic", "",
             f"git commit: {results['git_commit']}",
             f"K={K}, {N_COMPARISONS} comparisons per condition per split", ""]
    for split_name in ("train", "val"):
        s = results["splits"][split_name]
        lines += [f"## {split_name} (coverage {s['mean_coverage']:.3f})"]
        for condition, means in s["condition_means"].items():
            lines.append(f"- {condition}: median_r {means['median_r']:.4f}, "
                         f"logmad_r {means['logmad_r']:.4f}, "
                         f"combined {means['combined_r']:.4f}")
        for metric, d in s["same_minus_cross"].items():
            lines.append(f"- same-cross {metric}: {d['mean']:+.4f} "
                         f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
        lines.append("")
    lines += [f"## VERDICT: {'PASS' if results['passes'] else 'FAIL'}",
              f"({results['pass_rule']})", "", results["test_untouched"], ""]
    md_path = os.path.join(ART_DIR, "raw_region_diagnostic.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"diagnostic -> {md_path}", flush=True)
    print(f"PASSES: {results['passes']}", flush=True)


if __name__ == "__main__":
    main()
