# ============================================================================
# TEMPORARY DIAGNOSTIC -- safe to delete.
#
# Question: are CLASS and SECTOR correlated in the data the probe runs on?
#
#   If sector is recoverable from class ALONE (well above chance), then part of
#   the physics->sector 0.37 is UNAVOIDABLE -- a good class latent would predict
#   sector anyway -- and adversarially removing sector would also damage class.
#
#   If class->sector is ~chance, the 0.37 is genuine instrument leakage and it's
#   safe to remove adversarially.
#
# No model / no GPU -- just the parquet. Runs on the login node in seconds.
# Uses the full eval parquet (all rows), matching what the probes saw.
# ============================================================================
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, normalized_mutual_info_score

from src.data.data import CLASS_TO_IDX

DATA_PATH = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split/tess_classification_train.parquet"

df = pd.read_parquet(DATA_PATH)
labels  = df["label"].map(CLASS_TO_IDX).to_numpy()
sectors = df["sector"].astype(int).to_numpy()


def one_hot(x):
    vals = np.unique(x)
    idx = {v: i for i, v in enumerate(vals)}
    oh = np.zeros((len(x), len(vals)), dtype=np.float32)
    oh[np.arange(len(x)), [idx[v] for v in x]] = 1.0
    return oh


def label_probe(x_in, y, name):
    # same balanced-accuracy probe as the eval, but the INPUT is just the other
    # label (one-hot). Tells us how much of y is determined by x ALONE.
    uniq, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, uniq[counts >= 2])
    x_in, y = x_in[keep], y[keep]

    X = one_hot(x_in)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    probe = LogisticRegression(max_iter=5000)
    probe.fit(X_tr, y_tr)
    acc = balanced_accuracy_score(y_te, probe.predict(X_te))
    chance = 1 / len(np.unique(y))
    print(f"{name}: balanced accuracy = {acc:.4f}   (chance = {chance:.4f})")


def cramers_v(a, b):
    table = pd.crosstab(a, b).to_numpy()
    chi2 = chi2_contingency(table)[0]
    n = table.sum()
    r, k = table.shape
    return np.sqrt((chi2 / n) / (min(r, k) - 1))


print(f"n = {len(labels)} rows | {len(np.unique(labels))} classes | {len(np.unique(sectors))} sectors")
print(f"normalized mutual info(class, sector): {normalized_mutual_info_score(labels, sectors):.4f}   (0 = independent, 1 = identical)")
print(f"Cramer's V(class, sector):             {cramers_v(labels, sectors):.4f}   (0 = independent, 1 = fully associated)")
print("-" * 60)
label_probe(labels,  sectors, "CLASS  -> SECTOR (from label only)")
label_probe(sectors, labels,  "SECTOR -> CLASS  (from label only)")
