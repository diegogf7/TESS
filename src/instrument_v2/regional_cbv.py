from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from src.instrument_v2.area_commonmode_dataset import group_statistics
from src.instrument_v2.train_sector14_jepa import git_commit

def train_tic_hash(tics):

    return hashlib.sha256("\n".join(sorted(map(str, tics))).encode()).hexdigest()[:16]

def area_group_medians(X, M, rows, group_size, min_valid):


    medians = []
    valids = []

    n_groups = len(rows) // group_size

    for g in range(n_groups):

        group = rows[g * group_size:(g + 1) * group_size]
        median, _log_mad, valid, _n = group_statistics(X[group], M[group], min_valid)

        medians.append(median)
        valids.append(valid)

    empty = np.empty((0, X.shape[1]), dtype = np.float32)

    final = (np.stack(medians) if medians else empty, np.stack(valids) if valids else empty, n_groups)

    return final


def uncentered_area_basis(medians, valids, k):

    masked = (medians * valids).astype(np.float64)

    _, _s, vt = np.linalg.svd(masked, full_matrices = False) #just getting doing our singular value decomposition to do our pca later

    if vt.shape[0] < k:
        raise RuntimeError(f"Area SVD produced is wrong")

    return vt[:k].T.astype(np.float64)

def build_or_load_area_bases(X, M, areas, tics, k, cache_directive, group_size, min_valid):

    os.makedirs(cache_directive, exist_ok = True)

    # rank (r), group size (g) and min-valid (mv) are all independent; every one
    # is in the cache key so an old basis built under a different config can never
    # be silently loaded.
    tag = f"area_group_cbv_r{k}_g{group_size}_mv{min_valid}_{train_tic_hash(tics)}"

    npz = os.path.join(cache_directive, tag + ".npz")

    if os.path.exists(npz):

        data = np.load(npz, allow_pickle = True)

        return {int(a): data[f"B_{int(a)}"] for a in data["areas"]}
    
    order = np.argsort(np.asarray(tics, dtype = str))
    rank = np.empty(len(tics), dtype = np.int64)
    rank[order] = np.arange(len(tics))

    bases = {}
    store= {}
    n_groups_by_area = {}

    for area in sorted(np.unique(areas)):

        rows = np.flatnonzero(areas == area)
        rows = rows[np.argsort(rank[rows])]

        medians, valids, n_groups = area_group_medians(X, M, rows, group_size, min_valid)
        bases[int(area)] = uncentered_area_basis(medians, valids, k)

        n_groups_by_area[int(area)] = int(n_groups)
        store[f"B_{int(area)}"] = bases[int(area)]


    store["areas"] = np.array(sorted(bases))
    np.savez(npz, **store)

    with open(os.path.join(cache_directive, tag + ".json"), "w") as fh:
        json.dump({"train_tic_hash": train_tic_hash(tics), "k": k,
                   "group_size": group_size, "min_valid": min_valid,
                   "n_train_tics": len(tics), "n_areas": len(bases),
                   "n_groups_by_area": n_groups_by_area,
                   "git_commit": git_commit()}, fh, indent = 2)

    return bases

def ridge_reconstruct(median, valid, B, ridge_lambda):

    observed = valid >0 
    Bv = B[observed]
    mv = median[observed].astype(np.float64)

    A = Bv.T @ Bv + ridge_lambda * np.eye(B.shape[1])
    w = np.linalg.solve(A, Bv.T @ mv)

    final = (B @ w).astype(np.float32)
    return final

def cbv_fingerprint(X, M, rows, B, min_valid, ridge_lambda):

    median, log_mad, valid, _ = group_statistics(X[rows], M[rows], min_valid)

    instrument = ridge_reconstruct(median, valid, B, ridge_lambda)

    return instrument, log_mad, valid


