"""Frozen-representation stellar classification probe on PhyTS labels.

Every representation is frozen, the scaler and classifier are fit on TRAIN only,
and balanced accuracy is reported on val/test. Same split, same classifier, same
seeds for every arm, so the arms are comparable.

    python -m src.vicreg_jepa.classify --npz artifacts/vicreg_jepa/phyts_s15.npz \
        --part1 <part1_best.pt> --part2 <part2_best.pt>
"""
import argparse
import json
import os

import numpy as np
import torch

from .models import S4Encoder
from .part2_model import PhysicsEncoder
from .real_data import RealCurveSource


@torch.no_grad()
def encode(enc, src, device="cpu", batch=256, kind="part2"):
    enc.eval()
    flux = torch.from_numpy(np.asarray(src.flux, dtype=np.float32))
    obs = torch.from_numpy(np.asarray(src.observed, dtype=np.float32))
    out = []
    for i in range(0, len(flux), batch):
        f, o = flux[i:i + batch].to(device), obs[i:i + batch].to(device)
        z = enc(f, o, pool_mask=o) if kind == "part1" else enc(f, o)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def moment_features(src):
    """Cheap non-learned baseline: per-curve moments over observed cadences."""
    f, m = src.flux, src.observed.astype(bool)
    rows = []
    for i in range(len(f)):
        v = f[i][m[i]]
        if v.size < 8:
            rows.append(np.zeros(8)); continue
        q = np.percentile(v, [5, 25, 50, 75, 95])
        d = np.diff(v)
        rows.append(np.array([v.mean(), v.std(), *q, np.abs(d).mean()]))
    return np.asarray(rows, dtype=np.float64)


def probe(ztr, ytr, zev, yev, seed=0, max_iter=3000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=max_iter, random_state=seed,
                                           class_weight="balanced"))
    clf.fit(ztr, ytr)
    pred = clf.predict(zev)
    return (float(balanced_accuracy_score(yev, pred)),
            confusion_matrix(yev, pred).tolist(), pred)


def knn_probe(ztr, ytr, zev, yev, k=20):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(ztr)
    kn = KNeighborsClassifier(n_neighbors=k).fit(sc.transform(ztr), ytr)
    return float(balanced_accuracy_score(yev, kn.predict(sc.transform(zev))))


def main(a):
    device = os.environ.get("TESS_DEVICE", "cpu")
    srcs = {s: RealCurveSource(a.npz, s) for s in ("train", "val", "test")}
    lab = {s: srcs[s].label for s in srcs}
    keep = {s: np.flatnonzero(lab[s] >= 0) for s in srcs}
    for s in srcs:
        print(f"{s}: {len(keep[s])} labelled of {len(lab[s])}")

    classes = json.load(open(os.path.splitext(a.npz)[0] + "_manifest.json"))["label_classes"]
    names = [k for k, _ in sorted(classes.items(), key=lambda kv: kv[1])]

    # Drop classes so the run matches a prior benchmark's class set. The S14
    # benchmarks are 7-class with INSTRUMENT/JUNK removed; comparing an 8-class
    # balanced accuracy against them is not a like-for-like number.
    if a.drop_class:
        drop = {classes[c] for c in a.drop_class if c in classes}
        if drop:
            for s in srcs:
                keep[s] = np.array([i for i in keep[s] if lab[s][i] not in drop])
            names = [n for n in names if classes[n] not in drop]
            print(f"dropped {sorted(a.drop_class)} -> {len(names)} classes; "
                  f"labelled now " + ", ".join(f"{s}={len(keep[s])}" for s in srcs))

    arms = {}

    if a.part1:
        blob = torch.load(a.part1, map_location="cpu", weights_only=False)
        c = blob["cfg"]
        e = S4Encoder(c["latent_dim"], c["n_tokens"], c["d_model"], c["d_state"],
                      c["n_layers"], 0.0)
        e.load_state_dict(blob["encoder"]); e.eval().to(device)
        arms["part1_systematics"] = {s: encode(e, srcs[s], device, kind="part1") for s in srcs}

    if a.part2:
        blob = torch.load(a.part2, map_location="cpu", weights_only=False)
        c = blob["cfg"]
        for key, tag in (("target", "part2_ema"), ("online", "part2_online")):
            if key not in blob:
                continue
            e = PhysicsEncoder(c["latent_dim"], c["d_model"], c["n_layers"],
                               c["d_state"], 0.0)
            e.load_state_dict(blob[key]); e.eval().to(device)
            arms[tag] = {s: encode(e, srcs[s], device, kind="part2") for s in srcs}

    if a.random_control:
        torch.manual_seed(0)
        e = PhysicsEncoder(64, 64, 2, 32, 0.0); e.eval().to(device)
        arms["random_encoder"] = {s: encode(e, srcs[s], device, kind="part2") for s in srcs}

    arms["moments_baseline"] = {s: moment_features(srcs[s]) for s in srcs}

    results = {}
    for name, Z in arms.items():
        ztr, ytr = Z["train"][keep["train"]], lab["train"][keep["train"]]
        row = {"dim": int(ztr.shape[1])}
        for s in ("val", "test"):
            zev, yev = Z[s][keep[s]], lab[s][keep[s]]
            bacc, cm, _ = probe(ztr, ytr, zev, yev, a.seed)
            row[f"{s}_bacc_linear"] = bacc
            row[f"{s}_bacc_knn"] = knn_probe(ztr, ytr, zev, yev)
            if s == "test":
                row["test_confusion"] = cm
        results[name] = row

    print(f"\n{'representation':<22}{'dim':>5}{'val linear':>12}{'val knn':>10}"
          f"{'test linear':>13}{'test knn':>10}")
    print("-" * 72)
    for n, r in results.items():
        print(f"{n:<22}{r['dim']:>5}{r['val_bacc_linear']:>12.4f}{r['val_bacc_knn']:>10.4f}"
              f"{r['test_bacc_linear']:>13.4f}{r['test_bacc_knn']:>10.4f}")
    print(f"\nchance = {1/len(names):.4f} ({len(names)} classes, balanced accuracy)")
    print("classes: " + ", ".join(names))

    out = {"results": results, "classes": names, "npz": a.npz,
           "part1": a.part1, "part2": a.part2, "seed": a.seed}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="artifacts/vicreg_jepa/phyts_s15.npz")
    p.add_argument("--part1", default=None)
    p.add_argument("--part2", default=None)
    p.add_argument("--random-control", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drop-class", nargs="*", default=None,
                   help='class names to exclude, e.g. "INSTRUMENT/JUNK"')
    p.add_argument("--out", default="artifacts/vicreg_jepa/classification.json")
    main(p.parse_args())
