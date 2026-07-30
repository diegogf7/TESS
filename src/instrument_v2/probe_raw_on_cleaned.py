from __future__ import annotations
"""QUICK cross-domain eyeball (NOT a defensible A/B result).

Probes the EXISTING frozen RAW physics JEPA on raw vs new-model-CLEANED eval
curves -- i.e. the top-right control cell of the 2x2 (raw JEPA on cleaned
inputs). Because the raw encoder never trained on cleaned curves, any drop here
is confounded with DOMAIN SHIFT, not evidence that cleaning helps physics. Use
only as a rough look; the defensible test still needs a cleaned arm trained on
cleaned curves (stage train --arm cbv, then stage evaluate).

Reuses the pipeline's exact probe (same cached eval arrays, same encode_physics,
same KNN _classify, same TIC-disjoint split random_state=0) so the numbers are
directly comparable to run_tglc_physics_jepa_ab's raw_jepa_on_raw cell.

Requires: STAGE=prepare already run with the desired CLEAN_MODE / INST_CKPT /
GROUP_ART_DIR so OUT_DIR/prepared holds eval_raw_* and eval_cleaned_* arrays,
plus an existing raw JEPA at RAW_CKPT_DIR/physics_jepa_raw_s{SEED}_best.pth.

    OUT_DIR=... SEED=0 python -m src.instrument_v2.probe_raw_on_cleaned
"""

import json
import os

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from src.worked_folder.physics.latent_jepa import build_latent_jepa
from src.instrument_v2.eval_phyts_instrument_ab import DEVICE, encode_physics
from src.instrument_v2.run_tglc_physics_jepa_ab import (
    PREP_DIR, OUT_DIR, RAW_CKPT_DIR, CLASSES, _classify, _require_valid_prepared,
)

SEED = int(os.environ.get("SEED", "0"))


def main():
    manifest = _require_valid_prepared()
    em = np.load(os.path.join(PREP_DIR, "eval_meta.npz"), allow_pickle=True)
    y, tics = em["y"], em["tics"].astype(str)
    present = np.unique(y)
    X = {k: np.load(os.path.join(PREP_DIR, f"eval_{k}_X.npy")) for k in ("raw", "cleaned")}
    M = {k: np.load(os.path.join(PREP_DIR, f"eval_{k}_M.npy")) for k in ("raw", "cleaned")}

    ck = os.path.join(RAW_CKPT_DIR, f"physics_jepa_raw_s{SEED}_best.pth")
    if not os.path.exists(ck):
        raise RuntimeError(f"missing raw physics JEPA {ck}; train the raw arm first")
    model = build_latent_jepa().to(DEVICE)
    model.load_state_dict(torch.load(ck, map_location=DEVICE), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # identical TIC-disjoint split to stage_evaluate (random_state=0)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(len(tics)), y, groups=tics))

    acc = {}
    for dm in ("raw", "cleaned"):
        lat = encode_physics(model, X[dm], M[dm])
        acc[dm], _rec, _pred = _classify(lat, y, tr, te, present)
    delta = acc["cleaned"] - acc["raw"]

    print(f"RAW physics JEPA (frozen), seed {SEED}, clean_mode={manifest.get('clean_mode')}", flush=True)
    print(f"  raw_jepa_on_raw     : {acc['raw']:.4f}", flush=True)
    print(f"  raw_jepa_on_cleaned : {acc['cleaned']:.4f}   (CONTROL: off-distribution inputs)", flush=True)
    print(f"  cleaned - raw       : {delta:+.4f}   (negative = cleaning hurts the frozen raw encoder)", flush=True)

    out = {"seed": SEED, "clean_mode": manifest.get("clean_mode"),
           "raw_jepa_on_raw": acc["raw"], "raw_jepa_on_cleaned": acc["cleaned"],
           "cleaned_minus_raw": delta, "n_eval": int(len(tics)),
           "classes": [CLASSES[c] for c in present], "raw_ckpt": ck,
           "note": "cross-domain CONTROL cell only; raw encoder on cleaned inputs is "
                   "confounded by domain shift -- not a defensible cleaning A/B"}
    dest = os.path.join(OUT_DIR, f"probe_raw_on_cleaned_s{SEED}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
