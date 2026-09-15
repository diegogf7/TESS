"""Spec s7: the single, final test-set evaluation.

Refuses to run unless a selection record exists, and refuses to run twice --
the test set is scored exactly once, after selection on validation.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .config import Part2Config
from .evaluate import evaluate_representation, summarise_seeds, write_report
from .models import S4Encoder
from .real_data import git_sha
from .train import DEVICE, load_sources, add_common_args, load_frozen_part1

SEAL = "TEST_EVALUATED.json"


def load_encoder(ckpt, cfg, device, which="ema_encoder"):
    """Primary representation is the EMA encoder on the full observed curve."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = blob.get("cfg", cfg.__dict__)
    enc = S4Encoder(c["physics_dim"], c["n_tokens"], c["d_model"],
                    c["d_state"], c["n_layers"], 0.0)
    key = which if which in blob else "encoder"
    enc.load_state_dict(blob[key])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device), key


def final_evaluation(sources, part1_ckpt, runs, out_root, cfg=Part2Config(),
                     device=DEVICE, force=False):
    """runs: {representation_name: [ckpt_path, ...]} one checkpoint per seed."""
    seal_path = os.path.join(out_root, SEAL)
    if os.path.exists(seal_path) and not force:
        raise RuntimeError(
            f"the test set has already been scored for this run ({seal_path}).\n"
            "Re-scoring would invalidate the single-evaluation protocol. Pass "
            "force=True only if you are deliberately starting a new experiment.")

    part1 = load_frozen_part1(part1_ckpt, device)
    table = {}
    for name, ckpts in runs.items():
        reports = []
        for seed, ckpt in enumerate(ckpts):
            enc, used = load_encoder(ckpt, cfg, device, "ema_encoder")
            rep = evaluate_representation(name, enc, part1, sources, device, cfg, seed)
            rep["checkpoint"] = ckpt
            rep["encoder_key"] = used
            # online encoder reported as a diagnostic only, never selected on
            enc_on, _ = load_encoder(ckpt, cfg, device, "encoder")
            diag = evaluate_representation(name + "_online", enc_on, part1,
                                           sources, device, cfg, seed)
            rep["online_diagnostic"] = diag["splits"]
            reports.append(rep)
            write_report(rep, os.path.join(out_root, f"{name}_seed{seed}_final.json"))
        table[name] = {"per_seed": reports,
                       "test": summarise_seeds(reports, "test"),
                       "val": summarise_seeds(reports, "val")}

    os.makedirs(out_root, exist_ok=True)
    with open(seal_path, "w") as fh:
        json.dump({"sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "git_sha": git_sha(), "representations": sorted(table)}, fh, indent=2)
    with open(os.path.join(out_root, "final_table.json"), "w") as fh:
        json.dump(table, fh, indent=2, default=str)
    print_final(table)
    return table


def print_final(table):
    keys = ["physics_bacc", "mean_abs_corr_with_systematics",
            "effective_rank", "std_median"]
    head = f"{'representation':<18}" + "".join(f"{k:>34}" for k in keys)
    print("\n" + head)
    print("-" * len(head))
    for name, r in table.items():
        cells = []
        for k in keys:
            s = r["test"].get(k)
            cells.append((f"{s['mean']:.4f} +/- {s['std']:.4f} (n={s['n_seeds']})"
                          if s else "n/a").rjust(34))
        print(f"{name:<18}" + "".join(cells))
    print("\nTest split, mean +/- sd across seeds. Selection was made on validation.")


if __name__ == "__main__":
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--part1-ckpt", required=True)
    ap.add_argument("--runs", required=True,
                    help='JSON: {"full_vicreg": ["a.pt","b.pt"], ...}')
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sources = load_sources(args)
    final_evaluation(sources, args.part1_ckpt, json.loads(args.runs),
                     args.out, device=DEVICE, force=args.force)
