# All this code is from Claude
"""Stage C: frozen probes + final report for the fixed-teacher pilot.

  STAGE=probes  frozen student probes: camera, CCD, 16-way camCCD, area,
                ring-within-CCD. Validation only.
  STAGE=report  ALWAYS writes final_summary.{md,json}, whatever exists:
                teacher results, student trajectory, probes, matched LP-FT
                (best val LR per arm), diffs vs group-JEPA (0.5434) and
                scratch (0.5559), teacher-hash confirmation, test statement.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import build_fixed_teacher_jepa
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import git_commit

STAGE = os.environ.get("STAGE", "report")
SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
ART_DIR = os.environ.get(
    "FRT_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "fixed_regional_teacher_v1"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUPJEPA_REF = 0.5434
SCRATCH_REF = 0.5559
FT_ARMS = ("scratch", "groupjepa", "fixed_teacher")
SEEDS = (SEED,)  # seed-0 pilot


def maybe(path):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return None


def probes():
    student_sel = maybe(os.path.join(ART_DIR, f"selection_frtstudent_k{K}_s{SEED}.json"))
    if student_sel is None:
        print("SKIP probes: no student selection", flush=True)
        return
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", K)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", K)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics

    model = build_fixed_teacher_jepa().to(DEVICE)
    model.load_state_dict(torch.load(student_sel["checkpoint"],
                                     map_location=DEVICE))
    model.eval()
    train_z = individual_latents(model, train_ds, "online")
    val_z = individual_latents(model, val_ds, "online")

    def labels(ds):
        return {"camera": ds.chips // 4 + 1, "ccd": ds.chips % 4 + 1,
                "camccd_16way": ds.chips, "area": ds.areas,
                "ring_within_ccd": ds.areas % 10}

    train_y, val_y = labels(train_ds), labels(val_ds)
    results = {}
    for name in train_y:
        bacc = fast_probe(train_z, train_y[name], val_z, val_y[name])
        chance = 1.0 / len(np.unique(train_y[name]))
        results[name] = {"val_bacc": bacc, "chance": chance}
        print(f"frozen probe {name}: {bacc:.4f} (chance {chance:.4f})",
              flush=True)
    with open(os.path.join(ART_DIR, "frozen_probes.json"), "w") as handle:
        json.dump(results, handle, indent=2)


def student_trajectory():
    path = os.path.join(ART_DIR, f"metrics_frtstudent_k{K}_s{SEED}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return list(csv.DictReader(handle))


def best_ft(arm):
    paths = glob.glob(os.path.join(ART_DIR, f"result_lpft_{arm}_s{SEED}_lr*.json"))
    if not paths:
        return None
    results = []
    for p in paths:
        with open(p) as handle:
            results.append(json.load(handle))
    return max(results, key=lambda r: r["best_val_bacc16"])


def report():
    os.makedirs(ART_DIR, exist_ok=True)
    teacher = maybe(os.path.join(ART_DIR, f"selection_regteacher_k{K}_s{SEED}.json"))
    student = maybe(os.path.join(ART_DIR, f"selection_frtstudent_k{K}_s{SEED}.json"))
    frozen = maybe(os.path.join(ART_DIR, "frozen_probes.json"))
    trajectory = student_trajectory()
    finetune = {arm: best_ft(arm) for arm in FT_ARMS}
    ft_complete = all(finetune.values())

    summary = {"git_commit": git_commit(),
               "teacher": teacher, "student": student,
               "student_trajectory": trajectory,
               "frozen_probes": frozen,
               "finetune_best_per_arm": {
                   arm: (r["best_val_bacc16"] if r else None)
                   for arm, r in finetune.items()},
               "references": {"groupjepa_val": GROUPJEPA_REF,
                              "scratch_val": SCRATCH_REF},
               "teacher_frozen_confirmed": bool(
                   student and student.get("teacher_hash_verified_every_epoch")),
               "test_untouched": ("Test TICs were never loaded, selected on, "
                                  "or evaluated anywhere in this pilot; all "
                                  "numbers are validation-only.")}
    if ft_complete:
        ours = finetune["fixed_teacher"]["best_val_bacc16"]
        summary["diffs"] = {
            "fixed_teacher_minus_groupjepa":
                ours - finetune["groupjepa"]["best_val_bacc16"],
            "fixed_teacher_minus_scratch":
                ours - finetune["scratch"]["best_val_bacc16"]}

    with open(os.path.join(ART_DIR, "final_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    lines = ["# fixed_regional_teacher_v1 -- final summary (validation-only)",
             "", f"git commit: {summary['git_commit']}", ""]
    if teacher:
        lines += ["## Stage A: regional teacher",
                  f"passed gates: {teacher.get('passed_gates')}",
                  json.dumps(teacher.get("best"), indent=2), ""]
    else:
        lines += ["## Stage A: regional teacher NEVER RAN", ""]
    if student:
        lines += ["## Stage B: student (best epoch)",
                  json.dumps(student.get("best"), indent=2),
                  f"teacher hash verified every epoch: "
                  f"{summary['teacher_frozen_confirmed']}", ""]
    else:
        lines += ["## Stage B: student SKIPPED "
                  "(teacher gates refused or never ran)", ""]
    if trajectory:
        lines += ["## Student trajectory (camccd bacc per epoch)",
                  ", ".join(f"{float(r['val_camccd_bacc']):.4f}"
                            for r in trajectory), ""]
    if frozen:
        lines += ["## Frozen student probes"]
        for name, entry in frozen.items():
            lines.append(f"- {name}: {entry['val_bacc']:.4f} "
                         f"(chance {entry['chance']:.4f})")
        lines.append("")
    lines += ["## Matched LP-FT (best val backbone LR per arm)"]
    for arm in FT_ARMS:
        r = finetune[arm]
        lines.append(f"- {arm}: "
                     + (f"{r['best_val_bacc16']:.4f} (lr {r['backbone_lr']:g})"
                        if r else "missing"))
    if ft_complete:
        lines += ["",
                  f"fixed_teacher - groupjepa: "
                  f"{summary['diffs']['fixed_teacher_minus_groupjepa']:+.4f}",
                  f"fixed_teacher - scratch:   "
                  f"{summary['diffs']['fixed_teacher_minus_scratch']:+.4f}"]
    lines += ["", f"references: groupjepa {GROUPJEPA_REF}, scratch {SCRATCH_REF}",
              "", summary["test_untouched"], ""]
    md_path = os.path.join(ART_DIR, "final_summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"final report -> {md_path}", flush=True)


if __name__ == "__main__":
    if STAGE == "probes":
        probes()
    elif STAGE == "report":
        report()
    else:
        raise SystemExit(f"unknown STAGE {STAGE!r}")
