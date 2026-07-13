import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.environ.get("AREA_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_full_area.parquet")
SECTOR = int(os.environ.get("AREA_SECTOR", 14))

CAMERA = int(os.environ.get("AREA_CAMERA", 1))
CCD = int(os.environ.get("AREA_CCD", 1))
RING = int(os.environ.get("AREA_RING", 3))
N_CURVES = int(os.environ.get("AREA_NCURVES", 6))
SEED = int(os.environ.get("AREA_SEED", 0))

ROBUST_YLIM = os.environ.get("AREA_ROBUST_YLIM", "1") == "1"

AREA = CAMERA * 100 + CCD * 10 + RING
OUT = os.environ.get("AREA_OUT", f"area_curves_s{SECTOR}_a{AREA}.png")
df = pd.read_parquet(DATA)
sel = df[(df.sector == SECTOR) & (df.area == AREA)]

print(f"Sector {SECTOR}, camera {CAMERA}, ccd {CCD}, ring {RING} (area {AREA}): {len(sel)} curves")

rows = sel.sample(n = min(N_CURVES, len(sel)), random_state = SEED)

figure, axis = plt.subplots(len(rows), 1, figsize = (11, 2.5 * len(rows)), sharex = True)
axis = np.atleast_1d(axis)

for ax, (_, r) in zip(axis, rows.iterrows()):

    time = np.asarray(r["time"], dtype = float)
    flux = np.asarray(r["flux"], dtype = float)

    relative = flux / np.median(flux) - 1.0

    ax.plot(time, relative, lw = 0.5, color = "black")

    ax.set_ylabel("relative flux")
    ax.set_title(f"GAIA {r['GAIADR3']} camera distance: {r['cam_dist']:.1f} degrees", fontsize = 10)

axis[-1].set_xlabel("Time (BTJD)")
figure.suptitle(f"Raw TGLC: sector {SECTOR}, camera {CAMERA}, CCD {CCD}, REGION {RING}, area {AREA}", y = 1.0)
figure.tight_layout()
figure.savefig(OUT, dpi = 150)

print(f"Wrote {OUT}")
