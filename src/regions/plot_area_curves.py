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

rows = sel.sample(n = 1, random_state = SEED).iloc[0]


time = np.asarray(rows["time"], dtype = float)
flux = np.asarray(rows["flux"], dtype = float)

relative = flux / np.median(flux) - 1.0

figure, axis = plt.subplots(figsize = (11, 4))

axis.plot(time, relative, ".", ms = 2)
axis.set_ylabel("Relative flux")
axis.set_xlabel("Time [BTJD]")
figure.suptitle(f"Raw TGLC curve: sector {SECTOR}, area {AREA}, camera distance {r['cam_dist']:.1f} degrees", fontsize = 10)

figure.tight_layout()
figure.savefig(OUT, dpi = 150)

print(f"Wrote {OUT}")
