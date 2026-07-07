"""Inspect a TGLC light-curve FITS file and plot raw vs calibrated flux.

First run tells us the REAL column/header names (the parquet extractor gets
written against this output). The plot is the advisor's raw-vs-TGLC pair check.

Usage: python -m src.tglc.inspect_fits <file.fits> [raw_col] [cal_col]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

path = sys.argv[1]
raw_col = sys.argv[2] if len(sys.argv) > 2 else "aperture_flux"
cal_col = sys.argv[3] if len(sys.argv) > 3 else "cal_aper_flux"

with fits.open(path) as hdul:
    hdul.info()
    print("\ncolumns in HDU 1:", [c.name for c in hdul[1].columns])
    print("\nprimary header:")
    for card in hdul[0].header.cards[:40]:
        print(" ", card.image)

    data = hdul[1].data
    t = data["time"]
    raw = data[raw_col]
    cal = data[cal_col]

    good = (data["TESS_flags"] == 0) & (data["TGLC_flags"] == 0)   # NEW
    print(f"\nkept {good.sum()}/{good.size} cadences ({100*good.mean():.1f}%)")  # NEW
    t, raw, cal = t[good], raw[good], cal[good]                    # NEW


    tess_flags = data["TESS_flags"]
    tglc_flags = data["TGLC_flags"]

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t, raw, ".", ms=2)
axes[0].set_ylabel(f"Raw flux")
axes[1].plot(t, cal, ".", ms=2)
axes[1].set_ylabel(f"Calibrated flux")
axes[1].set_xlabel("time [BTJD]")
fig.suptitle(path.split("/")[-1], fontsize=9)
fig.tight_layout()
fig.savefig("tglc_pair.png", dpi=150)
print("\nwrote tglc_pair.png")
