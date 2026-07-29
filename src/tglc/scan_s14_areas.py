
import json
import os
import re

import numpy as np
import pandas as pd

from src.regions.areas import add_area

ROOT = os.environ.get("ROOT", "/orcd/scratch/orcd/006/diegogon/tglc_primary")
BULK = os.environ.get("BULK", os.path.join(ROOT, "s0014_full.sh"))
DENSE_V2 = os.environ.get("DENSE_V2", os.path.join(ROOT, "tglc_raw_cadence_s14_dense_v2.parquet"))
OUT_DIR = os.environ.get("OUT_DIR", ROOT)
TARGET_TOTAL = int(os.environ.get("TARGET_TOTAL", "1400"))   # downloaded/area to leave 1000 hopefully train after split
CHUNK = int(os.environ.get("GAIA_CHUNK", "20000"))           # Gaia ids per query
MAX_NEW = int(os.environ.get("MAX_NEW", "0"))                # >0 = stop after querying this many candidates (bound runtime)
CACHE = os.path.join(OUT_DIR, "s14_bulk_area_scan.parquet")   # resumable Gaia crossmatch cache

LINE_RE = re.compile(r"gaiaid-(\d+)-s0014-cam(\d)-ccd(\d)")


def parse_bulk(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.lstrip().startswith("curl"):
                continue
            m = LINE_RE.search(line)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), line.rstrip("\n")))
    df = pd.DataFrame(rows, columns=["GAIADR3", "camera", "ccd", "curl"])
    return df.drop_duplicates("GAIADR3").reset_index(drop=True)


def gaia_radec(source_ids):
    """ra/dec for a list of Gaia DR3 source ids, chunked."""
    from astroquery.gaia import Gaia
    out = {}
    ids = list(source_ids)
    for k in range(0, len(ids), CHUNK):
        chunk = ids[k:k + CHUNK]
        q = ("SELECT source_id, ra, dec FROM gaiadr3.gaia_source "
             f"WHERE source_id IN ({','.join(str(int(s)) for s in chunk)})")
        try:
            t = Gaia.launch_job(q).get_results()
            for sid, ra, dec in zip(t["source_id"], t["ra"], t["dec"]):
                out[int(sid)] = (float(ra), float(dec))
        except Exception as e:
            print(f"  Gaia chunk {k}-{k+len(chunk)} failed: {e}", flush=True)
        print(f"  Gaia {min(k + CHUNK, len(ids))}/{len(ids)} (matched {len(out)})", flush=True)
    return out


def main():
    print(f"bulk list : {BULK}", flush=True)
    bulk = parse_bulk(BULK)
    print(f"parsed {len(bulk)} unique S14 targets from the bulk list", flush=True)

    # what we already have, per area (dense_v2 already carries `area`)
    dv = pd.read_parquet(DENSE_V2, columns=["GAIADR3", "area"]).dropna()
    dv["GAIADR3"] = dv["GAIADR3"].astype("int64")
    downloaded_ids = set(dv["GAIADR3"].tolist())
    dl_counts = {int(a): int(n) for a, n in dv.groupby("area").size().items()}
    needed = {a: TARGET_TOTAL - n for a, n in dl_counts.items() if n < TARGET_TOTAL}   # short areas -> slots to fill
    print(f"already downloaded: {len(downloaded_ids)} stars in {len(dl_counts)} areas; "
          f"{len(needed)} areas below {TARGET_TOTAL} (need {sum(needed.values())} new)", flush=True)
    if not needed:
        print("all areas already at target; nothing to download", flush=True); return

    # candidates = NOT-yet-downloaded, shuffled so we don't march through one chip
    cand = bulk[~bulk["GAIADR3"].isin(downloaded_ids)].sample(frac=1, random_state=0).reset_index(drop=True)
    print(f"{len(cand)} candidate targets to consider (streaming, stop when short areas fill)", flush=True)

    radec = {}
    if os.path.exists(CACHE):
        c = pd.read_parquet(CACHE)
        radec = {int(g): (float(r), float(d)) for g, r, d in zip(c["GAIADR3"], c["ra"], c["dec"])}
        print(f"resumed {len(radec)} cached Gaia positions", flush=True)

    dl_lines, new_found, queried = [], {}, 0
    for k in range(0, len(cand), CHUNK):
        chunk = cand.iloc[k:k + CHUNK].copy()
        miss = [int(g) for g in chunk["GAIADR3"] if int(g) not in radec]
        if miss:
            radec.update(gaia_radec(miss))
            pd.DataFrame([(g, r, d) for g, (r, d) in radec.items()],
                         columns=["GAIADR3", "ra", "dec"]).to_parquet(CACHE)
        chunk["ra"] = chunk["GAIADR3"].map(lambda g: radec.get(int(g), (np.nan, np.nan))[0])
        chunk["dec"] = chunk["GAIADR3"].map(lambda g: radec.get(int(g), (np.nan, np.nan))[1])
        chunk = chunk.dropna(subset=["ra", "dec"])
        chunk["sector"] = 14
        chunk = add_area(chunk)
        for row in chunk.itertuples():                       # keep only short-area hits still needing slots
            a = int(row.area)
            if needed.get(a, 0) > 0:
                dl_lines.append(row.curl)
                needed[a] -= 1
                new_found[a] = new_found.get(a, 0) + 1
        queried += len(chunk)
        remaining = sum(v for v in needed.values() if v > 0)
        print(f"  queried {queried} | collected {len(dl_lines)} | short slots left {remaining}", flush=True)
        if remaining == 0:
            print("all short areas filled -- stopping early", flush=True); break
        if MAX_NEW and queried >= MAX_NEW:
            print(f"hit MAX_NEW={MAX_NEW} -- stopping (some areas may still be short)", flush=True); break

    rep = []
    for a in sorted(dl_counts):
        got = new_found.get(a, 0)
        rep.append({"area": a, "downloaded": dl_counts[a], "new_found": got,
                    "projected_total": dl_counts[a] + got,
                    "still_short": max(0, TARGET_TOTAL - dl_counts[a] - got)})
    rep = sorted(rep, key=lambda r: r["projected_total"])

    with open(os.path.join(OUT_DIR, "s14_area_availability.json"), "w") as fh:
        json.dump({"target_total_per_area": TARGET_TOTAL, "n_areas": len(rep), "areas": rep}, fh, indent=2)
    pd.DataFrame(rep).to_csv(os.path.join(OUT_DIR, "s14_area_availability.csv"), index=False)
    tgt = os.path.join(OUT_DIR, "targeted_download.sh")
    with open(tgt, "w") as fh:
        fh.write("#!/bin/bash -l\nset -u\ncd " + ROOT + "\n")
        fh.write("\n".join(l.replace("curl -f", "curl -sf") for l in dl_lines) + "\n")

    still_short = [r for r in rep if r["still_short"] > 0]
    print(f"\nTARGET (downloaded/area) = {TARGET_TOTAL}", flush=True)
    print(f"{'area':>5} {'downloaded':>10} {'new_found':>9} {'projected':>10} {'still_short':>11}", flush=True)
    for r in rep:
        print(f"{r['area']:>5} {r['downloaded']:>10} {r['new_found']:>9} "
              f"{r['projected_total']:>10} {r['still_short']:>11}", flush=True)
    print(f"\nto download: {len(dl_lines)} new targets -> {tgt}", flush=True)
    if still_short:
        print(f"STILL SHORT after this scan ({len(still_short)} areas -- ran out of candidates "
              f"or hit MAX_NEW): {[(r['area'], r['still_short']) for r in still_short]}", flush=True)
    else:
        print("all short areas filled to target", flush=True)
    print(f"wrote s14_area_availability.csv/json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
