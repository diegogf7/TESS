
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
CHUNK = int(os.environ.get("GAIA_CHUNK", "5000"))
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

    # ra/dec: reuse the downloaded parquet where we already have it
    known = pd.read_parquet(DENSE_V2, columns=["GAIADR3", "ra", "dec"]).dropna()
    known["GAIADR3"] = known["GAIADR3"].astype("int64")
    known = known.drop_duplicates("GAIADR3").set_index("GAIADR3")
    downloaded_ids = set(known.index.tolist())
    bulk["ra"] = bulk["GAIADR3"].map(known["ra"])
    bulk["dec"] = bulk["GAIADR3"].map(known["dec"])
    print(f"already downloaded (ra/dec known): {bulk['ra'].notna().sum()}/{len(bulk)}", flush=True)

    need = bulk[bulk["ra"].isna()]["GAIADR3"].tolist()
    radec = {}
    if os.path.exists(CACHE):
        c = pd.read_parquet(CACHE)
        radec = {int(g): (float(r), float(d)) for g, r, d in zip(c["GAIADR3"], c["ra"], c["dec"])}
        need = [g for g in need if g not in radec]
        print(f"resumed {len(radec)} cached Gaia positions; {len(need)} still to query", flush=True)
    if need:
        radec.update(gaia_radec(need))
        pd.DataFrame([(g, r, d) for g, (r, d) in radec.items()],
                     columns=["GAIADR3", "ra", "dec"]).to_parquet(CACHE)
    fill = bulk["ra"].isna()
    bulk.loc[fill, "ra"] = bulk.loc[fill, "GAIADR3"].map(lambda g: radec.get(g, (np.nan, np.nan))[0])
    bulk.loc[fill, "dec"] = bulk.loc[fill, "GAIADR3"].map(lambda g: radec.get(g, (np.nan, np.nan))[1])

    resolved = bulk.dropna(subset=["ra", "dec"]).copy()
    resolved["sector"] = 14
    resolved = add_area(resolved)
    resolved = resolved[resolved["area"] != -1]
    resolved["downloaded"] = resolved["GAIADR3"].isin(downloaded_ids)
    print(f"resolved ra/dec + area for {len(resolved)}/{len(bulk)} targets "
          f"({len(bulk) - len(resolved)} unresolved -- no Gaia match)", flush=True)

    # per-area ceiling + how many are already downloaded
    rep = []
    dl_lines = []
    for area, g in resolved.groupby("area"):
        avail = len(g)
        dl = int(g["downloaded"].sum())
        need_more = max(0, TARGET_TOTAL - dl)
        candidates = g[~g["downloaded"]]
        take = candidates.head(need_more) if need_more else candidates.head(0)
        dl_lines.extend(take["curl"].tolist())
        rep.append({"area": int(area), "downloaded": dl, "available_total": avail,
                    "available_new": int(len(candidates)), "will_download": int(len(take)),
                    "reachable_target": bool(avail >= TARGET_TOTAL),
                    "projected_total": int(dl + len(take))})
    rep = sorted(rep, key=lambda r: r["projected_total"])

    with open(os.path.join(OUT_DIR, "s14_area_availability.json"), "w") as fh:
        json.dump({"target_total_per_area": TARGET_TOTAL, "n_areas": len(rep), "areas": rep}, fh, indent=2)
    pd.DataFrame(rep).to_csv(os.path.join(OUT_DIR, "s14_area_availability.csv"), index=False)
    tgt = os.path.join(OUT_DIR, "targeted_download.sh")
    with open(tgt, "w") as fh:
        fh.write("#!/bin/bash -l\nset -u\ncd " + ROOT + "\n")
        fh.write("\n".join(l.replace("curl -f", "curl -sf") for l in dl_lines) + "\n")

    unreachable = [r for r in rep if not r["reachable_target"]]
    print(f"\nTARGET (downloaded/area) = {TARGET_TOTAL}", flush=True)
    print(f"{'area':>5} {'downloaded':>10} {'avail_total':>12} {'will_dl':>8} {'projected':>10} reachable", flush=True)
    for r in rep:
        print(f"{r['area']:>5} {r['downloaded']:>10} {r['available_total']:>12} "
              f"{r['will_download']:>8} {r['projected_total']:>10} {r['reachable_target']}", flush=True)
    print(f"\nto download: {len(dl_lines)} new targets -> {tgt}", flush=True)
    if unreachable:
        print(f"UNREACHABLE ({len(unreachable)} areas have < {TARGET_TOTAL} total available -- "
              f"physical ceiling): {[r['area'] for r in unreachable]}", flush=True)
    print(f"wrote s14_area_availability.csv/json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
