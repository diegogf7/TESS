"""A clickable UMAP: pick a point, see that star's raw and cleaned curve.

Writes ONE self-contained HTML file with the curves embedded, so it opens locally with
no server and no network. Built for eyeballing physics candidates for transit-like
dips -- the cleaned curve is the one a real transit should survive.

Curves are downsampled and rounded before embedding, purely to keep the file openable;
the underlying analysis is untouched. Cleaning uses a quiet reference built on each
star's OWN chip, and stars on a chip with no usable reference are shown raw-only and
labelled as such.

UMAP position is a visualization. Proximity in the plot is not evidence of anything,
and a high percentile is a tail-of-distribution pick, not a confirmed planet.

    python -m disentangle_attempt.export_umap_explorer \
      --checkpoint .../multichip_5sectors_v1/best.pt \
      --scores .../anomaly_analysis_20k_pca90/anomaly_scores.csv \
      --latents .../anomaly_analysis_20k_pca90/physics_latents.npy \
      --parquet ... --out .../umap_explorer.html
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from disentangle_attempt.dataset import (CrossSectorPatch, infer_require_cross_sector,
                                         target_from_checkpoint)
from disentangle_attempt.fit_anomaly_flows import THRESHOLD
from disentangle_attempt.infer import dual_context_prediction
from disentangle_attempt.masking import complementary_masks
from disentangle_attempt.model import DisentangleModel
from disentangle_attempt.reference_context import build_reference_context
from disentangle_attempt.train import DEFAULT_PARQUET, pick_device

UMAP_KWARGS = dict(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine",
                   random_state=42)

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Physics-latent UMAP explorer</title>
<style>
 body{font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;
      background:#0f1117;color:#e6e8ee}
 header{padding:12px 18px;border-bottom:1px solid #262a36}
 h1{font-size:15px;margin:0 0 4px}
 .note{color:#9aa3b5;font-size:12px}
 #wrap{display:flex;gap:14px;padding:14px 18px;flex-wrap:wrap}
 canvas{background:#151823;border:1px solid #262a36;border-radius:6px}
 #side{min-width:320px;flex:1}
 #meta{margin:8px 0 10px}
 #meta b{color:#fff}
 label{margin-right:14px;color:#9aa3b5}
 .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-right:6px}
 .phy{background:#7f1d2e;color:#ffd9df}.ins{background:#123c63;color:#cfe6ff}
 .both{background:#4a2a6b;color:#e8d9ff}.typ{background:#23262f;color:#9aa3b5}
</style></head><body>
<header>
 <h1>Physics-latent UMAP &mdash; click a point to see its light curve</h1>
 <div class="note">Colour = physics anomaly percentile. Ringed = &ge; __THRESHOLD__.
 UMAP position is visualization only; a high percentile is a tail-of-distribution pick,
 not a confirmed planet. Cleaned = raw &minus; correction, a quiet-context
 counterfactual. Only ringed candidates are clickable; faded points are context.</div>
</header>
<div id="wrap">
 <div>
  <canvas id="map" width="620" height="560"></canvas>
  <div style="margin-top:8px">
   <label><input type="checkbox" id="onlyCand"> show candidates only</label>
   <label><input type="checkbox" id="quietInst"> quiet instrument (&lt;0.5)</label>
   <label><input type="checkbox" id="clickAny"> allow clicking non-candidates</label>
  </div>
 </div>
 <div id="side">
  <canvas id="curve" width="720" height="300"></canvas>
  <div id="meta" class="note">Click a point.</div>
  <canvas id="corr" width="720" height="150"></canvas>
 </div>
</div>
<script>
const DATA = __DATA__;
const TH = __THRESHOLD__;
const map = document.getElementById('map'), mctx = map.getContext('2d');
const cur = document.getElementById('curve'), cctx = cur.getContext('2d');
const cor = document.getElementById('corr'), rctx = cor.getContext('2d');
const meta = document.getElementById('meta');
const onlyCand = document.getElementById('onlyCand'), quietInst = document.getElementById('quietInst');
const clickAny = document.getElementById('clickAny');
let sel = -1;

const xs = DATA.points.map(p => p.x), ys = DATA.points.map(p => p.y);
const xmin = Math.min(...xs), xmax = Math.max(...xs);
const ymin = Math.min(...ys), ymax = Math.max(...ys);
const PAD = 26;
const px = x => PAD + (x - xmin) / (xmax - xmin || 1) * (map.width - 2 * PAD);
const py = y => map.height - PAD - (y - ymin) / (ymax - ymin || 1) * (map.height - 2 * PAD);

function colour(v){                       // viridis-ish ramp
  const s = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const t = Math.max(0, Math.min(1, v)) * (s.length - 1);
  const i = Math.floor(t), f = t - i, a = s[i], b = s[Math.min(i + 1, s.length - 1)];
  return `rgb(${a.map((c,k)=>Math.round(c+(b[k]-c)*f)).join(',')})`;
}
function visible(p){
  if (onlyCand.checked && p.pp < TH) return false;
  if (quietInst.checked && p.ip >= 0.5) return false;
  return true;
}
function hasCurve(p){ return p.raw !== undefined; }
// Only ringed candidates are selectable unless the box is ticked.
function clickable(p){ return hasCurve(p) && visible(p) && (clickAny.checked || p.pp >= TH); }
function drawMap(){
  mctx.clearRect(0,0,map.width,map.height);
  DATA.points.forEach((p,i)=>{
    if(!visible(p)) return;
    mctx.globalAlpha = hasCurve(p) ? 1 : 0.45;
    mctx.fillStyle = colour(p.pp);
    mctx.beginPath(); mctx.arc(px(p.x),py(p.y), i===sel?5:1.7, 0, 7); mctx.fill();
    mctx.globalAlpha = 1;
    if(p.pp>=TH){ mctx.strokeStyle='#ff5d73'; mctx.lineWidth=0.9;
      mctx.beginPath(); mctx.arc(px(p.x),py(p.y),5.5,0,7); mctx.stroke(); }
    if(i===sel){ mctx.strokeStyle='#fff'; mctx.lineWidth=1.6;
      mctx.beginPath(); mctx.arc(px(p.x),py(p.y),8,0,7); mctx.stroke(); }
  });
}
function axes(ctx,w,h,lo,hi,label){
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#2b303d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(38,h-22); ctx.lineTo(w-8,h-22); ctx.stroke();
  ctx.fillStyle='#9aa3b5'; ctx.font='10px sans-serif';
  ctx.fillText(hi.toFixed(1), 4, 14); ctx.fillText(lo.toFixed(1), 4, h-26);
  ctx.fillText(label, 44, 12);
}
function series(ctx,w,h,vals,lo,hi,col){
  const n = vals.length;
  ctx.fillStyle = col;
  for(let i=0;i<n;i++){
    const v = vals[i];
    if(v===null) continue;
    const x = 38 + i/(n-1)*(w-46);
    const y = (h-22) - (v-lo)/((hi-lo)||1)*(h-34);
    ctx.fillRect(x,y,1.6,1.6);
  }
}
function show(i){
  sel = i; const p = DATA.points[i];
  const raw = p.raw, cl = p.cleaned;
  const finite = raw.filter(v=>v!==null);
  let lo = Math.min(...finite), hi = Math.max(...finite);
  if (cl){ const f2 = cl.filter(v=>v!==null); lo = Math.min(lo,...f2); hi = Math.max(hi,...f2); }
  const padv = (hi-lo)*0.08 || 1; lo-=padv; hi+=padv;
  axes(cctx,cur.width,cur.height,lo,hi,'normalized flux — grey raw, blue cleaned');
  series(cctx,cur.width,cur.height,raw,lo,hi,'#8b93a7');
  if(cl) series(cctx,cur.width,cur.height,cl,lo,hi,'#5aa9ff');
  if(cl){
    const d = raw.map((v,k)=> (v===null||cl[k]===null)?null:(v-cl[k]));
    const df = d.filter(v=>v!==null);
    let dl = Math.min(...df), dh = Math.max(...df);
    const dp = (dh-dl)*0.1 || 1; dl-=dp; dh+=dp;
    axes(rctx,cor.width,cor.height,dl,dh,'correction = raw − cleaned');
    series(rctx,cor.width,cor.height,d,dl,dh,'#ff6b6b');
  } else { rctx.clearRect(0,0,cor.width,cor.height); }
  const cls = {physics:'phy',instrument:'ins',both:'both',typical:'typ'}[p.cls]||'typ';
  meta.innerHTML = `<b>TIC ${p.tic}</b> &nbsp; chip ${p.chip} &nbsp; split ${p.split}
    <br><span class="pill ${cls}">${p.cls}</span>
    physics <b>${p.pp.toFixed(3)}</b> &nbsp; instrument <b>${p.ip.toFixed(3)}</b>
    &nbsp; valid cadences ${p.nv}
    ${cl?'':'<br><span class="note">no quiet reference on this chip — raw only</span>'}`;
  drawMap();
}
map.addEventListener('click', e=>{
  const r = map.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  let best=-1, bd=1e9;
  DATA.points.forEach((p,i)=>{ if(!clickable(p)) return;
    const d=(px(p.x)-mx)**2+(py(p.y)-my)**2; if(d<bd){bd=d;best=i;} });
  if(best>=0 && bd<900) show(best);
});
onlyCand.onchange = quietInst.onchange = clickAny.onchange = drawMap;
map.addEventListener('mousemove', e=>{
  const r = map.getBoundingClientRect();
  const mx = e.clientX-r.left, my = e.clientY-r.top;
  let near = false;
  DATA.points.forEach(p=>{ if(!clickable(p)) return;
    if((px(p.x)-mx)**2+(py(p.y)-my)**2 < 900) near = true; });
  map.style.cursor = near ? 'pointer' : 'default';
});
drawMap();
const first = DATA.points.findIndex(p=>hasCurve(p) && p.pp>=TH);
if(first>=0) show(first); else { const any=DATA.points.findIndex(hasCurve);
  if(any>=0) show(any); }
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--umap-points", type=int, default=10000,
                        help="points drawn in the scatter (coordinates only: cheap)")
    parser.add_argument("--max-curves", type=int, default=0,
                        help="cap on candidates embedded (0 = every candidate)")
    parser.add_argument("--background", type=int, default=1100,
                        help="non-candidates whose curves are embedded")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--require-cross-sector", default="auto",
                        choices=("auto", "yes", "no"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.scores)),
                                   "umap_explorer.html")
    table = pd.read_csv(args.scores)
    latents = np.load(args.latents)
    assert len(table) == len(latents), "scores and latents disagree in length"

    keep = np.ones(len(table), dtype=bool) if args.split == "all" \
        else (table["split"] == args.split).to_numpy()
    rng = np.random.default_rng(args.seed)

    # Every point in `shown` is drawn; only `chosen` carries an embedded light curve.
    # Coordinates cost ~100 bytes a star, a curve costs ~1 KB, so this keeps the scatter
    # as dense as the static figure while the file stays openable.
    eligible = np.flatnonzero(keep)
    is_candidate = table["physics_percentile"].to_numpy() >= THRESHOLD
    candidates = eligible[is_candidate[eligible]]
    others = eligible[~is_candidate[eligible]]
    if args.max_curves and len(candidates) > args.max_curves:
        order = np.argsort(-table["physics_nll"].to_numpy()[candidates])
        candidates = candidates[order[:args.max_curves]]

    # Every candidate is drawn AND carries a curve, so no ringed point is ever
    # unclickable. Background stars only fill out the shape of the embedding.
    room = max(args.umap_points - len(candidates), 0)
    background_shown = rng.permutation(others)[:room]
    shown = np.sort(np.concatenate([candidates, background_shown]))
    background_curves = rng.permutation(background_shown)[:args.background]
    chosen = set(int(v) for v in np.concatenate([candidates, background_curves]))
    print(f"{len(shown)} points drawn ({len(candidates)} candidates, all clickable) | "
          f"{len(chosen)} with embedded curves", flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    device = pick_device(config.get("device", "auto"))
    sector, camera, ccd = target_from_checkpoint(state, config)
    patch = CrossSectorPatch(
        args.parquet or config.get("parquet") or DEFAULT_PARQUET,
        target_sector=sector, camera=camera, ccd=ccd,
        curve_length=config["curve_length"], n_peers=config["n_peers"],
        min_valid_fraction=config.get("min_valid_fraction", 0.5),
        split_seed=config["seed"], max_eligible_anchors=config.get("max_eligible_anchors"),
        require_cross_sector=infer_require_cross_sector(config, args.require_cross_sector),
        verbose=False)
    model = DisentangleModel(d_model=config.get("d_model", 128),
                             n_layers=config.get("n_layers", 4), dropout=0.0,
                             n_peers=config["n_peers"], n_tokens=config["n_tokens"],
                             token_dim=config["token_dim"],
                             curve_length=config["curve_length"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    masks = complementary_masks(config["curve_length"], n_masks=4)

    print("fitting UMAP on the physics latents of the selected points", flush=True)
    import umap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = np.asarray(umap.UMAP(**UMAP_KWARGS).fit_transform(
            StandardScaler().fit_transform(latents[shown])))

    curve_index = np.asarray(sorted(chosen))
    rows = table["row"].to_numpy()[curve_index]
    splits = table["split"].to_numpy()[curve_index]
    chips = [(int(patch.sector[r]), int(patch.camera[r]), int(patch.ccd[r])) for r in rows]
    cleaned = {}
    for chip in sorted(set(chips)):
        members = [int(r) for r, c in zip(rows, chips) if c == chip]
        member_splits = [s for s, c in zip(splits, chips) if c == chip]
        try:
            reference = build_reference_context(patch, "train", config["n_peers"],
                                                chip=chip, verbose=False)
        except RuntimeError:
            print(f"  chip {chip}: no quiet reference, {len(members)} stars raw-only",
                  flush=True)
            continue
        peers = np.stack([patch.peers_for_row(r, s)[0]
                          for r, s in zip(members, member_splits)])
        quiet_flux = torch.from_numpy(patch.X[reference["rows"]]).unsqueeze(0)
        quiet_mask = torch.from_numpy(patch.M[reference["rows"]]).unsqueeze(0)
        actual, ref, _, _, _ = dual_context_prediction(
            model, torch.from_numpy(patch.X[members]), torch.from_numpy(patch.M[members]),
            torch.from_numpy(patch.X[peers]), torch.from_numpy(patch.M[peers]),
            quiet_flux.expand(len(members), -1, -1),
            quiet_mask.expand(len(members), -1, -1), masks, device)
        correction = (actual - ref).numpy()
        for k, row in enumerate(members):
            cleaned[row] = patch.X[row] - correction[k]

    step = max(int(args.downsample), 1)

    def encode(values, valid):
        """Gaps become null so the plot breaks rather than drawing a zero."""
        return [None if not valid[i] else round(float(values[i]), 2)
                for i in range(0, len(values), step)]

    def curve_count():
        return sum("raw" in p for p in points)

    curve_rows = {int(index): int(row) for index, row in zip(curve_index, rows)}
    points = []
    for k, index in enumerate(shown):
        record = table.iloc[index]
        point = {
            "tic": str(record["TIC"]), "chip": str(record["chip"]),
            "split": str(record["split"]), "cls": str(record["classification"]),
            "pp": float(record["physics_percentile"]),
            "ip": float(record["instrument_percentile"]),
            "nv": int(record["valid_cadences"]),
            "x": round(float(coords[k, 0]), 2), "y": round(float(coords[k, 1]), 2),
        }
        row = curve_rows.get(int(index))
        if row is not None:
            valid = patch.M[row]
            point["raw"] = encode(patch.X[row], valid)
            point["cleaned"] = encode(cleaned[row], valid) if row in cleaned else None
        points.append(point)

    html = TEMPLATE.replace("__DATA__", json.dumps({"points": points},
                                                   separators=(",", ":")))
    html = html.replace("__THRESHOLD__", f"{THRESHOLD}")
    with open(out, "w") as handle:
        handle.write(html)
    size = os.path.getsize(out) / 1e6
    with_curve = sum("raw" in p for p in points)
    with_clean = sum(p.get("cleaned") is not None for p in points)
    print(f"wrote {out} ({size:.1f} MB, {len(points)} points drawn, "
          f"{with_curve} with curves, {with_clean} with cleaned curves)")


if __name__ == "__main__":
    main()
