"""Visualization + diagnostics for PyramidExplainer's second-order (synergy) part.

This module is the pyramid-NATIVE companion to the grid-based Hessian-IG view.
The central point: pyramid's Delta lives on TREE NODES (nested regions of varying
size), not on a fixed k x k grid. Rasterizing Delta onto a grid throws away the
one thing pyramid computes -- the nesting. So the headline export here is an
interactive tree (icicle), where hovering a node traces its region back onto the
input image and shows the merge that produced the synergy.

Public API (imported by run_pyramid_interactions.py):
    summarize_synergy(res, k=10) -> str
    check_identity(res, tol=1e-3) -> bool
    plot_interactions(res, img01, leaf_masks=None) -> matplotlib Figure
    delta_localization_map(res, leaf_masks=None) -> (H,W) float array
    export_interactive_html(res, img01, path, leaf_masks=None) -> path   [the new one]

The interactive HTML is self-contained (input image inlined as base64, region
masks inlined as run-length-encoded rows). No server, no external assets.
"""
from __future__ import annotations

import base64
import io
import json
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# tree access helpers (work off the serialized extras['tree'])
# --------------------------------------------------------------------------- #
def _tree(res):
    return res.extras["tree"]


def _by_id(res):
    return {n["id"]: n for n in _tree(res)}


def _root_id(res):
    """Root = the node that is no one's child."""
    tree = _tree(res)
    child_ids = set()
    for n in tree:
        child_ids.update(n["child_ids"])
    roots = [n["id"] for n in tree if n["id"] not in child_ids]
    # there should be exactly one; if several (defensive), take largest area
    if len(roots) == 1:
        return roots[0]
    by_id = _by_id(res)
    return max(roots, key=lambda i: by_id[i]["area"])


def _depths(res):
    """Map node id -> depth (root = 0) via BFS over child_ids."""
    by_id = _by_id(res)
    root = _root_id(res)
    depth = {root: 0}
    stack = [root]
    while stack:
        nid = stack.pop()
        for c in by_id[nid]["child_ids"]:
            depth[c] = depth[nid] + 1
            stack.append(c)
    return depth


# --------------------------------------------------------------------------- #
# 1) summary: where interaction lives + how much of the story it is
# --------------------------------------------------------------------------- #
def summarize_synergy(res, k: int = 10) -> str:
    tree = _tree(res)
    by_id = _by_id(res)
    depth = _depths(res)

    leaves = [n for n in tree if n["is_leaf"]]
    internals = [n for n in tree if not n["is_leaf"]]

    sum_leaf_v = sum(n["v"] for n in leaves)
    sum_delta = sum(n["delta"] for n in internals)
    sum_abs_leaf = sum(abs(n["v"]) for n in leaves)
    sum_abs_delta = sum(abs(n["delta"]) for n in internals)
    denom = sum_abs_leaf + sum_abs_delta
    nai = (sum_abs_delta / denom) if denom > 0 else 0.0

    root = by_id[_root_id(res)]

    lines = []
    lines.append("=" * 60)
    lines.append("PYRAMID SYNERGY SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  nodes: {len(leaves)} leaves + {len(internals)} internal "
                 f"= {len(tree)} total   (tree depth {max(depth.values())})")
    lines.append(f"  v(root) = f(x)-f(b)           : {root['v']:+.5f}")
    lines.append(f"  sum_leaf_v   (additive part)  : {sum_leaf_v:+.5f}")
    lines.append(f"  sum_delta    (synergy part)   : {sum_delta:+.5f}")
    lines.append("")
    lines.append(f"  NAI = sum|D| / (sum|leaf v| + sum|D|) : {nai:.4f}")
    add_share = (abs(sum_leaf_v) / (abs(sum_leaf_v) + abs(sum_delta))
                 if (abs(sum_leaf_v) + abs(sum_delta)) > 0 else 0.0)
    lines.append(f"  --> additive part is {add_share:.1%} of the signed total; "
                 f"the rest is cooperation.")
    lines.append("")
    lines.append(f"  TOP {k} INTERACTION NODES (by |Delta|):")
    lines.append(f"    {'rank':>4} {'node':>6} {'depth':>5} {'area':>7} "
                 f"{'Delta':>10} {'kind':>10}")
    ranked = sorted(internals, key=lambda n: abs(n["delta"]), reverse=True)[:k]
    for r, n in enumerate(ranked, 1):
        kind = "coop" if n["delta"] > 0 else "redundant"
        lines.append(f"    {r:>4} {n['id']:>6} {depth[n['id']]:>5} "
                     f"{n['area']:>7} {n['delta']:>+10.5f} {kind:>10}")
    lines.append("=" * 60)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2) identity / trust check (telescoping completeness)
# --------------------------------------------------------------------------- #
def check_identity(res, tol: float = 1e-3) -> bool:
    ex = res.extras
    # prefer the values the explainer already computed
    if "identity_residual" in ex:
        resid = float(ex["identity_residual"])
        lhs = float(ex.get("identity_lhs", float("nan")))
        rhs = float(ex.get("identity_rhs", float("nan")))
    else:
        by_id = _by_id(res)
        tree = _tree(res)
        leaves = [n for n in tree if n["is_leaf"]]
        internals = [n for n in tree if not n["is_leaf"]]
        lhs = by_id[_root_id(res)]["v"]
        rhs = sum(n["v"] for n in leaves) + sum(n["delta"] for n in internals)
        resid = lhs - rhs
    ok = abs(resid) <= tol
    flag = "OK" if ok else "FAIL"
    print(f"[identity] v(root)={lhs:+.6f}  sum_leaf_v+sum_delta={rhs:+.6f}  "
          f"residual={resid:+.2e}  (tol={tol:.0e})  [{flag}]")
    return ok


# --------------------------------------------------------------------------- #
# spatial Delta map (density): each pixel = max |Delta| over nodes covering it,
# signed by the dominant node's sign. This is the rasterized view (kept for the
# static plot + numeric peek), explicitly NOT the headline -- see module docstr.
# --------------------------------------------------------------------------- #
def _node_mask(res, nid, leaf_masks, cache):
    if nid in cache:
        return cache[nid]
    by_id = _by_id(res)
    n = by_id[nid]
    if n["is_leaf"]:
        m = leaf_masks[nid].astype(bool)
    else:
        m = None
        for c in n["child_ids"]:
            cm = _node_mask(res, c, leaf_masks, cache)
            m = cm.copy() if m is None else (m | cm)
    cache[nid] = m
    return m


def delta_localization_map(res, leaf_masks=None) -> np.ndarray:
    if leaf_masks is None:
        leaf_masks = res.extras.get("leaf_masks")
    if leaf_masks is None:
        raise ValueError("need leaf_masks (apply the serialize patch)")
    leaf_masks = {int(k): np.asarray(v).astype(bool) for k, v in leaf_masks.items()}
    H, W = next(iter(leaf_masks.values())).shape
    tree = _tree(res)
    internals = [n for n in tree if not n["is_leaf"]]

    best_abs = np.zeros((H, W), dtype=np.float64)
    signed = np.zeros((H, W), dtype=np.float64)
    cache: dict = {}
    # process by ascending |delta| so larger magnitudes overwrite
    for n in sorted(internals, key=lambda n: abs(n["delta"])):
        d = n["delta"]
        ad = abs(d)
        if ad == 0:
            continue
        m = _node_mask(res, n["id"], leaf_masks, cache)
        area = max(int(m.sum()), 1)
        dens = ad / area
        upd = m & (dens > best_abs)
        best_abs[upd] = dens
        signed[upd] = np.sign(d) * dens
    return signed


# --------------------------------------------------------------------------- #
# 3) static matplotlib plot (input | additive leaf | synergy density | overlay)
#    kept as the quick-look; the interactive HTML is the real deliverable.
# --------------------------------------------------------------------------- #
def plot_interactions(res, img01, leaf_masks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if leaf_masks is None:
        leaf_masks = res.extras.get("leaf_masks")
    leaf_masks = {int(k): np.asarray(v).astype(bool) for k, v in leaf_masks.items()}

    ex = res.extras
    sum_leaf_v = ex.get("sum_leaf_v", float("nan"))
    sum_delta = ex.get("sum_delta", float("nan"))
    resid = ex.get("identity_residual", float("nan"))

    # additive leaf-v density (first-order map)
    H, W = next(iter(leaf_masks.values())).shape
    by_id = _by_id(res)
    leaf_v = np.zeros((H, W))
    for n in _tree(res):
        if n["is_leaf"]:
            m = leaf_masks[n["id"]]
            leaf_v[m] = n["v"] / max(int(m.sum()), 1)

    dmap = delta_localization_map(res, leaf_masks)
    dlim = np.abs(dmap).max() or 1.0

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    fig.suptitle(f"pyramid interactions | sum_leaf_v={sum_leaf_v:+.3f} "
                 f"sum_delta={sum_delta:+.3f} residual={resid:+.1e}")

    ax[0].imshow(img01); ax[0].set_title("input"); ax[0].axis("off")

    im1 = ax[1].imshow(leaf_v, cmap="viridis")
    ax[1].set_title("additive (leaf v)"); ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(dmap, cmap="bwr", vmin=-dlim, vmax=dlim)
    ax[2].set_title("synergy Delta density\n(red=coop, blue=redundant)")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046)

    ax[3].imshow(img01)
    ax[3].imshow(dmap, cmap="bwr", vmin=-dlim, vmax=dlim, alpha=0.5)
    ax[3].set_title("Delta overlay"); ax[3].axis("off")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# RLE + image encoding helpers for the interactive HTML
# --------------------------------------------------------------------------- #
def _rle_rows(mask: np.ndarray):
    """Row-wise run-length encode a boolean mask.

    Returns a list (one entry per row that has any True pixel):
        [row_index, [start0, len0, start1, len1, ...], ...]
    Compact and trivial to decode in JS by drawing horizontal spans.
    """
    H, W = mask.shape
    out = []
    for r in range(H):
        row = mask[r]
        if not row.any():
            continue
        runs = []
        c = 0
        while c < W:
            if row[c]:
                start = c
                while c < W and row[c]:
                    c += 1
                runs.append(start)
                runs.append(c - start)
            else:
                c += 1
        out.append([r, runs])
    return out


def _img_to_base64_png(img01: np.ndarray) -> str:
    """(H,W,3) float [0,1] -> base64 PNG string (no data: prefix)."""
    from PIL import Image
    arr = np.clip(img01, 0, 1)
    arr = (arr * 255).round().astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# the headline: interactive self-contained HTML (icicle + hover-to-trace)
# --------------------------------------------------------------------------- #
def export_interactive_html(res, img01, path: str, leaf_masks=None,
                            max_nodes_with_masks: Optional[int] = 400) -> str:
    """Write a self-contained interactive HTML visualization.

    Layout:
      LEFT  : the input image; hovering a tree node paints that node's region.
      RIGHT : an icicle of the region tree. Rows = depth levels (= merge scales).
              Each block's width is proportional to region area; color encodes
              Delta (red=cooperation, blue=redundancy, near-white=additive).
              Hovering a block (a) highlights it, (b) traces its region onto the
              image, (c) shows v / Delta / area / depth, and (d) marks its two
              children + parent so you SEE the merge that produced the synergy.

    The image is inlined as base64 PNG; region masks are inlined as row RLE.
    To keep file size sane on big trees, masks are attached to at most
    `max_nodes_with_masks` nodes (always: root, all leaves, and the top-|Delta|
    internal nodes); other internal nodes still appear in the icicle and report
    their numbers, and their region is reconstructed in-browser as the union of
    descendant leaf masks (so hover-trace still works for every node).
    """
    if leaf_masks is None:
        leaf_masks = res.extras.get("leaf_masks")
    if leaf_masks is None:
        raise ValueError("need leaf_masks (apply the serialize patch to "
                         "PyramidExplainer.explain)")
    leaf_masks = {int(k): np.asarray(v).astype(bool) for k, v in leaf_masks.items()}

    tree = _tree(res)
    by_id = _by_id(res)
    depth = _depths(res)
    root_id = _root_id(res)
    H, W = next(iter(leaf_masks.values())).shape

    # choose which nodes carry an explicit RLE mask (cost control); every LEAF
    # always carries one, so any node's region = union of its descendant leaves.
    internals = [n for n in tree if not n["is_leaf"]]
    top_internal = set(
        n["id"] for n in sorted(internals, key=lambda n: abs(n["delta"]),
                                reverse=True)[:max(0, (max_nodes_with_masks or 0))]
    )

    # build the JS node table
    js_nodes = {}
    for n in tree:
        nid = n["id"]
        entry = {
            "id": nid,
            "depth": depth[nid],
            "area": int(n["area"]),
            "v": float(n["v"]),
            "delta": float(n["delta"]),
            "is_leaf": bool(n["is_leaf"]),
            "children": [int(c) for c in n["child_ids"]],
        }
        # attach RLE for leaves, root, and top-delta internals
        if n["is_leaf"] or nid == root_id or nid in top_internal:
            entry["rle"] = _rle_rows(leaf_masks[nid]) if n["is_leaf"] \
                else _rle_rows(_node_mask(res, nid, leaf_masks, {}))
        js_nodes[nid] = entry

    # parent pointers (handy for hover "show parent")
    parent = {}
    for n in tree:
        for c in n["child_ids"]:
            parent[int(c)] = int(n["id"])

    img_b64 = _img_to_base64_png(img01)

    meta = {
        "H": H, "W": W,
        "root": int(root_id),
        "max_depth": int(max(depth.values())),
        "n_leaves": int(sum(1 for n in tree if n["is_leaf"])),
        "n_internal": int(len(internals)),
        "sum_leaf_v": float(res.extras.get("sum_leaf_v", float("nan"))),
        "sum_delta": float(res.extras.get("sum_delta", float("nan"))),
        "root_v": float(by_id[root_id]["v"]),
        "identity_residual": float(res.extras.get("identity_residual", float("nan"))),
        "sigma": float(res.extras.get("sigma", float("nan"))),
        "target": int(getattr(res, "target_class", -1) or -1),
        "target_name": str(getattr(res, "target_class_name", "")),
        "f_x": float(getattr(res, "f_x", float("nan"))),
        "f_b": float(getattr(res, "f_b", float("nan"))),
    }

    payload = {
        "meta": meta,
        "nodes": js_nodes,
        "parent": parent,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    html = _HTML_TEMPLATE.replace("__IMG_B64__", img_b64) \
                         .replace("__PAYLOAD__", payload_json)
    with open(path, "w") as fh:
        fh.write(html)
    return path


# --------------------------------------------------------------------------- #
# the HTML/JS/CSS template (single file, no external deps)
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyramidExplainer &mdash; interactive synergy tree</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --ink: #e8eaf0; --muted: #9aa3b2;
    --line: #2a2f3a; --coop: #e8463a; --redund: #3a6ee8; --accent: #f5c451;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
  }
  header h1 { font-size: 16px; margin: 0; font-weight: 650; letter-spacing: .2px; }
  header .meta { color: var(--muted); font-size: 12.5px; }
  header .meta b { color: var(--ink); font-weight: 600; }
  .wrap { display: grid; grid-template-columns: 420px 1fr; gap: 0; height: calc(100vh - 56px); }
  .left { padding: 18px; border-right: 1px solid var(--line); overflow: auto; }
  .right { padding: 14px 18px 24px; overflow: auto; }
  .imgbox { position: relative; width: 100%; max-width: 380px; }
  .imgbox img { width: 100%; display: block; border-radius: 8px; }
  .imgbox canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%;
    border-radius: 8px; pointer-events: none; }
  .readout { margin-top: 14px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px 14px; min-height: 110px; }
  .readout .big { font-size: 15px; font-weight: 650; margin-bottom: 6px; }
  .kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; font-size: 13px; }
  .kv .k { color: var(--muted); }
  .kv .v { font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 99px; font-size: 12px;
    font-weight: 600; }
  .pill.coop { background: rgba(232,70,58,.18); color: #ff9085; }
  .pill.redund { background: rgba(58,110,232,.18); color: #8fb0ff; }
  .pill.add { background: rgba(154,163,178,.18); color: var(--muted); }
  .legend { margin-top: 12px; font-size: 12px; color: var(--muted); }
  .legend .bar { height: 10px; border-radius: 4px; margin: 5px 0 3px;
    background: linear-gradient(90deg, var(--redund), #aab2c2 50%, var(--coop)); }
  .icicle-head { display:flex; align-items:baseline; gap:14px; margin-bottom: 8px; }
  .icicle-head h2 { font-size: 14px; margin: 0; font-weight: 600; }
  .icicle-head .hint { color: var(--muted); font-size: 12px; }
  .level { display: flex; align-items: stretch; height: 30px; margin-bottom: 3px;
    width: 100%; }
  .level-label { width: 56px; flex: 0 0 56px; color: var(--muted); font-size: 11px;
    display: flex; align-items: center; gap: 4px; }
  .level-row { position: relative; flex: 1 1 auto; display: flex; gap: 1px; }
  .blk { height: 100%; border-radius: 2px; cursor: pointer; position: relative;
    transition: outline-color .08s, filter .08s; outline: 2px solid transparent;
    outline-offset: -2px; min-width: 1px; }
  .blk:hover { filter: brightness(1.25); }
  .blk.sel { outline-color: var(--accent); z-index: 3; }
  .blk.kin { outline-color: #6de08a; z-index: 2; }
  .blk.par { outline-color: #c98bff; z-index: 2; }
  .controls { margin: 4px 0 14px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }
  .controls label { font-size:12.5px; color: var(--muted); display:flex; gap:6px; align-items:center; }
  .controls input[type=range]{ width: 160px; }
  .foot { color: var(--muted); font-size: 11.5px; margin-top: 18px; line-height: 1.5; }
  code { background:#0b0d11; padding:1px 5px; border-radius:4px; color:#cdd3df; }
</style>
</head>
<body>
<header>
  <h1>PyramidExplainer &middot; interactive synergy tree</h1>
  <div class="meta" id="meta"></div>
</header>

<div class="wrap">
  <div class="left">
    <div class="imgbox">
      <img id="img" alt="input" src="data:image/png;base64,__IMG_B64__">
      <canvas id="overlay"></canvas>
    </div>
    <div class="readout" id="readout">
      <div class="big">Hover a block &rarr;</div>
      <div class="kv"><span class="k">Each row of the icicle is a merge scale.</span><span class="v"></span></div>
      <div class="kv"><span class="k">Hovering traces that region onto the image.</span><span class="v"></span></div>
    </div>
    <div class="legend">
      synergy &Delta; &nbsp; (red = cooperation, blue = redundancy)
      <div class="bar"></div>
      <span style="float:left">more redundant</span><span style="float:right">more cooperative</span>
      <div style="clear:both"></div>
    </div>
  </div>

  <div class="right">
    <div class="icicle-head">
      <h2>Region tree (icicle)</h2>
      <span class="hint">row = depth / scale &nbsp;&bull;&nbsp; width &prop; area &nbsp;&bull;&nbsp;
        color = &Delta; &nbsp;&bull;&nbsp; hover: <span style="color:#f5c451">self</span>
        / <span style="color:#6de08a">children</span> / <span style="color:#c98bff">parent</span></span>
    </div>
    <div class="controls">
      <label>color gain
        <input type="range" id="gain" min="0.2" max="3" step="0.1" value="1">
        <span id="gainval">1.0&times;</span>
      </label>
      <label><input type="checkbox" id="sortdelta"> sort each row by |&Delta;|</label>
      <label><input type="checkbox" id="hidetiny" checked> hide area &lt; 0.2%</label>
    </div>
    <div id="icicle"></div>
    <div class="foot" id="foot"></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const IMG_B64 = "__IMG_B64__";
const M = DATA.meta, NODES = DATA.nodes, PARENT = DATA.parent;

// ---- decode RLE region -> draw on overlay canvas --------------------------- //
const img = document.getElementById('img');
const cv = document.getElementById('overlay');
const ctx = cv.getContext('2d');

function ensureCanvas() {
  // size the canvas to the natural image resolution for crisp masks
  if (cv.width !== M.W || cv.height !== M.H) { cv.width = M.W; cv.height = M.H; }
}
function clearOverlay() { ensureCanvas(); ctx.clearRect(0,0,cv.width,cv.height); }

// reconstruct a node's RLE: explicit if present, else union of descendant leaves
function nodeRLE(id) {
  const n = NODES[id];
  if (n.rle) return n.rle;
  // union of descendant leaf masks -> build a row map
  const rowmap = new Map(); // r -> array of [start,len]
  (function collect(k){
    const nn = NODES[k];
    if (nn.is_leaf) {
      for (const [r, runs] of (nn.rle||[])) {
        if (!rowmap.has(r)) rowmap.set(r, []);
        const arr = rowmap.get(r);
        for (let i=0;i<runs.length;i+=2) arr.push([runs[i], runs[i+1]]);
      }
    } else { for (const c of nn.children) collect(c); }
  })(id);
  // flatten + merge overlapping/adjacent runs per row
  const out = [];
  for (const [r, spans] of [...rowmap.entries()].sort((a,b)=>a[0]-b[0])) {
    spans.sort((a,b)=>a[0]-b[0]);
    const merged = [];
    let [cs, cl] = spans[0];
    for (let i=1;i<spans.length;i++){
      const [s,l]=spans[i];
      if (s <= cs+cl) { cl = Math.max(cl, s+l-cs); }
      else { merged.push(cs,cl); cs=s; cl=l; }
    }
    merged.push(cs,cl);
    out.push([r, merged]);
  }
  return out;
}

function paintRegion(id, rgba) {
  clearOverlay();
  const rle = nodeRLE(id);
  ctx.fillStyle = rgba;
  for (const [r, runs] of rle) {
    for (let i=0;i<runs.length;i+=2) ctx.fillRect(runs[i], r, runs[i+1], 1);
  }
}

// ---- color map for Delta --------------------------------------------------- //
let GAIN = 1.0;
let DMAX = 1e-9;
for (const id in NODES) { const d=Math.abs(NODES[id].delta); if (d>DMAX) DMAX=d; }

function deltaColor(d) {
  // diverging: blue (neg) - white (0) - red (pos), scaled by DMAX and GAIN
  const t = Math.max(-1, Math.min(1, (d / DMAX) * GAIN));
  const coop = [232,70,58], redund=[58,110,232], mid=[176,182,196];
  let c;
  if (t >= 0) c = mid.map((m,i)=> Math.round(m + (coop[i]-m)*t));
  else        c = mid.map((m,i)=> Math.round(m + (redund[i]-m)*(-t)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---- build the icicle ------------------------------------------------------ //
const icicle = document.getElementById('icicle');
const totalArea = M.W * M.H;
const hideTiny = document.getElementById('hidetiny');
const sortDelta = document.getElementById('sortdelta');

function buildIcicle() {
  icicle.innerHTML = '';
  // group node ids by depth
  const byDepth = [];
  for (const id in NODES) {
    const d = NODES[id].depth;
    (byDepth[d] = byDepth[d] || []).push(id);
  }
  const minFrac = hideTiny.checked ? 0.002 : 0;
  for (let d=0; d<byDepth.length; d++) {
    let ids = (byDepth[d]||[]).filter(id => NODES[id].area/totalArea >= minFrac);
    // order: by spatial position (so siblings stay near each other) unless sorting by |delta|
    if (sortDelta.checked) ids.sort((a,b)=> Math.abs(NODES[b].delta)-Math.abs(NODES[a].delta));
    else ids.sort((a,b)=> a-b); // id order ~ merge order, keeps it stable
    const areaSum = ids.reduce((s,id)=> s + NODES[id].area, 0) || 1;

    const level = document.createElement('div'); level.className='level';
    const lab = document.createElement('div'); lab.className='level-label';
    lab.textContent = (d===0?'root':'d'+d);
    const row = document.createElement('div'); row.className='level-row';
    level.appendChild(lab); level.appendChild(row);

    for (const id of ids) {
      const n = NODES[id];
      const b = document.createElement('div');
      b.className = 'blk'; b.dataset.id = id;
      b.style.flexGrow = String(n.area);
      b.style.background = n.is_leaf ? '#222732' : deltaColor(n.delta);
      b.title = ''; // we use custom readout
      b.addEventListener('mouseenter', ()=> onHover(id));
      b.addEventListener('mouseleave', ()=> onLeave());
      b.addEventListener('click', ()=> pin(id));
      row.appendChild(b);
    }
    icicle.appendChild(level);
  }
}

// ---- hover behavior -------------------------------------------------------- //
const readout = document.getElementById('readout');
let PINNED = null;

function classifyDelta(d) {
  if (Math.abs(d) < DMAX*1e-3) return ['add','additive (&Delta;&approx;0)'];
  return d>0 ? ['coop','cooperation (&Delta;&gt;0)'] : ['redund','redundancy (&Delta;&lt;0)'];
}

function markKin(id) {
  document.querySelectorAll('.blk').forEach(b=>b.classList.remove('sel','kin','par'));
  const self = document.querySelector(`.blk[data-id="${id}"]`);
  if (self) self.classList.add('sel');
  const n = NODES[id];
  for (const c of n.children) {
    const e = document.querySelector(`.blk[data-id="${c}"]`); if (e) e.classList.add('kin');
  }
  const p = PARENT[id];
  if (p!==undefined) { const e=document.querySelector(`.blk[data-id="${p}"]`); if(e) e.classList.add('par'); }
}

function showReadout(id) {
  const n = NODES[id];
  const [cls, lbl] = classifyDelta(n.delta);
  const kids = n.children.length;
  const childStr = kids ? n.children.map(c=>{
      const cn=NODES[c];
      const cd = cn.is_leaf ? 'leaf' : ('&Delta;='+cn.delta.toFixed(4));
      return `#${c} (v=${cn.v.toFixed(4)}, ${cd})`;
    }).join('<br>') : '&mdash; leaf &mdash;';
  readout.innerHTML = `
    <div class="big">node #${id} &nbsp; <span class="pill ${cls}">${lbl}</span></div>
    <div class="kv">
      <span class="k">depth</span><span class="v">${n.depth} ${n.depth===0?'(root)':''}</span>
      <span class="k">area</span><span class="v">${n.area} px (${(100*n.area/totalArea).toFixed(2)}%)</span>
      <span class="k">v(R) = f(&Phi;<sub>R</sub>)-f(b)</span><span class="v">${n.v.toFixed(5)}</span>
      <span class="k">&Delta;(R) synergy</span><span class="v">${n.is_leaf?'&mdash; (leaf)':n.delta.toFixed(5)}</span>
      <span class="k">children</span><span class="v">${childStr}</span>
    </div>`;
}

function onHover(id) {
  if (PINNED!==null) return;
  paintRegion(id, 'rgba(245,196,81,0.42)');
  // tint children differently so you SEE the merge
  const n = NODES[id];
  if (!n.is_leaf && n.children.length) {
    ctx.fillStyle='rgba(109,224,138,0.30)';
    for (const c of n.children) {
      const rle = nodeRLE(c);
      for (const [r,runs] of rle) for (let i=0;i<runs.length;i+=2) ctx.fillRect(runs[i],r,runs[i+1],1);
    }
    // re-stroke the parent outline lightly by repainting self at low alpha edges
  }
  markKin(id);
  showReadout(id);
}
function onLeave() {
  if (PINNED!==null) { onHover_pinned(); return; }
  clearOverlay();
  document.querySelectorAll('.blk').forEach(b=>b.classList.remove('sel','kin','par'));
}
function pin(id) {
  if (PINNED===id) { PINNED=null; onLeave(); return; }
  PINNED=id; onHover_pinned();
}
function onHover_pinned() {
  const id=PINNED;
  paintRegion(id,'rgba(245,196,81,0.42)');
  const n=NODES[id];
  if (!n.is_leaf && n.children.length) {
    ctx.fillStyle='rgba(109,224,138,0.30)';
    for (const c of n.children){ const rle=nodeRLE(c);
      for (const [r,runs] of rle) for(let i=0;i<runs.length;i+=2) ctx.fillRect(runs[i],r,runs[i+1],1);}
  }
  markKin(id); showReadout(id);
}

// ---- meta header + footer + controls --------------------------------------- //
document.getElementById('meta').innerHTML =
  `class <b>${M.target} (${M.target_name})</b> &nbsp;&bull;&nbsp; ` +
  `f(x)=<b>${M.f_x.toFixed(3)}</b> f(b)=<b>${M.f_b.toFixed(3)}</b> &nbsp;&bull;&nbsp; ` +
  `&sigma;=${M.sigma} &nbsp;&bull;&nbsp; ${M.n_leaves} leaves / ${M.n_internal} internal &nbsp;&bull;&nbsp; ` +
  `v(root)=<b>${M.root_v.toFixed(3)}</b> = sum_leaf_v <b>${M.sum_leaf_v.toFixed(3)}</b> + sum_&Delta; <b>${M.sum_delta.toFixed(3)}</b>`;

const addShare = Math.abs(M.sum_leaf_v)/(Math.abs(M.sum_leaf_v)+Math.abs(M.sum_delta)||1);
document.getElementById('foot').innerHTML =
  `Telescoping identity residual = <code>${M.identity_residual.toExponential(2)}</code> ` +
  `(should be &approx;0; if not, &Delta; values are untrustworthy). &nbsp; ` +
  `Additive (leaf) part is <b>${(100*addShare).toFixed(1)}%</b> of the signed total &mdash; ` +
  `the remaining <b>${(100*(1-addShare)).toFixed(1)}%</b> is cooperation that no additive ` +
  `explanation (LIME, IG) can represent (Prop. 1). &nbsp; Click a block to pin it.`;

document.getElementById('gain').addEventListener('input', e=>{
  GAIN=parseFloat(e.target.value);
  document.getElementById('gainval').textContent=GAIN.toFixed(1)+'\u00d7';
  buildIcicle(); if(PINNED!==null) onHover_pinned();
});
hideTiny.addEventListener('change', ()=>{ buildIcicle(); });
sortDelta.addEventListener('change', ()=>{ buildIcicle(); });

img.addEventListener('load', ()=>{ ensureCanvas(); });
ensureCanvas();
buildIcicle();
</script>
</body>
</html>
"""