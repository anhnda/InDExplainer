"""Interactive HTML export for HIMEExplainer results.

HIME produces a *graph on the grid*, not a containment tree, so this is built
around a spatial node-link overlay rather than the Pyramid icicle/dendrogram.

Four tabs share one inlined image + one overlay canvas (the Pyramid pattern):
  TAB 1  FIRST ORDER     LIME main-effect heatmap painted blocky on the grid.
  TAB 2  INTERACTION     per-cell aggregate of |Delta_ij| (sum/max toggle):
         DENSITY         "where does cooperation concentrate" at a glance.
  TAB 3  INTERACTIVE     hover/pin a cell -> fan out edges to every partner,
         GRAPH           color = sign (coop red / redundant blue), width = |Delta|,
                         opacity faded by stability. The core view.
  TAB 4  RANKED PAIRS    top-k pairs by |Delta|: two region thumbnails + signed
                         bar + stability score. Depth-independent direct answer.

Public entry point:
    export_hime_html(res, img01, out_path)         # from a HIME AttributionResult
    build_html(payload, img_b64)                   # from a raw payload dict

`res` must expose: .extras with keys
    grid              (gh, gw)
    interactions      list of (cell_i, cell_j, strength)   # i<j, strength signed
    interaction_stability  {"i-j": stability_float}        # optional
  and (optional, for tab 1) a per-cell main-effect vector. We recover it from
  res.attribution by reading one representative pixel per cell, so no extra field
  is required.
"""
from __future__ import annotations

import base64
import io
import json
import numpy as np


# --------------------------------------------------------------------------- #
# payload construction
# --------------------------------------------------------------------------- #
def _img_to_b64(img01: np.ndarray) -> str:
    """img01: (H,W,3) float in [0,1] -> base64 PNG."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("export_hime_html needs Pillow for PNG encoding.")
    arr = np.clip(img01 * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cell_to_main_effect(res, gh, gw):
    """Recover per-cell main effect from the blocky attribution map.

    The attribution is constant within each grid cell, so sampling the center
    pixel of each cell block recovers the coefficient. Falls back to zeros.
    """
    attr = getattr(res, "attribution", None)
    if attr is None:
        return [0.0] * (gh * gw)
    H, W = attr.shape
    main = [0.0] * (gh * gw)
    for cy in range(gh):
        for cx in range(gw):
            # center pixel of this cell block
            py = int((cy + 0.5) * H / gh)
            px = int((cx + 0.5) * W / gw)
            py = min(py, H - 1); px = min(px, W - 1)
            main[cy * gw + cx] = float(attr[py, px])
    return main


def build_payload(res):
    gh, gw = res.extras["grid"]
    n_cells = gh * gw
    interactions = res.extras.get("interactions", [])
    stab = res.extras.get("interaction_stability", {})

    main = _cell_to_main_effect(res, gh, gw)

    edges = []
    for (i, j, s) in interactions:
        i, j = int(i), int(j)
        if i > j:
            i, j = j, i
        key = f"{i}-{j}"
        edges.append({
            "i": i, "j": j, "s": float(s),
            "stab": float(stab.get(key, 1.0)),
        })

    # per-cell aggregates of |Delta| (sum and max), plus signed sum for tint
    agg_sum = [0.0] * n_cells
    agg_max = [0.0] * n_cells
    agg_signed = [0.0] * n_cells
    deg = [0] * n_cells
    for e in edges:
        a = abs(e["s"])
        for c in (e["i"], e["j"]):
            agg_sum[c] += a
            agg_max[c] = max(agg_max[c], a)
            agg_signed[c] += e["s"]
            deg[c] += 1

    meta = {
        "gh": gh, "gw": gw, "n_cells": n_cells,
        "n_edges": len(edges),
        "target": int(getattr(res, "target_class", -1) or -1),
        "target_name": str(getattr(res, "target_class_name", "")),
        "f_x": float(getattr(res, "f_x", float("nan"))),
        "n_samples": int(res.extras.get("n_samples", 0)),
        "n_active_cells": int(res.extras.get("n_active_cells", 0)),
        "candidate_pairs": int(res.extras.get("candidate_pairs", 0)),
        "main_min": float(min(main)) if main else 0.0,
        "main_max": float(max(main)) if main else 0.0,
    }
    return {
        "meta": meta,
        "main": main,
        "edges": edges,
        "agg_sum": agg_sum,
        "agg_max": agg_max,
        "agg_signed": agg_signed,
        "deg": deg,
    }


def build_html(payload, img_b64):
    payload_json = json.dumps(payload, separators=(",", ":"))
    return _TEMPLATE.replace("__IMG_B64__", img_b64).replace("__PAYLOAD__", payload_json)


def export_hime_html(res, img01, out_path):
    payload = build_payload(res)
    img_b64 = _img_to_b64(img01)
    html = build_html(payload, img_b64)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


# --------------------------------------------------------------------------- #
# template
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIME &mdash; interaction views</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaf0;--muted:#9aa3b2;--line:#2a2f3a;
    --coop:#e8463a;--redund:#3a6ee8;--accent:#f5c451;--hub:#6de08a;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  header{padding:13px 20px;border-bottom:1px solid var(--line);display:flex;
    align-items:baseline;gap:18px;flex-wrap:wrap;}
  header h1{font-size:16px;margin:0;font-weight:650;}
  header .meta{color:var(--muted);font-size:12.5px;}
  header .meta b{color:var(--ink);font-weight:600;}
  .wrap{display:grid;grid-template-columns:440px 1fr;height:calc(100vh - 53px);}
  .left{padding:18px;border-right:1px solid var(--line);overflow:auto;}
  .right{padding:12px 18px 24px;overflow:auto;display:flex;flex-direction:column;}
  .imgbox{position:relative;width:100%;max-width:400px;}
  .imgbox img{width:100%;display:block;border-radius:8px;}
  .imgbox canvas,.imgbox svg{position:absolute;left:0;top:0;width:100%;height:100%;border-radius:8px;}
  .imgbox canvas{pointer-events:none;}
  .imgbox svg{pointer-events:none;}
  .readout{margin-top:14px;background:var(--panel);border:1px solid var(--line);
    border-radius:8px;padding:12px 14px;min-height:110px;}
  .readout .big{font-size:15px;font-weight:650;margin-bottom:6px;}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-size:13px;}
  .kv .k{color:var(--muted);} .kv .v{font-variant-numeric:tabular-nums;}
  .pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;}
  .pill.coop{background:rgba(232,70,58,.18);color:#ff9085;}
  .pill.redund{background:rgba(58,110,232,.18);color:#8fb0ff;}
  .pill.add{background:rgba(154,163,178,.18);color:var(--muted);}
  .legend{margin-top:12px;font-size:12px;color:var(--muted);}
  .legend .bar{height:10px;border-radius:4px;margin:5px 0 3px;
    background:linear-gradient(90deg,var(--redund),#aab2c2 50%,var(--coop));}
  .tabs{display:flex;gap:4px;margin-bottom:10px;border-bottom:1px solid var(--line);flex-wrap:wrap;}
  .tab{padding:8px 14px;cursor:pointer;color:var(--muted);font-weight:600;font-size:13.5px;
    border-bottom:2px solid transparent;margin-bottom:-1px;}
  .tab:hover{color:var(--ink);}
  .tab.active{color:var(--ink);border-bottom-color:var(--accent);}
  .panel{display:none;flex:1 1 auto;min-height:0;}
  .panel.active{display:block;}
  .controls{margin:2px 0 12px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;}
  .controls label{font-size:12.5px;color:var(--muted);display:flex;gap:6px;align-items:center;}
  .controls input[type=range]{width:140px;}
  .hint{color:var(--muted);font-size:12px;margin:2px 0 12px;}
  .foot{color:var(--muted);font-size:11.5px;margin-top:16px;line-height:1.5;}
  code{background:#0b0d11;padding:1px 5px;border-radius:4px;color:#cdd3df;}
  /* grid views render into an aspect-ratio square */
  .gridwrap{position:relative;width:100%;max-width:640px;aspect-ratio:1/1;
    background:#0b0d11;border-radius:8px;overflow:hidden;}
  .gridwrap canvas{position:absolute;left:0;top:0;width:100%;height:100%;}
  .gcell{position:absolute;outline:1px solid rgba(0,0,0,0.15);cursor:pointer;
    transition:filter .08s;}
  .gcell:hover{filter:brightness(1.4);}
  .gcell.hub{outline:2px solid var(--hub);z-index:3;}
  /* ranked */
  #ranked .rk{display:flex;align-items:center;gap:10px;padding:5px 6px;border-radius:6px;cursor:pointer;}
  #ranked .rk:hover{background:var(--panel);}
  #ranked .rk.sel{background:rgba(245,196,81,.10);outline:1px solid rgba(245,196,81,.4);}
  #ranked .rnum{width:24px;color:var(--muted);font-size:12px;text-align:right;}
  #ranked canvas.thumb{width:46px;height:46px;border-radius:4px;background:#0b0d11;flex:0 0 46px;}
  #ranked .rbar-wrap{flex:1 1 auto;}
  #ranked .rbar-track{height:14px;background:#0b0d11;border-radius:7px;position:relative;overflow:hidden;}
  #ranked .rbar{position:absolute;top:0;height:100%;}
  #ranked .rmeta{font-size:12px;color:var(--muted);margin-top:3px;}
  #ranked .rmeta b{color:var(--ink);}
  .stabchip{font-size:11px;padding:0 6px;border-radius:99px;background:#0b0d11;color:var(--muted);}
</style></head>
<body>
<header>
  <h1>HIME &middot; high-order interaction views</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="wrap">
  <div class="left">
    <div class="imgbox">
      <img id="img" alt="input" src="data:image/png;base64,__IMG_B64__">
      <canvas id="overlay"></canvas>
      <svg id="edgesvg" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
    </div>
    <div class="readout" id="readout">
      <div class="big">Hover a cell &rarr;</div>
      <div class="kv"><span class="k">Tab 3: hovering a cell fans out its interaction edges.</span><span class="v"></span></div>
    </div>
    <div class="legend">interaction &Delta; (red = cooperation, blue = redundancy)
      <div class="bar"></div>
      <span style="float:left">more redundant</span><span style="float:right">more cooperative</span>
      <div style="clear:both"></div>
    </div>
    <div class="foot" id="foot"></div>
  </div>
  <div class="right">
    <div class="tabs">
      <div class="tab active" data-tab="first">First order</div>
      <div class="tab" data-tab="density">Interaction density</div>
      <div class="tab" data-tab="graph">Interactive graph</div>
      <div class="tab" data-tab="ranked">Ranked pairs</div>
    </div>
    <div class="controls">
      <label>color gain <input type="range" id="gain" min="0.2" max="4" step="0.1" value="1"><span id="gainval">1.0&times;</span></label>
      <label id="aggctl" style="display:none">density:
        <select id="aggmode"><option value="sum">sum |&Delta;|</option><option value="max">max |&Delta;|</option></select></label>
      <label id="stabctl" style="display:none">min stability
        <input type="range" id="stabmin" min="0" max="1" step="0.05" value="0"><span id="stabval">0.00</span></label>
    </div>

    <div class="panel active" id="panel-first">
      <div class="hint">LIME first-order map: each cell painted by its main-effect coefficient. Blocky by design (per-cell constant).</div>
      <div class="gridwrap" id="grid-first"></div>
    </div>
    <div class="panel" id="panel-density">
      <div class="hint">Per-cell aggregate of |&Delta;| over all its pairs &mdash; where interaction concentrates. Toggle sum (hub cells) vs max (sharpest single pair). Tint shows net sign.</div>
      <div class="gridwrap" id="grid-density"></div>
    </div>
    <div class="panel" id="panel-graph">
      <div class="hint">Hover any cell to fan out its interaction edges (also drawn on the left image). Edge color = sign, width = |&Delta;|, opacity faded by stability. Click a cell to pin.</div>
      <div class="gridwrap" id="grid-graph"><svg id="graphsvg" viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none;"></svg></div>
    </div>
    <div class="panel" id="panel-ranked">
      <div class="hint">Top pairs by |&Delta;|, depth-independent. Thumbnails = the two cells; bar = signed &Delta;; chip = stability.</div>
      <div id="ranked"></div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const M = DATA.meta, MAIN = DATA.main, EDGES = DATA.edges;
const AGG_SUM = DATA.agg_sum, AGG_MAX = DATA.agg_max, AGG_SIGNED = DATA.agg_signed, DEG = DATA.deg;
const GH = M.gh, GW = M.gw, NCELL = M.n_cells;
let GAIN = 1.0, PINNED = null, AGGMODE = "sum", STABMIN = 0.0;

// adjacency: cell -> list of edges touching it
const ADJ = Array.from({length:NCELL}, ()=>[]);
EDGES.forEach((e,idx)=>{ e.idx=idx; ADJ[e.i].push(e); ADJ[e.j].push(e); });

let DMAX = 1e-9;
for(const e of EDGES){ const a=Math.abs(e.s); if(a>DMAX)DMAX=a; }
let MAINMAX = Math.max(Math.abs(M.main_min), Math.abs(M.main_max), 1e-9);
let SUMMAX = Math.max(...AGG_SUM, 1e-9), MAXMAX = Math.max(...AGG_MAX, 1e-9);

// ---- color helpers --------------------------------------------------------- //
function signColor(d, scale){
  const t=Math.max(-1,Math.min(1,(d/scale)*GAIN));
  const coop=[232,70,58], redund=[58,110,232], mid=[176,182,196];
  const c=(t>=0)?mid.map((m,i)=>Math.round(m+(coop[i]-m)*t))
                :mid.map((m,i)=>Math.round(m+(redund[i]-m)*(-t)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function magColor(v, scale){   // 0..1 intensity, gold ramp for density magnitude
  const t=Math.max(0,Math.min(1,(v/scale)*GAIN));
  const lo=[16,18,24], hi=[245,196,81];
  const c=lo.map((l,i)=>Math.round(l+(hi[i]-l)*t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function cellXY(c){ return {cx:c%GW, cy:Math.floor(c/GW)}; }
function cellCenterPct(c){ const {cx,cy}=cellXY(c);
  return {x:(cx+0.5)/GW*100, y:(cy+0.5)/GH*100}; }

// ---- left-image region + edge painting ------------------------------------ //
const img=document.getElementById('img'), cv=document.getElementById('overlay'), ctx=cv.getContext('2d');
const edgesvg=document.getElementById('edgesvg');
function ensureCanvas(){ if(cv.width!==GW*16||cv.height!==GH*16){cv.width=GW*16;cv.height=GH*16;} }
function clearLeft(){ ensureCanvas(); ctx.clearRect(0,0,cv.width,cv.height); edgesvg.innerHTML=''; }
function fillCell(c,style){ const {cx,cy}=cellXY(c);
  const w=cv.width/GW, h=cv.height/GH; ctx.fillStyle=style; ctx.fillRect(cx*w,cy*h,w,h); }
function paintCellEdges(c){
  clearLeft();
  // highlight hub cell
  fillCell(c,'rgba(109,224,138,0.40)');
  // draw edges as SVG lines in 0..100 space
  const hub=cellCenterPct(c);
  let svg='';
  for(const e of ADJ[c]){
    if(e.stab<STABMIN) continue;
    const other=(e.i===c)?e.j:e.i;
    const p=cellCenterPct(other);
    const col=signColor(e.s, DMAX);
    const wpx=0.4+3.2*Math.abs(e.s)/DMAX;
    const op=0.25+0.7*e.stab;
    // tint partner cell faintly
    fillCell(other, col.replace('rgb','rgba').replace(')',`,${0.18+0.3*Math.abs(e.s)/DMAX})`));
    svg+=`<line x1="${hub.x}" y1="${hub.y}" x2="${p.x}" y2="${p.y}" stroke="${col}" stroke-width="${wpx}" stroke-linecap="round" opacity="${op}"/>`;
  }
  svg+=`<circle cx="${hub.x}" cy="${hub.y}" r="1.6" fill="#6de08a"/>`;
  edgesvg.innerHTML=svg;
}

// ---- readout --------------------------------------------------------------- //
const readout=document.getElementById('readout');
function cellName(c){ const {cx,cy}=cellXY(c); return `#${c} (r${cy},c${cx})`; }
function showReadout(c){
  const partners=ADJ[c].filter(e=>e.stab>=STABMIN)
        .slice().sort((a,b)=>Math.abs(b.s)-Math.abs(a.s));
  const rows=partners.slice(0,8).map(e=>{
    const o=(e.i===c)?e.j:e.i;
    const cls=e.s>0?'coop':'redund';
    return `<span class="k">&harr; ${cellName(o)}</span><span class="v"><span class="pill ${cls}">${e.s>=0?'+':''}${e.s.toFixed(4)}</span> stab ${e.stab.toFixed(2)}</span>`;
  }).join('');
  readout.innerHTML=`<div class="big">cell ${cellName(c)}</div>
    <div class="kv">
    <span class="k">main effect</span><span class="v">${MAIN[c].toFixed(5)}</span>
    <span class="k">partners (|&Delta;|&ge;stab)</span><span class="v">${partners.length} / ${ADJ[c].length}</span>
    <span class="k">&Sigma;|&Delta;|</span><span class="v">${AGG_SUM[c].toFixed(4)}</span>
    <span class="k">max|&Delta;|</span><span class="v">${AGG_MAX[c].toFixed(4)}</span>
    ${rows}</div>`;
}

// ---- generic grid builder (cells as positioned divs) ----------------------- //
function buildGrid(containerId, colorFn, withHover){
  const box=document.getElementById(containerId);
  // keep any existing svg child (graph tab), clear cell divs only
  box.querySelectorAll('.gcell').forEach(e=>e.remove());
  for(let c=0;c<NCELL;c++){
    const {cx,cy}=cellXY(c);
    const d=document.createElement('div'); d.className='gcell'; d.dataset.cell=c;
    d.style.left=(100*cx/GW)+'%'; d.style.top=(100*cy/GH)+'%';
    d.style.width=(100/GW)+'%'; d.style.height=(100/GH)+'%';
    d.style.background=colorFn(c);
    if(withHover){
      d.addEventListener('mouseenter',()=>hoverCell(c));
      d.addEventListener('mouseleave',leaveCell);
      d.addEventListener('click',()=>pinCell(c));
    }
    box.appendChild(d);
  }
}

// ---- tab 1: first order ---------------------------------------------------- //
function colorFirst(c){ return signColor(MAIN[c], MAINMAX); }

// ---- tab 2: density -------------------------------------------------------- //
function colorDensity(c){
  const v=(AGGMODE==='sum')?AGG_SUM[c]:AGG_MAX[c];
  const scale=(AGGMODE==='sum')?SUMMAX:MAXMAX;
  return magColor(v, scale);
}

// ---- tab 3: interactive graph --------------------------------------------- //
const graphsvg=document.getElementById('graphsvg');
function colorGraphBg(c){
  // faint density background so the user knows where to hover
  const v=AGG_SUM[c]; return magColor(v, SUMMAX).replace('rgb','rgba').replace(')',',0.55)');
}
function hoverCell(c){
  if(PINNED!==null) return;
  drawGraphEdges(c); paintCellEdges(c); showReadout(c); markHub(c);
}
function leaveCell(){
  if(PINNED!==null){ pinnedView(); return; }
  graphsvg.innerHTML=''; clearLeft();
  document.querySelectorAll('.gcell').forEach(e=>e.classList.remove('hub'));
}
function pinCell(c){ if(PINNED===c){PINNED=null;leaveCell();return;} PINNED=c; pinnedView(); }
function pinnedView(){ drawGraphEdges(PINNED); paintCellEdges(PINNED); showReadout(PINNED); markHub(PINNED); }
function markHub(c){
  document.querySelectorAll('.gcell').forEach(e=>e.classList.remove('hub'));
  document.querySelectorAll(`#grid-graph .gcell[data-cell="${c}"]`).forEach(e=>e.classList.add('hub'));
}
function drawGraphEdges(c){
  const hub=cellCenterPct(c); let svg='';
  for(const e of ADJ[c]){
    if(e.stab<STABMIN) continue;
    const o=(e.i===c)?e.j:e.i; const p=cellCenterPct(o);
    const col=signColor(e.s, DMAX);
    const w=0.4+3.2*Math.abs(e.s)/DMAX, op=0.25+0.7*e.stab;
    svg+=`<line x1="${hub.x}" y1="${hub.y}" x2="${p.x}" y2="${p.y}" stroke="${col}" stroke-width="${w}" stroke-linecap="round" opacity="${op}"/>`;
  }
  svg+=`<circle cx="${hub.x}" cy="${hub.y}" r="1.4" fill="#6de08a"/>`;
  graphsvg.innerHTML=svg;
}

// ---- tab 4: ranked pairs --------------------------------------------------- //
const ranked=document.getElementById('ranked');
let rankedBuilt=false;
function buildRanked(){
  ranked.innerHTML='';
  const es=EDGES.filter(e=>e.stab>=STABMIN).slice()
        .sort((a,b)=>Math.abs(b.s)-Math.abs(a.s)).slice(0,40);
  const maxAbs=Math.abs(es[0]?.s||1);
  es.forEach((e,i)=>{
    const row=document.createElement('div'); row.className='rk'; row.dataset.pair=`${e.i}-${e.j}`;
    const num=document.createElement('div'); num.className='rnum'; num.textContent=(i+1);
    const th=document.createElement('canvas'); th.className='thumb'; th.width=GW*8; th.height=GH*8;
    drawPairThumb(th,e.i,e.j);
    const wrap=document.createElement('div'); wrap.className='rbar-wrap';
    const track=document.createElement('div'); track.className='rbar-track';
    const bar=document.createElement('div'); bar.className='rbar';
    const frac=Math.abs(e.s)/maxAbs; bar.style.background=signColor(e.s,DMAX);
    if(e.s>=0){bar.style.left='50%';bar.style.width=(50*frac)+'%';}
    else{bar.style.right='50%';bar.style.left='auto';bar.style.width=(50*frac)+'%';}
    track.appendChild(bar);
    const tick=document.createElement('div'); tick.style.cssText='position:absolute;left:50%;top:0;width:1px;height:100%;background:#3a3f4a;';
    track.appendChild(tick);
    const meta=document.createElement('div'); meta.className='rmeta';
    const kind=e.s>0?'coop':'redundant';
    meta.innerHTML=`${cellName(e.i)} &harr; ${cellName(e.j)} &middot; <b>&Delta;=${e.s>=0?'+':''}${e.s.toFixed(4)}</b> (${kind}) <span class="stabchip">stab ${e.stab.toFixed(2)}</span>`;
    wrap.appendChild(track); wrap.appendChild(meta);
    row.appendChild(num); row.appendChild(th); row.appendChild(wrap);
    row.addEventListener('mouseenter',()=>{ paintPairLeft(e); });
    row.addEventListener('mouseleave',()=>{ if(PINNED===null) clearLeft(); });
    ranked.appendChild(row);
  });
  rankedBuilt=true;
}
function drawPairThumb(canvas,i,j){
  const c=canvas.getContext('2d');
  c.globalAlpha=0.55; c.drawImage(img,0,0,canvas.width,canvas.height); c.globalAlpha=1;
  const w=canvas.width/GW, h=canvas.height/GH;
  for(const [cell,style] of [[i,'rgba(245,196,81,0.8)'],[j,'rgba(109,224,138,0.8)']]){
    const {cx,cy}=cellXY(cell); c.fillStyle=style; c.fillRect(cx*w,cy*h,w,h);
  }
}
function paintPairLeft(e){
  clearLeft();
  fillCell(e.i,'rgba(245,196,81,0.5)'); fillCell(e.j,'rgba(109,224,138,0.5)');
  const a=cellCenterPct(e.i), b=cellCenterPct(e.j);
  const col=signColor(e.s,DMAX), w=0.6+3.2*Math.abs(e.s)/DMAX;
  edgesvg.innerHTML=`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${col}" stroke-width="${w}" stroke-linecap="round" opacity="0.9"/>`;
}

// ---- tabs ------------------------------------------------------------------ //
const aggctl=document.getElementById('aggctl'), stabctl=document.getElementById('stabctl');
function activateTab(id){
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${id}"]`).classList.add('active');
  document.getElementById('panel-'+id).classList.add('active');
  aggctl.style.display=(id==='density')?'flex':'none';
  stabctl.style.display=(id==='graph'||id==='ranked')?'flex':'none';
  if(id==='density') buildGrid('grid-density',colorDensity,false);
  if(id==='graph') buildGrid('grid-graph',colorGraphBg,true);
  if(id==='ranked'){ rankedBuilt=false; buildRanked(); }
  if(PINNED!==null && id==='graph') pinnedView();
}
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>activateTab(t.dataset.tab)));

// ---- controls -------------------------------------------------------------- //
document.getElementById('gain').addEventListener('input',e=>{
  GAIN=parseFloat(e.target.value);
  document.getElementById('gainval').textContent=GAIN.toFixed(1)+'\u00d7';
  rebuildActive();
});
document.getElementById('aggmode').addEventListener('change',e=>{ AGGMODE=e.target.value; buildGrid('grid-density',colorDensity,false); });
document.getElementById('stabmin').addEventListener('input',e=>{
  STABMIN=parseFloat(e.target.value);
  document.getElementById('stabval').textContent=STABMIN.toFixed(2);
  if(document.querySelector('.tab.active').dataset.tab==='ranked'){ buildRanked(); }
  if(PINNED!==null) pinnedView();
});
function rebuildActive(){
  const id=document.querySelector('.tab.active').dataset.tab;
  if(id==='first') buildGrid('grid-first',colorFirst,false);
  if(id==='density') buildGrid('grid-density',colorDensity,false);
  if(id==='graph') buildGrid('grid-graph',colorGraphBg,true);
  if(id==='ranked') buildRanked();
}

// ---- header + footer ------------------------------------------------------- //
document.getElementById('meta').innerHTML=
  `class <b>${M.target} (${M.target_name})</b> &bull; f(x)=<b>${M.f_x.toFixed(3)}</b> &bull; `+
  `grid ${GH}&times;${GW} &bull; ${M.n_edges} interaction pairs &bull; `+
  `${M.n_active_cells} active cells / ${M.candidate_pairs} candidates &bull; N=${M.n_samples}`;
document.getElementById('foot').innerHTML=
  `HIME: first-order main effects + degree-2 interactions recovered by LASSO support recovery. `+
  `Edges are undirected (&Delta;<sub>ij</sub>=&Delta;<sub>ji</sub>); width=|&Delta;|, opacity=stability. `+
  `Use the <b>min stability</b> slider to hide low-confidence pairs. Click a cell in the graph tab to pin it.`;

ensureCanvas();
buildGrid('grid-first',colorFirst,false);
</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
# synthetic demo so you can open something immediately (no Torch / model)
# --------------------------------------------------------------------------- #
def _demo():
    import types
    rng = np.random.default_rng(3)
    gh = gw = 12
    H = W = 384
    n_cells = gh * gw

    # fake blocky main-effect attribution: a couple of hot regions
    main = np.zeros(n_cells)
    for c in [40, 41, 52, 53, 77, 90]:
        main[c] = rng.uniform(0.4, 1.0)
    for c in [10, 11, 130]:
        main[c] = -rng.uniform(0.3, 0.6)
    attr = np.zeros((H, W))
    for c in range(n_cells):
        cy, cx = c // gw, c % gw
        attr[cy*H//gh:(cy+1)*H//gh, cx*W//gw:(cx+1)*W//gw] = main[c]

    # fake interactions: hierarchy-respecting pairs among active cells + noise
    active = [c for c in range(n_cells) if abs(main[c]) > 0.2]
    inter, stab = [], {}
    for _ in range(28):
        i, j = sorted(rng.choice(active, 2, replace=False).tolist())
        s = rng.uniform(-1, 1) * 0.5
        inter.append((i, j, float(s)))
        stab[f"{i}-{j}"] = float(np.clip(0.5 + abs(s) + rng.uniform(-0.2, 0.3), 0, 1))

    res = types.SimpleNamespace(
        attribution=attr, method="hime", target_class=497,
        target_class_name="church", f_x=0.83,
        extras=dict(grid=(gh, gw), n_samples=2500, n_active_cells=len(active),
                    candidate_pairs=len(active)*(len(active)-1)//2,
                    interactions=inter, interaction_stability=stab),
    )

    # synthetic input image (smooth color field) so the overlay has something
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    img01 = np.stack([0.4+0.4*np.sin(6*xx), 0.4+0.3*np.cos(5*yy),
                      0.5+0.3*np.sin(4*(xx+yy))], axis=-1)
    img01 = np.clip(img01, 0, 1)

    out = "/home/claude/hime_demo.html"
    export_hime_html(res, img01, out)
    print("wrote", out)


if __name__ == "__main__":
    _demo()