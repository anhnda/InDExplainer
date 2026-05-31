"""HTML template (v2) for PyramidExplainer: three tabbed, pyramid-native views.

Replaces the single flat-icicle template. Three tabs share one inlined image +
one node table (RLE masks) and one set of controls; hovering any node in any
tab traces its region onto the input image.

  TAB 1  NESTED ICICLE   true containment layout: a parent spans exactly its
                         children's combined width and sits directly above them,
                         so vertical reading = the tree. Row = depth. Optional
                         "collapse trivial merges" folds big-blob+crumb chains.
  TAB 2  RANKED |Delta|  the direct answer to "where is the cooperation": nodes
                         sorted by |Delta|, each a bar + region thumbnail + the
                         merge it came from. Depth-independent, so it ignores the
                         caterpillar entirely.
  TAB 3  DENDROGRAM      x = node's holistic value v(R); a merge is drawn as two
                         child stems joining at the parent's v, and the vertical
                         jump at the join = Delta (the joint value that appeared
                         only on merging). The big synergy reads as a tall step.

This module exposes build_html_v2(...) used by export_interactive_html.
"""
from __future__ import annotations
import base64, io, json
from typing import Optional
import numpy as np


# this file is imported by pyramid_interactions; it only owns the v2 template
# and the payload builder. RLE / image helpers are passed in to avoid dup.

def build_payload(res, img_b64, leaf_masks, tree_helpers, max_nodes_with_masks):
    _tree, _by_id, _root_id, _depths, _node_mask, _rle_rows = tree_helpers
    tree = _tree(res); by_id = _by_id(res); depth = _depths(res)
    root_id = _root_id(res)
    H, W = next(iter(leaf_masks.values())).shape

    internals = [n for n in tree if not n["is_leaf"]]
    top_internal = set(
        n["id"] for n in sorted(internals, key=lambda n: abs(n["delta"]),
                                reverse=True)[:max(0, (max_nodes_with_masks or 0))]
    )
    js_nodes = {}
    for n in tree:
        nid = n["id"]
        entry = {
            "id": nid, "depth": depth[nid], "area": int(n["area"]),
            "v": float(n["v"]), "delta": float(n["delta"]),
            "is_leaf": bool(n["is_leaf"]),
            "children": [int(c) for c in n["child_ids"]],
        }
        if n["is_leaf"] or nid == root_id or nid in top_internal:
            entry["rle"] = (_rle_rows(leaf_masks[nid]) if n["is_leaf"]
                            else _rle_rows(_node_mask(res, nid, leaf_masks, {})))
        js_nodes[nid] = entry

    parent = {}
    for n in tree:
        for c in n["child_ids"]:
            parent[int(c)] = int(n["id"])

    meta = {
        "H": H, "W": W, "root": int(root_id),
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
    return {"meta": meta, "nodes": js_nodes, "parent": parent}


def render(res, img_b64, payload):
    payload_json = json.dumps(payload, separators=(",", ":"))
    return _TEMPLATE.replace("__IMG_B64__", img_b64).replace("__PAYLOAD__", payload_json)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyramidExplainer &mdash; synergy views</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaf0;--muted:#9aa3b2;--line:#2a2f3a;
    --coop:#e8463a;--redund:#3a6ee8;--accent:#f5c451;--kin:#6de08a;--par:#c98bff;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  header{padding:13px 20px;border-bottom:1px solid var(--line);display:flex;
    align-items:baseline;gap:18px;flex-wrap:wrap;}
  header h1{font-size:16px;margin:0;font-weight:650;}
  header .meta{color:var(--muted);font-size:12.5px;}
  header .meta b{color:var(--ink);font-weight:600;}
  .wrap{display:grid;grid-template-columns:420px 1fr;height:calc(100vh - 53px);}
  .left{padding:18px;border-right:1px solid var(--line);overflow:auto;}
  .right{padding:12px 18px 24px;overflow:auto;display:flex;flex-direction:column;}
  .imgbox{position:relative;width:100%;max-width:382px;}
  .imgbox img{width:100%;display:block;border-radius:8px;}
  .imgbox canvas{position:absolute;left:0;top:0;width:100%;height:100%;border-radius:8px;pointer-events:none;}
  .readout{margin-top:14px;background:var(--panel);border:1px solid var(--line);
    border-radius:8px;padding:12px 14px;min-height:120px;}
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
  .tabs{display:flex;gap:4px;margin-bottom:10px;border-bottom:1px solid var(--line);}
  .tab{padding:8px 16px;cursor:pointer;color:var(--muted);font-weight:600;font-size:13.5px;
    border-bottom:2px solid transparent;margin-bottom:-1px;}
  .tab:hover{color:var(--ink);}
  .tab.active{color:var(--ink);border-bottom-color:var(--accent);}
  .panel{display:none;flex:1 1 auto;min-height:0;}
  .panel.active{display:block;}
  .controls{margin:2px 0 12px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;}
  .controls label{font-size:12.5px;color:var(--muted);display:flex;gap:6px;align-items:center;}
  .controls input[type=range]{width:150px;}
  .hint{color:var(--muted);font-size:12px;margin:2px 0 10px;}
  .foot{color:var(--muted);font-size:11.5px;margin-top:16px;line-height:1.5;}
  code{background:#0b0d11;padding:1px 5px;border-radius:4px;color:#cdd3df;}
  /* icicle */
  #icicle .level{display:flex;align-items:stretch;height:26px;margin-bottom:2px;width:100%;}
  #icicle .level-label{width:48px;flex:0 0 48px;color:var(--muted);font-size:11px;
    display:flex;align-items:center;}
  #icicle .level-row{position:relative;flex:1 1 auto;}
  .blk{position:absolute;top:0;height:100%;border-radius:2px;cursor:pointer;
    outline:2px solid transparent;outline-offset:-2px;transition:filter .08s;min-width:1px;}
  .blk:hover{filter:brightness(1.3);}
  .blk.sel{outline-color:var(--accent);z-index:3;}
  .blk.kin{outline-color:var(--kin);z-index:2;}
  .blk.par{outline-color:var(--par);z-index:2;}
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
  /* dendrogram */
  #dendro svg{width:100%;height:auto;display:block;}
  #dendro .edge{stroke:#566;stroke-width:1;fill:none;}
  #dendro .join{stroke-width:2.5;}
  #dendro .nodedot{cursor:pointer;}
  #dendro .axis{stroke:var(--line);stroke-width:1;}
  #dendro .axlab{fill:var(--muted);font-size:10px;}
</style></head>
<body>
<header>
  <h1>PyramidExplainer &middot; synergy views</h1>
  <div class="meta" id="meta"></div>
</header>
<div class="wrap">
  <div class="left">
    <div class="imgbox">
      <img id="img" alt="input" src="data:image/png;base64,__IMG_B64__">
      <canvas id="overlay"></canvas>
    </div>
    <div class="readout" id="readout">
      <div class="big">Hover a node &rarr;</div>
      <div class="kv"><span class="k">Any view: hovering traces the region onto the image.</span><span class="v"></span></div>
    </div>
    <div class="legend">synergy &Delta; (red = cooperation, blue = redundancy)
      <div class="bar"></div>
      <span style="float:left">more redundant</span><span style="float:right">more cooperative</span>
      <div style="clear:both"></div>
    </div>
    <div class="foot" id="foot"></div>
  </div>
  <div class="right">
    <div class="tabs">
      <div class="tab active" data-tab="icicle">Nested icicle</div>
      <div class="tab" data-tab="ranked">Ranked |&Delta;|</div>
      <div class="tab" data-tab="dendro">Dendrogram</div>
    </div>
    <div class="controls">
      <label>color gain <input type="range" id="gain" min="0.2" max="4" step="0.1" value="1"><span id="gainval">1.0&times;</span></label>
      <label><input type="checkbox" id="collapse"> collapse trivial merges (big&nbsp;blob&nbsp;+&nbsp;crumb)</label>
    </div>

    <div class="panel active" id="panel-icicle">
      <div class="hint">True containment: a parent spans exactly its children and sits directly above them &mdash; read the tree top-to-bottom. Caterpillar trees look like a staircase; that's real, not a bug.</div>
      <div id="icicle"></div>
    </div>
    <div class="panel" id="panel-ranked">
      <div class="hint">The direct answer to &ldquo;where is the cooperation?&rdquo; &mdash; nodes by |&Delta;|, depth-independent. Thumbnail = the region; bar = signed &Delta;.</div>
      <div id="ranked"></div>
    </div>
    <div class="panel" id="panel-dendro">
      <div class="hint">x = holistic value v(R). Two child stems join at the parent; the horizontal jump at the join is &Delta; (value that appeared only on merging). Big synergy = a long step.</div>
      <div id="dendro"></div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const M = DATA.meta, NODES = DATA.nodes, PARENT = DATA.parent;
const totalArea = M.W * M.H;
let GAIN = 1.0, PINNED = null, COLLAPSE = false;

let DMAX = 1e-9;
for (const id in NODES){const d=Math.abs(NODES[id].delta); if(d>DMAX)DMAX=d;}

// ---- region paint ---------------------------------------------------------- //
const img=document.getElementById('img'), cv=document.getElementById('overlay'), ctx=cv.getContext('2d');
function ensureCanvas(){if(cv.width!==M.W||cv.height!==M.H){cv.width=M.W;cv.height=M.H;}}
function clearOverlay(){ensureCanvas();ctx.clearRect(0,0,cv.width,cv.height);}
function nodeRLE(id){
  const n=NODES[id]; if(n.rle)return n.rle;
  const rowmap=new Map();
  (function collect(k){const nn=NODES[k];
    if(nn.is_leaf){for(const [r,runs] of (nn.rle||[])){if(!rowmap.has(r))rowmap.set(r,[]);
      const a=rowmap.get(r);for(let i=0;i<runs.length;i+=2)a.push([runs[i],runs[i+1]]);}}
    else{for(const c of nn.children)collect(c);}})(id);
  const out=[];
  for(const [r,spans] of [...rowmap.entries()].sort((a,b)=>a[0]-b[0])){
    spans.sort((a,b)=>a[0]-b[0]); const mg=[]; let [cs,cl]=spans[0];
    for(let i=1;i<spans.length;i++){const [s,l]=spans[i];
      if(s<=cs+cl)cl=Math.max(cl,s+l-cs);else{mg.push(cs,cl);cs=s;cl=l;}}
    mg.push(cs,cl); out.push([r,mg]);}
  return out;
}
function paintNode(id){
  clearOverlay();
  const rle=nodeRLE(id); ctx.fillStyle='rgba(245,196,81,0.42)';
  for(const [r,runs] of rle)for(let i=0;i<runs.length;i+=2)ctx.fillRect(runs[i],r,runs[i+1],1);
  const n=NODES[id];
  if(!n.is_leaf&&n.children.length){ctx.fillStyle='rgba(109,224,138,0.30)';
    for(const c of n.children){const cr=nodeRLE(c);
      for(const [r,runs] of cr)for(let i=0;i<runs.length;i+=2)ctx.fillRect(runs[i],r,runs[i+1],1);}}
}

// ---- color ----------------------------------------------------------------- //
function deltaColor(d){
  const t=Math.max(-1,Math.min(1,(d/DMAX)*GAIN));
  const coop=[232,70,58],redund=[58,110,232],mid=[176,182,196];
  const c=(t>=0)?mid.map((m,i)=>Math.round(m+(coop[i]-m)*t))
                :mid.map((m,i)=>Math.round(m+(redund[i]-m)*(-t)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function classifyDelta(d){
  if(Math.abs(d)<DMAX*1e-3)return['add','additive (&Delta;&approx;0)'];
  return d>0?['coop','cooperation (&Delta;&gt;0)']:['redund','redundancy (&Delta;&lt;0)'];
}

// ---- readout + kin highlight (works across tabs) --------------------------- //
const readout=document.getElementById('readout');
function showReadout(id){
  const n=NODES[id], [cls,lbl]=classifyDelta(n.delta);
  const childStr=n.children.length?n.children.map(c=>{const cn=NODES[c];
    const cd=cn.is_leaf?'leaf':('&Delta;='+cn.delta.toFixed(4));
    return `#${c} (v=${cn.v.toFixed(4)}, ${cd})`;}).join('<br>'):'&mdash; leaf &mdash;';
  readout.innerHTML=`<div class="big">node #${id} &nbsp;<span class="pill ${cls}">${lbl}</span></div>
    <div class="kv">
    <span class="k">depth</span><span class="v">${n.depth}${n.depth===0?' (root)':''}</span>
    <span class="k">area</span><span class="v">${n.area} px (${(100*n.area/totalArea).toFixed(2)}%)</span>
    <span class="k">v(R)=f(&Phi;<sub>R</sub>)-f(b)</span><span class="v">${n.v.toFixed(5)}</span>
    <span class="k">&Delta;(R) synergy</span><span class="v">${n.is_leaf?'&mdash; (leaf)':n.delta.toFixed(5)}</span>
    <span class="k">children</span><span class="v">${childStr}</span></div>`;
}
function markKin(id){
  document.querySelectorAll('.blk,.rk,.nodedot').forEach(e=>e.classList.remove('sel','kin','par'));
  document.querySelectorAll(`[data-id="${id}"]`).forEach(e=>e.classList.add('sel'));
  for(const c of NODES[id].children)
    document.querySelectorAll(`[data-id="${c}"]`).forEach(e=>e.classList.add('kin'));
  const p=PARENT[id]; if(p!==undefined)
    document.querySelectorAll(`[data-id="${p}"]`).forEach(e=>e.classList.add('par'));
}
function hover(id){ if(PINNED!==null)return; paintNode(id); markKin(id); showReadout(id); }
function leave(){ if(PINNED!==null){pinnedView();return;} clearOverlay();
  document.querySelectorAll('.blk,.rk,.nodedot').forEach(e=>e.classList.remove('sel','kin','par')); }
function pin(id){ if(PINNED===id){PINNED=null;leave();return;} PINNED=id; pinnedView(); }
function pinnedView(){ paintNode(PINNED); markKin(PINNED); showReadout(PINNED); }

// ---- collapse: skip internal nodes whose smaller child is a tiny crumb ----- //
// A node is "trivial" if min(child areas)/area < CRUMB and that child is a leaf
// or itself trivial. Collapsing splices grandchildren up, so the staircase of
// big-blob+crumb merges folds into the one merge that actually combines mass.
const CRUMB=0.04;
function effectiveChildren(id){
  // returns child ids after collapsing trivial intermediate merges
  if(!COLLAPSE) return NODES[id].children;
  const out=[];
  for(const c of NODES[id].children){
    let cur=c;
    while(true){
      const n=NODES[cur];
      if(n.is_leaf){out.push(cur);break;}
      const areas=n.children.map(k=>NODES[k].area);
      const mn=Math.min(...areas), frac=mn/n.area;
      if(frac<CRUMB){
        // splice: keep the big child, drop the crumb up a level by descending
        const big=n.children[areas[0]>=areas[1]?0:1];
        const small=n.children[areas[0]>=areas[1]?1:0];
        out.push(small);          // crumb becomes a direct child here
        cur=big;                  // keep walking down the big side
      } else { out.push(cur); break; }
    }
  }
  return out;
}

// ============================ TAB 1: NESTED ICICLE ========================== //
// True containment via recursive x-interval layout. Each node occupies an
// [x0,x1] fraction of the width = its share of the root area, positioned so it
// sits under its parent and over its children.
const icicle=document.getElementById('icicle');
function buildIcicle(){
  icicle.innerHTML='';
  const rootA=NODES[M.root].area;
  // assign x-intervals by DFS; row=depth (post-collapse depth recomputed)
  const placed=[]; // {id, depth, x0, x1}
  (function place(id,x0,x1,depth){
    placed.push({id,depth,x0,x1});
    const kids=effectiveChildren(id);
    if(!kids.length)return;
    const tot=kids.reduce((s,k)=>s+NODES[k].area,0)||1;
    let cx=x0;
    for(const k of kids){const w=(x1-x0)*NODES[k].area/tot; place(k,cx,cx+w,depth+1); cx+=w;}
  })(M.root,0,1,0);
  const maxd=Math.max(...placed.map(p=>p.depth));
  const rows=[]; for(let d=0;d<=maxd;d++)rows[d]=[];
  for(const p of placed)rows[p.depth].push(p);
  for(let d=0;d<=maxd;d++){
    const level=document.createElement('div');level.className='level';
    const lab=document.createElement('div');lab.className='level-label';
    lab.textContent=(d===0?'root':'d'+d);
    const row=document.createElement('div');row.className='level-row';
    level.appendChild(lab);level.appendChild(row);
    for(const p of rows[d]){
      const n=NODES[p.id]; const b=document.createElement('div');
      b.className='blk'; b.dataset.id=p.id;
      b.style.left=(100*p.x0)+'%'; b.style.width=(100*(p.x1-p.x0))+'%';
      b.style.background=n.is_leaf?'#222732':deltaColor(n.delta);
      b.addEventListener('mouseenter',()=>hover(p.id));
      b.addEventListener('mouseleave',leave);
      b.addEventListener('click',()=>pin(p.id));
      row.appendChild(b);
    }
    icicle.appendChild(level);
  }
}

// ============================ TAB 2: RANKED |Δ| ============================= //
const ranked=document.getElementById('ranked');
let rankedBuilt=false;
function buildRanked(){
  ranked.innerHTML='';
  const internals=Object.values(NODES).filter(n=>!n.is_leaf);
  internals.sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));
  const top=internals.slice(0,30);
  const maxAbs=Math.abs(top[0]?.delta||1);
  top.forEach((n,i)=>{
    const row=document.createElement('div');row.className='rk';row.dataset.id=n.id;
    const num=document.createElement('div');num.className='rnum';num.textContent=(i+1);
    const th=document.createElement('canvas');th.className='thumb';th.width=M.W;th.height=M.H;
    drawThumb(th,n.id);
    const wrap=document.createElement('div');wrap.className='rbar-wrap';
    const track=document.createElement('div');track.className='rbar-track';
    const bar=document.createElement('div');bar.className='rbar';
    const frac=Math.abs(n.delta)/maxAbs;
    bar.style.width=(100*frac)+'%';
    bar.style.background=deltaColor(n.delta);
    if(n.delta>=0){bar.style.left='50%';bar.style.width=(50*frac)+'%';}
    else{bar.style.right='50%';bar.style.left='auto';bar.style.width=(50*frac)+'%';}
    track.appendChild(bar);
    // center tick
    const tick=document.createElement('div');tick.style.cssText='position:absolute;left:50%;top:0;width:1px;height:100%;background:#3a3f4a;';
    track.appendChild(tick);
    const meta=document.createElement('div');meta.className='rmeta';
    const kind=n.delta>0?'coop':'redundant';
    meta.innerHTML=`#${n.id} &middot; <b>&Delta;=${n.delta>=0?'+':''}${n.delta.toFixed(4)}</b> (${kind}) &middot; depth ${n.depth} &middot; ${(100*n.area/totalArea).toFixed(1)}% area`;
    wrap.appendChild(track);wrap.appendChild(meta);
    row.appendChild(num);row.appendChild(th);row.appendChild(wrap);
    row.addEventListener('mouseenter',()=>hover(n.id));
    row.addEventListener('mouseleave',leave);
    row.addEventListener('click',()=>pin(n.id));
    ranked.appendChild(row);
  });
  rankedBuilt=true;
}
function drawThumb(canvas,id){
  const c=canvas.getContext('2d');
  c.fillStyle='#0b0d11';c.fillRect(0,0,canvas.width,canvas.height);
  // draw the base image faintly then region
  c.globalAlpha=0.5;c.drawImage(img,0,0,canvas.width,canvas.height);c.globalAlpha=1;
  const n=NODES[id];
  c.fillStyle='rgba(245,196,81,0.55)';
  for(const [r,runs] of nodeRLE(id))for(let i=0;i<runs.length;i+=2)c.fillRect(runs[i],r,runs[i+1],1);
}

// ============================ TAB 3: DENDROGRAM ============================= //
const dendro=document.getElementById('dendro');
let dendroBuilt=false;
function buildDendro(){
  dendro.innerHTML='';
  // leaves ordered by a DFS so stems don't cross; y = leaf order, x = v(R)
  const order=[]; (function dfs(id){const n=NODES[id];
    if(n.is_leaf){order.push(id);return;} for(const c of n.children)dfs(c);})(M.root);
  const yOf={}; order.forEach((id,i)=>yOf[id]=i);
  const allV=Object.values(NODES).map(n=>n.v);
  const vmin=Math.min(0,...allV), vmax=Math.max(...allV);
  const Wd=900, Hd=Math.max(360, order.length*4.2), padL=8, padR=120, padT=10, padB=26;
  const sx=v=>padL+(Wd-padL-padR)*(v-vmin)/((vmax-vmin)||1);
  const sy=i=>padT+(Hd-padT-padB)*i/((order.length-1)||1);
  // y of internal = mean of children's y
  const yPos={}; for(const id of order)yPos[id]=sy(yOf[id]);
  (function setY(id){const n=NODES[id]; if(n.is_leaf)return yPos[id];
    const cy=n.children.map(setY); yPos[id]=cy.reduce((a,b)=>a+b,0)/cy.length; return yPos[id];})(M.root);

  let svg=`<svg viewBox="0 0 ${Wd} ${Hd}" xmlns="http://www.w3.org/2000/svg">`;
  // axis
  svg+=`<line class="axis" x1="${padL}" y1="${Hd-padB}" x2="${Wd-padR}" y2="${Hd-padB}"/>`;
  for(let t=0;t<=4;t++){const v=vmin+(vmax-vmin)*t/4; const x=sx(v);
    svg+=`<line class="axis" x1="${x}" y1="${Hd-padB}" x2="${x}" y2="${Hd-padB+4}"/>`;
    svg+=`<text class="axlab" x="${x}" y="${Hd-padB+15}" text-anchor="middle">${v.toFixed(2)}</text>`;}
  svg+=`<text class="axlab" x="${(Wd-padR)/2}" y="${Hd-2}" text-anchor="middle">v(R) = f(&#934;_R) - f(b)</text>`;
  // edges: for each internal node, draw child stems to parent's x, join colored by Δ
  function draw(id){const n=NODES[id]; if(n.is_leaf)return;
    const px=sx(n.v), py=yPos[id];
    for(const c of n.children){const cx=sx(NODES[c].v), cy=yPos[c];
      // elbow: horizontal from child x to parent x at child y, then vertical to parent y
      svg+=`<path class="edge" d="M ${cx} ${cy} H ${px} V ${py}"/>`;
      draw(c);}
    // the join marker: vertical extent shows Δ contribution; color by Δ
    const col=deltaColor(n.delta);
    const r=Math.max(2.2,Math.min(9,2.2+18*Math.abs(n.delta)/DMAX));
    svg+=`<circle class="nodedot" data-id="${id}" cx="${px}" cy="${py}" r="${r}" fill="${col}" stroke="#0b0d11" stroke-width="1"/>`;
  }
  draw(M.root);
  svg+=`</svg>`;
  dendro.innerHTML=svg;
  dendro.querySelectorAll('.nodedot').forEach(el=>{
    const id=+el.dataset.id;
    el.addEventListener('mouseenter',()=>hover(id));
    el.addEventListener('mouseleave',leave);
    el.addEventListener('click',()=>pin(id));
  });
  dendroBuilt=true;
}

// ---- tabs ------------------------------------------------------------------ //
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    const id=t.dataset.tab;
    document.getElementById('panel-'+id).classList.add('active');
    if(id==='ranked'&&!rankedBuilt)buildRanked();
    if(id==='dendro'&&!dendroBuilt)buildDendro();
    if(PINNED!==null)pinnedView();
  });
});

// ---- controls -------------------------------------------------------------- //
document.getElementById('gain').addEventListener('input',e=>{
  GAIN=parseFloat(e.target.value);
  document.getElementById('gainval').textContent=GAIN.toFixed(1)+'\u00d7';
  buildIcicle(); rankedBuilt=false; dendroBuilt=false;
  if(document.querySelector('.tab.active').dataset.tab==='ranked')buildRanked();
  if(document.querySelector('.tab.active').dataset.tab==='dendro')buildDendro();
  if(PINNED!==null)pinnedView();
});
document.getElementById('collapse').addEventListener('change',e=>{
  COLLAPSE=e.target.checked; buildIcicle(); if(PINNED!==null)pinnedView();
});

// ---- header + footer ------------------------------------------------------- //
document.getElementById('meta').innerHTML=
  `class <b>${M.target} (${M.target_name})</b> &bull; f(x)=<b>${M.f_x.toFixed(3)}</b> f(b)=<b>${M.f_b.toFixed(3)}</b> &bull; `+
  `&sigma;=${M.sigma} &bull; ${M.n_leaves} leaves / ${M.n_internal} internal &bull; depth ${M.max_depth} &bull; `+
  `v(root)=<b>${M.root_v.toFixed(3)}</b> = leaf <b>${M.sum_leaf_v.toFixed(3)}</b> + &Delta; <b>${M.sum_delta.toFixed(3)}</b>`;
const addShare=Math.abs(M.sum_leaf_v)/(Math.abs(M.sum_leaf_v)+Math.abs(M.sum_delta)||1);
document.getElementById('foot').innerHTML=
  `Identity residual = <code>${M.identity_residual.toExponential(2)}</code> (&approx;0 &rarr; &Delta; trustworthy). `+
  `Additive part is <b>${(100*addShare).toFixed(1)}%</b>; the other <b>${(100*(1-addShare)).toFixed(1)}%</b> is cooperation `+
  `no additive method can represent (Prop. 1). Click any node to pin. Depth ${M.max_depth} on ${M.n_leaves} leaves &rarr; `+
  `this is an unbalanced (caterpillar) tree, so synergy concentrates at a few deep merges, not the top &mdash; the Ranked tab finds them directly.`;

ensureCanvas(); buildIcicle();
</script>
</body></html>
"""