"""Demonstrate the non-additivity of LIME and motivate PyramidExplainer.

The weak-regime trap: two tiny cells on a fine grid carry almost no signal, so
v(joint) ~ 0 and any synergy is at the noise floor. To produce a convincing
demonstration we (a) select cells that actually move the model and (b) report
synergy normalized by the total explainable response f(x)-f(b).

Selection strategies (--select):
  lime   : top-k cells by LIME coefficient        (original; can be near-dead)
  value  : top-k cells by individual reveal v(c)   (cells that matter alone)
  pair   : the *pair* (i,j) with the largest |g| = |v(i,j) - v(i) - v(j)|
           among a candidate shortlist  (best for SHOWING synergy)

For each selected coalition we measure, with the package's Phi completion
operator (selected cells sharp, rest = blur reference b):
    v(c)     = f(Phi_c)     - f(b)     per cell
    v(joint) = f(Phi_joint) - f(b)     all selected cells together
    g        = v(joint) - sum_c v(c)   the group non-additivity term
so that                       v(A,B) = v(A) + v(B) + g(A,B)   (exact, by defn).

Usage:
    python verify_nonadditivity.py --image img/church.JPEG --out nonadd_out
        [--select pair] [--grid 6 6] [--topk 2] [--n-candidates 24]
        [--target 497] [--sigma 50] [--device cpu] [--n-samples 1000]
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_suff.backbone import (
    denormalize,
    get_class_names,
    load_backbone,
    load_image,
)
from xai_suff.explainers import LIMEExplainer, blur_reference


# --------------------------------------------------------------------------- #
# core mechanics
# --------------------------------------------------------------------------- #
def cell_id_map(H: int, W: int, grid) -> torch.Tensor:
    """Replicate LIMEExplainer._cell_id_map: pixel -> flat cell index in [0,gh*gw)."""
    gh, gw = grid
    ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
    xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
    return ys.view(-1, 1) * gw + xs.view(1, -1)  # (H,W)


@torch.no_grad()
def reveal_value(model, x, b, cell_ids, keep_cells, target, f0):
    """v(S) = f(Phi_S(x)) - f0, revealing only the cells in `keep_cells` (a set).

    Phi_S(x) = m_S * x + (1 - m_S) * b   (selected cells sharp, rest blurred).
    Returns (v, f_phi).
    """
    keep = torch.zeros_like(cell_ids, dtype=torch.float32)  # (H,W)
    for c in keep_cells:
        keep = keep + (cell_ids == c).float()
    keep = keep.clamp(0, 1).view(1, 1, *cell_ids.shape)     # (1,1,H,W)
    comp = keep * x + (1 - keep) * b
    f_phi = float(F.softmax(model(comp), dim=1)[0, target])
    return f_phi - f0, f_phi


@torch.no_grad()
def all_single_values(model, x, b, cell_ids, n_cells, target, f0):
    """v(c) for every cell c (one forward pass each). Returns np array (n_cells,)."""
    vs = np.zeros(n_cells, dtype=np.float64)
    for c in range(n_cells):
        if (cell_ids == c).any():
            vs[c], _ = reveal_value(model, x, b, cell_ids, {c}, target, f0)
    return vs


# --------------------------------------------------------------------------- #
# cell / coalition selection
# --------------------------------------------------------------------------- #
def select_coalition(args, model, x, b, cell_ids, n_cells, coefs, target, f0):
    """Return (selected_cells:list, single_v:dict[c]->v, note:str)."""
    if args.select == "lime":
        order = np.argsort(-coefs)
        cells = [int(c) for c in order[: args.topk]]
        sv = {}
        for c in cells:
            sv[c], _ = reveal_value(model, x, b, cell_ids, {c}, target, f0)
        return cells, sv, "top-k by LIME coefficient"

    if args.select == "value":
        vs = all_single_values(model, x, b, cell_ids, n_cells, target, f0)
        order = np.argsort(-vs)
        cells = [int(c) for c in order[: args.topk]]
        sv = {int(c): float(vs[c]) for c in cells}
        return cells, sv, "top-k by individual reveal v(c)"

    if args.select == "pair":
        # shortlist by individual value, then brute-force |g| over all pairs.
        vs = all_single_values(model, x, b, cell_ids, n_cells, target, f0)
        cand = [int(c) for c in np.argsort(-vs)[: args.n_candidates]]
        best = None  # (abs_g, (i,j), v_i, v_j, v_ij, g)
        for i, j in itertools.combinations(cand, 2):
            v_ij, _ = reveal_value(model, x, b, cell_ids, {i, j}, target, f0)
            g = v_ij - vs[i] - vs[j]
            if best is None or abs(g) > best[0]:
                best = (abs(g), (i, j), float(vs[i]), float(vs[j]),
                        float(v_ij), float(g))
        (_, (i, j), v_i, v_j, _, _) = best
        sv = {i: v_i, j: v_j}
        return [i, j], sv, f"max-|g| pair among top-{args.n_candidates} by v(c)"

    raise ValueError(args.select)


# --------------------------------------------------------------------------- #
# visualization
# --------------------------------------------------------------------------- #
def save_panel(out_path, x, b, cell_ids, sel_cells, coefs, grid, target,
               target_name, results, total, per_cell, f0):
    """Synergy panel.

    Row of reveals: [input w/ A,B outlined] [reveal A: f(A)] [reveal B: f(B)]
                    [reveal A,B: f(A,B)]   then a synergy bar:
                    v(A)+v(B)  vs  v(A,B), with the gap g shaded.
    Generalizes to k>2 (one reveal column per cell, capped for layout).
    """
    img = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
    Hc, Wc = cell_ids.shape
    labels = [chr(ord("A") + i) for i in range(len(sel_cells))]

    def reveal_img(cells):
        keep = torch.zeros_like(cell_ids, dtype=torch.float32)
        for c in cells:
            keep = keep + (cell_ids == c).float()
        keep = keep.clamp(0, 1).view(1, 1, Hc, Wc).to(x.device)
        comp = keep * x + (1 - keep) * b
        return denormalize(comp)[0].permute(1, 2, 0).cpu().numpy()

    def outline(ax, c, color):
        """Draw a rectangle around cell c's region."""
        ys, xs = torch.where(cell_ids.cpu() == c)
        if len(ys) == 0:
            return
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   fill=False, edgecolor=color, linewidth=2.5))

    v_joint = results["v_joint"]; v_sum = results["v_sum"]; g = results["synergy"]
    cell_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    # ---- font sizes (bump these to scale the whole figure) ----
    FS_TITLE = 18      # per-axis subplot titles
    FS_BADGE = 26      # A / B region labels on the input
    FS_AXLABEL = 16    # y-axis label
    FS_TICK = 15       # x/y tick labels
    FS_LEGEND = 14     # synergy-bar legend
    FS_SUPTITLE = 20   # figure suptitle

    n_reveal = len(sel_cells)
    n_cols = 1 + n_reveal + 1 + 1  # input | per-cell reveals | joint | synergy bar
    fig, ax = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.8))

    # --- input with all selected cells outlined ---
    ax[0].imshow(img)
    for k, c in enumerate(sel_cells):
        outline(ax[0], c, cell_colors[k % len(cell_colors)])
        ys, xs = torch.where(cell_ids.cpu() == c)
        ax[0].text(int(xs.float().mean()), int(ys.float().mean()), labels[k],
                   color="white", fontsize=FS_BADGE, fontweight="bold", ha="center",
                   va="center", bbox=dict(boxstyle="round,pad=0.2",
                                          fc=cell_colors[k % len(cell_colors)], ec="none"))
    ax[0].set_title(f"input: regions {', '.join(labels)}", fontsize=FS_TITLE)
    ax[0].axis("off")

    # --- each cell revealed alone ---
    for k, c in enumerate(sel_cells):
        a = ax[1 + k]
        a.imshow(reveal_img([c]))
        d = per_cell[c]
        a.set_title(f"reveal {labels[k]} only\nf({labels[k]})={d['f_phi']:.3f}  "
                    f"v={d['v']:+.3f}",
                    color=cell_colors[k % len(cell_colors)], fontsize=FS_TITLE)
        a.axis("off")

    # --- joint reveal ---
    aj = ax[1 + n_reveal]
    aj.imshow(reveal_img(sel_cells))
    joint_lab = ",".join(labels)
    aj.set_title(f"reveal {joint_lab}\nf({joint_lab})={results['f_phi_joint']:.3f}  "
                 f"v={v_joint:+.3f}", color="black", fontsize=FS_TITLE)
    aj.axis("off")

    # --- synergy bar: additive prediction vs actual joint ---
    ab = ax[-1]
    bottoms = 0.0
    for k, c in enumerate(sel_cells):
        vk = per_cell[c]["v"]
        ab.bar(0, vk, bottom=bottoms, width=0.6,
               color=cell_colors[k % len(cell_colors)],
               label=f"v({labels[k]})={vk:+.3f}")
        bottoms += vk
    # synergy stacked on top of the additive sum
    ab.bar(0, g, bottom=v_sum, width=0.6, color="#999999", hatch="//",
           label=f"g={g:+.3f}")
    ab.bar(1, v_joint, width=0.6, color="black", alpha=0.85,
           label=f"v(joint)={v_joint:+.3f}")
    ab.set_xticks([0, 1])
    ab.set_xticklabels(["sum v(c)\n+ g", "actual\nv(joint)"], fontsize=FS_TICK)
    ab.tick_params(axis="y", labelsize=FS_TICK)
    ab.set_ylabel("target-prob gain over b", fontsize=FS_AXLABEL)
    ab.set_title(f"g = {g:+.3f}  ({g / (total + 1e-12) * 100:+.1f}% of f(x)-f(b))",
                 fontsize=FS_TITLE)
    ab.legend(fontsize=FS_LEGEND, loc="upper left")
    ab.axhline(0, color="k", lw=0.6)

    fig.suptitle(
        f"class={target} ({target_name})   "
        f"v(joint)={v_joint:+.4f}   sum v(c)={v_sum:+.4f}   g={g:+.4f}   "
        f"({'COOPERATION (whole>parts)' if g > 0 else 'REDUNDANCY (whole<parts)'})",
        fontsize=FS_SUPTITLE,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="nonadd_out")
    ap.add_argument("--target", type=int, default=None,
                    help="target class index; default = model top-1")
    ap.add_argument("--grid", type=int, nargs=2, default=[6, 6],
                    help="coarser grid => each cell carries real signal")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--select", choices=["lime", "value", "pair"], default="pair")
    ap.add_argument("--n-candidates", type=int, default=24,
                    help="pair-mode: shortlist size (by v(c)) before brute-force |g|")
    ap.add_argument("--sigma", type=float, default=50.0)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    grid = tuple(args.grid)

    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)
    b = blur_reference(x, args.sigma)
    _, _, H, W = x.shape

    with torch.no_grad():
        top1 = int(model(x).argmax(1).item())
    target = args.target if args.target is not None else top1
    target_name = class_names[target] if target < len(class_names) else str(target)

    with torch.no_grad():
        f0 = float(F.softmax(model(b), dim=1)[0, target])
        f_x = float(F.softmax(model(x), dim=1)[0, target])
    total = f_x - f0
    print(f"[verify] target={target} ({target_name})  f(x)={f_x:.4f}  "
          f"f(b)={f0:.4f}  total=f(x)-f(b)={total:+.4f}")

    # ---- LIME coefficients (always computed, for the panel + lime-select) ----
    lime = LIMEExplainer(
        model, target_class=target, device=device, class_names=class_names,
        grid=grid, n_samples=args.n_samples, sigma=args.sigma,
    )
    lime_res = lime.explain(x)
    cell_ids = cell_id_map(H, W, grid).to(device)
    n_cells = grid[0] * grid[1]
    attr = torch.tensor(lime_res.attribution, device=device)
    coefs = np.zeros(n_cells, dtype=np.float64)
    for c in range(n_cells):
        mask = (cell_ids == c)
        if mask.any():
            coefs[c] = float(attr[mask][0])

    # ---- select coalition ----
    sel_cells, per_v, note = select_coalition(
        args, model, x, b, cell_ids, n_cells, coefs, target, f0
    )
    print(f"[verify] selection ({args.select}): cells {sel_cells}  [{note}]")

    v_sum = 0.0
    per_cell = {}
    for c in sel_cells:
        v_c = per_v[c]
        _, f_phi_c = reveal_value(model, x, b, cell_ids, {c}, target, f0)
        per_cell[c] = dict(v=v_c, f_phi=f_phi_c, lime_coef=float(coefs[c]))
        v_sum += v_c
        print(f"[verify]   cell {c:4d}: LIME w={coefs[c]:+.4f}  "
              f"v(c)={v_c:+.4f}  f(Phi)={f_phi_c:.4f}")

    v_joint, f_phi_joint = reveal_value(
        model, x, b, cell_ids, set(sel_cells), target, f0
    )
    g = v_joint - v_sum
    g_frac = g / (total + 1e-12)
    g_rel = g / (abs(v_joint) + 1e-12)

    print(f"[verify] v(joint)={v_joint:+.4f}  sum v(c)={v_sum:+.4f}  g={g:+.4f}")
    print(f"[verify] g as % of total f(x)-f(b): {g_frac * 100:+.1f}%")
    print(f"[verify] g as % of v(joint):        {g_rel * 100:+.1f}%")
    if abs(g) < 1e-4:
        print("[verify] additive within tolerance on this coalition (weak regime)")
    else:
        kind = "COOPERATION (whole>parts)" if g > 0 else "REDUNDANCY (whole<parts)"
        print(f"[verify] NON-ADDITIVE: {kind}")

    results = dict(v_joint=v_joint, v_sum=v_sum, synergy=g, f_phi_joint=f_phi_joint)

    panel_path = os.path.join(args.out, "nonadditivity.png")
    save_panel(panel_path, x, b, cell_ids, sel_cells, coefs, grid,
               target, target_name, results, total, per_cell, f0)

    lines = [
        f"image: {args.image}",
        f"target: {target} ({target_name})",
        f"grid: {grid[0]}x{grid[1]}   select: {args.select} ({note})   "
        f"sigma: {args.sigma}",
        f"f(x)={f_x:.4f}  f(b)={f0:.4f}  total explainable = {total:+.4f}",
        "",
        "per-cell (only that cell revealed, rest = blur b):",
    ]
    for c in sel_cells:
        d = per_cell[c]
        lines.append(f"  cell {c}: LIME_w={d['lime_coef']:+.4f}  "
                     f"v(c)={d['v']:+.4f}  f(Phi)={d['f_phi']:.4f}")
    lines += [
        "",
        f"v(joint)          = {v_joint:+.6f}",
        f"sum_c v(c)        = {v_sum:+.6f}",
        f"synergy g         = {g:+.6f}",
        f"  g / (f(x)-f(b)) = {g_frac * 100:+.2f}%",
        f"  g / v(joint)    = {g_rel * 100:+.2f}%",
        "",
        "identity:  v(A,B) = v(A) + v(B) + g(A,B)",
        f"  check:   {v_joint:+.6f} = {v_sum:+.6f} + {g:+.6f}  "
        f"(residual {v_joint - (v_sum + g):.2e})",
        "",
        "interpretation:",
        "  LIME predicts v(A,B) ~ w_A + w_B (purely additive, g forced to 0).",
        "  The measured g is the group non-additivity term PyramidExplainer",
        "  keeps as the node synergy Delta(R).",
    ]
    summary_path = os.path.join(args.out, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[verify] panel   -> {panel_path}")
    print(f"[verify] summary -> {summary_path}")


if __name__ == "__main__":
    main()