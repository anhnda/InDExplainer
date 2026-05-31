"""Demonstrate the non-additivity of LIME and motivate PyramidExplainer.

Pipeline:
  1. Run grid-LIME (default 14x14) to get a linear coefficient w_c per cell.
  2. Rank cells by coefficient; take the top-k (default k=2).
  3. For each selected cell c, measure its *holistic* value v(c) via ONE forward
     pass with ONLY cell c revealed (sharp) and everything else set to the blur
     reference b  --  this is exactly the package's Phi_R(x) completion operator.
  4. Measure the joint value v(A,B,...) revealing ALL top-k cells together.
  5. Verify additivity fails:   v(A,B) != v(A) + v(B),
     and report the synergy    g(A,B) = v(A,B) - sum_c v(c),
     so that                    v(A,B) = sum_c v(c) + g(A,B)   (exact, by defn).

This is the single-merge analogue of PyramidExplainer's node synergy Delta(R),
shown directly against the per-cell additive credits LIME assigns.

Usage:
    python verify_nonadditivity.py --image path/to/img.jpg --out out_dir
        [--target 207] [--grid 14 14] [--topk 2] [--sigma 50] [--device cpu]
        [--n-samples 1000]
"""
from __future__ import annotations

import argparse
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
    Returns (v, f_phi) where f_phi = f(Phi_S(x)) is the raw target probability.
    """
    keep = torch.zeros_like(cell_ids, dtype=torch.float32)  # (H,W)
    for c in keep_cells:
        keep = keep + (cell_ids == c).float()
    keep = keep.clamp(0, 1).view(1, 1, *cell_ids.shape)     # (1,1,H,W)
    comp = keep * x + (1 - keep) * b
    f_phi = float(F.softmax(model(comp), dim=1)[0, target])
    return f_phi - f0, f_phi


# --------------------------------------------------------------------------- #
# visualization
# --------------------------------------------------------------------------- #
def save_panel(out_path, x, b, cell_ids, top_cells, coefs, grid, target,
               target_name, results):
    """input | blur | LIME coef map | top-k reveal composite."""
    img = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
    blur = denormalize(b)[0].permute(1, 2, 0).cpu().numpy()

    coef_t = torch.tensor(coefs, dtype=torch.float32)
    coef_map = coef_t[cell_ids.cpu()].numpy()
    m = np.abs(coef_map).max() + 1e-12

    # composite revealing only top-k cells
    keep = torch.zeros_like(cell_ids, dtype=torch.float32)
    for c in top_cells:
        keep = keep + (cell_ids == c).float()
    keep = keep.clamp(0, 1).view(1, 1, *cell_ids.shape).to(x.device)
    comp = (keep * x + (1 - keep) * b)
    comp_img = denormalize(comp)[0].permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(blur); ax[1].set_title("blur reference b"); ax[1].axis("off")
    im = ax[2].imshow(coef_map, cmap="seismic", vmin=-m, vmax=m)
    ax[2].set_title(f"LIME coefficients ({grid[0]}x{grid[1]})"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].imshow(comp_img)
    ax[3].set_title(f"top-{len(top_cells)} reveal (rest=b)"); ax[3].axis("off")

    v_joint = results["v_joint"]
    v_sum = results["v_sum"]
    g = results["synergy"]
    fig.suptitle(
        f"class={target} ({target_name})   "
        f"v(joint)={v_joint:+.4f}   sum v(c)={v_sum:+.4f}   "
        f"g={g:+.4f}   "
        f"({'cooperation' if g > 0 else 'redundancy'})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
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
    ap.add_argument("--grid", type=int, nargs=2, default=[14, 14])
    ap.add_argument("--topk", type=int, default=2)
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

    # baseline f0 = f(b): the empty-reveal value used in v(R) = f(Phi_R) - f0
    with torch.no_grad():
        f0 = float(F.softmax(model(b), dim=1)[0, target])
        f_x = float(F.softmax(model(x), dim=1)[0, target])

    print(f"[verify] target={target} ({target_name})  f(x)={f_x:.4f}  f(b)={f0:.4f}")

    # ---- 1. run LIME to get per-cell coefficients ----
    lime = LIMEExplainer(
        model, target_class=target, device=device, class_names=class_names,
        grid=grid, n_samples=args.n_samples, sigma=args.sigma,
    )
    lime_res = lime.explain(x)
    # recover per-cell coefficients from the painted (H,W) map
    cell_ids = cell_id_map(H, W, grid).to(device)
    n_cells = grid[0] * grid[1]
    attr = torch.tensor(lime_res.attribution, device=device)  # (H,W)
    coefs = np.zeros(n_cells, dtype=np.float64)
    for c in range(n_cells):
        mask = (cell_ids == c)
        if mask.any():
            coefs[c] = float(attr[mask][0])  # cell is constant within its region

    # ---- 2. rank by coefficient, take top-k ----
    order = np.argsort(-coefs)
    top_cells = [int(c) for c in order[: args.topk]]
    print(f"[verify] top-{args.topk} cells by LIME coef: {top_cells}")

    # ---- 3. individual holistic values v(c) ----
    per_cell = {}
    v_sum = 0.0
    for c in top_cells:
        v_c, f_phi_c = reveal_value(model, x, b, cell_ids, {c}, target, f0)
        per_cell[c] = dict(v=v_c, f_phi=f_phi_c, lime_coef=float(coefs[c]))
        v_sum += v_c
        print(f"[verify]   cell {c:4d}: LIME w={coefs[c]:+.4f}  "
              f"v(c)={v_c:+.4f}  f(Phi)={f_phi_c:.4f}")

    # ---- 4. joint holistic value v(top-k together) ----
    v_joint, f_phi_joint = reveal_value(
        model, x, b, cell_ids, set(top_cells), target, f0
    )

    # ---- 5. synergy: v(joint) = sum v(c) + g ----
    g = v_joint - v_sum
    print(f"[verify] v(joint) = {v_joint:+.4f}   "
          f"sum_c v(c) = {v_sum:+.4f}   "
          f"g = v(joint)-sum v(c) = {g:+.4f}")
    print(f"[verify] additive prediction error |v(joint) - sum v(c)| = {abs(g):.4f}")
    if abs(g) < 1e-6:
        print("[verify] (additive within tolerance on this coalition)")
    else:
        kind = "COOPERATION (whole > parts)" if g > 0 else "REDUNDANCY (whole < parts)"
        print(f"[verify] NON-ADDITIVE: {kind} -> g must be a separate interaction term")

    results = dict(
        v_joint=v_joint, v_sum=v_sum, synergy=g,
        f_phi_joint=f_phi_joint, per_cell=per_cell,
    )

    # ---- panel + summary ----
    panel_path = os.path.join(args.out, "nonadditivity.png")
    save_panel(panel_path, x, b, cell_ids, top_cells, coefs, grid,
               target, target_name, results)

    lines = [
        f"image: {args.image}",
        f"target: {target} ({target_name})",
        f"grid: {grid[0]}x{grid[1]}   topk: {args.topk}   sigma: {args.sigma}",
        f"f(x)={f_x:.4f}   f(b)={f0:.4f}   total explainable = f(x)-f(b) = {f_x - f0:+.4f}",
        "",
        "per-cell (only that cell revealed, rest = blur b):",
    ]
    for c in top_cells:
        d = per_cell[c]
        lines.append(f"  cell {c}: LIME_w={d['lime_coef']:+.4f}  "
                     f"v(c)={d['v']:+.4f}  f(Phi)={d['f_phi']:.4f}")
    lines += [
        "",
        f"v(joint top-{args.topk})       = {v_joint:+.6f}",
        f"sum_c v(c)              = {v_sum:+.6f}",
        f"synergy g = v_joint - sum = {g:+.6f}",
        "",
        "identity:  v(A,B) = v(A) + v(B) + g(A,B)",
        f"  check:   {v_joint:+.6f} = {v_sum:+.6f} + {g:+.6f}  "
        f"(residual {v_joint - (v_sum + g):.2e})",
        "",
        "interpretation:",
        "  LIME predicts v(A,B) ~ w_A + w_B (purely additive, g forced to 0).",
        "  The measured g != 0 is exactly the group non-additivity term that",
        "  PyramidExplainer keeps as the node synergy Delta(R).",
    ]
    summary_path = os.path.join(args.out, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[verify] panel   -> {panel_path}")
    print(f"[verify] summary -> {summary_path}")


if __name__ == "__main__":
    main()