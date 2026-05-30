"""Regional insertion / deletion faithfulness scoring.

A drop-in extension of the pixel-level `insertion_deletion` in `evaluate.py`.

WHY REGIONAL
------------
The standard RISE insertion/deletion reveals/removes pixels one at a time in
attribution order. For a *pixel* method (IG) that is fine, but it has two
problems that bias the comparison against *region* methods (pyramid, LIME):

  1. Off-manifold ordering. Revealing the single highest-attribution pixels
     first produces a scattered, high-frequency composite -- exactly the
     off-manifold scatter the method note warns about. The curve then measures
     how well f tolerates OOD inputs, not how good the attribution is.
  2. Unit mismatch. A region method assigns one value to a whole superpixel.
     Ranking its pixels individually just shuffles ties; the per-pixel order
     inside a region is meaningless, so the fine-grained curve is noise.

REGIONAL FIX
------------
Tile the image into CELLS, each cell ~ `cell_frac` of the total features
(default 5% => ~20 cells). Score each cell by the AGGREGATE attribution mass
inside it (sum of |attr|). Then insert / delete WHOLE CELLS in importance order.
The composite stays coherent (block reveals, not pixel scatter) and the unit
matches how region methods actually attribute.

  - Regional insertion: start from blur b, paste back whole sharp cells most-
    important first. Early rise => a few coherent regions already suffice.
  - Regional deletion: start from sharp x, blur out whole cells most-important
    first. Early drop => those regions were necessary.

AUC is reported the same way (higher insertion-AUC / lower deletion-AUC =>
more faithful). Because every method is scored against the SAME cell grid and
the SAME blur reference, the comparison is fair across pixel and region methods.

PARAMS (all overridable)
------------------------
  regional   : bool  -- regional (True, default) vs per-pixel (False) scoring.
  cell_frac  : float -- fraction of total features per cell (default 0.05 = 5%).
                        Cell side ~= round(sqrt(cell_frac) * H). Number of cells
                        ~= 1 / cell_frac (so 5% => ~20 cells along the budget).
  steps      : int   -- in regional mode, ignored if it exceeds the cell count;
                        the natural resolution is "one step per cell".
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# cell partition
# --------------------------------------------------------------------------- #
def _cell_grid(H: int, W: int, cell_frac: float) -> tuple[np.ndarray, int]:
    """Tile (H,W) into square-ish cells of ~`cell_frac` of total features.

    Returns (cell_id_map, n_cells):
      cell_id_map -- (H,W) int array, each pixel labelled with its cell index.
      n_cells     -- number of distinct cells.

    Cell side length s is chosen so s*s ~= cell_frac * H * W, i.e.
    s = round(sqrt(cell_frac * H * W)). With cell_frac=0.05 on 224x224 this
    gives s ~= 50 px and ~ 20 cells, matching "5% of features per cell".
    """
    cell_frac = float(np.clip(cell_frac, 1e-4, 1.0))
    side = max(1, int(round(np.sqrt(cell_frac * H * W))))
    rows = int(np.ceil(H / side))
    cols = int(np.ceil(W / side))
    ys = np.arange(H) // side          # row block index per pixel
    xs = np.arange(W) // side          # col block index per pixel
    cell_id = (ys[:, None] * cols + xs[None, :]).astype(np.int64)  # (H,W)
    # relabel to a contiguous 0..n-1 (edge cells may be partial but still valid)
    uniq = np.unique(cell_id)
    remap = {old: new for new, old in enumerate(uniq)}
    cell_id = np.vectorize(remap.get)(cell_id).astype(np.int64)
    return cell_id, len(uniq)


def _cell_scores(attr: np.ndarray, cell_id: np.ndarray, n_cells: int) -> np.ndarray:
    """Aggregate |attribution| mass per cell. Returns (n_cells,) score vector."""
    flat_attr = np.abs(attr).reshape(-1)
    flat_cell = cell_id.reshape(-1)
    scores = np.zeros(n_cells, dtype=np.float64)
    np.add.at(scores, flat_cell, flat_attr)
    return scores


# --------------------------------------------------------------------------- #
# regional insertion / deletion
# --------------------------------------------------------------------------- #
@torch.no_grad()
def regional_insertion_deletion(
    model,
    x: torch.Tensor,
    attr: np.ndarray,
    target: int,
    b: torch.Tensor,
    cell_frac: float = 0.05,
    batch: int = 16,
):
    """Regional insertion/deletion curves: reveal/remove WHOLE CELLS in order.

    The budget axis is the fraction of *cells* revealed/removed, which (since
    cells are equal-area) equals the fraction of features revealed/removed.

    Returns the same dict shape as the pixel-level version so it slots straight
    into `_plot_curves` and `curves.json`:
        fractions, insertion, deletion, insertion_auc, deletion_auc
    plus regional metadata (n_cells, cell_frac, cell_side).
    """
    device = x.device
    _, C, H, W = x.shape
    n = H * W

    cell_id, n_cells = _cell_grid(H, W, cell_frac)
    scores = _cell_scores(attr, cell_id, n_cells)
    cell_order = np.argsort(-scores)  # most-important cell first

    cell_id_t = torch.as_tensor(cell_id.reshape(-1), device=device)

    x_flat = x.view(C, n)
    b_flat = b.view(C, n)

    # Precompute a boolean pixel-mask for "top-k cells" cumulatively.
    # fractions[k] = k cells revealed (k = 0..n_cells), so a curve of n_cells+1
    # points. We measure the fraction of *features* covered for the x-axis so
    # partial edge cells are reflected honestly.
    fractions = []
    keep_masks = []  # each (n,) bool over flattened pixels
    covered = torch.zeros(n, dtype=torch.bool, device=device)
    fractions.append(0.0)
    keep_masks.append(covered.clone())
    for k in range(n_cells):
        c = int(cell_order[k])
        covered = covered | (cell_id_t == c)
        keep_masks.append(covered.clone())
        fractions.append(float(covered.float().mean().item()))
    fractions = np.asarray(fractions)

    def _build(masks_subset, base_flat, fill_flat):
        imgs = []
        for m in masks_subset:
            comp = base_flat.clone()
            if m.any():
                comp[:, m] = fill_flat[:, m]
            imgs.append(comp.view(1, C, H, W))
        return torch.cat(imgs, dim=0)

    ins_probs = np.zeros(len(keep_masks))
    del_probs = np.zeros(len(keep_masks))

    # insertion: base=blur, fill=sharp (reveal top cells)
    # deletion : base=sharp, fill=blur (remove top cells)
    for base, fill, out in (
        (b_flat, x_flat, ins_probs),
        (x_flat, b_flat, del_probs),
    ):
        for start in range(0, len(keep_masks), batch):
            subset = keep_masks[start:start + batch]
            comp = _build(subset, base, fill)
            p = F.softmax(model(comp), dim=1)[:, target].cpu().numpy()
            out[start:start + len(subset)] = p

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    ins_auc = float(_trapz(ins_probs, fractions))
    del_auc = float(_trapz(del_probs, fractions))

    return {
        "fractions": fractions.tolist(),
        "insertion": ins_probs.tolist(),
        "deletion": del_probs.tolist(),
        "insertion_auc": ins_auc,
        "deletion_auc": del_auc,
        "regional": True,
        "n_cells": int(n_cells),
        "cell_frac": float(cell_frac),
        "cell_side": int(round(np.sqrt(cell_frac * H * W))),
    }


# --------------------------------------------------------------------------- #
# unified entry point: regional by default, falls back to per-pixel
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_insertion_deletion(
    model,
    x: torch.Tensor,
    attr: np.ndarray,
    target: int,
    b: torch.Tensor,
    regional: bool = True,        # REGIONAL BY DEFAULT
    cell_frac: float = 0.05,      # 5% of features per cell BY DEFAULT
    steps: int = 50,              # used only in per-pixel mode
    batch: int = 16,
):
    """Faithfulness curves. Regional (default) or per-pixel.

    regional=True  -> reveal/remove whole cells of `cell_frac` features each.
    regional=False -> per-pixel RISE ordering (the original behaviour).
    """
    if regional:
        return regional_insertion_deletion(
            model, x, attr, target, b, cell_frac=cell_frac, batch=batch
        )
    # ---- per-pixel fallback (original evaluate.py logic) ------------------ #
    device = x.device
    _, C, H, W = x.shape
    n = H * W
    order = np.argsort(-np.abs(attr).reshape(-1))
    fractions = np.linspace(0, 1, steps + 1)
    counts = (fractions * n).astype(int)
    x_flat = x.view(C, n)
    b_flat = b.view(C, n)

    def _build(keep_idx_lists, base_flat, fill_flat):
        imgs = []
        for idx in keep_idx_lists:
            comp = base_flat.clone()
            if len(idx) > 0:
                idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
                comp[:, idx_t] = fill_flat[:, idx_t]
            imgs.append(comp.view(1, C, H, W))
        return torch.cat(imgs, dim=0)

    ins_probs = np.zeros(len(counts))
    del_probs = np.zeros(len(counts))
    for base, fill, out in (
        (b_flat, x_flat, ins_probs),
        (x_flat, b_flat, del_probs),
    ):
        for start in range(0, len(counts), batch):
            chunk = counts[start:start + batch]
            keep_lists = [order[:k] for k in chunk]
            comp = _build(keep_lists, base, fill)
            p = F.softmax(model(comp), dim=1)[:, target].cpu().numpy()
            out[start:start + len(chunk)] = p

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        "fractions": fractions.tolist(),
        "insertion": ins_probs.tolist(),
        "deletion": del_probs.tolist(),
        "insertion_auc": float(_trapz(ins_probs, fractions)),
        "deletion_auc": float(_trapz(del_probs, fractions)),
        "regional": False,
    }