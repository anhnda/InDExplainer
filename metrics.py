"""Metrics for pyramid vs LIME vs regional-IG, against the xai_suff harness.

THREE THINGS, AT TWO ORDERS
---------------------------
FIRST-ORDER faithfulness (insertion/deletion via regional_eval): fair on a
  shared cell grid, and BECAUSE it is fair, LIME is a strong baseline. Pyramid's
  additive map (sum_leaf_v, ~1% leftover) ties or loses here. Report honestly.
  Use del_fill="mean" (non-circular) for the headline number; del_fill="blur"
  shares pyramid's Phi and a win there is partly circular (the harness docstring
  says so). Insertion/deletion is FIRST-ORDER: it sums marginal region effects
  and has no slot for "A and B only matter together", so it cannot reward an
  interaction method for finding interaction. Do not rank by the Delta map here.

SECOND-ORDER interaction (NAI + pairwise matrix + Delta-faithfulness): the
  quantity a linear surrogate sets to zero. LIME predicts a region as the sum of
  its leaf weights -- implied interaction = identity, zero off-diagonal -- so it
  CANNOT represent cooperation regardless of fit. Pyramid measures it directly.
  This is the defensible distinction: not "pyramid beats LIME at LIME's job",
  but "pyramid measures an order LIME's model class omits".

DELTA-FAITHFULNESS (delta_faithfulness): the right way to use Delta. Not by
  stuffing the Delta map into first-order insertion/deletion (wrong quantity,
  and circular through Phi), but by a COALITION-REVEAL test: for node R, compare
  the additive prediction sum_c v(c) against the interaction prediction v(R), and
  validate against a direct forward pass on R's region. If the model's true joint
  reveal matches v(R) and not sum_c v(c), the synergy is faithful. The additive
  model's error is exactly |Delta(R)| -- the gap LIME is stuck with.

Consumes the same package as evaluate.py:
  from regional_eval import score_insertion_deletion
  from xai_suff.explainers import blur_reference
"""
from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import torch
import torch.nn.functional as F


# =========================================================================== #
#  SECOND-ORDER, tree-restricted: NAI  (pyramid headline; LIME has no analog)
# =========================================================================== #
def non_additivity_index(result) -> float:
    """NAI = sum|Delta| / (sum|leaf v| + sum|Delta|) in [0,1].

    Fraction of attributed signal that is non-additive along pyramid's tree.
    Proposition: an additive model predicts region R as sum of leaf weights; its
    irreducible error on R is bounded below by |Delta(R)|. Mean NAI is a lower
    bound on the additive class's error on the data. Church: 0.989.
    """
    tree = result.extras["tree"]
    sum_leaf = sum(abs(n["v"]) for n in tree if n["is_leaf"])
    sum_delta = sum(abs(n["delta"]) for n in tree if not n["is_leaf"])
    denom = sum_leaf + sum_delta
    return float(sum_delta / denom) if denom > 0 else 0.0


def additive_error_lower_bound(result) -> float:
    """sum|Delta|: absolute signal no additive model can represent (this image,
    shared Phi). Un-normalized companion to NAI."""
    tree = result.extras["tree"]
    return float(sum(abs(n["delta"]) for n in tree if not n["is_leaf"]))


# =========================================================================== #
#  DELTA-FAITHFULNESS: the correct use of Delta (coalition-reveal, validated)
# =========================================================================== #
@torch.no_grad()
def delta_faithfulness(
    result,
    model=None,
    x: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    target: Optional[int] = None,
    validate: bool = True,
):
    """Per-node: additive-prediction error vs interaction-prediction error.

    For each internal node R with children c:
      additive prediction  v_add(R) = sum_c v(c)         (what LIME-type models give)
      interaction pred.     v_int(R) = sum_c v(c) + Delta(R) = v(R)  (pyramid, exact)
      additive error        |v(R) - v_add(R)| = |Delta(R)|

    If validate=True and (model,x,b,target) are given, also RE-MEASURE v(R) by a
    fresh forward pass revealing R's region under the same Phi, and report how
    well each prediction matches the freshly measured truth. This is the part
    that makes it faithfulness and not bookkeeping: it confirms the model's
    ACTUAL joint reveal matches the interaction prediction, not the additive one.

    Returns dict with mean additive error, mean interaction error, and (if
    validated) mean |measured - additive| and mean |measured - interaction|.
    Lower interaction error than additive error = the synergy is real and the
    additive model is missing it. Validation needs result.extras['leaf_masks'].
    """
    tree = result.extras["tree"]
    by_id = {n["id"]: n for n in tree}
    internals = [n for n in tree if not n["is_leaf"]]

    add_err, int_err = [], []
    for n in internals:
        v_add = sum(by_id[c]["v"] for c in n["child_ids"])
        v_int = v_add + n["delta"]
        add_err.append(abs(n["v"] - v_add))   # == |Delta|
        int_err.append(abs(n["v"] - v_int))   # ~ 0 by identity

    out = {
        "mean_additive_error": float(np.mean(add_err)) if add_err else 0.0,
        "mean_interaction_error": float(np.mean(int_err)) if int_err else 0.0,
        "n_internal": len(internals),
    }

    if validate and model is not None and x is not None and b is not None \
            and target is not None and "leaf_masks" in result.extras:
        device = x.device
        _, C, H, W = x.shape
        leaf_masks = result.extras["leaf_masks"]
        cache: dict = {}

        def node_mask(nid):
            if nid in cache:
                return cache[nid]
            nn = by_id[nid]
            if nn["is_leaf"]:
                m = leaf_masks[nid]
            else:
                m = None
                for c in nn["child_ids"]:
                    cm = node_mask(c)
                    m = cm.copy() if m is None else (m | cm)
            cache[nid] = m
            return m

        def phi_prob(mask_bool):
            m = torch.as_tensor(mask_bool, dtype=x.dtype, device=device).view(1, 1, H, W)
            comp = m * x + (1.0 - m) * b
            return float(F.softmax(model(comp), dim=1)[:, target].item())

        x0 = phi_prob(np.zeros((H, W), dtype=bool))
        meas_vs_add, meas_vs_int = [], []
        # sample a subset of internal nodes if there are many (cost control)
        sample = internals if len(internals) <= 40 else \
            sorted(internals, key=lambda n: abs(n["delta"]), reverse=True)[:40]
        for n in sample:
            v_meas = phi_prob(node_mask(n["id"])) - x0
            v_add = sum(by_id[c]["v"] for c in n["child_ids"])
            v_int = n["v"]
            meas_vs_add.append(abs(v_meas - v_add))
            meas_vs_int.append(abs(v_meas - v_int))
        out["mean_measured_vs_additive"] = float(np.mean(meas_vs_add))
        out["mean_measured_vs_interaction"] = float(np.mean(meas_vs_int))
        out["n_validated"] = len(sample)

    return out


# =========================================================================== #
#  SECOND-ORDER, exact: pairwise interaction matrix over a shared grid
# =========================================================================== #
def _grid_cells(H: int, W: int, k: int) -> list[np.ndarray]:
    cells = []
    ys = np.linspace(0, H, k + 1).astype(int)
    xs = np.linspace(0, W, k + 1).astype(int)
    for i in range(k):
        for j in range(k):
            m = np.zeros((H, W), dtype=bool)
            m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True
            cells.append(m)
    return cells


@torch.no_grad()
def pairwise_interaction_matrix(model, x, b, target, k: int = 4):
    """Harsanyi/Shapley pairwise interaction over a k x k grid, under blur-Phi.

      v(S) = f(Phi_S(x)) - f(x0),  x0 = full blur b   (matches pyramid's v)
      I[i,i] = v({i})                              (the part LIME fits)
      I[i,j] = v({i,j}) - v({i}) - v({j})          (cooperation, LIME omits)

    LIME's implied matrix is diagonal-only by construction. Returns the matrix
    and off_diag_ratio = sum_{i!=j}|I| / sum_all|I| -- the cooperation share.
    """
    device = x.device
    _, C, H, W = x.shape
    cells = _grid_cells(H, W, k)
    n = len(cells)

    def phi_prob(mask_bool):
        m = torch.as_tensor(mask_bool, dtype=x.dtype, device=device).view(1, 1, H, W)
        comp = m * x + (1.0 - m) * b
        return float(F.softmax(model(comp), dim=1)[:, target].item())

    x0 = phi_prob(np.zeros((H, W), dtype=bool))
    v_single = np.array([phi_prob(c) - x0 for c in cells])
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = v_single[i]
    for i in range(n):
        for j in range(i + 1, n):
            v_ij = phi_prob(cells[i] | cells[j]) - x0
            inter = v_ij - v_single[i] - v_single[j]
            I[i, j] = I[j, i] = inter
    diag = float(np.sum(np.abs(np.diag(I))))
    off = float(np.sum(np.abs(I)) - diag)
    total = diag + off
    return {
        "I": I,
        "off_diag_ratio": float(off / total) if total > 0 else 0.0,
        "diag_mass": diag, "off_diag_mass": off, "k": k, "n_cells": n,
        "convention": "reveal-against-blur (empty coalition), matches v(R)",
        "note": "LIME's implied matrix is diagonal-only by construction.",
    }


# =========================================================================== #
#  FIRST-ORDER faithfulness: thin wrapper over the real harness
# =========================================================================== #
def faithfulness(score_fn, model, x, attr, target, b,
                 cell_frac=0.05, del_fill="mean", steps=50):
    """regional_eval.score_insertion_deletion, defaulting to the NON-CIRCULAR
    mean-fill. Pass score_fn=regional_eval.score_insertion_deletion to avoid a
    circular import. Expect pyramid (additive map) ~ LIME here; that is fine and
    honest -- the pyramid story is second-order, measured above."""
    return score_fn(model, x, attr, target, b,
                    regional=True, cell_frac=cell_frac,
                    del_fill=del_fill, steps=steps)


def query_cost(result) -> int:
    """Forward passes pyramid spent (one per tree-node value query)."""
    return int(result.extras.get("n_value_queries", -1))