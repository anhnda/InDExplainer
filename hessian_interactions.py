"""Verdict utilities: is Hessian-IG or pyramid the better second-order explainer?

The only honest way to say "better" is against a ground truth. The ground truth
here is the *model-actual* pairwise interaction on a k x k grid, built by direct
reveal (no surrogate, no gradient): ``M.pairwise_interaction_matrix`` returns it
as ``I`` (K x K, K = k*k). Both contenders are projected onto that same grid and
scored against it cell-for-cell.

Two complementary criteria (reported side by side, per your call):

A. AGREEMENT vs model-actual matrix
   - Pearson r over off-diagonal entries (does the *shape* of interaction match?)
   - sign-agreement fraction over off-diagonal (cooperation vs redundancy called
     correctly?)
   - off-diagonal mass recovery ratio (does it capture the right *amount* of
     non-additivity?)

B. DELTA-FAITHFULNESS (predict held-out region reveals)
   - reuse ``M.delta_faithfulness`` for pyramid (tree-native).
   - for Hessian-IG, predict the measured 2-cell reveal value v({i,j}) from the
     additive part v(i)+v(j) plus the Hessian off-diagonal Phi_{ij}, and compare
     the additive vs interaction-augmented prediction error against direct
     reveals. Lower interaction error than additive error => the Hessian term is
     carrying real signal.

Projection of pyramid onto the grid
-----------------------------------
Pyramid's interaction is tree-relative (Delta per node), not pairwise on a fixed
grid. To compare, we render pyramid's synergy to a per-pixel density map (via the
leaf masks + node Deltas) and *pool* it onto the k x k grid, giving a diagonal-
dominant matrix P with P_ii = pooled synergy mass in cell i. This is the fairest
projection: pyramid does not claim a specific (i,j) pair, so its mass lands on the
diagonal, exactly as LIME's does in the existing interaction_matrix.png panel.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #
def _cell_slices(H: int, W: int, k: int):
    ys = np.linspace(0, H, k + 1).round().astype(int)
    xs = np.linspace(0, W, k + 1).round().astype(int)
    return [
        (slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1]))
        for r in range(k) for c in range(k)
    ]


def _off_diag(M: np.ndarray) -> np.ndarray:
    """Flattened off-diagonal entries of a square matrix."""
    mask = ~np.eye(M.shape[0], dtype=bool)
    return M[mask]


# --------------------------------------------------------------------------- #
# project pyramid synergy -> k x k diagonal matrix
# --------------------------------------------------------------------------- #
def pyramid_grid_matrix(res, leaf_masks: dict, k: int) -> np.ndarray:
    """Pool pyramid's per-pixel synergy density onto a k x k diagonal matrix.

    Pyramid synergy is tree-relative; we render Delta to a per-pixel density and
    sum within each cell. Off-diagonals are zero by construction (pyramid asserts
    no specific cell-pair), mirroring how LIME appears on this panel.
    """
    tree = res.extras["tree"]
    by_id = {n["id"]: n for n in tree}
    # accumulate each internal node's Delta as a density over its pixels,
    # distributed to leaves under it (same convention as delta_localization_map)
    # Here we approximate node->pixels via union of descendant leaf masks.
    H, W = next(iter(leaf_masks.values())).shape
    dens = np.zeros((H, W), dtype=np.float64)

    # map node id -> set of leaf ids under it
    def leaves_under(nid):
        node = by_id[nid]
        if node["is_leaf"]:
            return [nid]
        out = []
        for c in node["child_ids"]:
            out.extend(leaves_under(c))
        return out

    for node in tree:
        if node["is_leaf"] or node["delta"] == 0.0:
            continue
        leaf_ids = leaves_under(node["id"])
        area = sum(int(leaf_masks[lid].sum()) for lid in leaf_ids) or 1
        share = node["delta"] / area
        for lid in leaf_ids:
            dens[leaf_masks[lid]] += share

    slices = _cell_slices(H, W, k)
    P = np.zeros((k * k, k * k), dtype=np.float64)
    for idx, (rs, cs) in enumerate(slices):
        P[idx, idx] = float(dens[rs, cs].sum())
    return P


# --------------------------------------------------------------------------- #
# criterion A: agreement vs model-actual matrix
# --------------------------------------------------------------------------- #
def agreement_scores(I_method: np.ndarray, I_actual: np.ndarray) -> dict:
    """Compare a method matrix to the model-actual matrix on shared off-diagonal.

    Returns an `applicable` flag: if the method has essentially no off-diagonal
    mass (e.g. pyramid, diagonal by construction), off-diagonal correlation is
    undefined, not zero -- the method simply does not make pairwise claims, so it
    must NOT be scored as if it tried and failed. Callers should treat
    applicable=False as "N/A", never as a loss or a win.
    """
    a = _off_diag(I_method).astype(np.float64)
    g = _off_diag(I_actual).astype(np.float64)

    mass_actual = float(np.abs(g).sum())
    mass_method = float(np.abs(a).sum())
    # method makes no off-diagonal claim -> correlation is N/A, not 0
    applicable = mass_method > 1e-9 * (mass_actual + 1e-12)

    if not applicable or a.std() < 1e-12 or g.std() < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, g)[0, 1])

    sig = np.abs(g) > (1e-3 * (np.abs(g).max() or 1.0))
    if not applicable or sig.sum() == 0:
        sign_agree = float("nan")
    else:
        sign_agree = float((np.sign(a[sig]) == np.sign(g[sig])).mean())

    mass_ratio = float(mass_method / mass_actual) if mass_actual > 0 else 0.0

    return {
        "applicable": bool(applicable),
        "offdiag_pearson_r": r,
        "offdiag_sign_agreement": sign_agree,
        "offdiag_mass_ratio": mass_ratio,   # |1.0| = right magnitude budget
        "offdiag_mass_actual": mass_actual,
        "offdiag_mass_method": mass_method,
    }


# --------------------------------------------------------------------------- #
# criterion B: delta-faithfulness for Hessian-IG (predict 2-cell reveals)
# --------------------------------------------------------------------------- #
def hessian_delta_faithfulness(
    res_hess, res_pyr, model, x, b, target, k: int,
    n_regions: int = 40, seed: int = 0,
):
    """Do Hessian terms improve prediction of measured REGION reveals?

    Earlier this probed single k=14 cells, where v(cell) ~ 0 (a single 1/196 tile
    barely moves the logit), so additive error was ~0 by triviality and the test
    was uninformative. Fix per request: reuse pyramid's tree nodes as the reveal
    regions -- the same regions pyramid's own delta-faithfulness validates -- so
    both methods are tested on identical, non-trivial coalitions.

    For each internal node R covering cell-set C(R):
      measured     v(R)      = f(reveal R) - f(b)              [direct forward]
      additive     v_add(R)  = sum_{c in C(R)} Gamma_cc        [Hessian main effects]
      interaction  v_int(R)  = sum_{c} Gamma_cc
                               + sum_{c<c' in C(R)} 2*Gamma_cc' [+ pairwise terms]
    The factor 2 is because interaction-completeness counts each unordered pair
    twice (Gamma is symmetric; sum_i sum_j includes (i,j) and (j,i)).

    Lower interaction error than additive error => the Hessian off-diagonals carry
    real, faithful signal at region scale. Needs res_pyr.extras['leaf_masks'].
    """
    import torch
    import torch.nn.functional as F

    if "leaf_masks" not in res_pyr.extras:
        return {"skipped": "pyramid extras['leaf_masks'] missing"}

    I = res_hess.extras["interaction_matrix"]          # (K,K) Gamma
    leaf_masks = res_pyr.extras["leaf_masks"]
    tree = res_pyr.extras["tree"]
    by_id = {n["id"]: n for n in tree}
    _, _, H, W = x.shape
    slices = _cell_slices(H, W, k)
    K = len(slices)
    delta = (x - b)

    # map each grid cell to a (H,W) boolean; precompute cell centroids? no --
    # assign a node's pixels to cells by majority overlap.
    cell_of_pixel = -np.ones((H, W), dtype=np.int64)
    for idx, (rs, cs) in enumerate(slices):
        cell_of_pixel[rs, cs] = idx

    # leaves under a node (cached)
    cache = {}
    def leaves_under(nid):
        if nid in cache:
            return cache[nid]
        n = by_id[nid]
        s = [nid] if n["is_leaf"] else []
        for c in n["child_ids"]:
            s += leaves_under(c)
        cache[nid] = s
        return s

    def node_mask(nid):
        m = np.zeros((H, W), dtype=bool)
        for lid in leaves_under(nid):
            m |= leaf_masks[lid]
        return m

    def node_cells(mask):
        # cells whose pixels are (mostly) inside the node's mask
        cells = []
        for idx, (rs, cs) in enumerate(slices):
            tile = mask[rs, cs]
            if tile.mean() > 0.5:          # majority of the cell is in the region
                cells.append(idx)
        return cells

    def reveal_value(mask):
        m = torch.as_tensor(mask, dtype=x.dtype, device=x.device).view(1, 1, H, W)
        X = b + m * delta
        with torch.no_grad():
            p = F.softmax(model(X), dim=1)[0, target].item()
        return float(p)

    with torch.no_grad():
        f_b = F.softmax(model(b), dim=1)[0, target].item()

    internals = [n for n in tree if not n["is_leaf"]]
    # prefer larger-|delta| nodes (more interaction to test), cap for cost
    sample = sorted(internals, key=lambda n: abs(n["delta"]), reverse=True)
    sample = sample[:min(n_regions, len(sample))]

    add_err, int_err = [], []
    for n in sample:
        mask = node_mask(n["id"])
        cells = node_cells(mask)
        if len(cells) < 2:
            continue
        v_meas = reveal_value(mask) - f_b
        v_add = float(sum(I[c, c] for c in cells))
        pair_sum = 0.0
        for a_i in range(len(cells)):
            for b_i in range(a_i + 1, len(cells)):
                pair_sum += 2.0 * I[cells[a_i], cells[b_i]]
        v_int = v_add + pair_sum
        add_err.append(abs(v_meas - v_add))
        int_err.append(abs(v_meas - v_int))

    if not add_err:
        return {"skipped": "no multi-cell regions found", "n_regions": 0}

    add_err = np.array(add_err)
    int_err = np.array(int_err)
    return {
        "n_regions": int(len(add_err)),
        "mean_additive_error": float(add_err.mean()),
        "mean_interaction_error": float(int_err.mean()),
        "interaction_helps_fraction": float((int_err < add_err).mean()),
        "median_error_reduction": float(np.median(add_err - int_err)),
    }


# --------------------------------------------------------------------------- #
# combined verdict
# --------------------------------------------------------------------------- #
def verdict(agree_hess: dict, agree_pyr: dict,
            dfaith_hess: dict, dfaith_pyr: dict,
            completeness_residual: float, completeness_rhs: float,
            tol_frac: float = 0.10) -> dict:
    """Adjudicate, but refuse to crown a winner on unreliable numbers.

    GATE 0 -- Hessian validity. If interaction-completeness fails (residual not
    small vs |f(x)-f(b)|), the integrated Hessian is under-resolved or the
    smoothing is wrong; its matrix is not trustworthy and NO agreement/faithful
    comparison is meaningful. We say so instead of comparing noise.
    """
    reasons = []

    rhs_scale = abs(completeness_rhs) + 1e-9
    completeness_ok = abs(completeness_residual) <= tol_frac * rhs_scale
    reasons.append(
        f"interaction-completeness: residual={completeness_residual:+.4f} "
        f"vs |f(x)-f(b)|={rhs_scale:.4f} "
        f"({'OK' if completeness_ok else 'FAILED -> Hessian unreliable'})"
    )
    if not completeness_ok:
        return {
            "winner_agreement": "unreliable (completeness failed)",
            "winner_delta_faithfulness": "see below",
            "overall": "Hessian-IG not validated; increase hess_steps or lower "
                       "softplus_beta before comparing",
            "reasons": reasons,
        }

    # criterion A: agreement vs model-actual. Only meaningful where applicable.
    if not agree_hess.get("applicable", False):
        winner_A = "neither (no off-diagonal claims)"
        reasons.append("agreement: hessian makes no off-diagonal claim -> N/A")
    elif not agree_pyr.get("applicable", False):
        # hessian makes pairwise claims, pyramid (by construction) does not.
        # Score hessian on its own merits: does it POSITIVELY track the actual?
        r = agree_hess["offdiag_pearson_r"]
        sign = agree_hess["offdiag_sign_agreement"]
        mr = agree_hess["offdiag_mass_ratio"]
        good = (r > 0.2) and (sign > 0.6) and (0.3 < mr < 3.0)
        winner_A = "hessian_ig" if good else "neither (hessian tracks actual poorly)"
        reasons.append(
            f"agreement: pyramid N/A (diagonal by construction); hessian "
            f"r={r:+.3f} sign={sign:.2f} mass_ratio={mr:.2f} -> "
            f"{'tracks actual' if good else 'does NOT track actual'}"
        )
    else:
        a_h = (agree_hess["offdiag_pearson_r"], agree_hess["offdiag_sign_agreement"])
        a_p = (agree_pyr["offdiag_pearson_r"], agree_pyr["offdiag_sign_agreement"])
        winner_A = "hessian_ig" if a_h >= a_p else "pyramid"
        reasons.append(
            f"agreement: hessian r={a_h[0]:+.3f}/sign={a_h[1]:.2f} vs "
            f"pyramid r={a_p[0]:+.3f}/sign={a_p[1]:.2f} -> {winner_A}"
        )

    # criterion B: relative error reduction, guarded against tiny baselines.
    def red(d):
        if d.get("skipped"):
            return None
        a = d.get("mean_additive_error", 0.0)
        i = d.get("mean_interaction_error", 0.0)
        if a < 1e-6:               # additive already exact -> reduction undefined
            return None
        return (a - i) / a
    r_h, r_p = red(dfaith_hess), red(dfaith_pyr)

    if r_h is None and r_p is None:
        winner_B = "neither (additive baseline already exact / no regions)"
        reasons.append("delta-faithfulness: undefined (additive error ~0) -> N/A")
    elif r_h is None:
        winner_B = "pyramid"
        reasons.append(f"delta-faithfulness: hessian undefined; pyramid {r_p:+.2%}")
    elif r_p is None:
        winner_B = "hessian_ig" if r_h > 0 else "neither"
        reasons.append(f"delta-faithfulness: pyramid undefined; hessian {r_h:+.2%}")
    else:
        winner_B = "hessian_ig" if r_h >= r_p else "pyramid"
        reasons.append(
            f"delta-faithfulness reduction: hessian {r_h:+.2%} vs "
            f"pyramid {r_p:+.2%} -> {winner_B}"
        )

    # overall: only a clean winner if both real criteria agree on a real method
    real = {"hessian_ig", "pyramid"}
    if winner_A in real and winner_A == winner_B:
        overall = winner_A
    elif winner_A in real and winner_B not in real:
        overall = f"{winner_A} (on agreement; faithfulness inconclusive)"
    elif winner_B in real and winner_A not in real:
        overall = f"{winner_B} (on faithfulness; agreement inconclusive)"
    else:
        overall = "inconclusive / different estimands (see criteria)"

    return {"winner_agreement": winner_A,
            "winner_delta_faithfulness": winner_B,
            "overall": overall,
            "reasons": reasons}


# --------------------------------------------------------------------------- #
# 3-panel plot: actual | hessian-IG | pyramid-projected
# --------------------------------------------------------------------------- #
def plot_three_matrices(I_actual, I_hess, P_pyr, k, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = [("model actual\n(direct reveal)", I_actual),
            ("Hessian-IG\n(integrated mixed partials)", I_hess),
            ("pyramid\n(synergy pooled -> diagonal)", P_pyr)]
    lim = max(np.abs(M).max() or 1.0 for _, M in mats)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    im = None
    for a, (title, M) in zip(ax, mats):
        im = a.imshow(M, cmap="bwr", vmin=-lim, vmax=lim)
        a.set_title(title)
        a.set_xlabel(f"cell (of {k*k})")
        a.set_ylabel("cell")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Second-order interaction under blur-Phi: contenders vs ground truth")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)