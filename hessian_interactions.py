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
    """Compare a method matrix to the model-actual matrix on shared off-diagonal."""
    a = _off_diag(I_method).astype(np.float64)
    g = _off_diag(I_actual).astype(np.float64)

    # Pearson r (guard against zero variance, e.g. a diagonal-only matrix)
    if a.std() < 1e-12 or g.std() < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(a, g)[0, 1])

    # sign agreement on entries where the actual is non-trivial
    sig = np.abs(g) > (1e-3 * (np.abs(g).max() or 1.0))
    if sig.sum() == 0:
        sign_agree = 0.0
    else:
        sign_agree = float((np.sign(a[sig]) == np.sign(g[sig])).mean())

    # off-diagonal mass recovery: how much of the actual non-additive mass the
    # method accounts for (ratio of L1 masses; 1.0 = same magnitude budget)
    mass_actual = float(np.abs(g).sum())
    mass_method = float(np.abs(a).sum())
    mass_ratio = float(mass_method / mass_actual) if mass_actual > 0 else 0.0

    return {
        "offdiag_pearson_r": r,
        "offdiag_sign_agreement": sign_agree,
        "offdiag_mass_ratio": mass_ratio,
        "offdiag_mass_actual": mass_actual,
        "offdiag_mass_method": mass_method,
    }


# --------------------------------------------------------------------------- #
# criterion B: delta-faithfulness for Hessian-IG (predict 2-cell reveals)
# --------------------------------------------------------------------------- #
def hessian_delta_faithfulness(
    res_hess, model, x, b, target, k: int, n_pairs: int = 24, seed: int = 0
):
    """Do Hessian off-diagonals improve prediction of measured 2-cell reveals?

    For random cell pairs (i,j): measure v(S) = f(reveal S) - f(b) for
    S in {i}, {j}, {i,j} by direct blur-completion reveal. Compare:
        additive prediction      v_hat_add = v(i) + v(j)
        interaction prediction   v_hat_int = v(i) + v(j) + Phi_ij
    against the measured v({i,j}). Lower interaction error => Hessian term real.
    """
    import torch
    import torch.nn.functional as F

    I = res_hess.extras["interaction_matrix"]
    _, _, H, W = x.shape
    slices = _cell_slices(H, W, k)
    delta = (x - b)

    def reveal_value(cell_ids):
        field = torch.zeros((1, 1, H, W), dtype=x.dtype, device=x.device)
        for cid in cell_ids:
            rs, cs = slices[cid]
            field[..., rs, cs] = 1.0
        X = b + field * delta
        with torch.no_grad():
            p = F.softmax(model(X), dim=1)[0, target].item()
        return float(p)

    with torch.no_grad():
        f_b = F.softmax(model(b), dim=1)[0, target].item()

    rng = np.random.default_rng(seed)
    K = len(slices)
    pairs = set()
    while len(pairs) < min(n_pairs, K * (K - 1) // 2):
        i, j = rng.integers(0, K, size=2)
        if i != j:
            pairs.add((min(i, j), max(i, j)))

    add_err, int_err = [], []
    for (i, j) in pairs:
        vi = reveal_value([i]) - f_b
        vj = reveal_value([j]) - f_b
        vij = reveal_value([i, j]) - f_b
        v_add = vi + vj
        v_int = vi + vj + I[i, j]
        add_err.append(abs(vij - v_add))
        int_err.append(abs(vij - v_int))

    add_err = np.array(add_err)
    int_err = np.array(int_err)
    return {
        "n_pairs": int(len(pairs)),
        "mean_additive_error": float(add_err.mean()),
        "mean_interaction_error": float(int_err.mean()),
        "interaction_helps_fraction": float((int_err < add_err).mean()),
        "median_error_reduction": float(np.median(add_err - int_err)),
    }


# --------------------------------------------------------------------------- #
# combined verdict
# --------------------------------------------------------------------------- #
def verdict(agree_hess: dict, agree_pyr: dict,
            dfaith_hess: dict, dfaith_pyr: dict) -> dict:
    """Pick a winner per criterion + an overall call, with reasons."""
    reasons = []

    # criterion A: prefer higher |r|, then higher sign agreement
    a_h = (abs(agree_hess["offdiag_pearson_r"]),
           agree_hess["offdiag_sign_agreement"])
    a_p = (abs(agree_pyr["offdiag_pearson_r"]),
           agree_pyr["offdiag_sign_agreement"])
    winner_A = "hessian_ig" if a_h >= a_p else "pyramid"
    reasons.append(
        f"agreement: hessian r={agree_hess['offdiag_pearson_r']:+.3f} "
        f"sign={agree_hess['offdiag_sign_agreement']:.2f} vs "
        f"pyramid r={agree_pyr['offdiag_pearson_r']:+.3f} "
        f"sign={agree_pyr['offdiag_sign_agreement']:.2f} -> {winner_A}"
    )

    # criterion B: prefer larger relative error reduction (add - int)/add
    def red(d):
        a = d.get("mean_additive_error", 0.0)
        i = d.get("mean_interaction_error", 0.0)
        return (a - i) / a if a > 1e-12 else 0.0
    r_h, r_p = red(dfaith_hess), red(dfaith_pyr)
    winner_B = "hessian_ig" if r_h >= r_p else "pyramid"
    reasons.append(
        f"delta-faithfulness error reduction: hessian {r_h:+.2%} vs "
        f"pyramid {r_p:+.2%} -> {winner_B}"
    )

    overall = winner_A if winner_A == winner_B else "split (see criteria)"
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