"""Show the interaction-driven part of a PyramidExplainer result.

The per-pixel `attribution` map is only the *additive* part (leaf v's).
The interesting interaction signal is the per-node synergy

    Delta(R) = v(R) - sum_j v(child_j)

which lives in result.extras["tree"] but is never rendered. This module:

  1) ranks internal nodes by |Delta|  (where does interaction live?)
  2) localizes Delta on the image     (where on the church?)
  3) checks the telescoping identity   (are the books balanced?)

USAGE
-----
    from pyramid_interactions import (
        summarize_synergy, delta_localization_map, plot_interactions,
        check_identity,
    )

    res = PyramidExplainer(model, ...).explain(x)
    print(summarize_synergy(res, k=10))
    check_identity(res)
    plot_interactions(res, img01)   # img01 = denormalize(x)[0].permute(1,2,0).cpu().numpy()

NOTE ON MASKS
-------------
The default serialize() in pyramidexplainer.py does NOT store node masks,
only child_ids. delta_localization_map() therefore rebuilds each node's mask
by unioning its descendant leaves' masks -- but leaf masks aren't serialized
either. Two options:

  (A) RECOMMENDED: add masks to the tree at explain() time (see the
      `serialize` patch in the chat). Then leaf_masks is read directly.
  (B) If you still have `labels` (the SLIC leaf label map) and the tree's
      leaf ids correspond to label values, pass leaf_label_map=labels and
      this module reconstructs leaf masks from it.

If neither is available, ranking/identity (steps 1 & 3) still work fully;
only the spatial map (step 2) needs masks.
"""
from __future__ import annotations

from typing import Optional
import numpy as np


# --------------------------------------------------------------------------- #
# 1. rank internal nodes by synergy
# --------------------------------------------------------------------------- #
def summarize_synergy(result, k: int = 10) -> str:
    """Return a human-readable table of the top-k internal nodes by |Delta|.

    Delta > 0  -> cooperation (whole > sum of parts): the region matters
                  more revealed together than its pieces do alone.
    Delta < 0  -> redundancy  (whole < sum of parts): pieces double-count;
                  revealing them together adds less than separately.
    """
    tree = result.extras["tree"]
    internal = [n for n in tree if not n["is_leaf"]]
    internal.sort(key=lambda n: abs(n["delta"]), reverse=True)

    total_abs = sum(abs(n["delta"]) for n in internal) or 1.0
    sum_leaf_v = result.extras["sum_leaf_v"]
    sum_delta = result.extras["sum_delta"]

    lines = []
    lines.append("Interaction summary (PyramidExplainer)")
    lines.append("-" * 60)
    lines.append(f"  sum of leaf v (additive part) : {sum_leaf_v:+.4f}")
    lines.append(f"  sum of Delta  (interaction)   : {sum_delta:+.4f}")
    frac = abs(sum_delta) / (abs(sum_leaf_v) + abs(sum_delta) or 1.0)
    lines.append(f"  interaction share |dD|/(|v|+|dD|) : {frac:6.1%}")
    lines.append("")
    lines.append(f"  Top {k} internal nodes by |Delta|:")
    lines.append(f"  {'id':>5} {'area':>7} {'v':>9} {'Delta':>9} {'kind':>11} {'%|dD|':>7}")
    for n in internal[:k]:
        kind = "cooperative" if n["delta"] > 0 else "redundant"
        share = abs(n["delta"]) / total_abs
        lines.append(
            f"  {n['id']:>5} {n['area']:>7} {n['v']:>+9.4f} "
            f"{n['delta']:>+9.4f} {kind:>11} {share:>6.1%}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. localize Delta on the image
# --------------------------------------------------------------------------- #
def _leaf_masks_from_labels(tree, leaf_label_map: np.ndarray) -> dict[int, np.ndarray]:
    """Build {leaf_id: bool mask} assuming leaf ids == label values.

    PyramidExplainer assigns leaf ids 0..L-1 in the order np.unique(labels)
    returns, which is sorted -- so leaf id i corresponds to the i-th unique
    label. We map accordingly rather than assuming id == raw label value.
    """
    leaf_ids = sorted(n["id"] for n in tree if n["is_leaf"])
    uniq = np.unique(leaf_label_map)
    if len(leaf_ids) != len(uniq):
        raise ValueError(
            f"{len(leaf_ids)} leaves in tree but {len(uniq)} labels in map; "
            "ids and labels are out of sync -- use option (A) instead."
        )
    return {lid: (leaf_label_map == lab) for lid, lab in zip(leaf_ids, uniq)}


def _node_mask(node_id, tree_by_id, leaf_masks, _cache):
    """Union of descendant-leaf masks for any node (memoized)."""
    if node_id in _cache:
        return _cache[node_id]
    node = tree_by_id[node_id]
    if node["is_leaf"]:
        m = leaf_masks[node_id]
    else:
        m = None
        for cid in node["child_ids"]:
            cm = _node_mask(cid, tree_by_id, leaf_masks, _cache)
            m = cm.copy() if m is None else (m | cm)
    _cache[node_id] = m
    return m


def delta_localization_map(
    result,
    leaf_label_map: Optional[np.ndarray] = None,
    leaf_masks: Optional[dict] = None,
    signed: bool = True,
    density: bool = True,
) -> np.ndarray:
    """(H,W) map painting each internal node's Delta onto its region footprint.

    For overlapping ancestors, deeper (smaller-area) nodes are painted last so
    the most local interaction wins a pixel. With density=True each node's
    Delta is spread over its area (comparable to the additive density map);
    with density=False the raw Delta is painted.

    Provide EITHER leaf_masks={leaf_id: bool (H,W)} (option A) OR
    leaf_label_map=labels (option B).
    """
    tree = result.extras["tree"]
    tree_by_id = {n["id"]: n for n in tree}

    if leaf_masks is None:
        if leaf_label_map is None:
            raise ValueError("Provide leaf_masks or leaf_label_map (see module docstring).")
        leaf_masks = _leaf_masks_from_labels(tree, leaf_label_map)

    any_mask = next(iter(leaf_masks.values()))
    H, W = any_mask.shape
    out = np.zeros((H, W), dtype=np.float64)

    internal = [n for n in tree if not n["is_leaf"]]
    # paint large regions first, small (deep) regions last
    internal.sort(key=lambda n: n["area"], reverse=True)

    cache: dict = {}
    for n in internal:
        m = _node_mask(n["id"], tree_by_id, leaf_masks, cache)
        val = n["delta"]
        if density:
            val = val / max(n["area"], 1)
        out[m] = val if signed else abs(val)
    return out


# --------------------------------------------------------------------------- #
# 3. identity / bookkeeping check
# --------------------------------------------------------------------------- #
def check_identity(result, tol: float = 1e-4, verbose: bool = True) -> bool:
    """Verify the telescoping identity v(root) == sum leaf v + sum Delta."""
    e = result.extras
    res = e["identity_residual"]
    ok = abs(res) <= tol
    if verbose:
        print("Telescoping identity check")
        print(f"  v(root)            = {e['identity_lhs']:+.6f}")
        print(f"  sum leaf v + sum dD= {e['identity_rhs']:+.6f}")
        print(f"  residual           = {res:+.2e}   ({'OK' if ok else 'FAIL'} @ tol {tol:g})")
    return ok


# --------------------------------------------------------------------------- #
# plotting (matplotlib optional)
# --------------------------------------------------------------------------- #
def plot_interactions(result, img01: np.ndarray, leaf_label_map=None,
                      leaf_masks=None, k: int = 10):
    """Four-panel figure: input | additive (leaf v) | signed Delta | overlay."""
    import matplotlib.pyplot as plt

    add_map = result.attribution
    dmap = delta_localization_map(
        result, leaf_label_map=leaf_label_map, leaf_masks=leaf_masks, signed=True
    )
    lim = np.abs(dmap).max() or 1.0

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    ax[0].imshow(img01); ax[0].set_title("input"); ax[0].axis("off")

    a1 = ax[1].imshow(add_map, cmap="viridis")
    ax[1].set_title("additive (leaf v)"); ax[1].axis("off")
    fig.colorbar(a1, ax=ax[1], fraction=0.046)

    a2 = ax[2].imshow(dmap, cmap="bwr", vmin=-lim, vmax=lim)
    ax[2].set_title("synergy  Delta  (red=coop, blue=redundant)"); ax[2].axis("off")
    fig.colorbar(a2, ax=ax[2], fraction=0.046)

    ax[3].imshow(img01)
    ax[3].imshow(dmap, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.55)
    ax[3].set_title("Delta overlay"); ax[3].axis("off")

    e = result.extras
    fig.suptitle(
        f"pyramid interactions | sum_leaf_v={e['sum_leaf_v']:+.3f} "
        f"sum_delta={e['sum_delta']:+.3f} residual={e['identity_residual']:+.1e}"
    )
    fig.tight_layout()
    return fig