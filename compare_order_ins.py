"""Ordered regional insertion/deletion: nested-chain (pyramid) vs LIME order.

WHAT THIS COMPARES
------------------
The standard `regional_insertion_deletion` in `regional_eval.py` ranks cells by
aggregate |attribution| mass. That is fine for a heat-map method, but it throws
away the one thing the *nested* method actually produces: an ORDER. The pyramid
explainer (name is temporal -- the object is a nested-partition / caterpillar
chain) grows a coalition S_1 c S_2 c ... c S_n by greedy joint-score steps, and
that growth order IS the explanation. LIME, on the same leaf vocabulary, induces
its own order: leaves sorted by descending surrogate coefficient.

So this script holds the GRID FIXED (one SLIC label map, shared by both methods)
and compares only the ORDERING:

  * pyramid : insert leaves in nested-chain order  (extras["chain"])
  * lime    : insert leaves in decreasing LIME-coefficient order

Both curves are built with the SAME regional composite logic and the SAME
removal field as `regional_eval.py`, so insertion is the exact inverse of
deletion and AUCs are directly comparable. The only moving part is the order.

Reading the curves
------------------
  * Insertion (blur/mean -> sharp), reveal leaves in method order, most-important
    first. Faster early rise  => the order front-loads sufficient evidence.
  * Deletion  (sharp -> blur/mean), remove leaves in method order, most-important
    first. Faster early drop   => the order front-loads necessary evidence.
  Higher insertion-AUC / lower deletion-AUC = the better ordering.

WHY THIS IS A FAIR FIGHT
------------------------
Same leaves, same reference, same metric. If the nested greedy order does real
work, its insertion curve rises faster than LIME's pure-surrogate order. If it
does not, the two curves coincide -- which is itself a useful (negative) result
and exactly the control the pyramid docstring asks for.

USAGE
-----
    python compare_ordered_insertion.py --image church.jpg --out ord_out \
        [--sigma 11] [--n-segments 144] [--compactness 2] [--target N] \
        [--del-fill mean] [--k-lime 2]

Outputs in --out/:
    ordered_curves.png   insertion + deletion, both methods overlaid
    ordered_curves.json  raw fractions / curves / AUCs
    summary.txt          AUC table + which order wins

NOTE: requires the same environment as the rest of the suite (torch + the
xai_suff package). Nothing is installed here; run it where those already live.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_suff.backbone import (
    denormalize, get_class_names, load_backbone, load_image,
)
from xai_suff.explainers import PyramidExplainer, blur_reference


# --------------------------------------------------------------------------- #
# removal field (mirrors regional_eval._removal_field exactly)
# --------------------------------------------------------------------------- #
def _removal_field(x: torch.Tensor, b: torch.Tensor, del_fill: str) -> torch.Tensor:
    """Field that replaces a removed region. 'blur' shares Phi (circular for a
    blur-Phi method); 'mean' is independent of Phi (non-circular)."""
    if del_fill == "blur":
        return b
    if del_fill == "mean":
        mean_c = x.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        return mean_c.expand_as(x).contiguous()        # (1,C,H,W)
    raise ValueError(f"unknown del_fill={del_fill!r}; use 'blur' or 'mean'")


# --------------------------------------------------------------------------- #
# ordered regional insertion/deletion on a FIXED leaf grid
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ordered_insertion_deletion(
    model,
    x: torch.Tensor,
    b: torch.Tensor,
    labels: np.ndarray,          # (H,W) shared SLIC leaf label map
    leaf_order: list,            # leaf labels, most-important first
    target: int,
    del_fill: str = "mean",
    batch: int = 16,
):
    """Reveal/remove WHOLE LEAVES in `leaf_order`. Returns the same dict shape as
    regional_eval so it plots identically. The x-axis is the fraction of FEATURES
    covered (so unequal-area SLIC leaves are reflected honestly, not assumed
    equal-area)."""
    device = x.device
    _, C, H, W = x.shape
    n = H * W

    labels_t = torch.as_tensor(labels.reshape(-1), device=device)
    x_flat = x.view(C, n)
    r = _removal_field(x, b, del_fill)
    r_flat = r.reshape(C, n)

    # cumulative pixel masks: 0,1,...,len(order) leaves revealed
    fractions = [0.0]
    keep_masks = []
    covered = torch.zeros(n, dtype=torch.bool, device=device)
    keep_masks.append(covered.clone())
    for lab in leaf_order:
        covered = covered | (labels_t == int(lab))
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

    # insertion: base = removal field, fill = sharp  (reveal top leaves)
    # deletion : base = sharp,         fill = removal field  (remove top leaves)
    for base, fill, out in (
        (r_flat, x_flat, ins_probs),
        (x_flat, r_flat, del_probs),
    ):
        for start in range(0, len(keep_masks), batch):
            subset = keep_masks[start:start + batch]
            comp = _build(subset, base, fill)
            p = F.softmax(model(comp), dim=1)[:, target].cpu().numpy()
            out[start:start + len(subset)] = p

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        "fractions": fractions.tolist(),
        "insertion": ins_probs.tolist(),
        "deletion": del_probs.tolist(),
        "insertion_auc": float(_trapz(ins_probs, fractions)),
        "deletion_auc": float(_trapz(del_probs, fractions)),
        "n_leaves": int(len(leaf_order)),
        "del_fill": del_fill,
    }


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _plot(curves: dict, out_path: str, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in curves.items():
        f = c["fractions"]
        ax[0].plot(f, c["insertion"], marker="", lw=2,
                   label=f"{name} (AUC={c['insertion_auc']:.3f})")
        ax[1].plot(f, c["deletion"], marker="", lw=2,
                   label=f"{name} (AUC={c['deletion_auc']:.3f})")
    ax[0].set_title("Insertion (reference -> sharp, leaves in method order)")
    ax[0].set_xlabel("fraction of features revealed"); ax[0].set_ylabel("target prob")
    ax[1].set_title("Deletion (sharp -> reference, leaves in method order)")
    ax[1].set_xlabel("fraction of features removed"); ax[1].set_ylabel("target prob")
    for a in ax:
        a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="ord_out")
    ap.add_argument("--target", type=int, default=None,
                    help="class index; default = model top-1")
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--n-segments", type=int, default=144)
    ap.add_argument("--compactness", type=float, default=2.0)
    ap.add_argument("--del-fill", choices=["blur", "mean"], default="mean",
                    help="removal operator: mean = non-circular (default), "
                         "blur = shares Phi (circular)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k-lime", type=int, default=10,
                    help="LIME seed leaves used by the pyramid chain")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    model = load_backbone(device)
    names = get_class_names()
    x = load_image(args.image, device)
    b = blur_reference(x, args.sigma).to(device)

    with torch.no_grad():
        top1 = int(model(x).argmax(1).item())
    target = args.target if args.target is not None else top1
    print(f"[ord] target = {target} ({names[target]})  del_fill={args.del_fill}")

    # --- ONE pyramid run gives BOTH orderings on ONE shared grid ----------- #
    # The pyramid builds the SLIC leaves, the nested greedy chain, AND a LIME
    # ranking over the very same leaves. We pull all three from one explain()
    # call so the grid is provably identical for both methods.
    expl = PyramidExplainer(
        model, target_class=target, device=device, class_names=names,
        sigma=args.sigma, n_segments=args.n_segments,
        compactness=args.compactness, k_lime=args.k_lime,
    )
    res = expl.explain(x)

    nested_order = [int(l) for l in res.extras["chain"]]        # S_1..S_n order
    lime_order = [int(l) for l in res.extras["lime_ranked"]]    # desc. coeff

    # rebuild the exact SLIC labels the explainer used (same seed/params -> same
    # map). We reuse the explainer's own leaf segmentation to avoid any drift.
    img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
    labels = res.extras["leaf_labels"]   # exact map the chain indexes; do NOT re-segment
    # sanity: the two orders must be permutations of the same leaf set
    assert set(nested_order) == set(lime_order) == set(int(l) for l in np.unique(labels)), \
        "nested / lime / label leaf sets disagree -- grid is not shared"
    print(f"[ord] shared grid: {len(nested_order)} leaves; "
          f"nested chain seeded by top-{res.extras['k_lime_used']} LIME leaves")

    curves = {}
    print("[ord] pyramid (nested-chain order) ...")
    curves["pyramid (nested)"] = ordered_insertion_deletion(
        model, x, b, labels, nested_order, target, del_fill=args.del_fill)
    print(f"[ord]   ins-AUC={curves['pyramid (nested)']['insertion_auc']:.3f} "
          f"del-AUC={curves['pyramid (nested)']['deletion_auc']:.3f}")

    print("[ord] LIME (descending-coefficient order) ...")
    curves["LIME (score order)"] = ordered_insertion_deletion(
        model, x, b, labels, lime_order, target, del_fill=args.del_fill)
    print(f"[ord]   ins-AUC={curves['LIME (score order)']['insertion_auc']:.3f} "
          f"del-AUC={curves['LIME (score order)']['deletion_auc']:.3f}")

    # --- save ---------------------------------------------------------------- #
    _plot(curves, os.path.join(args.out, "ordered_curves.png"),
          f"Ordered regional insertion/deletion  class={target} ({names[target]})  "
          f"fill={args.del_fill}")
    with open(os.path.join(args.out, "ordered_curves.json"), "w") as fh:
        json.dump({"target": target, "target_name": names[target],
                   "sigma": args.sigma, "del_fill": args.del_fill,
                   "n_leaves": len(nested_order),
                   "k_lime_used": res.extras["k_lime_used"],
                   "curves": curves}, fh, indent=2)

    pn, pl = curves["pyramid (nested)"], curves["LIME (score order)"]
    ins_win = "pyramid" if pn["insertion_auc"] > pl["insertion_auc"] else "LIME"
    del_win = "pyramid" if pn["deletion_auc"] < pl["deletion_auc"] else "LIME"
    lines = [
        f"image: {args.image}   class: {target} ({names[target]})",
        f"sigma: {args.sigma}   del_fill: {args.del_fill} "
        f"({'non-circular' if args.del_fill == 'mean' else 'CIRCULAR (shares Phi)'})",
        f"shared SLIC grid: {len(nested_order)} leaves "
        f"(n_segments={args.n_segments}, compactness={args.compactness})",
        f"pyramid chain seeded by top-{res.extras['k_lime_used']} LIME leaves "
        f"(name 'pyramid' is temporal; object is a nested-partition chain)",
        "",
        "ORDERED INSERTION / DELETION  (same leaves, same reference, only the",
        "ordering differs -- nested greedy chain vs descending LIME coefficient):",
        f"  {'order':>22} {'ins-AUC':>9} {'del-AUC':>9}",
        f"  {'pyramid (nested)':>22} {pn['insertion_auc']:>9.3f} {pn['deletion_auc']:>9.3f}",
        f"  {'LIME (score order)':>22} {pl['insertion_auc']:>9.3f} {pl['deletion_auc']:>9.3f}",
        "",
        f"  insertion winner (higher AUC): {ins_win}",
        f"  deletion  winner (lower  AUC): {del_win}",
        "",
        "  Note: the first few nested leaves are LIME-seeded, so the two curves",
        "  start together and diverge once the greedy joint-score steps take over.",
        "  If they do NOT diverge, the greedy phase adds nothing beyond LIME.",
    ]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(txt)
    print("\n" + txt)
    print(f"[ord] -> {args.out}/")


if __name__ == "__main__":
    main()