"""Compare LIME, pyramid, and regional-IG on first- and second-order metrics.

    python compare_methods.py --image church.jpg --out cmp_out [--sigma 11] [--k 4]

Produces:
  cmp_out/faithfulness.json   first-order ins/del AUC per method (shared grid,
                              mean-fill = non-circular). Expect pyramid ~ LIME.
  cmp_out/interaction.json    second-order: NAI, additive-error lower bound,
                              Delta-faithfulness (additive vs interaction pred,
                              validated by direct reveal), pairwise off-diag ratio.
  cmp_out/interaction_matrix.png   LIME-diagonal vs model-actual side by side.
  cmp_out/summary.txt         the table you put in the paper.

The honest headline: first-order faithfulness is COMPARABLE across methods
(LIME is a strong first-order estimator on a shared grid); the model's response
is non-additive by fraction NAI, which LIME's additive model represents as zero.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_suff.backbone import get_class_names, load_backbone, load_image
from xai_suff.explainers import (
    LIMEExplainer, IGExplainer, PyramidExplainer, blur_reference,
)
from regional_eval import score_insertion_deletion
import metrics as M


def _plot_matrices(I, k, out_path):
    lim = np.abs(I).max() or 1.0
    diag_only = np.diag(np.diag(I))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(diag_only, cmap="bwr", vmin=-lim, vmax=lim)
    ax[0].set_title("LIME implied interaction\n(diagonal only, by construction)")
    im = ax[1].imshow(I, cmap="bwr", vmin=-lim, vmax=lim)
    ax[1].set_title("model actual interaction\n(diagonal + cooperation)")
    for a in ax:
        a.set_xlabel(f"cell (of {k*k})"); a.set_ylabel("cell")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Pairwise region interaction under blur-Phi")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="cmp_out")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--cell-frac", type=float, default=0.05)
    ap.add_argument("--del-fill", choices=["blur", "mean"], default="mean",
                    help="mean = non-circular (default, headline); blur shares Phi")
    ap.add_argument("--k", type=int, default=4, help="grid side for pairwise matrix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"[cmp] target = {target} ({names[target]})")

    methods = {
        "lime": LIMEExplainer(model, target_class=target, device=device,
                              class_names=names, sigma=args.sigma),
        "ig": IGExplainer(model, target_class=target, device=device,
                          class_names=names, sigma=args.sigma),
        "pyramid": PyramidExplainer(model, target_class=target, device=device,
                                    class_names=names, sigma=args.sigma),
    }

    # ---- first-order faithfulness (shared grid, non-circular fill) -------- #
    faith = {}
    results = {}
    for name, exp in methods.items():
        print(f"[cmp] {name} ...")
        res = exp.explain(x)
        results[name] = res
        c = M.faithfulness(score_insertion_deletion, model, x, res.attribution,
                           target, b, cell_frac=args.cell_frac,
                           del_fill=args.del_fill)
        faith[name] = {"insertion_auc": c["insertion_auc"],
                       "deletion_auc": c["deletion_auc"],
                       "del_fill": c.get("del_fill"),
                       "map": "additive (leaf v)" if name == "pyramid" else "default"}
        print(f"[cmp]   {name}: ins-AUC={c['insertion_auc']:.3f} "
              f"del-AUC={c['deletion_auc']:.3f}")

    # ---- pyramid scored with the SYNERGY map (Delta-derived ranking) ------ #
    # This is the "use synergy as the insertion/deletion input" option. It is
    # LOCKED to mean-fill (non-circular): scoring a Delta-ranking with blur-fill
    # would reuse pyramid's own Phi as the removal operator and be circular.
    try:
        syn_attr = M.synergy_attribution_map(results["pyramid"])
        cs = M.faithfulness(score_insertion_deletion, model, x, syn_attr,
                            target, b, cell_frac=args.cell_frac,
                            del_fill="mean")  # forced non-circular
        faith["pyramid_synergy"] = {"insertion_auc": cs["insertion_auc"],
                                    "deletion_auc": cs["deletion_auc"],
                                    "del_fill": "mean (forced non-circular)",
                                    "map": "synergy (max|Delta|)"}
        print(f"[cmp]   pyramid_synergy: ins-AUC={cs['insertion_auc']:.3f} "
              f"del-AUC={cs['deletion_auc']:.3f}  [Delta-ranked, mean-fill]")
    except ValueError as e:
        print(f"[cmp]   pyramid_synergy skipped: {e}")
    with open(os.path.join(args.out, "faithfulness.json"), "w") as fh:
        json.dump(faith, fh, indent=2)

    # ---- second-order interaction (pyramid + model-level matrix) ---------- #
    pyr = results["pyramid"]
    inter = {
        "nai": M.non_additivity_index(pyr),
        "additive_error_lower_bound": M.additive_error_lower_bound(pyr),
        "pyramid_query_cost": M.query_cost(pyr),
        "delta_faithfulness": M.delta_faithfulness(
            pyr, model=model, x=x, b=b, target=target, validate=True),
    }
    pim = M.pairwise_interaction_matrix(model, x, b, target, k=args.k)
    inter["pairwise_off_diag_ratio"] = pim["off_diag_ratio"]
    inter["pairwise_diag_mass"] = pim["diag_mass"]
    inter["pairwise_off_diag_mass"] = pim["off_diag_mass"]
    _plot_matrices(pim["I"], pim["k"], os.path.join(args.out, "interaction_matrix.png"))
    with open(os.path.join(args.out, "interaction.json"), "w") as fh:
        json.dump(inter, fh, indent=2)

    # ---- summary table ---------------------------------------------------- #
    df = inter["delta_faithfulness"]
    lines = [
        f"image: {args.image}   class: {target} ({names[target]})   sigma: {args.sigma}",
        f"faithfulness fill: {args.del_fill} ({'non-circular' if args.del_fill=='mean' else 'CIRCULAR-shares-Phi'})",
        "",
        "FIRST-ORDER faithfulness (shared grid; LIME is a strong baseline):",
        f"  {'method':>8} {'ins-AUC':>9} {'del-AUC':>9}",
    ]
    for name in methods:
        lines.append(f"  {name:>8} {faith[name]['insertion_auc']:>9.3f} "
                     f"{faith[name]['deletion_auc']:>9.3f}")
    if "pyramid_synergy" in faith:
        ps = faith["pyramid_synergy"]
        lines.append(f"  {'pyr-syn':>8} {ps['insertion_auc']:>9.3f} "
                     f"{ps['deletion_auc']:>9.3f}   <- Delta-ranked map, mean-fill")
    lines += [
        "",
        "SECOND-ORDER interaction (LIME = zero by construction):",
        f"  NAI (tree)                       : {inter['nai']:.4f}",
        f"  additive-error lower bound sum|D|: {inter['additive_error_lower_bound']:.4f}",
        f"  pairwise off-diagonal ratio (k={args.k}): {inter['pairwise_off_diag_ratio']:.4f}",
        "",
        "DELTA-FAITHFULNESS (additive vs interaction prediction of v(R)):",
        f"  mean additive error  |v(R)-sum v(c)| : {df['mean_additive_error']:.4f}  (= LIME's error)",
        f"  mean interaction err |v(R)-v_pred|   : {df['mean_interaction_error']:.2e}  (~0, identity)",
    ]
    if "mean_measured_vs_additive" in df:
        lines += [
            f"  validated on {df['n_validated']} nodes by direct reveal:",
            f"    mean |measured - additive|    : {df['mean_measured_vs_additive']:.4f}",
            f"    mean |measured - interaction| : {df['mean_measured_vs_interaction']:.4f}",
        ]
    lines += [
        "",
        f"COST: pyramid forward passes = {inter['pyramid_query_cost']} "
        f"(report this; pyramid is expensive)",
    ]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(txt)
    print("\n" + txt)
    print(f"[cmp] -> {args.out}/")


if __name__ == "__main__":
    main()