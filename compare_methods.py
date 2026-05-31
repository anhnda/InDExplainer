"""Compare LIME, pyramid, regional-IG, and Hessian-IG on first- and
second-order metrics, and adjudicate Hessian-IG vs pyramid on the second order.

    python compare_methods.py --image church.jpg --out cmp_out [--sigma 11] [--k 14]

Produces (additions over the original marked [NEW]):
  cmp_out/faithfulness.json       first-order ins/del AUC per method.
  cmp_out/interaction.json        second-order pyramid metrics + pairwise referee.
  cmp_out/interaction_matrix.png  LIME-diagonal vs model-actual (unchanged).
  cmp_out/verdict.json            [NEW] hessian-IG vs pyramid, both criteria.
  cmp_out/second_order_matrices.png [NEW] actual | hessian | pyramid, side by side.
  cmp_out/summary.txt             the paper table, now with the verdict block.

Headline (unchanged + extended): first-order faithfulness is COMPARABLE across
methods; LIME's additive model reports zero second-order signal; pyramid and
Hessian-IG both estimate the non-additive part, and we score *which* tracks the
model-actual interaction better, on the same k x k grid.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_suff.backbone import get_class_names, load_backbone, load_image, denormalize
from xai_suff.explainers import (
    LIMEExplainer, IGExplainer, PyramidExplainer, blur_reference,
)
from xai_suff.explainers import HessianIGExplainer  # [NEW]
from regional_eval import score_insertion_deletion
import metrics as M
import hessian_interactions as HV  # [NEW] verdict helpers


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
    ap.add_argument("--k", type=int, default=14, help="grid side for pairwise + Hessian")
    ap.add_argument("--hess-steps", type=int, default=16,
                    help="Riemann steps for the integrated Hessian")  # [NEW]
    ap.add_argument("--verdict-pairs", type=int, default=24,
                    help="random cell pairs for Hessian delta-faithfulness")  # [NEW]
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
        "hessian_ig": HessianIGExplainer(model, target_class=target, device=device,  # [NEW]
                                         class_names=names, sigma=args.sigma,
                                         k=args.k, hess_steps=args.hess_steps),
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
                       "map": "additive (leaf v)" if name == "pyramid"
                              else ("IG (blur-baseline)" if name in ("ig", "hessian_ig")
                                    else "default")}
        print(f"[cmp]   {name}: ins-AUC={c['insertion_auc']:.3f} "
              f"del-AUC={c['deletion_auc']:.3f}")

    # ---- pyramid scored with the SYNERGY map (Delta-derived ranking) ------ #
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

    # ====================================================================== #
    # [NEW] SECOND-ORDER VERDICT: Hessian-IG vs pyramid, on the SAME grid
    # ====================================================================== #
    I_actual = pim["I"]                                  # model-actual (K x K)
    hess = results["hessian_ig"]
    I_hess = hess.extras["interaction_matrix"]           # Hessian-IG (K x K)

    # project pyramid synergy onto the same grid (diagonal-dominant, fair)
    if "leaf_masks" in pyr.extras:
        P_pyr = HV.pyramid_grid_matrix(pyr, pyr.extras["leaf_masks"], k=args.k)
    else:
        print("[cmp][verdict] pyramid extras missing 'leaf_masks'; "
              "pyramid projected as zeros. Apply the serialize patch.")
        P_pyr = np.zeros_like(I_actual)

    # criterion A: agreement vs model-actual
    agree_hess = HV.agreement_scores(I_hess, I_actual)
    agree_pyr = HV.agreement_scores(P_pyr, I_actual)

    # criterion B: delta-faithfulness
    dfaith_hess = HV.hessian_delta_faithfulness(
        hess, model, x, b, target, k=args.k, n_pairs=args.verdict_pairs)
    dfaith_pyr = inter["delta_faithfulness"]  # pyramid's tree-native version

    verdict = HV.verdict(agree_hess, agree_pyr, dfaith_hess, dfaith_pyr)

    HV.plot_three_matrices(I_actual, I_hess, P_pyr, args.k,
                           os.path.join(args.out, "second_order_matrices.png"))

    verdict_blob = {
        "grid_k": args.k,
        "agreement_vs_actual": {"hessian_ig": agree_hess, "pyramid": agree_pyr},
        "delta_faithfulness": {"hessian_ig": dfaith_hess, "pyramid": dfaith_pyr},
        "hessian_query_cost": {"n_forward": hess.extras["n_forward"],
                               "n_backward": hess.extras["n_backward"]},
        "verdict": verdict,
    }
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(verdict_blob, fh, indent=2)

    # ---- summary table ---------------------------------------------------- #
    df = inter["delta_faithfulness"]
    lines = [
        f"image: {args.image}   class: {target} ({names[target]})   sigma: {args.sigma}",
        f"faithfulness fill: {args.del_fill} ({'non-circular' if args.del_fill=='mean' else 'CIRCULAR-shares-Phi'})",
        "",
        "FIRST-ORDER faithfulness (shared grid; LIME is a strong baseline):",
        f"  {'method':>10} {'ins-AUC':>9} {'del-AUC':>9}",
    ]
    for name in methods:
        lines.append(f"  {name:>10} {faith[name]['insertion_auc']:>9.3f} "
                     f"{faith[name]['deletion_auc']:>9.3f}")
    if "pyramid_synergy" in faith:
        ps = faith["pyramid_synergy"]
        lines.append(f"  {'pyr-syn':>10} {ps['insertion_auc']:>9.3f} "
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
    # ---- [NEW] verdict block ---------------------------------------------- #
    lines += [
        "",
        "=" * 64,
        "VERDICT  --  Hessian-IG vs pyramid (second order, same k x k grid)",
        "=" * 64,
        "",
        "A) AGREEMENT with model-actual pairwise matrix (off-diagonal):",
        f"  {'metric':>22} {'hessian_ig':>12} {'pyramid':>12}",
        f"  {'pearson r':>22} {agree_hess['offdiag_pearson_r']:>12.3f} "
        f"{agree_pyr['offdiag_pearson_r']:>12.3f}",
        f"  {'sign agreement':>22} {agree_hess['offdiag_sign_agreement']:>12.2f} "
        f"{agree_pyr['offdiag_sign_agreement']:>12.2f}",
        f"  {'off-diag mass ratio':>22} {agree_hess['offdiag_mass_ratio']:>12.3f} "
        f"{agree_pyr['offdiag_mass_ratio']:>12.3f}",
        "  (pyramid lands on the diagonal by construction -> low off-diag r is",
        "   expected; it asserts no specific cell-pair, like LIME.)",
        "",
        "B) DELTA-FAITHFULNESS (does the interaction term beat pure additive?):",
        f"  hessian-IG: add-err {dfaith_hess['mean_additive_error']:.4f} -> "
        f"int-err {dfaith_hess['mean_interaction_error']:.4f}  "
        f"(helps on {dfaith_hess['interaction_helps_fraction']:.0%} of {dfaith_hess['n_pairs']} pairs)",
        f"  pyramid   : add-err {df['mean_additive_error']:.4f} -> "
        f"int-err {df['mean_interaction_error']:.2e}  (telescoping identity)",
        "",
        f"  winner (agreement)          : {verdict['winner_agreement']}",
        f"  winner (delta-faithfulness) : {verdict['winner_delta_faithfulness']}",
        f"  OVERALL                     : {verdict['overall']}",
        "",
        "  why:",
    ]
    for r in verdict["reasons"]:
        lines.append(f"    - {r}")
    lines += [
        "",
        "COST:",
        f"  pyramid forward passes   = {inter['pyramid_query_cost']}",
        f"  hessian-IG forward       = {hess.extras['n_forward']}",
        f"  hessian-IG backward      = {hess.extras['n_backward']} "
        f"(K backward/step x hess_steps; K={args.k*args.k})",
    ]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(txt)
    print("\n" + txt)
    print(f"[cmp] -> {args.out}/")


if __name__ == "__main__":
    main()