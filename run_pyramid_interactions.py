"""Run PyramidExplainer on an image and export an interactive synergy tree.

Requires the package patch (already described in chat):
  In PyramidExplainer.explain(), before constructing AttributionResult, add
      leaf_masks = {leaf.id: leaf.mask for leaf in leaves}
  and include  "leaf_masks": leaf_masks  in the extras dict.
(Your metrics.py / compare_methods.py already assume this patch.)

USAGE
    python run_pyramid_interactions.py path/to/church.jpg
    python run_pyramid_interactions.py path/to/church.jpg --out church_pyramid.html

Outputs:
  - <out>.html               self-contained interactive icicle (hover -> trace region)
  - pyramid_interactions.png  the 4-panel static quick-look (input|leaf|Delta|overlay)
The class index is left at None so the explainer uses the model's own top-1
(497 = church for your panels).
"""
import sys
import argparse
import numpy as np

import torch
from xai_suff.backbone import load_backbone, get_class_names, load_image, denormalize

# adjust this import to your package layout
from xai_suff.explainers import PyramidExplainer

from pyramid_interactions import (
    summarize_synergy, check_identity, plot_interactions,
    delta_localization_map, export_interactive_html,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="pyramid_tree.html",
                    help="path for the interactive HTML")
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--n-segments", type=int, default=144)
    ap.add_argument("--compactness", type=float, default=2.0)
    ap.add_argument("--target", type=int, default=None,
                    help="class index; default = model top-1")
    ap.add_argument("--static-png", default="pyramid_interactions.png")
    ap.add_argument("--mask-budget", type=int, default=400,
                    help="max internal nodes carrying an explicit RLE mask "
                         "(others reconstruct from descendant leaves in-browser)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)

    expl = PyramidExplainer(
        model, target_class=args.target, device=device, class_names=class_names,
        sigma=args.sigma, n_segments=args.n_segments, compactness=args.compactness,
    )
    res = expl.explain(x)

    print(f"\nmethod={res.method} target={res.target_class} "
          f"({res.target_class_name})  f_x={res.f_x:.3f} f_b={res.f_b:.3f}\n")

    # 1) trust check FIRST -- if this fails, the Delta values are unreliable
    ok = check_identity(res)
    print()

    # 2) where does interaction live, and how much of the story is it?
    print(summarize_synergy(res, k=10))

    if not ok:
        print("\n[!] identity residual exceeds tol; treat Delta numbers with care.")

    if "leaf_masks" not in res.extras:
        print("\n[!] extras['leaf_masks'] missing -- apply the serialize patch to "
              "PyramidExplainer.explain. Ranking/identity above still hold, but the "
              "spatial map and interactive HTML need it.")
        return

    img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()

    # 3) static quick-look (input | additive leaf | Delta density | overlay)
    fig = plot_interactions(res, img01, leaf_masks=res.extras["leaf_masks"])
    fig.savefig(args.static_png, dpi=130, bbox_inches="tight")
    print(f"\nwrote {args.static_png}")

    # 4) the headline: interactive self-contained HTML
    path = export_interactive_html(
        res, img01, args.out, leaf_masks=res.extras["leaf_masks"],
        max_nodes_with_masks=args.mask_budget,
    )
    print(f"wrote {path}  (open in any browser; no server needed)")

    # numeric peek
    dmap = delta_localization_map(res, leaf_masks=res.extras["leaf_masks"])
    print(f"\nmax cooperation (density) = {dmap.max():+.6f}")
    print(f"max redundancy  (density) = {dmap.min():+.6f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()