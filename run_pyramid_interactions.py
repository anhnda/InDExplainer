"""Run PyramidExplainer on an image and show its interaction-driven part.

Requires the package patch described in chat:
  In PyramidExplainer.explain(), before constructing AttributionResult, add
      leaf_masks = {leaf.id: leaf.mask for leaf in leaves}
  and include  "leaf_masks": leaf_masks  in the extras dict.

USAGE
    python run_pyramid_interactions.py path/to/church.jpg

The class index is left at None so the explainer uses the model's own top-1
(which for this image was 497 = church, per your panels).
"""
import sys
import numpy as np

import torch
from xai_suff.backbone import load_backbone, get_class_names, load_image, denormalize

# adjust this import to your package layout
from xai_suff.explainers import PyramidExplainer

from pyramid_interactions import (
    summarize_synergy, check_identity, plot_interactions, delta_localization_map,
)


def main(img_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(img_path, device)

    expl = PyramidExplainer(
        model, target_class=None, device=device, class_names=class_names,
        sigma=11.0, n_segments=144, compactness=2,
    )
    res = expl.explain(x)

    print(f"\nmethod={res.method} target={res.target_class} "
          f"({res.target_class_name})  f_x={res.f_x:.3f} f_b={res.f_b:.3f}\n")

    # 3) trust check FIRST -- if this fails, the Delta values are unreliable
    ok = check_identity(res)
    print()

    # 1) where does interaction live, and how much of the story is it?
    print(summarize_synergy(res, k=10))

    if not ok:
        print("\n[!] identity residual exceeds tol; treat Delta numbers with care.")

    # 2) localize on the image
    if "leaf_masks" not in res.extras:
        print("\n[!] extras['leaf_masks'] missing -- apply the serialize patch "
              "to render the spatial Delta map. Ranking/identity above still hold.")
        return

    img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
    fig = plot_interactions(res, img01, leaf_masks=res.extras["leaf_masks"])
    fig.savefig("pyramid_interactions.png", dpi=130, bbox_inches="tight")
    print("\nwrote pyramid_interactions.png")

    # numeric peek: most cooperative and most redundant region
    dmap = delta_localization_map(res, leaf_masks=res.extras["leaf_masks"])
    print(f"max cooperation (density) = {dmap.max():+.5f}")
    print(f"max redundancy  (density) = {dmap.min():+.5f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])