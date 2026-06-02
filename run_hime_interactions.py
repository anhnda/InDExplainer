"""Run HIMEExplainer on an image and export the interactive interaction views.

USAGE
    python run_hime_interactions.py path/to/church.jpg
    python run_hime_interactions.py path/to/church.jpg --out church_hime.html

Outputs:
  - <out>.html                 self-contained 4-tab interactive viewer
                               (first order | density | interactive graph | ranked pairs)
  - hime_interactions.png      static quick-look (input | first-order | density | top edges)

The class index is left at None so the explainer uses the model's own top-1.
HTML is self-contained: open in any browser, no server needed.
"""
import sys
import argparse
import numpy as np

import torch
from xai_suff.backbone import load_backbone, get_class_names, load_image, denormalize

# adjust this import to your package layout
from xai_suff.explainers import HIMEExplainer

from hime_interactions import export_hime_html, build_payload


def _static_panel(res, img01, out_png):
    """input | first-order map | interaction-density map | top-edge overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gh, gw = res.extras["grid"]
    payload = build_payload(res)
    H, W = img01.shape[:2]

    # density map painted blocky
    dens = np.zeros((H, W))
    agg = payload["agg_sum"]
    smax = max(agg) or 1.0
    for c in range(gh * gw):
        cy, cx = c // gw, c % gw
        dens[cy*H//gh:(cy+1)*H//gh, cx*W//gw:(cx+1)*W//gw] = agg[c] / smax

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img01); ax[0].set_title("input"); ax[0].axis("off")

    fmap = res.attribution
    m = np.abs(fmap).max() + 1e-12
    ax[1].imshow(fmap, cmap="seismic", vmin=-m, vmax=m)
    ax[1].set_title("first order (main effect)"); ax[1].axis("off")

    ax[2].imshow(dens, cmap="inferno", vmin=0, vmax=1)
    ax[2].set_title("interaction density  $\\Sigma|\\Delta|$"); ax[2].axis("off")

    # top edges drawn on the image
    ax[3].imshow(img01)
    edges = sorted(payload["edges"], key=lambda e: -abs(e["s"]))[:25]
    dmax = max((abs(e["s"]) for e in edges), default=1.0)
    for e in edges:
        (ci, cj, s) = e["i"], e["j"], e["s"]
        ay, ax_ = (ci // gw + 0.5) * H / gh, (ci % gw + 0.5) * W / gw
        by, bx_ = (cj // gw + 0.5) * H / gh, (cj % gw + 0.5) * W / gw
        col = "#e8463a" if s > 0 else "#3a6ee8"
        ax[3].plot([ax_, bx_], [ay, by], color=col,
                   lw=0.5 + 3.0 * abs(s) / dmax, alpha=0.5 + 0.4 * e["stab"])
    ax[3].set_title("top interaction pairs"); ax[3].axis("off")

    title = (f"hime  |  class={res.target_class} ({res.target_class_name})  "
             f"f_x={res.f_x:.3f}  |  {payload['meta']['n_edges']} pairs  "
             f"N={payload['meta']['n_samples']}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="hime_tree.html",
                    help="path for the interactive HTML")
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--grid", type=int, nargs=2, default=(12, 12),
                    help="superpixel grid, e.g. --grid 16 16")
    ap.add_argument("--n-samples", type=int, default=2500,
                    help="number of mask queries (theory: N ~ C*s*log p / gamma^2)")
    ap.add_argument("--target", type=int, default=None,
                    help="class index; default = model top-1")
    ap.add_argument("--static-png", default="hime_interactions.png")
    ap.add_argument("--max-active-cells", type=int, default=40)
    ap.add_argument("--stability-runs", type=int, default=20)
    ap.add_argument("--stability-thresh", type=float, default=0.8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)

    expl = HIMEExplainer(
        model, target_class=args.target, device=device, class_names=class_names,
        sigma=args.sigma, grid=tuple(args.grid), n_samples=args.n_samples,
        max_active_cells=args.max_active_cells,
        stability_runs=args.stability_runs, stability_thresh=args.stability_thresh,
    )
    res = expl.explain(x)

    print(f"\nmethod={res.method} target={res.target_class} "
          f"({res.target_class_name})  f_x={res.f_x:.3f}\n")

    ex = res.extras
    print(f"grid              = {ex['grid']}")
    print(f"n_samples (N)     = {ex.get('n_samples')}")
    print(f"active cells (m') = {ex.get('n_active_cells')}")
    print(f"candidate pairs   = {ex.get('candidate_pairs')}  "
          f"(dim p = {ex.get('candidate_dim_p')})")
    print(f"lambda            = {ex.get('lambda', float('nan')):.5f}")

    inter = ex.get("interactions", [])
    print(f"\nrecovered {len(inter)} interaction pairs. top 10 by |Delta|:")
    for (i, j, s) in inter[:10]:
        kind = "coop " if s > 0 else "redund"
        stab = ex.get("interaction_stability", {}).get(
            f"{min(i,j)}-{max(i,j)}", float("nan"))
        print(f"  cell {i:>3} <-> {j:>3}   Delta={s:+.4f}  ({kind})  stab={stab:.2f}")

    img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()

    # static quick-look
    png = _static_panel(res, img01, args.static_png)
    print(f"\nwrote {png}")

    # interactive HTML
    path = export_hime_html(res, img01, args.out)
    print(f"wrote {path}  (open in any browser; no server needed)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()