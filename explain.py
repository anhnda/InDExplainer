"""Main explanation driver.

Usage:
    python explain.py --image path/to/img.jpg --out out_dir [--target 207]
                      [--methods lime ig sufficiency] [--device cpu]

Loads the frozen ResNet-50 backbone, builds the strong-blur self-reference once
(used by IG + sufficiency), runs each requested explainer, and writes a SEPARATE
output file per method:  <out>/<method>.png  (original | heatmap | overlay)
plus a <out>/summary.txt with per-method diagnostics (f_x, f_b, f_phi, ...).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_suff.backbone import (
    denormalize,
    get_class_names,
    load_backbone,
    load_image,
)
from xai_suff.explainers import (
    IGExplainer,
    LIMEExplainer,
    SufficiencyExplainer,
    blur_reference,
)

METHODS = {
    "lime": LIMEExplainer,
    "ig": IGExplainer,
    "sufficiency": SufficiencyExplainer,
}


def _normalize_map(a: np.ndarray, signed: bool) -> np.ndarray:
    if signed:
        m = np.abs(a).max() + 1e-12
        return np.clip(a / m, -1, 1)  # keep sign, scale to [-1,1]
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)  # [0,1]


def _save_panel(out_path, x_norm, b_norm, result):
    img = denormalize(x_norm)[0].permute(1, 2, 0).cpu().numpy()
    blur = denormalize(b_norm)[0].permute(1, 2, 0).cpu().numpy()
    a = result.attribution
    signed = result.method == "ig"  # IG is signed; LIME/sufficiency >=0-ish
    amap = _normalize_map(a, signed)
    cmap = "seismic" if signed else "jet"

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(blur); ax[1].set_title("blur reference b"); ax[1].axis("off")
    im = ax[2].imshow(amap, cmap=cmap, vmin=(-1 if signed else 0), vmax=1)
    ax[2].set_title(f"{result.method} attribution"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ov = _normalize_map(a, False)
    ax[3].imshow(img); ax[3].imshow(ov, cmap="jet", alpha=0.5)
    ax[3].set_title("overlay"); ax[3].axis("off")

    title = (f"{result.method}  |  class={result.target_class} "
             f"({result.target_class_name})  f_x={result.f_x:.3f}")
    if result.f_b is not None:
        title += f"  f_b={result.f_b:.3f}"
    if result.f_phi is not None:
        title += f"  f_phi={result.f_phi:.3f}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--target", type=int, default=None,
                    help="target class index; default = model top-1")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument("--sigma", type=float, default=20.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stochastic", action="store_true",
                    help="use Bernoulli-sampled masks in the sufficiency method")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device

    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)
    b = blur_reference(x, args.sigma)

    with torch.no_grad():
        top1 = int(model(x).argmax(1).item())
    target = args.target if args.target is not None else top1
    print(f"[explain] target class = {target} ({class_names[target]})")

    summary_lines = [f"image: {args.image}",
                     f"target: {target} ({class_names[target]})",
                     f"sigma: {args.sigma}", ""]

    for name in args.methods:
        kwargs = dict(target_class=target, device=device, class_names=class_names,
                      sigma=args.sigma)
        if name == "sufficiency":
            kwargs["stochastic"] = args.stochastic
        explainer = METHODS[name](model, **kwargs)
        print(f"[explain] running {name} ...")
        result = explainer.explain(x)

        out_path = os.path.join(args.out, f"{name}.png")
        _save_panel(out_path, x, b, result)
        print(f"[explain]   -> {out_path}")

        line = f"{name}: f_x={result.f_x:.4f}"
        if result.f_b is not None:
            line += f" f_b={result.f_b:.4f}"
        if result.f_phi is not None:
            line += (f" f_phi={result.f_phi:.4f}"
                     f" mass={result.extras.get('final_mass', float('nan')):.4f}")
        summary_lines.append(line)

    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"[explain] summary -> {os.path.join(args.out, 'summary.txt')}")


if __name__ == "__main__":
    main()