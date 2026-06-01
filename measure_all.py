"""Main explanation driver + multi-baseline faithfulness measurement.

Extends explain.py with two flags:

    --all       run over EVERY infill/baseline mode (blur, black, white,
                noise, corners) instead of just the one passed to --infill.
    --measure   for each (method, infill) pair also compute insertion/deletion
                faithfulness, BOTH regional and per-pixel, and dump a summary
                table + JSON.

Usage:
    # original single-infill behaviour (unchanged)
    python explain_full.py --image img.jpg --out out_dir [--target 207]
                           [--methods lime ig sufficiency pyramid]
                           [--infill blur] [--device cpu]

    # measure faithfulness for the chosen infill (regional + per-pixel)
    python explain_full.py --image img.jpg --out out_dir --measure

    # sweep every baseline AND measure faithfulness for each
    python explain_full.py --image img.jpg --out out_dir --all --measure

The reference `b` used for insertion/deletion is the SAME infill reference that
the attribution method consumed, so insertion is the exact inverse construction
of the composite (no OOD gray/black fill mismatch). For "blur" the reference is
the strong self-blur; for the constant/noise modes it is that constant/noise
field. Both regional (whole-cell ordering) and per-pixel (RISE) curves are
reported because region methods (lime, pyramid) and pixel methods (ig) are
biased differently by the two orderings.
"""
from __future__ import annotations

import argparse
import json
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
    PyramidExplainer,
    make_reference,
)
from regional_eval import score_insertion_deletion

METHODS = {
    "lime": LIMEExplainer,
    "ig": IGExplainer,
    "sufficiency": SufficiencyExplainer,
    "pyramid": PyramidExplainer,
}

INFILLS = ["blur", "black", "white", "noise", "corners"]


# --------------------------------------------------------------------------- #
# plotting helpers (shared with explain.py)
# --------------------------------------------------------------------------- #
def _normalize_map(a: np.ndarray, signed: bool) -> np.ndarray:
    if signed:
        m = np.abs(a).max() + 1e-12
        return np.clip(a / m, -1, 1)  # keep sign, scale to [-1,1]
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)  # [0,1]


def _save_panel(out_path, x_norm, b_norm, result, infill="blur"):
    img = denormalize(x_norm)[0].permute(1, 2, 0).cpu().numpy()
    blur = denormalize(b_norm)[0].permute(1, 2, 0).cpu().numpy()
    a = result.attribution
    signed = result.method == "ig"  # IG is signed; LIME/sufficiency >=0-ish
    amap = _normalize_map(a, signed)
    cmap = "seismic" if signed else "jet"

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(img); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(blur); ax[1].set_title(f"{infill} reference b"); ax[1].axis("off")
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


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _measure(model, x, attr, target, b, cell_frac, steps, del_fill="blur"):
    """Return both regional and per-pixel insertion/deletion scores for one map.

    Regional uses whole-cell ordering on the shared cell grid; per-pixel uses
    RISE pixel ordering. del_fill controls the deletion removal operator:
    "blur" shares the infill operator (circular for blur-infill); "mean" is
    Phi-independent. We report blur-fill here to stay consistent with each
    method's own reference; pass del_fill="mean" for the non-circular check.
    """
    reg = score_insertion_deletion(
        model, x, attr, target, b,
        regional=True, cell_frac=cell_frac, del_fill=del_fill, steps=steps,
    )
    pix = score_insertion_deletion(
        model, x, attr, target, b,
        regional=False, cell_frac=cell_frac, del_fill=del_fill, steps=steps,
    )
    return reg, pix


def _plot_measure_curves(curves, out_path, title):
    """curves: list of dicts each with keys method, infill, regional/pixel."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for c in curves:
        tag = f"{c['method']}/{c['infill']}"
        reg, pix = c["regional"], c["pixel"]
        ax[0, 0].plot(reg["fractions"], reg["insertion"],
                      label=f"{tag} (AUC={reg['insertion_auc']:.3f})")
        ax[0, 1].plot(reg["fractions"], reg["deletion"],
                      label=f"{tag} (AUC={reg['deletion_auc']:.3f})")
        ax[1, 0].plot(pix["fractions"], pix["insertion"],
                      label=f"{tag} (AUC={pix['insertion_auc']:.3f})")
        ax[1, 1].plot(pix["fractions"], pix["deletion"],
                      label=f"{tag} (AUC={pix['deletion_auc']:.3f})")
    ax[0, 0].set_title("Regional insertion (blur->sharp)")
    ax[0, 1].set_title("Regional deletion (sharp->blur)")
    ax[1, 0].set_title("Per-pixel insertion")
    ax[1, 1].set_title("Per-pixel deletion")
    for a in ax.ravel():
        a.set_xlabel("fraction"); a.set_ylabel("target prob")
        a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--target", type=int, default=None,
                    help="target class index; default = model top-1")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument("--sigma", type=float, default=50.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stochastic", action="store_true",
                    help="use Bernoulli-sampled masks in the sufficiency method")
    ap.add_argument("--infill", default="blur", choices=INFILLS,
                    help="reference/infill mode (default: blur)")
    ap.add_argument("--all", dest="all_infills", action="store_true",
                    help="run over every infill baseline (overrides --infill)")
    ap.add_argument("--measure", action="store_true",
                    help="also compute regional + per-pixel insertion/deletion")
    ap.add_argument("--cell-frac", type=float, default=0.05,
                    help="fraction of total features per regional cell")
    ap.add_argument("--steps", type=int, default=50,
                    help="per-pixel insertion/deletion curve resolution")
    ap.add_argument("--del-fill", choices=["blur", "mean"], default="blur",
                    help="deletion removal operator (blur=shares infill, mean=independent)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device

    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)
    with torch.no_grad():
        top1 = int(model(x).argmax(1).item())
    target = args.target if args.target is not None else top1
    print(f"[explain] target class = {target} ({class_names[target]})")

    infills = INFILLS if args.all_infills else [args.infill]

    summary_lines = [f"image: {args.image}",
                     f"target: {target} ({class_names[target]})",
                     f"sigma: {args.sigma}",
                     f"infills: {infills}",
                     f"measure: {args.measure} (del_fill={args.del_fill})", ""]

    measure_rows = []   # flat records for JSON
    measure_curves = []  # curve dicts for plotting

    for infill in infills:
        b = make_reference(x, mode=infill, sigma=args.sigma).to(device)
        for name in args.methods:
            kwargs = dict(target_class=target, device=device,
                          class_names=class_names, sigma=args.sigma, infill=infill)
            if name == "sufficiency":
                kwargs["stochastic"] = args.stochastic
            explainer = METHODS[name](model, **kwargs)
            print(f"[explain] running {name} (infill={infill}) ...")
            result = explainer.explain(x)

            # panel: <method>_<infill>.png  (or <method>.png for single-infill runs)
            stem = name if len(infills) == 1 else f"{name}_{infill}"
            out_path = os.path.join(args.out, f"{stem}.png")
            _save_panel(out_path, x, b, result, infill=infill)
            print(f"[explain]   -> {out_path}")

            line = f"{name} [{infill}]: f_x={result.f_x:.4f}"
            if result.f_b is not None:
                line += f" f_b={result.f_b:.4f}"
            if result.f_phi is not None:
                line += (f" f_phi={result.f_phi:.4f}"
                         f" mass={result.extras.get('final_mass', float('nan')):.4f}")

            if args.measure:
                reg, pix = _measure(model, x, result.attribution, target, b,
                                    args.cell_frac, args.steps, args.del_fill)
                line += (f"\n    regional: ins-AUC={reg['insertion_auc']:.4f} "
                         f"del-AUC={reg['deletion_auc']:.4f}"
                         f"  | per-pixel: ins-AUC={pix['insertion_auc']:.4f} "
                         f"del-AUC={pix['deletion_auc']:.4f}")
                print(f"[explain]   {line.splitlines()[-1].strip()}")
                measure_rows.append({
                    "method": name, "infill": infill,
                    "f_x": result.f_x, "f_b": result.f_b, "f_phi": result.f_phi,
                    "regional_insertion_auc": reg["insertion_auc"],
                    "regional_deletion_auc": reg["deletion_auc"],
                    "pixel_insertion_auc": pix["insertion_auc"],
                    "pixel_deletion_auc": pix["deletion_auc"],
                    "del_fill": args.del_fill,
                })
                measure_curves.append({
                    "method": name, "infill": infill,
                    "regional": reg, "pixel": pix,
                })

            summary_lines.append(line)

    # write summary.txt
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"[explain] summary -> {os.path.join(args.out, 'summary.txt')}")

    # measurement artifacts
    if args.measure:
        with open(os.path.join(args.out, "measure.json"), "w") as fh:
            json.dump(measure_rows, fh, indent=2)

        # compact AUC table sorted by regional insertion-AUC (higher = better)
        hdr = (f"{'method':<12}{'infill':<9}"
               f"{'reg_ins':>9}{'reg_del':>9}{'pix_ins':>9}{'pix_del':>9}")
        table = [hdr, "-" * len(hdr)]
        for r in sorted(measure_rows,
                        key=lambda d: d["regional_insertion_auc"], reverse=True):
            table.append(
                f"{r['method']:<12}{r['infill']:<9}"
                f"{r['regional_insertion_auc']:>9.4f}"
                f"{r['regional_deletion_auc']:>9.4f}"
                f"{r['pixel_insertion_auc']:>9.4f}"
                f"{r['pixel_deletion_auc']:>9.4f}")
        table_str = "\n".join(table)
        with open(os.path.join(args.out, "measure_table.txt"), "w") as fh:
            fh.write(table_str + "\n")
        print("\n" + table_str)

        out_png = os.path.join(args.out, "measure_curves.png")
        _plot_measure_curves(
            measure_curves, out_png,
            f"Insertion/Deletion  class={target} ({class_names[target]})  "
            f"del_fill={args.del_fill}")
        print(f"[explain] measure -> {out_png}, measure.json, measure_table.txt")


if __name__ == "__main__":
    main()