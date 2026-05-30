"""Evaluation: insertion/deletion faithfulness curves + sigma sensitivity sweep.

Two evaluations, both consuming the same package as `explain.py`.

1. Insertion / Deletion curves (Petsiuk et al., RISE)
   Rank pixels by an attribution map. Reveal (insertion) or remove (deletion)
   them from most- to least-important in fractional steps, measuring the target
   probability at each budget.
     - Insertion: start from the blurred reference, progressively paste sharp
       pixels in importance order. Sharp early rise => compact sufficient evidence.
     - Deletion: start from the sharp image, progressively blur out pixels in
       importance order. Sharp early drop => the region was truly necessary.
   We report the area under each curve (AUC). Higher insertion-AUC and lower
   deletion-AUC both indicate a more faithful map. We use the SAME blur reference
   as the attribution background, so insertion is the exact inverse construction
   of the sufficiency composite (consistent baseline, no OOD gray/black fill).

2. Sigma sensitivity sweep (sufficiency method)
   The blur strength sigma is the one real knob. For each sigma we report:
   f_b (blur-floor neutrality), f_phi (achieved sufficiency), final mask mass,
   and insertion-AUC of the resulting map. Lets you pick a sigma where the blur
   is class-neutral (low f_b) yet the optimizer finds a compact, faithful mask.

Usage:
    python evaluate.py curves --image img.jpg --out eval_out [--method sufficiency]
    python evaluate.py sweep  --image img.jpg --out eval_out \
                              --sigmas 5 8 11 15 21
    python evaluate.py both   --image img.jpg --out eval_out
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

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


# --------------------------------------------------------------------------- #
# pixel ordering
# --------------------------------------------------------------------------- #
def _rank_pixels(attr: np.ndarray) -> np.ndarray:
    """Return flat pixel indices ordered most -> least important.

    Magnitude ranking so signed maps (IG) are handled by importance, not sign.
    """
    flat = np.abs(attr).reshape(-1)
    return np.argsort(-flat)  # descending


# --------------------------------------------------------------------------- #
# insertion / deletion
# --------------------------------------------------------------------------- #
@torch.no_grad()
def insertion_deletion(
    model,
    x: torch.Tensor,
    attr: np.ndarray,
    target: int,
    b: torch.Tensor,
    steps: int = 50,
    batch: int = 16,
):
    """Compute insertion and deletion probability curves.

    Returns dict with fractions, insertion probs, deletion probs, and AUCs.
    """
    device = x.device
    _, C, H, W = x.shape
    n = H * W
    order = _rank_pixels(attr)  # most important first
    fractions = np.linspace(0, 1, steps + 1)
    counts = (fractions * n).astype(int)

    x_flat = x.view(C, n)
    b_flat = b.view(C, n)

    def _build(keep_idx_lists, base_flat, fill_flat):
        """Build a batch of composites: start from base, overwrite keep idx w/ fill."""
        imgs = []
        for idx in keep_idx_lists:
            comp = base_flat.clone()
            if len(idx) > 0:
                idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
                comp[:, idx_t] = fill_flat[:, idx_t]
            imgs.append(comp.view(1, C, H, W))
        return torch.cat(imgs, dim=0)

    ins_probs = np.zeros(len(counts))
    del_probs = np.zeros(len(counts))

    # insertion: base = blur, fill = sharp, revealing top pixels
    # deletion : base = sharp, fill = blur, removing top pixels
    for which, (base, fill, out) in {
        "ins": (b_flat, x_flat, ins_probs),
        "del": (x_flat, b_flat, del_probs),
    }.items():
        for start in range(0, len(counts), batch):
            chunk = counts[start:start + batch]
            keep_lists = [order[:k] for k in chunk]
            comp = _build(keep_lists, base, fill)
            p = F.softmax(model(comp), dim=1)[:, target].cpu().numpy()
            out[start:start + len(chunk)] = p

    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2.x/1.x
    ins_auc = float(_trapz(ins_probs, fractions))
    del_auc = float(_trapz(del_probs, fractions))
    return {
        "fractions": fractions.tolist(),
        "insertion": ins_probs.tolist(),
        "deletion": del_probs.tolist(),
        "insertion_auc": ins_auc,
        "deletion_auc": del_auc,
    }


def _plot_curves(curves_by_method: dict, out_path: str, title: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in curves_by_method.items():
        f = c["fractions"]
        ax[0].plot(f, c["insertion"],
                   label=f"{name} (AUC={c['insertion_auc']:.3f})")
        ax[1].plot(f, c["deletion"],
                   label=f"{name} (AUC={c['deletion_auc']:.3f})")
    ax[0].set_title("Insertion (blur -> sharp, top-down)")
    ax[0].set_xlabel("fraction inserted"); ax[0].set_ylabel("target prob")
    ax[1].set_title("Deletion (sharp -> blur, top-down)")
    ax[1].set_xlabel("fraction deleted"); ax[1].set_ylabel("target prob")
    for a in ax:
        a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
def _make_explainer(name, model, target, device, class_names, sigma, stochastic):
    kw = dict(target_class=target, device=device, class_names=class_names,
              sigma=sigma)
    if name == "sufficiency":
        kw["stochastic"] = stochastic
    return METHODS[name](model, **kw)


def run_curves(args, model, x, target, class_names, device):
    b = blur_reference(x, args.sigma).to(device)
    methods = args.methods or list(METHODS)
    curves = {}
    for name in methods:
        exp = _make_explainer(name, model, target, device, class_names,
                              args.sigma, args.stochastic)
        print(f"[eval] curves: running {name} ...")
        res = exp.explain(x)
        c = insertion_deletion(model, x, res.attribution, target, b,
                               steps=args.steps)
        curves[name] = c
        print(f"[eval]   {name}: ins-AUC={c['insertion_auc']:.3f} "
              f"del-AUC={c['deletion_auc']:.3f}")
    out_png = os.path.join(args.out, "insertion_deletion.png")
    _plot_curves(curves, out_png,
                 f"Insertion/Deletion  class={target} ({class_names[target]})")
    with open(os.path.join(args.out, "curves.json"), "w") as fh:
        json.dump(curves, fh, indent=2)
    print(f"[eval] -> {out_png}")
    return curves


def run_sweep(args, model, x, target, class_names, device):
    rows = []
    for sigma in args.sigmas:
        exp = SufficiencyExplainer(
            model, target_class=target, device=device,
            class_names=class_names, sigma=sigma, stochastic=args.stochastic)
        print(f"[eval] sweep: sigma={sigma} ...")
        res = exp.explain(x)
        b = blur_reference(x, sigma).to(device)
        c = insertion_deletion(model, x, res.attribution, target, b,
                               steps=args.steps)
        rows.append({
            "sigma": float(sigma),
            "f_x": res.f_x,
            "f_b": res.f_b,
            "f_phi": res.f_phi,
            "mass": float(res.extras["final_mass"]),
            "gap": float(res.extras["gap"]),
            "insertion_auc": c["insertion_auc"],
            "deletion_auc": c["deletion_auc"],
        })
        print(f"[eval]   sigma={sigma}: f_b={res.f_b:.3f} f_phi={res.f_phi:.3f} "
              f"mass={res.extras['final_mass']:.3f} "
              f"ins-AUC={c['insertion_auc']:.3f}")

    # plot sweep
    s = [r["sigma"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(s, [r["f_b"] for r in rows], "o-", label="f_b (blur floor)")
    ax[0].plot(s, [r["f_x"] for r in rows], "k--", label="f_x")
    ax[0].plot(s, [r["f_phi"] for r in rows], "s-", label="f_phi")
    ax[0].set_title("neutrality & sufficiency"); ax[0].set_xlabel("sigma")
    ax[0].set_ylabel("target prob"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(s, [r["mass"] for r in rows], "o-")
    ax[1].set_title("final mask mass |m|_1 / d"); ax[1].set_xlabel("sigma")
    ax[1].grid(alpha=0.3)
    ax[2].plot(s, [r["insertion_auc"] for r in rows], "o-", label="insertion AUC")
    ax[2].plot(s, [r["deletion_auc"] for r in rows], "s-", label="deletion AUC")
    ax[2].set_title("faithfulness vs sigma"); ax[2].set_xlabel("sigma")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle(f"Sigma sweep (sufficiency)  class={target} ({class_names[target]})")
    fig.tight_layout()
    out_png = os.path.join(args.out, "sigma_sweep.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(args.out, "sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"[eval] -> {out_png}")

    # guidance: flag sigmas where blur isn't neutral
    bad = [r["sigma"] for r in rows if r["f_b"] >= r["f_x"] - 0.05]
    if bad:
        print(f"[eval] WARNING: blur not class-neutral at sigma={bad} "
              f"(f_b >= f_x); maps there are unreliable.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["curves", "sweep", "both"])
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS))
    ap.add_argument("--sigma", type=float, default=11.0)
    ap.add_argument("--sigmas", nargs="+", type=float,
                    default=[5, 8, 11, 15, 21])
    ap.add_argument("--steps", type=int, default=50,
                    help="insertion/deletion curve resolution")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)
    with torch.no_grad():
        top1 = int(model(x).argmax(1).item())
    target = args.target if args.target is not None else top1
    print(f"[eval] target class = {target} ({class_names[target]})")

    if args.mode in ("curves", "both"):
        run_curves(args, model, x, target, class_names, device)
    if args.mode in ("sweep", "both"):
        run_sweep(args, model, x, target, class_names, device)


if __name__ == "__main__":
    main()