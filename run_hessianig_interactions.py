"""Run HessianIGExplainer on an image and inspect its second-order interactions.

USAGE
    python run_hessianig_interactions.py path/to/church.jpg
        [--exact] [--k 14] [--hess-steps 8] [--beta 10] [--n-probes 16]

The class index is left at None so the explainer uses the model's own top-1
(497 = church for the panel image).

WHAT IT PRINTS, IN ORDER
------------------------
0) VALIDITY GATE -- interaction-completeness  sum_ij Gamma_ij == f(x)-f(b).
   This is the trust check, printed FIRST, exactly like pyramid's check_identity:
   if the residual is not small relative to |f(x)-f(b)|, the Gamma matrix is
   unreliable and every downstream number should be ignored. A large residual
   here is NOT fixed by reading the rest -- it means raise --hess-steps, lower
   --beta, try --exact, or check ReLU-swap coverage (n_relu_swapped).

1) WHERE interaction lives: the most cooperative (Gamma_ij > 0) and most
   redundant (Gamma_ij < 0) off-diagonal cell pairs, plus the cooperation share
   (off-diagonal mass / total mass).

2) LOCALIZE on the image: per-cell net interaction (row-sum of off-diagonal
   Gamma) rendered as a heat map over the k x k grid, saved to a PNG.

Mirrors run_pyramid_interactions.py in spirit; the pyramid runner reports a
tree-relative Delta map, this reports a grid-relative Gamma matrix.
"""
import argparse
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_suff.backbone import load_backbone, get_class_names, load_image, denormalize

# adjust this import to your package layout
from xai_suff.explainers import HessianIGExplainer


# --------------------------------------------------------------------------- #
# 0) validity gate
# --------------------------------------------------------------------------- #
def check_completeness(res, tol_frac: float = 0.10) -> bool:
    """Print interaction-completeness and return whether it holds.

    Analog of pyramid's check_identity: sum_ij Gamma_ij should equal f(x)-f(b)
    (both on the softplus-smoothed surface the Hessian was integrated on).
    """
    e = res.extras
    lhs = e["completeness_lhs"]          # sum_ij Gamma_ij
    rhs = e["completeness_rhs"]          # f(x)-f(b) on the softplus net
    resid = e["completeness_residual"]
    scale = abs(rhs) + 1e-9
    ok = abs(resid) <= tol_frac * scale
    print("interaction-completeness (sum_ij Gamma == f(x)-f(b)):")
    print(f"  sum Gamma      = {lhs:+.5f}")
    print(f"  f(x)-f(b) [sp] = {rhs:+.5f}")
    print(f"  residual       = {resid:+.5f}   ({abs(resid)/scale:.1%} of |f(x)-f(b)|)")
    print(f"  softplus_beta={e['softplus_beta']}  relus_swapped={e['n_relu_swapped']}"
          f"  mode={e['mode']}")
    print(f"  => {'OK' if ok else 'FAILED -- Gamma unreliable, do not trust below'}")
    return ok


# --------------------------------------------------------------------------- #
# helpers to read the Gamma matrix
# --------------------------------------------------------------------------- #
def _off_diag_pairs(I, k):
    """Yield (value, i, j) for i<j, sorted by |value| descending."""
    pairs = []
    K = I.shape[0]
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((float(I[i, j]), i, j))
    pairs.sort(key=lambda t: abs(t[0]), reverse=True)
    return pairs


def _rc(idx, k):
    """cell index -> (row, col) on the k x k grid (row-major)."""
    return idx // k, idx % k


def summarize_interactions(res, top: int = 10) -> str:
    e = res.extras
    I = e["interaction_matrix"]
    k = e["k"]
    diag = np.diag(I)
    off = I - np.diag(diag)
    diag_mass = float(np.abs(diag).sum())
    off_mass = float(np.abs(off).sum())
    total = diag_mass + off_mass
    coop_share = off_mass / total if total > 0 else 0.0

    lines = [
        f"GRID k={k} ({k*k} cells)   cooperation share (off-diag mass / total) = {coop_share:.3f}",
        f"  diagonal (main-effect) mass = {diag_mass:.4f}",
        f"  off-diagonal (interaction) mass = {off_mass:.4f}",
        "",
        f"top {top} interacting cell pairs (Gamma_ij, + cooperation / - redundancy):",
    ]
    pairs = _off_diag_pairs(I, k)[:top]
    for val, i, j in pairs:
        ri, ci = _rc(i, k)
        rj, cj = _rc(j, k)
        tag = "coop " if val > 0 else "redun"
        lines.append(f"  {tag} {val:+.5f}   cell {i}({ri},{ci}) <-> cell {j}({rj},{cj})")
    return "\n".join(lines)


def per_cell_net_interaction(res) -> np.ndarray:
    """(k,k) map: each cell's net off-diagonal interaction (row-sum of off-diag)."""
    e = res.extras
    I = e["interaction_matrix"]
    k = e["k"]
    off = I - np.diag(np.diag(I))
    net = off.sum(axis=1)                 # signed net interaction per cell
    return net.reshape(k, k)


# --------------------------------------------------------------------------- #
# 2) localize: overlay per-cell net interaction on the image
# --------------------------------------------------------------------------- #
def plot_interactions(res, img01, out_path):
    netmap = per_cell_net_interaction(res)
    k = res.extras["k"]
    lim = np.abs(netmap).max() or 1.0

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(img01)
    ax[0].set_title("input")
    ax[0].axis("off")

    # upsample the k x k net-interaction map to overlay
    H, W = img01.shape[:2]
    up = np.kron(netmap, np.ones((H // k + 1, W // k + 1)))[:H, :W]
    ax[1].imshow(img01)
    im = ax[1].imshow(up, cmap="bwr", vmin=-lim, vmax=lim, alpha=0.55)
    ax[1].set_title("net interaction per cell\n(red=cooperation, blue=redundancy)")
    ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle("Hessian-IG second-order interaction")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--k", type=int, default=14)
    ap.add_argument("--hess-steps", type=int, default=8)
    ap.add_argument("--beta", type=float, default=10.0, help="softplus sharpness")
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--exact", action="store_true",
                    help="exact full Hessian (slow); default fast Hutchinson")
    ap.add_argument("--sigma", type=float, default=11.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(device)
    class_names = get_class_names()
    x = load_image(args.image, device)

    expl = HessianIGExplainer(
        model, target_class=args.target, device=device, class_names=class_names,
        sigma=args.sigma, k=args.k, hess_steps=args.hess_steps,
        softplus_beta=args.beta, fast=not args.exact, n_probes=args.n_probes,
    )
    res = expl.explain(x)

    print(f"\nmethod={res.method} target={res.target_class} "
          f"({res.target_class_name})  f_x={res.f_x:.3f} f_b={res.f_b:.3f}\n")

    # 0) validity gate FIRST -- everything below depends on it
    ok = check_completeness(res)
    print()

    # 1) where does interaction live, how much of the story is it?
    print(summarize_interactions(res, top=10))

    if not ok:
        print("\n[!] completeness residual exceeds tol; the Gamma numbers above "
              "are unreliable. Try --exact, raise --hess-steps, lower --beta, or "
              "check n_relu_swapped covers your backbone's activations.")
        # still save the map, but the caveat stands
    # cost
    e = res.extras
    print(f"\ncost: forward={e['n_forward']} backward={e['n_backward']} "
          f"(mode={e['mode']}"
          + (f", n_probes={e['n_probes']}" if e['mode'] == 'fast_hutchinson' else "")
          + f", hess_steps={e['hess_steps_per_axis']}/axis)")

    # 2) localize on the image
    img01 = denormalize(x)[0].permute(1, 2, 0).cpu().numpy()
    plot_interactions(res, img01, "hessianig_interactions.png")
    print("\nwrote hessianig_interactions.png")

    netmap = per_cell_net_interaction(res)
    print(f"max cooperation (cell net) = {netmap.max():+.5f}")
    print(f"max redundancy  (cell net) = {netmap.min():+.5f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()