"""Select the best global infill for a fixed classifier — GPU-resident, batched.

Everything (features, energy, Mahalanobis, masks, composites) stays on the GPU.
No per-image Python loops in the hot path; references are built batched.

Usage:
    python select_infill.py --calib ./benchmark_50 \
        --candidates blur black white noise corners --sigma 50

Classifier-only: no autoencoder / flow / diffusion anywhere.
Adjust get_feature_layer() and make_reference_batched() for your backbone.
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from xai_suff.backbone import load_backbone, load_image, denormalize, normalize_pixel
from xai_suff.explainers.base import gaussian_blur   # reuse the separable blur


# --------------------------------------------------------------------------- #
# Feature hook
# --------------------------------------------------------------------------- #
def get_feature_layer(model):
    """ResNet-50: model.avgpool. ViT: pre-head norm. Edit for your backbone."""
    if hasattr(model, "avgpool"):
        return model.avgpool
    raise ValueError("Set get_feature_layer() for this backbone.")


class _FeatHook:
    def __init__(self, module):
        self.feat = None
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        z = out
        if z.dim() == 4:        z = z.mean(dim=(2, 3))   # CNN GAP
        elif z.dim() == 3:      z = z.mean(dim=1)        # ViT token mean
        self.feat = z.detach()

    def remove(self):
        self._h.remove()


@torch.no_grad()
def fwd_feats_energy(model, xb, hook):
    """Batched forward -> (feats (B,D), energy (B,)) — both GPU tensors."""
    logits = model(xb)
    energy = -torch.logsumexp(logits, dim=1)             # (B,)
    return hook.feat.flatten(1), energy                  # stay on device


# --------------------------------------------------------------------------- #
# Batched infill construction (GPU, no per-image loop)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def make_reference_batched(xb_norm, mode="blur", sigma=50.0, gen=None):
    """Build references for a whole batch at once. xb_norm: (B,3,H,W) normalized."""
    x01 = denormalize(xb_norm)                           # (B,3,H,W) in [0,1]
    if mode == "blur":
        b01 = gaussian_blur(x01, sigma).clamp(0, 1)      # separable, batched
    elif mode == "black":
        b01 = torch.zeros_like(x01)
    elif mode == "white":
        b01 = torch.ones_like(x01)
    elif mode == "noise":
        b01 = torch.rand(x01.shape, generator=gen, device=x01.device,
                         dtype=x01.dtype)
    elif mode == "corners":
        H, W = x01.shape[-2:]
        ph, pw = max(1, round(0.10 * H)), max(1, round(0.10 * W))
        patches = torch.cat([x01[..., :ph, :pw],  x01[..., :ph, -pw:],
                             x01[..., -ph:, :pw], x01[..., -ph:, -pw:]], dim=-1)
        mean_c = patches.mean(dim=(-2, -1)).view(x01.shape[0], -1, 1, 1)
        b01 = mean_c.expand_as(x01).clamp(0, 1)
    else:
        raise ValueError(f"unknown infill mode {mode!r}")
    return normalize_pixel(b01)


# --------------------------------------------------------------------------- #
# Calibration band (GPU tensors throughout)
# --------------------------------------------------------------------------- #
@dataclass
class ManifoldBand:
    mu: torch.Tensor            # (D,)
    prec: torch.Tensor          # (D,D) Sigma^{-1}, or (D,) if diagonal
    tau_M: torch.Tensor         # scalar
    tau_E: torch.Tensor         # scalar
    diag: bool = False

    def maha(self, F_):         # F_: (B,D) -> (B,)
        d = F_ - self.mu
        if self.diag:
            return (d * self.prec * d).sum(1)
        return torch.einsum("bd,de,be->b", d, self.prec, d)

    def passes(self, F_, E_):   # -> (B,) bool
        return (self.maha(F_) <= self.tau_M) & (E_ <= self.tau_E)


@torch.no_grad()
def _shrinkage_precision(F_, eps=1e-5):
    """Ledoit-Wolf-style shrinkage precision on GPU (no sklearn / no CPU copy).
    Shrinks empirical covariance toward a scaled identity, then inverts."""
    N, D = F_.shape
    mu = F_.mean(0)
    Xc = F_ - mu
    cov = (Xc.t() @ Xc) / max(N - 1, 1)                  # (D,D)
    alpha = 0.0
    I = torch.eye(D, device=F_.device, dtype=F_.dtype)
    if N <= D:
        mu_t = cov.diagonal().mean()
        # closed-form LW shrinkage intensity (batched, on GPU)
        var_cov = (((Xc.unsqueeze(2) * Xc.unsqueeze(1)) ** 2).mean(0).sum()) / N
        denom = ((cov - mu_t * I) ** 2).sum()
        alpha = float(torch.clamp(var_cov / (denom + eps), 0.0, 1.0))
        cov = (1 - alpha) * cov + alpha * mu_t * I
    cov = cov + eps * I
    return mu, torch.linalg.inv(cov), alpha


@torch.no_grad()
def calibrate_band(model, calib_batches, hook, quantile=0.95, force_diag=False):
    feats, energies = [], []
    for xb in calib_batches:
        f, e = fwd_feats_energy(model, xb, hook)
        feats.append(f); energies.append(e)
    F_ = torch.cat(feats, 0)                             # (N,D) on GPU
    E_ = torch.cat(energies, 0)                          # (N,)
    N, D = F_.shape

    if force_diag or N < 8:
        mu = F_.mean(0)
        prec = 1.0 / (F_.var(0) + 1e-5)                  # (D,) diagonal
        diag = True
        print(f"[calib] diagonal covariance (N={N}, D={D})")
    else:
        mu, prec, alpha = _shrinkage_precision(F_)
        diag = False
        print(f"[calib] shrinkage cov alpha={alpha:.3f} (N={N}, D={D})")

    band = ManifoldBand(mu, prec, torch.tensor(0., device=F_.device),
                        torch.tensor(0., device=F_.device), diag)
    M_cal = band.maha(F_)                                # in-sample (see note)
    band.tau_M = torch.quantile(M_cal, quantile)
    band.tau_E = torch.quantile(E_, quantile)
    print(f"[calib] tau_M={band.tau_M:.2f} tau_E={band.tau_E:.3f} (q{quantile})")
    return band


# --------------------------------------------------------------------------- #
# Data (loaded once, kept on GPU)
# --------------------------------------------------------------------------- #
def load_calib_batches(folder, device, batch_size=16, pattern="*.JPEG"):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"no {pattern} in {folder}")
    print(f"[data] {len(paths)} images")
    batches, buf = [], []
    for p in paths:
        buf.append(load_image(p, device))                # (1,3,H,W) on device
        if len(buf) == batch_size:
            batches.append(torch.cat(buf, 0)); buf = []
    if buf:
        batches.append(torch.cat(buf, 0))
    return batches                                       # list of GPU tensors


@torch.no_grad()
def sample_masks(n, shape, p_keep, gen, device):
    _, _, H, W = shape
    return (torch.rand(n, 1, H, W, generator=gen, device=device) < p_keep).float()


# --------------------------------------------------------------------------- #
# Candidate scoring + selection
# --------------------------------------------------------------------------- #
@dataclass
class CandidateScore:
    mode: str
    on_manifold: float
    f_b: float
    entropy: float
    removal: float
    score: float = 0.0


@torch.no_grad()
def score_candidate(model, band, hook, calib_batches, mode, device,
                    sigma, n_masks, p_keep, gen):
    pass_n = tot = 0
    fb_sum = ent_sum = rem_sum = 0.0
    n_img = 0
    for xb in calib_batches:
        B = xb.shape[0]; n_img += B
        logits = model(xb)
        tgt = logits.argmax(1)                           # (B,)
        f_x = F.softmax(logits, 1).gather(1, tgt[:, None]).squeeze(1)

        refs = make_reference_batched(xb, mode=mode, sigma=sigma, gen=gen)
        pr = F.softmax(model(refs), 1)
        f_b = pr.gather(1, tgt[:, None]).squeeze(1)
        ent = -(pr * (pr + 1e-12).log()).sum(1)
        fb_sum += f_b.sum().item()
        ent_sum += ent.sum().item()
        rem_sum += (f_x - f_b).sum().item()

        # stack all masks into one big batch -> single forward per image-batch
        m = sample_masks(n_masks * B, xb.shape, p_keep, gen, device)  # (n*B,1,H,W)
        xrep = xb.repeat(n_masks, 1, 1, 1)               # (n*B,3,H,W)
        rrep = refs.repeat(n_masks, 1, 1, 1)
        xp = m * xrep + (1 - m) * rrep
        f, e = fwd_feats_energy(model, xp, hook)
        pass_n += int(band.passes(f, e).sum().item())
        tot += xp.shape[0]

    return CandidateScore(mode, pass_n / max(tot, 1),
                          fb_sum / n_img, ent_sum / n_img, rem_sum / n_img)


@torch.no_grad()
def get_infill_ind(model, band, hook, calib_batches, candidates, device,
                   sigma=50.0, n_masks=8, p_keep=0.5, seed=0,
                   w_manifold=1.0, w_neutral=1.0, w_removal=0.5,
                   neutral_cap=None, verbose=True):
    """Score each candidate infill, return (best_index, ranked_scores).

    score = w_manifold*on_manifold - w_neutral*f_b + w_removal*removal
    neutral_cap: hard f_b ceiling; candidates above it are disqualified first.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = []
    for mode in candidates:
        s = score_candidate(model, band, hook, calib_batches, mode, device,
                             sigma, n_masks, p_keep, gen)
        s.score = w_manifold * s.on_manifold - w_neutral * s.f_b \
                  + w_removal * s.removal
        scores.append(s)
        if verbose:
            print(f"[score] {mode:8s} on_manifold={s.on_manifold:6.2%} "
                  f"f_b={s.f_b:6.3f} H={s.entropy:6.3f} "
                  f"removal={s.removal:+6.3f} -> {s.score:+.4f}")

    elig = list(range(len(scores)))
    if neutral_cap is not None:
        kept = [i for i in elig if scores[i].f_b <= neutral_cap]
        elig = kept if kept else elig
        if not kept:
            print(f"[select] none under neutral_cap={neutral_cap}; ignoring it.")
    best = max(elig, key=lambda i: scores[i].score)
    if verbose:
        print(f"[select] best = '{scores[best].mode}' (idx {best}, "
              f"score={scores[best].score:+.4f})")
    return best, scores


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="./benchmark_50")
    ap.add_argument("--pattern", default="*.JPEG")
    ap.add_argument("--candidates", nargs="+",
                    default=["blur", "black", "white", "noise", "corners"])
    ap.add_argument("--sigma", type=float, default=50.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_masks", type=int, default=8)
    ap.add_argument("--p_keep", type=float, default=0.5)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--w_manifold", type=float, default=1.0)
    ap.add_argument("--w_neutral", type=float, default=1.0)
    ap.add_argument("--w_removal", type=float, default=0.5)
    ap.add_argument("--neutral_cap", type=float, default=None)
    ap.add_argument("--force_diag", action="store_true",
                    help="force diagonal covariance (fastest, crudest)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device == "cpu":
        print("[warn] running on CPU — no GPU detected.")

    model = load_backbone(args.device).eval()
    hook = _FeatHook(get_feature_layer(model))

    calib = load_calib_batches(args.calib, args.device,
                               args.batch_size, args.pattern)
    band = calibrate_band(model, calib, hook, args.quantile, args.force_diag)

    best, scores = get_infill_ind(
        model, band, hook, calib, args.candidates, args.device,
        sigma=args.sigma, n_masks=args.n_masks, p_keep=args.p_keep,
        w_manifold=args.w_manifold, w_neutral=args.w_neutral,
        w_removal=args.w_removal, neutral_cap=args.neutral_cap,
    )
    hook.remove()
    print(f"\nBest infill: {scores[best].mode}")


if __name__ == "__main__":
    main()