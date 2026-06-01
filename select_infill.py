"""Select the best global infill for a fixed classifier — GPU-resident, batched.

Measures TWO independent axes (do not collapse them into one scalar):

  1. OOD  : k-NN distance in penultimate-feature space to a small calibration
            set of real images. Lower = the composite's features look like real
            images = on the model's familiar regime (on-R_f).
            -> k-NN, NOT a Gaussian: no covariance to estimate, no N<<D
               singularity, captures manifold shape with only ~50 points.
            -> NOT energy: a class-neutral infill (the goal) has flat logits,
               which an energy detector misreads as OOD. Energy conflates
               off-manifold with class-ambiguous, so it is the wrong signal here.

  2. NEUTRALITY : softmax entropy of the raw reference (high = flat logits =
                  carries no class signal). This is a SEPARATE axis you want
                  HIGH, the opposite of what energy would reward.

Selection objective (higher = better):
    score = - w_ood     * ood_dist        # feature k-NN distance, lower better
            + w_neutral  * entropy          # flat logits, higher better
            + w_removal  * removal          # f_x - f_b, higher better
with an optional hard f_b ceiling (--neutral_cap).

Everything (features, k-NN distances, masks, composites) stays on the GPU.
References are built batched; masks are stacked into one forward per batch.
Classifier-only: no autoencoder / flow / diffusion anywhere.

Usage:
    python select_infill.py --calib ./benchmark_50 \
        --candidates blur black white noise corners --sigma 50

Adjust get_feature_layer() for your backbone (ResNet-50: avgpool; ViT: pre-head).
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
def fwd_feats_logits(model, xb, hook):
    """Batched forward -> (feats (B,D), logits (B,C)) — both GPU tensors."""
    logits = model(xb)
    return hook.feat.flatten(1), logits


# --------------------------------------------------------------------------- #
# Batched infill construction (GPU, no per-image loop)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def make_reference_batched(xb_norm, mode="blur", sigma=50.0, gen=None):
    """Build references for a whole batch at once. xb_norm: (B,3,H,W) normalized."""
    x01 = denormalize(xb_norm)                           # (B,3,H,W) in [0,1]
    if mode == "blur":
        b01 = gaussian_blur(x01, sigma).clamp(0, 1)
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
# Calibration band: k-NN feature distance (GPU)
# --------------------------------------------------------------------------- #
@dataclass
class KNNBand:
    feats: torch.Tensor         # (N,D) calibration features (L2-normalized)
    k: int
    d_med: float                # median LOO k-NN distance on calib (for z-norm)
    d_mad: float
    tau: float                  # q-quantile LOO distance (pass/fail threshold)

    def _knn_dist(self, q, exclude_self=False):
        """Mean distance from each query (B,D) to its k nearest calib features."""
        q = F.normalize(q, dim=1)
        d = torch.cdist(q, self.feats)                   # (B,N) euclidean
        kk = self.k + (1 if exclude_self else 0)
        kk = min(kk, d.shape[1])
        vals, _ = d.topk(kk, dim=1, largest=False)       # (B,kk)
        if exclude_self:
            vals = vals[:, 1:]                            # drop self (dist 0)
        return vals.mean(dim=1)                           # (B,)

    def ood_dist(self, q):
        """Robust-z of k-NN distance. ~0 = as in-distribution as a calib image."""
        d = self._knn_dist(q, exclude_self=False)
        return (d - self.d_med) / self.d_mad

    def passes(self, q):
        return self._knn_dist(q, exclude_self=False) <= self.tau


@torch.no_grad()
def calibrate_band(model, calib_batches, hook, k=5, quantile=0.95):
    feats = []
    for xb in calib_batches:
        f, _ = fwd_feats_logits(model, xb, hook)
        feats.append(f)
    F_ = torch.cat(feats, 0)                             # (N,D)
    F_ = F.normalize(F_, dim=1)                          # cosine-geometry kNN
    N, D = F_.shape
    k = min(k, max(1, N - 1))

    band = KNNBand(F_, k, 0.0, 1.0, 0.0)
    # leave-one-out: each calib point's distance to its k nearest OTHERS
    d_loo = band._knn_dist(F_, exclude_self=True)        # (N,)
    band.d_med = float(d_loo.median())
    band.d_mad = float((d_loo - d_loo.median()).abs().median() + 1e-6)
    band.tau = float(d_loo.quantile(quantile))
    print(f"[calib] N={N} D={D} k={k}  kNN: d_med={band.d_med:.4f} "
          f"tau={band.tau:.4f} (q{quantile})")
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
    return batches


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
    ood_dist: float             # mean k-NN OOD distance (LOWER better)
    pass_rate: float            # fraction of composites inside the band
    f_b: float                  # target prob on reference (lower better)
    entropy: float              # softmax entropy of reference (HIGHER = neutral)
    removal: float              # f_x - f_b (higher better)
    score: float = 0.0


@torch.no_grad()
def score_candidate(model, band, hook, calib_batches, mode, device,
                    sigma, n_masks, p_keep, gen):
    ood_sum = 0.0
    pass_n = comp_n = 0
    fb_sum = ent_sum = rem_sum = 0.0
    n_img = 0
    for xb in calib_batches:
        B = xb.shape[0]; n_img += B
        feats_x, logits = fwd_feats_logits(model, xb, hook)
        tgt = logits.argmax(1)                           # (B,)
        f_x = F.softmax(logits, 1).gather(1, tgt[:, None]).squeeze(1)

        refs = make_reference_batched(xb, mode=mode, sigma=sigma, gen=gen)
        _, rlogits = fwd_feats_logits(model, refs, hook)
        pr = F.softmax(rlogits, 1)
        f_b = pr.gather(1, tgt[:, None]).squeeze(1)
        ent = -(pr * (pr + 1e-12).log()).sum(1)
        fb_sum += f_b.sum().item()
        ent_sum += ent.sum().item()
        rem_sum += (f_x - f_b).sum().item()

        # all masks in one stacked batch -> single forward per image-batch
        m = sample_masks(n_masks * B, xb.shape, p_keep, gen, device)
        xrep = xb.repeat(n_masks, 1, 1, 1)
        rrep = refs.repeat(n_masks, 1, 1, 1)
        xp = m * xrep + (1 - m) * rrep
        fcomp, _ = fwd_feats_logits(model, xp, hook)
        ood_sum += band.ood_dist(fcomp).sum().item()     # continuous, key signal
        pass_n += int(band.passes(fcomp).sum().item())
        comp_n += xp.shape[0]

    return CandidateScore(
        mode=mode,
        ood_dist=ood_sum / max(comp_n, 1),
        pass_rate=pass_n / max(comp_n, 1),
        f_b=fb_sum / n_img, entropy=ent_sum / n_img, removal=rem_sum / n_img,
    )


@torch.no_grad()
def get_infill_ind(model, band, hook, calib_batches, candidates, device,
                   sigma=50.0, n_masks=8, p_keep=0.5, seed=0,
                   w_ood=1.0, w_neutral=0.2, w_removal=0.5,
                   neutral_cap=None, verbose=True):
    """Score each candidate infill, return (best_index, ranked_scores).

    score = - w_ood*ood_dist + w_neutral*entropy + w_removal*removal
      ood_dist : k-NN feature distance (lower better) -> enters with minus
      entropy  : reference logit flatness (higher = more neutral) -> plus
      removal  : f_x - f_b (higher better) -> plus
    neutral_cap: hard f_b ceiling; candidates above it are disqualified first.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = []
    for mode in candidates:
        s = score_candidate(model, band, hook, calib_batches, mode, device,
                             sigma, n_masks, p_keep, gen)
        s.score = (-w_ood * s.ood_dist
                   + w_neutral * s.entropy
                   + w_removal * s.removal)
        scores.append(s)
        if verbose:
            print(f"[score] {mode:8s} ood={s.ood_dist:+7.3f} "
                  f"pass={s.pass_rate:5.1%} f_b={s.f_b:6.3f} "
                  f"H={s.entropy:6.3f} removal={s.removal:+6.3f} "
                  f"-> {s.score:+.4f}")

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
    ap.add_argument("--k", type=int, default=5, help="k for k-NN OOD distance")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--w_ood", type=float, default=1.0)
    ap.add_argument("--w_neutral", type=float, default=0.2)
    ap.add_argument("--w_removal", type=float, default=0.5)
    ap.add_argument("--neutral_cap", type=float, default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device == "cpu":
        print("[warn] running on CPU — no GPU detected.")

    model = load_backbone(args.device).eval()
    hook = _FeatHook(get_feature_layer(model))

    calib = load_calib_batches(args.calib, args.device,
                               args.batch_size, args.pattern)
    band = calibrate_band(model, calib, hook, k=args.k, quantile=args.quantile)

    best, scores = get_infill_ind(
        model, band, hook, calib, args.candidates, args.device,
        sigma=args.sigma, n_masks=args.n_masks, p_keep=args.p_keep,
        w_ood=args.w_ood, w_neutral=args.w_neutral, w_removal=args.w_removal,
        neutral_cap=args.neutral_cap,
    )
    hook.remove()
    print(f"\nBest infill: {scores[best].mode}")


if __name__ == "__main__":
    main()