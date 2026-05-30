"""LIME for images, grid-superpixel variant (default 16x16).

Perturbs the image by turning grid cells "on" (kept) or "off" (replaced by the
blur reference, so perturbations stay on-manifold rather than gray-filled),
fits a weighted linear surrogate of the target probability on the binary on/off
vectors, and paints each cell with its surrogate coefficient.

Per-segment constants (the known LIME limitation) -> blocky map by design.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import AttributionResult, Explainer, blur_reference


class LIMEExplainer(Explainer):
    name = "lime"

    def __init__(self, *args, grid=(12, 12), n_samples=1000, sigma=11.0,
                 kernel_width=0.25, seed=0, batch_size=64, **kw):
        super().__init__(*args, **kw)
        self.grid = grid
        self.n_samples = n_samples
        self.sigma = sigma
        self.kernel_width = kernel_width
        self.seed = seed
        self.batch_size = batch_size

    def _cell_id_map(self, H, W):
        gh, gw = self.grid
        ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
        xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
        ids = ys.view(-1, 1) * gw + xs.view(1, -1)  # (H,W) in [0, gh*gw)
        return ids

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        _, _, H, W = x.shape
        b = blur_reference(x, self.sigma)

        cell_ids = self._cell_id_map(H, W).to(self.device)  # (H,W)
        n_cells = self.grid[0] * self.grid[1]

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        # binary "interpretable" features: 1 = keep sharp, 0 = blur
        Z = (torch.rand(self.n_samples, n_cells, generator=g) > 0.5).float()
        Z[0] = 1.0  # include the all-on sample (full image)

        probs = np.zeros(self.n_samples, dtype=np.float64)
        for start in range(0, self.n_samples, self.batch_size):
            zb = Z[start:start + self.batch_size].to(self.device)  # (B,n_cells)
            # build per-pixel keep masks from cell vectors
            # gather cell value at each pixel: (B,H,W)
            keep = zb[:, cell_ids]  # (B,H,W)
            keep = keep.unsqueeze(1)  # (B,1,H,W)
            comp = keep * x + (1 - keep) * b  # broadcast over batch
            p = F.softmax(self.model(comp), dim=1)[:, target]
            probs[start:start + zb.shape[0]] = p.detach().cpu().numpy()

        # distance-based sample weights (cosine distance to all-on vector)
        all_on = np.ones(n_cells)
        Znp = Z.cpu().numpy()
        d = 1.0 - (Znp @ all_on) / (
            np.linalg.norm(Znp, axis=1) * np.linalg.norm(all_on) + 1e-12
        )
        weights = np.exp(-(d ** 2) / (self.kernel_width ** 2))

        # weighted ridge regression: probs ~ Znp @ w + b0
        coefs = _weighted_ridge(Znp, probs, weights, alpha=1.0)

        # paint coefficients back to pixels
        coef_t = torch.tensor(coefs, dtype=torch.float32, device=self.device)
        attr = coef_t[cell_ids].cpu().numpy()  # (H,W)

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=float(self._probs(x)[0, target]),
            extras={"grid": self.grid, "n_samples": self.n_samples},
        )


def _weighted_ridge(Z, y, w, alpha=1.0):
    """Closed-form weighted ridge; returns per-feature coefficients (no intercept term returned)."""
    n, d = Z.shape
    Zb = np.concatenate([Z, np.ones((n, 1))], axis=1)  # add intercept
    Wd = w[:, None]
    A = Zb.T @ (Wd * Zb)
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0  # don't regularize intercept
    A += reg
    b = Zb.T @ (w * y)
    sol = np.linalg.solve(A, b)
    return sol[:-1]  # drop intercept