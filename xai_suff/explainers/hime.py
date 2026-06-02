"""HIME: High-order Interaction Model Explanations.

Extends grid LIME to degree-2 interactions between grid cells.

Pipeline (matches the verified support-recovery theory):
  1. SCREEN  -- fit main effects; keep cells with non-negligible effect
                (hierarchy assumption: a true pair has detectable main effects,
                 so screening on mains is lossless for cooperative / non-XOR f).
  2. RECOVER -- LASSO + stability selection over [mains + candidate pairs]
                restricted to screened cells. Surviving pair coefficients are
                the estimated high-order interactions. Sample complexity
                N >~ C * s * log p / gamma^2 (verified ~ C in [2,3]).
  3. SOLVE   -- weighted re-fit on the selected support (mains + pairs) for
                clean, de-biased coefficients.

Per-cell main effects paint a blocky 2D map (as in LIME). Pairwise
interactions cannot be a pixel value, so they are returned in
extras["interactions"] as (cell_i, cell_j, strength) triples.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LassoLars

from .base import AttributionResult, Explainer, blur_reference


class HIMEExplainer(Explainer):
    name = "hime"

    def __init__(self, *args, grid=(12, 12), n_samples=2500, sigma=11.0,
                 kernel_width=0.25, seed=0, batch_size=64,
                 screen_quantile=0.6, max_active_cells=40,
                 stability_runs=20, stability_thresh=0.8,
                 lasso_lambda_scale=1.0, **kw):
        """
        screen_quantile   : keep cells whose |main effect| is above this quantile.
        max_active_cells  : hard cap m' on screened cells (controls p = m'+C(m',2)).
        stability_runs     : bootstrap resamples for stability selection.
        stability_thresh   : keep a pair if selected in >= this fraction of runs.
        lasso_lambda_scale : multiplier on lambda ~ sigma*sqrt(2 log p / N).
        """
        super().__init__(*args, **kw)
        self.grid = grid
        self.n_samples = n_samples
        self.sigma = sigma
        self.kernel_width = kernel_width
        self.seed = seed
        self.batch_size = batch_size
        self.screen_quantile = screen_quantile
        self.max_active_cells = max_active_cells
        self.stability_runs = stability_runs
        self.stability_thresh = stability_thresh
        self.lasso_lambda_scale = lasso_lambda_scale

    # ---- identical cell map to LIME ----
    def _cell_id_map(self, H, W):
        gh, gw = self.grid
        ys = (torch.arange(H) * gh // H).clamp(max=gh - 1)
        xs = (torch.arange(W) * gw // W).clamp(max=gw - 1)
        return ys.view(-1, 1) * gw + xs.view(1, -1)

    # ---- query f on a batch of on/off cell vectors ----
    @torch.no_grad()
    def _query(self, Z, x, b, cell_ids, target):
        probs = np.zeros(Z.shape[0], dtype=np.float64)
        for start in range(0, Z.shape[0], self.batch_size):
            zb = Z[start:start + self.batch_size].to(self.device)
            keep = zb[:, cell_ids].unsqueeze(1)            # (B,1,H,W)
            comp = keep * x + (1 - keep) * b
            p = F.softmax(self.model(comp), dim=1)[:, target]
            probs[start:start + zb.shape[0]] = p.detach().cpu().numpy()
        return probs

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        _, _, H, W = x.shape
        b = blur_reference(x, self.sigma)

        cell_ids = self._cell_id_map(H, W).to(self.device)
        n_cells = self.grid[0] * self.grid[1]

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        # Rademacher-style on/off design (Bernoulli-1/2), all-on baseline first.
        Z = (torch.rand(self.n_samples, n_cells, generator=g) > 0.5).float()
        Z[0] = 1.0
        probs = self._query(Z, x, b, cell_ids, target)
        Znp = Z.cpu().numpy()

        # distance-based weights, same kernel as LIME (used in final solve).
        all_on = np.ones(n_cells)
        d = 1.0 - (Znp @ all_on) / (
            np.linalg.norm(Znp, axis=1) * np.linalg.norm(all_on) + 1e-12)
        weights = np.exp(-(d ** 2) / (self.kernel_width ** 2))

        # ---------- STAGE 1: screen main effects ----------
        # cheap weighted-ridge main-effect fit -> rank cells.
        main_coefs = _weighted_ridge(Znp, probs, weights, alpha=1.0)
        mag = np.abs(main_coefs)
        thr = np.quantile(mag, self.screen_quantile)
        active = np.where(mag >= thr)[0]
        # cap to max_active_cells by magnitude (controls candidate dimension p)
        if active.size > self.max_active_cells:
            active = active[np.argsort(mag[active])[::-1][:self.max_active_cells]]
        active = np.sort(active)
        pair_list = list(combinations(active.tolist(), 2))

        # ---------- STAGE 2: LASSO support recovery + stability selection ----------
        # design over [active mains | candidate pairs]; center to absorb intercept.
        Zc = (Znp - 0.5)                       # +/-0.5 coding, zero-mean features
        main_block = Zc[:, active]             # (N, m')
        pair_block = np.empty((self.n_samples, len(pair_list)))
        for c, (i, j) in enumerate(pair_list):
            pair_block[:, c] = Zc[:, i] * Zc[:, j]
        X = np.hstack([main_block, pair_block])
        m_act = active.size
        p_dim = X.shape[1]
        y = probs - probs.mean()

        sigma_hat = max(np.std(y), 1e-6)
        lam = self.lasso_lambda_scale * sigma_hat * np.sqrt(2 * np.log(p_dim) / self.n_samples)

        rng = np.random.default_rng(self.seed)
        sel_count = np.zeros(p_dim)
        for _ in range(self.stability_runs):
            idx = rng.choice(self.n_samples, self.n_samples, replace=True)
            m = LassoLars(alpha=lam, fit_intercept=True, max_iter=4000)
            m.fit(X[idx], y[idx])
            sel_count += (np.abs(m.coef_) > 0).astype(float)
        stab = sel_count / self.stability_runs
        selected = stab >= self.stability_thresh

        # ---------- STAGE 3: re-solve on selected support ----------
        sel_idx = np.where(selected)[0]
        if sel_idx.size:
            Xs = X[:, sel_idx]
            beta = _weighted_ridge(Xs, y, weights, alpha=1.0)
        else:
            beta = np.zeros(0)

        # split recovered coefficients back into mains and pairs
        main_effect = np.zeros(n_cells)
        interactions = []
        for k, gi in enumerate(sel_idx):
            coef = beta[k]
            if gi < m_act:                       # main effect column
                main_effect[active[gi]] = coef
            else:                                # pair column
                ci, cj = pair_list[gi - m_act]
                interactions.append((int(ci), int(cj), float(coef)))
        interactions.sort(key=lambda t: -abs(t[2]))

        # paint first-order map (blocky, like LIME)
        coef_t = torch.tensor(main_effect, dtype=torch.float32, device=self.device)
        attr = coef_t[cell_ids].cpu().numpy()

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=float(self._probs(x)[0, target]),
            extras={
                "grid": self.grid,
                "n_samples": self.n_samples,
                "n_active_cells": int(m_act),
                "candidate_pairs": len(pair_list),
                "candidate_dim_p": int(p_dim),
                "lambda": float(lam),
                "interactions": interactions,          # (cell_i, cell_j, strength)
                "interaction_stability": {
                    f"{pair_list[gi - m_act][0]}-{pair_list[gi - m_act][1]}": float(stab[gi])
                    for gi in sel_idx if gi >= m_act
                },
            },
        )


def _weighted_ridge(Z, y, w, alpha=1.0):
    """Closed-form weighted ridge; returns per-feature coefficients (no intercept)."""
    n, d = Z.shape
    Zb = np.concatenate([Z, np.ones((n, 1))], axis=1)
    Wd = w[:, None]
    A = Zb.T @ (Wd * Zb)
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0
    A += reg
    b = Zb.T @ (w * y)
    sol = np.linalg.solve(A, b)
    return sol[:-1]