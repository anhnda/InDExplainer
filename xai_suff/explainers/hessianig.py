"""Regional second-order Integrated Hessian (Hessian-IG) on a grid of cells.

First order (for parity with the rest of the suite)
---------------------------------------------------
Same blur-baseline IG as ``IGExplainer``: a (H,W) attribution map

    IG(x) = (x - b) * integral_0^1 grad f(b + a (x - b)) da

so this method drops into the existing first-order faithfulness table.

Second order (the point of this file)
--------------------------------------
We want a *regional* interaction between grid cells, comparable cell-for-cell to
the model-actual matrix produced by ``M.pairwise_interaction_matrix(..., k)``.

Partition the image into K = k*k axis-aligned cells. Define a per-cell scalar
gate g in R^K, with g = 1 meaning "cell fully sharp", g = 0 meaning "cell fully
blurred", and the composite input

    X(g) = sum_c g_c * (cell_c of x) + (1 - g_c) * (cell_c of b)
         = b + sum_c g_c * (x - b)|_cell_c

This is exactly the blur-completion family restricted to cell granularity, so it
shares Phi with the pyramid / pairwise referee (same removal operator => the
comparison is non-circular only against the *reveal* matrix, see note below).

The Integrated Hessian interaction between cells i, j (Janizek et al. 2021,
"Explaining Explanations: Axiomatic Feature Interactions") is

    Phi_{ij} = (g_i^* - g_i^0)(g_j^* - g_j^0)
               * integral_0^1 a * d^2 f(X(g^0 + a(g^* - g^0))) / dg_i dg_j  da

with g^0 = 0 (all blurred = baseline b) and g^* = 1 (all sharp = x). Since the
gate endpoints are 0 -> 1, the prefactor is 1 and

    Phi_{ij} = integral_0^1 a * (d^2 f / dg_i dg_j)|_{g = a*1}  da

The diagonal Phi_{ii} is the cell's *self* (curvature) term; off-diagonal
Phi_{ij} (i != j) is the pairwise interaction. This matrix is directly
comparable to the reveal-based ground-truth matrix on the same k x k grid.

Exactness
---------
The mixed partials d^2 f / dg_i dg_j are computed by double autograd (grad of
grad), giving the full K x K Hessian of f w.r.t. the K cell gates at each
interpolation point -- exact, not Hutchinson/finite-difference. For k = 14,
K = 196: one Hessian is 196 x 196, obtained with K backward passes per
integration step (one per row of the Hessian). That is the documented cost.

Symmetry: the Hessian is symmetrized (H + H^T)/2 to wash out autograd asymmetry.

Reference dependence (b = blur_sigma(x), grid k) is logged in ``extras``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import AttributionResult, Explainer, blur_reference


class HessianIGExplainer(Explainer):
    name = "hessian_ig"

    def __init__(
        self,
        *args,
        steps: int = 32,          # Riemann steps for the IG path (first order)
        hess_steps: int = 16,     # Riemann steps for the integrated Hessian
        sigma: float = 11.0,      # blur strength of the reference b (shared Phi)
        k: int = 14,              # grid side; K = k*k cells for the interaction
        batch_size: int = 16,     # batching for the first-order IG path
        **kw,
    ):
        super().__init__(*args, **kw)
        self.steps = steps
        self.hess_steps = hess_steps
        self.sigma = sigma
        self.k = k
        self.batch_size = batch_size
        # bookkeeping for query_cost-style reporting
        self._n_forward = 0
        self._n_backward = 0

    # ------------------------------------------------------------------ #
    # cell geometry
    # ------------------------------------------------------------------ #
    def _cell_slices(self, H: int, W: int):
        """Row/col boundaries for a k x k near-uniform partition of (H,W)."""
        ys = np.linspace(0, H, self.k + 1).round().astype(int)
        xs = np.linspace(0, W, self.k + 1).round().astype(int)
        slices = []
        for r in range(self.k):
            for c in range(self.k):
                slices.append((slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1])))
        return slices  # length K, row-major (matches pairwise_interaction_matrix)

    def _gate_field(self, g: torch.Tensor, slices, H: int, W: int) -> torch.Tensor:
        """Broadcast a length-K gate vector g to a (1,1,H,W) per-pixel field."""
        field = torch.zeros((1, 1, H, W), dtype=g.dtype, device=g.device)
        for idx, (rs, cs) in enumerate(slices):
            field[..., rs, cs] = g[idx]
        return field

    # ------------------------------------------------------------------ #
    # first-order IG map (same recipe as IGExplainer, for table parity)
    # ------------------------------------------------------------------ #
    def _first_order_ig(self, x, b, target):
        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        grad_accum = torch.zeros_like(x)
        for start in range(0, self.steps, self.batch_size):
            a = alphas[start:start + self.batch_size].view(-1, 1, 1, 1)
            interp = (b + a * (x - b)).clone().requires_grad_(True)
            logits = self.model(interp)
            self._n_forward += interp.shape[0]
            score = F.log_softmax(logits, dim=1)[:, target].sum()
            grad = torch.autograd.grad(score, interp)[0]
            self._n_backward += interp.shape[0]
            grad_accum += grad.sum(dim=0, keepdim=True)
        avg_grad = grad_accum / self.steps
        ig = (x - b) * avg_grad
        return ig.sum(dim=1).squeeze(0).detach().cpu().numpy()  # (H,W)

    # ------------------------------------------------------------------ #
    # integrated cell-pair Hessian (exact, double autograd)
    # ------------------------------------------------------------------ #
    def _integrated_hessian(self, x, b, target, slices, H, W) -> np.ndarray:
        """Return the K x K integrated-Hessian interaction matrix Phi_{ij}.

        Phi_{ij} = integral_0^1 a * d^2 f(X(a*1)) / dg_i dg_j  da,   g in [0,1]^K.
        Riemann sum with midpoint-ish nodes over hess_steps; endpoints 0->1.
        """
        K = len(slices)
        delta = (x - b)  # (1,3,H,W); cell_c contribution is delta restricted there
        Phi = torch.zeros((K, K), dtype=torch.float64, device=self.device)

        alphas = (torch.arange(self.hess_steps, device=self.device) + 0.5) / self.hess_steps
        dα = 1.0 / self.hess_steps

        for a in alphas:
            # gate vector g = a * 1  (all cells partially revealed by fraction a)
            g = torch.full((K,), float(a), device=self.device, requires_grad=True)
            field = self._gate_field(g, slices, H, W)         # (1,1,H,W)
            X = b + field * delta                              # composite input
            logits = self.model(X)
            self._n_forward += 1
            score = F.log_softmax(logits, dim=1)[:, target]    # scalar (batch 1)

            # first derivatives df/dg  (length K), create graph for 2nd order
            grads = torch.autograd.grad(score, g, create_graph=True)[0]  # (K,)
            self._n_backward += 1

            # full Hessian row by row: d/dg_i (df/dg_j) -> K backward passes
            rows = []
            for i in range(K):
                row_i = torch.autograd.grad(
                    grads[i], g, retain_graph=True
                )[0]  # (K,)
                self._n_backward += 1
                rows.append(row_i.detach().to(torch.float64))
            Hmat = torch.stack(rows, dim=0)                    # (K,K)
            Hmat = 0.5 * (Hmat + Hmat.t())                     # symmetrize

            Phi += float(a) * Hmat * dα

        return Phi.cpu().numpy()

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        b = blur_reference(x, self.sigma).to(self.device)

        _, _, H, W = x.shape
        slices = self._cell_slices(H, W)
        K = len(slices)

        # first-order map (drops into the existing faithfulness table)
        attr = self._first_order_ig(x, b, target)

        # second-order interaction matrix on the SAME k x k grid as the referee
        I_hess = self._integrated_hessian(x, b, target, slices, H, W)

        # diagnostics comparable to pairwise_interaction_matrix's fields
        diag = np.diag(I_hess)
        off = I_hess - np.diag(diag)
        diag_mass = float(np.abs(diag).sum())
        off_diag_mass = float(np.abs(off).sum())
        total = diag_mass + off_diag_mass
        off_diag_ratio = float(off_diag_mass / total) if total > 0 else 0.0

        f_x = float(self._probs(x)[0, target])
        f_b = float(self._probs(b)[0, target])

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=f_x,
            f_b=f_b,
            extras={
                "sigma": self.sigma,
                "k": self.k,
                "n_cells": K,
                "steps": self.steps,
                "hess_steps": self.hess_steps,
                "interaction_matrix": I_hess,        # (K,K) integrated Hessian
                "diag_mass": diag_mass,
                "off_diag_mass": off_diag_mass,
                "off_diag_ratio": off_diag_ratio,
                "n_forward": self._n_forward,
                "n_backward": self._n_backward,
                "reference": "blur_completion",
            },
        )