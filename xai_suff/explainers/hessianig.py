"""Regional second-order Integrated Hessians (Hessian-IG) on a grid of cells.

Faithful to Janizek, Sturmfels & Lee, "Explaining Explanations: Axiomatic
Feature Interactions for Deep Networks" (arXiv:2002.04138). Two corrections over
a naive single-path Hessian, both required for the method to mean anything:

1. DOUBLE INTEGRAL (eq. 3 / discrete eq. 25). For i != j the interaction is

       Gamma_ij = (x_i - x'_i)(x_j - x'_j)
                  * int_0^1 int_0^1  a*b * d^2 f(x' + a*b (x - x')) / dx_i dx_j  da db

   i.e. Integrated Gradients applied to itself: phi_j(phi_i(x)). The Hessian is
   evaluated along the *product* path a*b, weighted by a*b, integrated over BOTH
   a and b. A single integral with weight a (the earlier draft) is a different,
   wrong quantity -- it does not satisfy interaction completeness and its scale
   is meaningless (this is what produced the ~78x off-diagonal mass blow-up).

   The discrete approximation (eq. 25), with k = m steps, is

       Gamma_ij ~ (dx_i)(dx_j) * sum_l sum_p (l/k)(p/m)
                  * d^2 f(x' + (l/k)(p/m) dx) / dx_i dx_j  * 1/(k m)

2. SOFTPLUS SMOOTHING (section 3). ResNet-50 is ReLU; ReLU networks are
   piecewise linear and have d^2 f = 0 almost everywhere, so the exact Hessian
   is identically zero and Integrated Hessians degenerates. The paper's remedy
   is to replace ReLU with SoftPlus_beta at explanation time (no retraining);
   larger beta -> closer to ReLU, smaller beta -> smoother, faster convergence.
   We swap activations in a temporary functional wrapper and restore after.

Interaction completeness (eq. 4) gives us a real correctness check, analogous to
pyramid's telescoping identity:

       sum_i sum_j Gamma_ij  =  f(x) - f(x')

We report the residual of this identity in extras['completeness_residual']; if it
is not near zero, the integration is under-resolved (raise hess_steps) or beta is
too large (smooth more). This is the honest analog of pyramid's identity_residual.

Regional gates
--------------
Partition the image into K = k*k cells. Gate vector g in [0,1]^K, with the
composite X(g) = b + sum_c g_c (x - b)|cell_c. Baseline g' = 0 (all blur = b),
target g* = 1 (all sharp = x), so the per-cell prefactor (g*_i - g'_i) = 1 and
the gate-space Gamma_ij is directly the K x K cell interaction matrix, on the
SAME grid and SAME blur-Phi as metrics.pairwise_interaction_matrix.

First-order map: standard blur-baseline IG over the *pixels* (unchanged), so the
method still drops into the first-order faithfulness table.

fast=True (default): Hutchinson HVP estimate of the per-(a,b)-point Hessian.
fast=False          : exact full Hessian via double autograd (K backward/point).

Reference dependence (b = blur_sigma(x), grid k, softplus beta) logged in extras.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AttributionResult, Explainer, blur_reference


# --------------------------------------------------------------------------- #
# SoftPlus smoothing: temporarily replace ReLU with SoftPlus_beta for 2nd-order
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _softplus_relus(model: nn.Module, beta: float):
    """Context manager: swap every nn.ReLU in `model` for nn.Softplus(beta).

    Restores the originals on exit. Functional F.relu calls inside forward() are
    NOT caught by this (they aren't modules); ResNet-50's torchvision impl uses
    nn.ReLU modules for the main activations, which this covers. We note the
    limitation in extras so a downstream user knows the smoothing is partial if
    their backbone uses functional relus.
    """
    originals = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            originals[name] = m
    # swap by reattaching Softplus modules in place of ReLU modules
    modules = dict(model.named_modules())
    replaced = []
    for name, m in list(originals.items()):
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr = name.rsplit(".", 1)[1] if "." in name else name
        parent = modules[parent_name] if parent_name else model
        setattr(parent, attr, nn.Softplus(beta=beta))
        replaced.append((parent, attr, m))
    try:
        yield len(replaced)
    finally:
        for parent, attr, m in replaced:
            setattr(parent, attr, m)


class HessianIGExplainer(Explainer):
    name = "hessian_ig"

    def __init__(
        self,
        *args,
        steps: int = 32,          # Riemann steps for the IG path (first order)
        hess_steps: int = 8,      # midpoint steps PER AXIS for the double integral
        sigma: float = 11.0,      # blur strength of the reference b (shared Phi)
        k: int = 14,              # grid side; K = k*k cells for the interaction
        batch_size: int = 16,     # batching for the first-order IG path
        softplus_beta: float = 10.0,  # SoftPlus sharpness for 2nd-order smoothing
        fast: bool = True,        # fast (Hutchinson HVP) by default; False = exact
        n_probes: int = 16,       # # random probes per point in fast mode
        seed: int = 0,            # RNG seed for the Hutchinson probes
        **kw,
    ):
        super().__init__(*args, **kw)
        self.steps = steps
        self.hess_steps = hess_steps
        self.sigma = sigma
        self.k = k
        self.batch_size = batch_size
        self.softplus_beta = softplus_beta
        self.fast = fast
        self.n_probes = n_probes
        self.seed = seed
        self._n_forward = 0
        self._n_backward = 0

    # ------------------------------------------------------------------ #
    # cell geometry
    # ------------------------------------------------------------------ #
    def _cell_slices(self, H: int, W: int):
        """Row/col boundaries for a k x k partition; row-major to match
        metrics._grid_cells (np.linspace(0, H, k+1).astype(int))."""
        ys = np.linspace(0, H, self.k + 1).astype(int)
        xs = np.linspace(0, W, self.k + 1).astype(int)
        slices = []
        for r in range(self.k):
            for c in range(self.k):
                slices.append((slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1])))
        return slices

    def _gate_field(self, g: torch.Tensor, slices, H: int, W: int) -> torch.Tensor:
        field = torch.zeros((1, 1, H, W), dtype=g.dtype, device=g.device)
        for idx, (rs, cs) in enumerate(slices):
            field[..., rs, cs] = g[idx]
        return field

    # ------------------------------------------------------------------ #
    # first-order IG map (unchanged; ReLU model, no smoothing needed)
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
        return ig.sum(dim=1).squeeze(0).detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # per-point first derivatives df/dg at gate g = t * 1   (t = a*b product)
    # ------------------------------------------------------------------ #
    def _point_grads(self, x, b, target, slices, H, W, t):
        K = len(slices)
        delta = (x - b)
        g = torch.full((K,), float(t), device=self.device,
                       dtype=torch.float32, requires_grad=True)
        field = self._gate_field(g, slices, H, W)
        X = b + field * delta
        logits = self.model(X)
        self._n_forward += 1
        # Integrate the Hessian of the SAME f used by the reveal matrix and the
        # completeness RHS: softmax PROBABILITY, not log-softmax. log-softmax has
        # large, sign-variable second derivatives and integrates to a different
        # quantity -- it breaks interaction-completeness (sum Gamma vs f(x)-f(b)).
        score = F.softmax(logits, dim=1)[:, target]
        grads = torch.autograd.grad(score, g, create_graph=True)[0]
        self._n_backward += 1
        return g, grads

    # ------------------------------------------------------------------ #
    # integrated cell-pair Hessian -- double integral (eq. 25), dispatcher
    # ------------------------------------------------------------------ #
    def _integrated_hessian(self, x, b, target, slices, H, W) -> np.ndarray:
        """Gamma (K x K), Integrated Hessians on the cell grid.

        Off-diagonal (eq. 13):  Gamma_ij = int int a*b * d2f/dg_i dg_j da db.
        Diagonal (eq. 17): Gamma_ii additionally gets the first-order term
            int int  df/dg_i  da db
        so that interaction-completeness sum_ij Gamma_ij = f(x)-f(b) can close.
        Per-cell prefactor (g*-g')=1 by gate construction. fast vs exact differ
        only in how each per-point Hessian is formed.
        """
        K = len(slices)
        steps = self.hess_steps
        # Midpoint nodes on BOTH axes: t_node = (l-0.5)/steps. Midpoint converges
        # O(1/steps^2) vs right-Riemann's O(1/steps), and is exact for the
        # bilinear weight a*b -- so completeness converges fast and hess_steps can
        # be small. Product path t=(la)(pb), 2nd-order weight (la)(pb); the
        # diagonal first-order term has weight 1 under da db.
        axis = (torch.arange(1, steps + 1, device=self.device).float() - 0.5) / steps
        Gamma = torch.zeros((K, K), dtype=torch.float64, device=self.device)
        first_order_diag = torch.zeros((K,), dtype=torch.float64, device=self.device)
        norm = 1.0 / (steps * steps)

        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        for la in axis:
            for pb in axis:
                t = float(la * pb)
                w2 = float(la * pb)              # second-order weight a*b
                g, grads = self._point_grads(x, b, target, slices, H, W, t)
                # accumulate diagonal first-order term: int int df/dg_i da db
                first_order_diag += grads.detach().to(torch.float64) * norm
                if self.fast:
                    Hmat = self._hvp_hessian(g, grads, K, gen)
                else:
                    Hmat = self._exact_hessian(g, grads, K)
                Gamma += w2 * Hmat * norm

        # add the first-order term onto the diagonal (eq. 17)
        Gamma += torch.diag(first_order_diag)
        return Gamma.cpu().numpy()

    def _exact_hessian(self, g, grads, K) -> torch.Tensor:
        rows = []
        for i in range(K):
            row_i = torch.autograd.grad(grads[i], g, retain_graph=True)[0]
            self._n_backward += 1
            rows.append(row_i.detach().to(torch.float64))
        Hmat = torch.stack(rows, dim=0)
        return 0.5 * (Hmat + Hmat.t())

    def _hvp_hessian(self, g, grads, K, gen) -> torch.Tensor:
        """Hutchinson estimate: H ~ mean_p (H v_p) v_p^T, v Rademacher."""
        Hacc = torch.zeros((K, K), dtype=torch.float64, device=self.device)
        for p in range(self.n_probes):
            v = (torch.randint(0, 2, (K,), generator=gen, dtype=torch.float32)
                 * 2 - 1).to(self.device)
            retain = p < self.n_probes - 1
            Hv = torch.autograd.grad(grads, g, grad_outputs=v,
                                     retain_graph=retain)[0]
            self._n_backward += 1
            Hacc += torch.outer(Hv.to(torch.float64), v.to(torch.float64))
        Hmat = Hacc / self.n_probes
        return 0.5 * (Hmat + Hmat.t())

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

        # first-order map on the ReLU model (no smoothing)
        attr = self._first_order_ig(x, b, target)

        f_x = float(self._probs(x)[0, target])
        f_b = float(self._probs(b)[0, target])

        # second-order: smooth ReLU -> SoftPlus for the duration of the Hessian
        with _softplus_relus(self.model, self.softplus_beta) as n_swapped:
            I_hess = self._integrated_hessian(x, b, target, slices, H, W)

        # interaction-completeness check (eq. 4): sum_ij Gamma_ij == f(x)-f(x')
        # NOTE: f(x), f(x') here are the *softplus* model's probs, to match the
        # surface the Hessian was integrated on.
        with _softplus_relus(self.model, self.softplus_beta):
            f_x_sp = float(self._probs(x)[0, target])
            f_b_sp = float(self._probs(b)[0, target])
        completeness_lhs = float(I_hess.sum())
        completeness_rhs = f_x_sp - f_b_sp
        completeness_residual = completeness_lhs - completeness_rhs

        diag = np.diag(I_hess)
        off = I_hess - np.diag(diag)
        diag_mass = float(np.abs(diag).sum())
        off_diag_mass = float(np.abs(off).sum())
        total = diag_mass + off_diag_mass
        off_diag_ratio = float(off_diag_mass / total) if total > 0 else 0.0

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
                "hess_steps_per_axis": self.hess_steps,
                "softplus_beta": self.softplus_beta,
                "n_relu_swapped": int(n_swapped),
                "mode": "fast_hutchinson" if self.fast else "exact",
                "n_probes": self.n_probes if self.fast else None,
                "interaction_matrix": I_hess,        # (K,K) Gamma
                "diag_mass": diag_mass,
                "off_diag_mass": off_diag_mass,
                "off_diag_ratio": off_diag_ratio,
                "completeness_lhs": completeness_lhs,   # sum_ij Gamma_ij
                "completeness_rhs": completeness_rhs,   # f(x)-f(b) on softplus net
                "completeness_residual": completeness_residual,  # ~0 if resolved
                "f_x_softplus": f_x_sp,
                "f_b_softplus": f_b_sp,
                "n_forward": self._n_forward,
                "n_backward": self._n_backward,
                "reference": "blur_completion",
            },
        )