"""Integrated Gradients with the strong-blur self-reference as baseline.

IG(x) = (x - b) * integral_0^1 d/dx f(b + a (x - b)) da

Approximated with a Riemann sum over `steps` interpolation points. Using the
blur reference b (instead of a black image) keeps the path on-manifold and makes
the attribution "what's gained going from blurred -> sharp", consistent with the
sufficiency method's baseline. Pixel-channel gradients are summed to a (H,W) map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import AttributionResult, Explainer, blur_reference


class IGExplainer(Explainer):
    name = "ig"

    def __init__(self, *args, steps=64, sigma=11.0, batch_size=16, **kw):
        super().__init__(*args, **kw)
        self.steps = steps
        self.sigma = sigma
        self.batch_size = batch_size

    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        b = blur_reference(x, self.sigma).to(self.device)

        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        grad_accum = torch.zeros_like(x)

        for start in range(0, self.steps, self.batch_size):
            a = alphas[start:start + self.batch_size].view(-1, 1, 1, 1)
            interp = (b + a * (x - b)).clone().requires_grad_(True)  # (B,3,H,W)
            logits = self.model(interp)
            score = F.log_softmax(logits, dim=1)[:, target].sum()
            grad = torch.autograd.grad(score, interp)[0]  # (B,3,H,W)
            grad_accum += grad.sum(dim=0, keepdim=True)

        avg_grad = grad_accum / self.steps
        ig = (x - b) * avg_grad  # (1,3,H,W)
        attr = ig.sum(dim=1).squeeze(0).detach().cpu().numpy()  # (H,W)

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=float(self._probs(x)[0, target]),
            f_b=float(self._probs(b)[0, target]),
            extras={"steps": self.steps, "sigma": self.sigma},
        )