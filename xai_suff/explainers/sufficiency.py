"""In-distribution minimal-sufficient-region attribution (the proposed method).

Solve, per instance, for the smallest feathered sharp region whose composite
against a blurred self-reference keeps the target probability near f(x):

    min_m  |m|_1 + lambda * TV(m)   s.t.  f(Phi(x,m)) >= (1-eps) f(x)
    Phi(x,m) = m * x + (1-m) * blur_sigma(x)

solved by a Lagrangian with dual ascent on mu >= 0:

    L(m,mu) = |m|_1 + lambda*TV(m) + mu * (target_gap - drop_recovered)

Two design choices from the design discussion are built in:
  * f(b)-relative constraint: "drop" is measured as recovery of the gap from the
    blur floor f(b) up to f(x), not implicitly relative to zero. We log f(b) so
    blur neutrality is auditable rather than assumed.
  * feathered mask: m is a smoothed sigmoid of free logits (Gaussian-blurred
    pre-activation) so edges are soft -> no boundary/shape leak.

Output is a continuous per-pixel sufficiency map in [0,1]. Optional stochastic
masking (Bernoulli sampling of the soft mask) defends against brittle/adversarial
masks by constraining the *expected* recovery.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import AttributionResult, Explainer, blur_reference, gaussian_blur


class SufficiencyExplainer(Explainer):
    name = "sufficiency"

    def __init__(
        self,
        *args,
        sigma=11.0,           # blur strength for the reference field
        lam=0.05,             # TV weight
        eps=0.10,             # tolerance: keep >= (1-eps) of the recoverable gap
        steps=300,            # optimization steps
        lr=0.05,              # mask learning rate
        rho=0.5,              # dual ascent rate
        feather=2.0,          # gaussian sigma for feathering the mask logits
        mu_init=5.0,
        mu_max=50.0,          # clamp dual to avoid runaway
        stochastic=False,     # sample Bernoulli hard masks for the constraint
        n_mc=4,               # MC samples when stochastic
        seed=0,
        **kw,
    ):
        super().__init__(*args, **kw)
        self.sigma = sigma
        self.lam = lam
        self.eps = eps
        self.steps = steps
        self.lr = lr
        self.rho = rho
        self.feather = feather
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.stochastic = stochastic
        self.n_mc = n_mc
        self.seed = seed

    def _feathered_mask(self, logits):
        """Soft mask in [0,1]: blur the logits then sigmoid -> smooth edges."""
        if self.feather > 0:
            logits = gaussian_blur(logits, self.feather)
        return torch.sigmoid(logits)

    @staticmethod
    def _tv(m):
        dh = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
        dw = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
        return dh + dw

    def _target_prob(self, comp, target):
        return F.softmax(self.model(comp), dim=1)[:, target]

    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        b = blur_reference(x, self.sigma).to(self.device)

        with torch.no_grad():
            f_x = float(self._target_prob(x, target).item())
            f_b = float(self._target_prob(b, target).item())
        # recoverable gap: how much probability the sharp evidence can add over blur
        gap = max(f_x - f_b, 1e-6)
        target_recovery = (1.0 - self.eps) * gap  # must recover at least this much

        # free logits parameterizing the mask (start mildly positive ~ mostly on)
        torch.manual_seed(self.seed)
        logits = torch.zeros_like(x[:, :1, :, :]).requires_grad_(True)  # (1,1,H,W)
        opt = torch.optim.Adam([logits], lr=self.lr)
        mu = self.mu_init

        history = {"loss": [], "mass": [], "recovery": [], "mu": []}

        for step in range(self.steps):
            opt.zero_grad()
            m = self._feathered_mask(logits)  # (1,1,H,W) in [0,1]

            if self.stochastic:
                # expected recovery over Bernoulli-sampled hard masks
                rec = 0.0
                for _ in range(self.n_mc):
                    hard = torch.bernoulli(m.detach())
                    # straight-through: use soft m for gradient, hard for forward effect
                    m_st = hard + (m - m.detach())
                    comp = m_st * x + (1 - m_st) * b
                    rec = rec + (self._target_prob(comp, target) - f_b)
                recovery = rec / self.n_mc
            else:
                comp = m * x + (1 - m) * b
                recovery = self._target_prob(comp, target) - f_b  # (1,)

            mass = m.mean()
            tv = self._tv(m)
            # constraint violation: positive when we haven't recovered enough
            violation = target_recovery - recovery  # scalar tensor
            loss = mass + self.lam * tv + mu * violation.mean()

            loss.backward()
            opt.step()

            # dual ascent on mu (clamped)
            with torch.no_grad():
                mu = float(np.clip(mu + self.rho * float(violation.mean().item()),
                                   0.0, self.mu_max))

            history["loss"].append(float(loss.item()))
            history["mass"].append(float(mass.item()))
            history["recovery"].append(float(recovery.mean().item()))
            history["mu"].append(mu)

        with torch.no_grad():
            m_final = self._feathered_mask(logits)
            comp = m_final * x + (1 - m_final) * b
            f_phi = float(self._target_prob(comp, target).item())
            attr = m_final.squeeze().cpu().numpy()  # (H,W)

        return AttributionResult(
            attribution=attr,
            method=self.name,
            target_class=target,
            target_class_name=self._class_name(target),
            f_x=f_x,
            f_b=f_b,
            f_phi=f_phi,
            extras={
                "gap": gap,
                "target_recovery": target_recovery,
                "final_mass": float(attr.mean()),
                "mu_final": mu,
                "stochastic": self.stochastic,
                "history": history,
            },
        )