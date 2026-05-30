"""In-distribution minimal-sufficient-region attribution (binary-mask version).

Solve, per instance, for the smallest BINARY sharp region whose composite
against a blurred self-reference keeps the target probability near f(x):

    min_m  |m|_1 + lambda * TV(m)   s.t.  f(Phi(x,m)) >= (1-eps) f(x)
    Phi(x,m) = m * x + (1-m) * blur_sigma(x),   m in {0,1}

solved by a Lagrangian with dual ascent on mu >= 0:

    L(m,mu) = |m|_1 + lambda*TV(m) + mu * (target_gap - drop_recovered)

KEY CHANGE vs. the soft version
-------------------------------
The mask is BINARY in the forward pass. Each pixel is either fully sharp
(m=1, original x) or fully replaced by the blur reference (m=0, b). There is
NO soft blend and NO feather floor, so a pixel can no longer "leak" a dimmed
copy of itself through a fractional mask value. This makes localization
mandatory: a mask that sits on the background reveals background sharp, blurs
the object away, and f collapses -- the optimizer can no longer cheat by
dimming the whole image.

The mask is non-differentiable, so we use a straight-through estimator:
forward uses hard (m>0.5); backward uses the smooth sigmoid gradient. A
temperature `tau` sharpens the sigmoid so the soft proxy tracks the hard
decision. L1 (budget) and TV (coherence) regularize the SOFT logits to keep
gradients informative.

Pair this with an ON-MANIFOLD blur reference (sigma ~ 11), so that "removing"
a pixel replaces it with wombat-colored blur (destroying evidence) rather than
darkening it. Binary mask + on-manifold b together close both leak paths.

f(b)-relative constraint is kept: recovery is measured from the blur floor
f(b) up to f(x), and f(b) is logged so blur neutrality is audited.
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
        sigma=11.0,           # blur strength for the reference field (keep on-manifold)
        lam=0.05,             # TV weight
        eps=0.10,             # tolerance: keep >= (1-eps) of the recoverable gap
        steps=300,            # optimization steps
        lr=0.05,              # mask learning rate
        rho=0.5,              # dual ascent rate
        tau=0.5,              # sigmoid temperature for the straight-through proxy
        logit_smooth=0.0,     # optional gaussian smoothing of logits (coherence prior;
                              # 0 = off. NOT a feather floor -- mask is still binary)
        mu_init=5.0,
        mu_max=50.0,          # clamp dual to avoid runaway
        stochastic=False,     # expected recovery over sampled binary masks
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
        self.tau = tau
        self.logit_smooth = logit_smooth
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.stochastic = stochastic
        self.n_mc = n_mc
        self.seed = seed

    # ------------------------------------------------------------------ #
    # mask
    # ------------------------------------------------------------------ #
    def _binary_mask(self, logits):
        """Hard {0,1} mask in the forward pass, soft gradient in the backward pass.

        Returns (m_hard, m_soft):
          m_hard -- straight-through binary mask used in the composite (forward
                    value is exactly 0 or 1; gradient flows through m_soft).
          m_soft -- smooth sigmoid proxy, used for the L1 / TV regularizers so
                    they stay differentiable and informative.
        """
        z = logits
        if self.logit_smooth > 0:
            # optional coherence prior on the logits -- this smooths the *decision
            # boundary location*, NOT the mask values; the output is still binary.
            z = gaussian_blur(z, self.logit_smooth)
        m_soft = torch.sigmoid(z / self.tau)
        m_hard = (m_soft > 0.5).float()
        # straight-through: binary forward, sigmoid-gradient backward
        m_st = m_hard + (m_soft - m_soft.detach())
        return m_st, m_soft

    def _stochastic_binary_mask(self, logits):
        """Bernoulli-sampled binary mask with straight-through gradient.

        Each pixel is independently kept with probability sigmoid(z/tau). Used
        when stochastic=True to constrain the EXPECTED recovery over sampled
        binary masks (a stronger defense against brittle single masks).
        """
        z = logits
        if self.logit_smooth > 0:
            z = gaussian_blur(z, self.logit_smooth)
        m_soft = torch.sigmoid(z / self.tau)
        hard = torch.bernoulli(m_soft.detach())
        m_st = hard + (m_soft - m_soft.detach())
        return m_st, m_soft

    @staticmethod
    def _tv(m):
        dh = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs().mean()
        dw = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs().mean()
        return dh + dw

    def _target_prob(self, comp, target):
        return F.softmax(self.model(comp), dim=1)[:, target]

    # ------------------------------------------------------------------ #
    # explain
    # ------------------------------------------------------------------ #
    def explain(self, x: torch.Tensor) -> AttributionResult:
        x = x.to(self.device)
        target = self._resolve_target(x)
        b = blur_reference(x, self.sigma).to(self.device)

        with torch.no_grad():
            f_x = float(self._target_prob(x, target).item())
            f_b = float(self._target_prob(b, target).item())
        gap = max(f_x - f_b, 1e-6)
        target_recovery = (1.0 - self.eps) * gap

        torch.manual_seed(self.seed)
        # start slightly positive so the mask begins mostly "on", then shrinks
        logits = torch.full_like(x[:, :1, :, :], 0.5).requires_grad_(True)  # (1,1,H,W)
        opt = torch.optim.Adam([logits], lr=self.lr)
        mu = self.mu_init

        history = {"loss": [], "mass": [], "recovery": [], "mu": []}

        for step in range(self.steps):
            opt.zero_grad()

            if self.stochastic:
                # expected recovery over Bernoulli-sampled BINARY masks
                rec = 0.0
                m_soft_acc = None
                for _ in range(self.n_mc):
                    m_st, m_soft = self._stochastic_binary_mask(logits)
                    comp = m_st * x + (1 - m_st) * b
                    rec = rec + (self._target_prob(comp, target) - f_b)
                    m_soft_acc = m_soft if m_soft_acc is None else m_soft_acc
                recovery = rec / self.n_mc
                m_for_reg = m_soft_acc            # regularize the soft proxy
            else:
                m_st, m_soft = self._binary_mask(logits)
                comp = m_st * x + (1 - m_st) * b  # BINARY composite: sharp OR blur
                recovery = self._target_prob(comp, target) - f_b
                m_for_reg = m_soft

            # budget (L1) and coherence (TV) on the soft proxy -> differentiable
            mass = m_for_reg.mean()
            tv = self._tv(m_for_reg)

            violation = target_recovery - recovery  # >0 when under-recovered
            loss = mass + self.lam * tv + mu * violation.mean()

            loss.backward()
            opt.step()

            with torch.no_grad():
                mu = float(np.clip(mu + self.rho * float(violation.mean().item()),
                                   0.0, self.mu_max))

            history["loss"].append(float(loss.item()))
            history["mass"].append(float(mass.item()))
            history["recovery"].append(float(recovery.mean().item()))
            history["mu"].append(mu)

        # final evaluation uses the HARD binary mask -- f_phi is now honest:
        # it is the probability recovered by revealing ONLY the kept pixels.
        with torch.no_grad():
            m_hard, _ = self._binary_mask(logits)
            comp = m_hard * x + (1 - m_hard) * b
            f_phi = float(self._target_prob(comp, target).item())
            attr = m_hard.squeeze().cpu().numpy()  # (H,W) in {0,1}

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
                "binary": True,
                "tau": self.tau,
                "history": history,
            },
        )