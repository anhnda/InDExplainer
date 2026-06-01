"""Explainer interface + shared utilities.

The contract every method implements:

    Explainer(model, target_class=None, device=...)
    explainer.explain(x) -> AttributionResult

where `x` is a normalized (1,3,H,W) tensor. `target_class=None` means "use the
model's own top-1 prediction on x". The returned attribution is a single-channel
(H,W) score map in numpy.

The strong-blur self-reference (`blur_reference`) is defined once here because
both IG (as its baseline) and the sufficiency method (as its background field)
consume it. This is the on-manifold, approximately class-neutral reference:
b = blur_sigma(x), built in *pixel* space then re-normalized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..backbone import denormalize, normalize_pixel


@dataclass
class AttributionResult:
    """Container for an attribution map + diagnostics."""

    attribution: np.ndarray  # (H,W) float score map
    method: str
    target_class: int
    target_class_name: str = ""
    # optional diagnostics (sufficiency method populates these)
    f_x: Optional[float] = None  # target prob on original
    f_b: Optional[float] = None  # target prob on blur reference (neutrality check)
    f_phi: Optional[float] = None  # target prob on final composite
    extras: dict = field(default_factory=dict)


def _gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a (1,3,H,W) tensor (operates per channel)."""
    if sigma <= 0:
        return x
    k1d = _gaussian_kernel1d(sigma, x.device)
    n = k1d.numel()
    pad = n // 2
    c = x.shape[1]
    kx = k1d.view(1, 1, 1, n).repeat(c, 1, 1, 1)
    ky = k1d.view(1, 1, n, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, pad), groups=c)
    x = F.conv2d(x, ky, padding=(pad, 0), groups=c)
    return x


# def blur_reference(x_norm: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
#     """Strong-blur self-reference of a *normalized* image.

#     Blurs in [0,1] pixel space (so color statistics stay physical) then
#     re-normalizes. Large sigma => destroys object high-frequency structure,
#     keeps low-frequency color/layout => approximately class-neutral, on-manifold.
#     """
#     x01 = denormalize(x_norm)
#     b01 = gaussian_blur(x01, sigma).clamp(0, 1)
#     return normalize_pixel(b01)
def blur_reference(x_norm: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
    """Strong-blur self-reference of a *normalized* image (kept for back-compat)."""
    return make_reference(x_norm, mode="blur", sigma=sigma)
def make_reference(
    x_norm: torch.Tensor,
    mode: str = "blur",
    sigma: float = 100.0,
    noise_std: float = 0.25,
    seed: int = 0,
) -> torch.Tensor:
    """Build an infill reference for a *normalized* image.

    Modes (all built in [0,1] pixel space, then re-normalized):
      - "blur":   strong Gaussian self-blur (default; on-manifold, ~class-neutral)
      - "black":  constant 0
      - "white":  constant 1
      - "noise":  per-pixel uniform noise in [0,1] (seeded)
      - "corners": constant fill = mean color of the four corner pixels
    """
    x01 = denormalize(x_norm)

    if mode == "blur":
        b01 = gaussian_blur(x01, sigma).clamp(0, 1)
    elif mode == "black":
        b01 = torch.zeros_like(x01)
    elif mode == "white":
        b01 = torch.ones_like(x01)
    elif mode == "noise":
        gen = torch.Generator(device=x01.device).manual_seed(seed)
        b01 = torch.rand(x01.shape, generator=gen, device=x01.device, dtype=x01.dtype)
    elif mode == "corners":
        # mean color over four 10%x10% corner patches, per channel -> (1,C,1,1) fill
        H, W = x01.shape[-2:]
        ph = max(1, int(round(0.10 * H)))
        pw = max(1, int(round(0.10 * W)))
        patches = torch.cat(
            [
                x01[..., :ph, :pw],     # top-left
                x01[..., :ph, -pw:],    # top-right
                x01[..., -ph:, :pw],    # bottom-left
                x01[..., -ph:, -pw:],   # bottom-right
            ],
            dim=-1,
        )  # (1,C,ph, 4*pw)
        mean_c = patches.mean(dim=(-2, -1)).view(1, -1, 1, 1)  # (1,C,1,1)
        b01 = mean_c.expand_as(x01).clamp(0, 1)
    else:
        raise ValueError(
            f"unknown infill mode {mode!r}; "
            "expected one of: blur, black, white, noise, corners"
        )

    return normalize_pixel(b01)

class Explainer:
    """Base class. Subclasses implement `explain`."""

    name = "base"

    def __init__(
        self,
        model: torch.nn.Module,
        target_class: Optional[int] = None,
        device: str | torch.device = "cpu",
        class_names: Optional[list[str]] = None,
    ):
        self.model = model
        self.target_class = target_class
        self.device = device
        self.class_names = class_names or []

    # -- shared helpers ----------------------------------------------------- #
    @torch.no_grad()
    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(x), dim=1)

    def _resolve_target(self, x: torch.Tensor) -> int:
        if self.target_class is not None:
            return int(self.target_class)
        with torch.no_grad():
            return int(self.model(x).argmax(dim=1).item())

    def _class_name(self, idx: int) -> str:
        return self.class_names[idx] if idx < len(self.class_names) else str(idx)

    def explain(self, x: torch.Tensor) -> AttributionResult:  # pragma: no cover
        raise NotImplementedError