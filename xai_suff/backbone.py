"""ResNet-50 ImageNet-1k backbone loader + image preprocessing utilities.

Single source of truth for: the frozen classifier, the ImageNet normalization
constants, and the path image -> normalized tensor pipeline. Everything else in
the package consumes `load_backbone()` and `load_image()`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

# ImageNet-1k channel statistics (what the pretrained weights expect).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def load_backbone(device: str | torch.device = "cpu") -> nn.Module:
    """Return a frozen, eval-mode ResNet-50 with ImageNet-1k weights."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def get_class_names() -> list[str]:
    """Human-readable ImageNet-1k class names, indexed by logit position."""
    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


# ---- preprocessing -------------------------------------------------------- #

_resize = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),  # -> [0,1], CHW
    ]
)
_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def load_image(path: str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load an image file -> normalized model-ready tensor of shape (1,3,H,W).

    We keep normalization *inside* the model call chain conceptually, but here
    return the already-normalized tensor for convenience. Methods that need the
    raw [0,1] image (e.g. for blurring in pixel space) can use `to_pixel_space`.
    """
    img = Image.open(path).convert("RGB")
    x01 = _resize(img)  # (3,H,W) in [0,1]
    x = _normalize(x01).unsqueeze(0)  # (1,3,H,W) normalized
    return x.to(device)


def load_image_pixel(path: str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Load an image file -> *un-normalized* [0,1] tensor of shape (1,3,H,W)."""
    img = Image.open(path).convert("RGB")
    x01 = _resize(img).unsqueeze(0)
    return x01.to(device)


def normalize_pixel(x01: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a [0,1] tensor (1,3,H,W) or (3,H,W)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x01.device).view(1, 3, 1, 1)
    if x01.dim() == 3:
        x01 = x01.unsqueeze(0)
    return (x01 - mean) / std


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization back to [0,1] pixel space."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return (x * std + mean).clamp(0, 1)


def model_size() -> Tuple[int, int]:
    return INPUT_SIZE, INPUT_SIZE