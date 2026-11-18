"""Configuration and output structures for V-JEPA 2 SSL."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class VJEPA2SSLConfig:
    """Configuration for V-JEPA 2 self-supervised domain adaptation and optimization."""

    model_name_or_path: str = "facebook/vjepa2-vitl-fpc64-256"
    frames_per_clip: int = 16
    crop_size: int = 256
    num_targets: int = 4
    spatial_scale: tuple[float, float] = (0.15, 0.35)
    temporal_scale: tuple[float, float] = (0.50, 1.00)
    aspect_ratio: tuple[float, float] = (0.75, 1.50)
    ema_momentum_range: tuple[float, float] = (0.996, 0.9995)
    loss_beta: float = 1.0
    normalize_targets: bool = True
    lr: float = 1e-5
    weight_decay: float = 0.04


@dataclass
class VJEPA2SSLOutput:
    """Structured output returned by `VJEPA2SSL.forward`."""

    loss: torch.Tensor
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    context_ratio: float
    target_ratio: float
