#!/usr/bin/env python3
"""Self-supervised V-JEPA 2 domain adaptation module (`torch.nn.Module` style).

Architecture (following Assran et al., V-JEPA / V-JEPA 2):
- `context_encoder` (`VJEPA2Encoder`): processes ONLY unmasked 3D space-time context tokens
  using 3D-RoPE `position_mask` indices (~4-6x faster & lower VRAM than full-token encoding).
- `predictor` (`VJEPA2Predictor`): predicts target latent representations from context latents +
  learnable mask tokens at target 3D positions.
- `target_encoder` (`VJEPA2Encoder`): stop-gradient Exponential Moving Average (EMA) copy of
  `context_encoder` that encodes the full unmasked clip to produce stable latent targets.

Supports distributed / multi-GPU mixed-precision training via Hugging Face `Accelerate` and
honors `HF_ENDPOINT` (e.g. `https://hf-mirror.com`) and `HF_TOKEN` automatically.

Example CLI usage (single GPU or 2x RTX A5000):

    uv run python -m models.vjepa2_ssl --data-dir data/unlabeled/clips --output-dir checkpoints/vjepa2_bay
    uv run accelerate launch -m models.vjepa2_ssl --data-dir data/unlabeled/clips --output-dir checkpoints/vjepa2_bay
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoVideoProcessor, VJEPA2Config, VJEPA2Model
from transformers.models.vjepa2.modeling_vjepa2 import apply_masks

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass
class VJEPA2SSLConfig:
    """Configuration for V-JEPA 2 self-supervised domain adaptation."""

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


@dataclass
class VJEPA2SSLOutput:
    """Structured output returned by `VJEPA2SSLModule.forward`."""

    loss: torch.Tensor
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    context_ratio: float
    target_ratio: float


class MultiBlockMaskGenerator3D(nn.Module):
    """3D space-time multi-block mask generator producing `(context_mask, target_mask)` index tensors."""

    def __init__(
        self,
        num_targets: int = 4,
        spatial_scale: tuple[float, float] = (0.15, 0.35),
        temporal_scale: tuple[float, float] = (0.50, 1.00),
        aspect_ratio: tuple[float, float] = (0.75, 1.50),
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.spatial_scale = spatial_scale
        self.temporal_scale = temporal_scale
        self.aspect_ratio = aspect_ratio

    def _sample_block_size(
        self,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        generator: torch.Generator | None = None,
    ) -> tuple[int, int, int]:
        rand = torch.rand(3, generator=generator).tolist()
        t_scale = self.temporal_scale[0] + rand[0] * (self.temporal_scale[1] - self.temporal_scale[0])
        s_scale = self.spatial_scale[0] + rand[1] * (self.spatial_scale[1] - self.spatial_scale[0])
        log_ar = math.log(self.aspect_ratio[0]) + rand[2] * (
            math.log(self.aspect_ratio[1]) - math.log(self.aspect_ratio[0])
        )
        ar = math.exp(log_ar)

        bt = max(1, min(grid_t, int(round(grid_t * t_scale))))
        spatial_tokens = grid_h * grid_w * s_scale
        bh = max(1, min(grid_h - 1 if grid_h > 1 else 1, int(round(math.sqrt(spatial_tokens * ar)))))
        bw = max(1, min(grid_w - 1 if grid_w > 1 else 1, int(round(math.sqrt(spatial_tokens / ar)))))
        return bt, bh, bw

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Sample shared block sizes across the batch so index tensors have uniform token counts."""
        bt, bh, bw = self._sample_block_size(grid_t, grid_h, grid_w, generator=generator)
        block_tokens = bt * bh * bw
        total_tokens = grid_t * grid_h * grid_w

        target_masks_per_block: list[list[torch.Tensor]] = [[] for _ in range(self.num_targets)]
        context_candidates: list[torch.Tensor] = []

        for _ in range(batch_size):
            occupied = torch.zeros((grid_t, grid_h, grid_w), dtype=torch.bool)
            for m_idx in range(self.num_targets):
                t0 = int(torch.randint(0, grid_t - bt + 1, (1,), generator=generator).item())
                h0 = int(torch.randint(0, grid_h - bh + 1, (1,), generator=generator).item())
                w0 = int(torch.randint(0, grid_w - bw + 1, (1,), generator=generator).item())

                block_mask = torch.zeros((grid_t, grid_h, grid_w), dtype=torch.bool)
                block_mask[t0 : t0 + bt, h0 : h0 + bh, w0 : w0 + bw] = True
                occupied |= block_mask

                flat_target = torch.nonzero(block_mask.reshape(-1), as_tuple=False).squeeze(-1)
                target_masks_per_block[m_idx].append(flat_target)

            unmasked = torch.nonzero((~occupied).reshape(-1), as_tuple=False).squeeze(-1)
            if unmasked.numel() == 0:
                # Guarantee non-empty context even on tiny unit-test grids
                unmasked = torch.zeros(1, dtype=torch.long)
            context_candidates.append(unmasked)

        # Collate context indices to common minimum visible count across batch
        min_ctx = min(int(c.numel()) for c in context_candidates)
        ctx_batch = torch.stack([c[:min_ctx] for c in context_candidates], dim=0).to(device)
        target_batches = [
            torch.stack(block_list, dim=0).to(device)
            for block_list in target_masks_per_block
            if block_tokens <= total_tokens
        ]
        return [ctx_batch], target_batches


class VJEPA2SSLModule(nn.Module):
    """PyTorch `nn.Module` encapsulating V-JEPA 2 Context Encoder, Predictor, and EMA Target Encoder."""

    def __init__(
        self,
        backbone: VJEPA2Model,
        ssl_config: VJEPA2SSLConfig | None = None,
    ) -> None:
        super().__init__()
        self.ssl_config = ssl_config or VJEPA2SSLConfig()
        self.config: VJEPA2Config = backbone.config

        # Online trainable modules
        self.context_encoder = backbone.encoder
        self.predictor = backbone.predictor

        # Momentum teacher (stop-gradient EMA copy of context_encoder)
        self.target_encoder = copy.deepcopy(backbone.encoder)
        self.target_encoder.eval()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.mask_generator = MultiBlockMaskGenerator3D(
            num_targets=self.ssl_config.num_targets,
            spatial_scale=self.ssl_config.spatial_scale,
            temporal_scale=self.ssl_config.temporal_scale,
            aspect_ratio=self.ssl_config.aspect_ratio,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "facebook/vjepa2-vitl-fpc64-256",
        ssl_config: VJEPA2SSLConfig | None = None,
        torch_dtype: torch.dtype | None = None,
        **hf_kwargs,
    ) -> VJEPA2SSLModule:
        """Load encoder + predictor weights via Hugging Face (`HF_ENDPOINT` / local checkpoint)."""
        cfg = ssl_config or VJEPA2SSLConfig(model_name_or_path=model_name_or_path)
        backbone = VJEPA2Model.from_pretrained(model_name_or_path, dtype=torch_dtype, **hf_kwargs)
        return cls(backbone=backbone, ssl_config=cfg)

    def token_grid_shape(self, pixel_values_videos: torch.Tensor) -> tuple[int, int, int]:
        """Return `(grid_t, grid_h, grid_w)` patch-token dimensions for `pixel_values_videos` `(B, T, C, H, W)`."""
        _, num_frames, _, height, width = pixel_values_videos.shape
        eff_frames = max(num_frames, self.config.tubelet_size)
        grid_t = eff_frames // self.config.tubelet_size
        grid_h = height // self.config.patch_size
        grid_w = width // self.config.patch_size
        return grid_t, grid_h, grid_w

    def encode_context(self, pixel_values_videos: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """Encode ONLY visible `context_idx` `(B, N_ctx)` patches while preserving 3D-RoPE positions."""
        patch_tokens = self.context_encoder.embeddings(pixel_values_videos)
        visible_tokens = apply_masks(patch_tokens, [context_idx])
        hidden_states = visible_tokens
        for layer_module in self.context_encoder.layer:
            hidden_states = layer_module(hidden_states, position_mask=context_idx)[0]
        return self.context_encoder.layernorm(hidden_states)

    def predict_from_masked_context(
        self,
        context_hidden_states: torch.Tensor,
        context_idx: torch.Tensor,
        target_mask: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict all target blocks from `context_hidden_states` `(B, N_ctx, D)`.

        In V-JEPA, each of the `M` target blocks is predicted independently from the visible
        context tokens (`context_mask` repeated `M` times along batch dim -> `(M * B, ...)`),
        while selecting each block's learnable `mask_tokens` embedding.
        """
        num_targets = len(target_mask)
        batch_size, n_ctx, _ = context_hidden_states.shape
        context_proj = self.predictor.embeddings.predictor_embeddings(context_hidden_states)

        # Build learnable mask token embeddings `(M * B, N_tgt_per_block, D_pred)`
        n_mask_tokens = self.predictor.embeddings.num_mask_tokens
        target_tokens_list = []
        for m_idx, t_mask in enumerate(target_mask):
            tok_idx = (m_idx + 1) % n_mask_tokens if len(target_mask) == 1 else m_idx % n_mask_tokens
            tok = self.predictor.embeddings.mask_tokens[tok_idx]  # (1, 1, D_pred)
            target_tokens_list.append(tok.expand(batch_size, t_mask.size(1), -1))
        target_proj = torch.cat(target_tokens_list, dim=0)

        context_rep = context_proj.repeat(num_targets, 1, 1)
        hidden_states = torch.cat([context_rep, target_proj], dim=1)

        ctx_pos = context_idx.repeat(num_targets, 1)
        tgt_pos = torch.cat(target_mask, dim=0)
        position_masks = torch.cat([ctx_pos, tgt_pos], dim=1)

        argsort = torch.argsort(position_masks, dim=1)
        hidden_states, position_masks = self.predictor.sort_tokens(hidden_states, position_masks, argsort)
        for layer_module in self.predictor.layer:
            hidden_states = layer_module(hidden_states, position_masks)[0]
        hidden_states = self.predictor.layernorm(hidden_states)
        hidden_states = self.predictor.unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, n_ctx:]
        return self.predictor.proj(hidden_states)

    @torch.no_grad()
    def encode_target(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Encode full unmasked video with the stop-gradient EMA `target_encoder`."""
        self.target_encoder.eval()
        target_out = self.target_encoder(pixel_values_videos=pixel_values_videos).last_hidden_state
        if self.ssl_config.normalize_targets:
            target_out = F.layer_norm(target_out, (target_out.size(-1),))
        return target_out

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.998) -> None:
        """Exponential Moving Average update: `θ_target <- m * θ_target + (1 - m) * θ_context`."""
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"EMA momentum must be in [0, 1], got {momentum}")
        for param_ctx, param_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_tgt.data.mul_(momentum).add_(param_ctx.detach().data, alpha=1.0 - momentum)

    @torch.no_grad()
    def extract_features(self, pixel_values_videos: torch.Tensor, use_target_encoder: bool = True) -> torch.Tensor:
        """Extract dense space-time features `(B, N_tokens, D)` for downstream Attentive Probe / retrieval."""
        encoder = self.target_encoder if use_target_encoder else self.context_encoder
        return encoder(pixel_values_videos=pixel_values_videos).last_hidden_state

    def save_backbone_pretrained(self, save_directory: str | Path, use_target_encoder: bool = True) -> None:
        """Export an HF-compatible `VJEPA2Model` directory ready for `VJEPA2ForVideoClassification.from_pretrained`."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_model = VJEPA2Model(self.config)
        chosen_encoder = self.target_encoder if use_target_encoder else self.context_encoder
        export_model.encoder.load_state_dict(chosen_encoder.state_dict())
        export_model.predictor.load_state_dict(self.predictor.state_dict())
        export_model.save_pretrained(out_dir)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_mask: list[torch.Tensor] | None = None,
        target_mask: list[torch.Tensor] | None = None,
    ) -> VJEPA2SSLOutput:
        """Compute V-JEPA 2 masked latent prediction loss on `pixel_values_videos` `(B, T, C, H, W)`."""
        batch_size = pixel_values_videos.size(0)
        grid_t, grid_h, grid_w = self.token_grid_shape(pixel_values_videos)
        total_tokens = grid_t * grid_h * grid_w

        if context_mask is None or target_mask is None:
            context_mask, target_mask = self.mask_generator(
                batch_size=batch_size,
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                device=pixel_values_videos.device,
            )

        with torch.no_grad():
            full_targets = self.encode_target(pixel_values_videos)
            target_latents = apply_masks(full_targets, target_mask)

        context_hidden = self.encode_context(pixel_values_videos, context_mask[0])
        predicted_latents = self.predict_from_masked_context(context_hidden, context_mask[0], target_mask)
        if self.ssl_config.normalize_targets:
            predicted_latents = F.layer_norm(predicted_latents, (predicted_latents.size(-1),))

        loss = F.smooth_l1_loss(predicted_latents, target_latents, beta=self.ssl_config.loss_beta)
        ctx_ratio = float(context_mask[0].size(1)) / max(total_tokens, 1)
        tgt_ratio = float(target_mask[0].size(1) * len(target_mask)) / max(total_tokens, 1)
        return VJEPA2SSLOutput(
            loss=loss,
            predicted_latents=predicted_latents,
            target_latents=target_latents,
            context_ratio=ctx_ratio,
            target_ratio=tgt_ratio,
        )


def read_video_clip_rgb(path: Path, num_frames: int = 16) -> np.ndarray:
    """Uniformly sample `num_frames` RGB frames `(T, H, W, 3)` from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video clip: {path}")
    frames_rgb: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames_rgb:
        raise RuntimeError(f"decoded 0 frames from video clip: {path}")
    indices = np.linspace(0, len(frames_rgb) - 1, num=num_frames).round().astype(int)
    return np.stack([frames_rgb[i] for i in indices], axis=0)


class UnlabeledClipDataset(Dataset):
    """Unlabeled video clip dataset scanning any folder recursively (no class subfolders needed)."""

    def __init__(self, root_dir: str | Path, frames_per_clip: int = 16) -> None:
        self.root_dir = Path(root_dir)
        self.frames_per_clip = frames_per_clip
        self.paths = sorted(
            p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"no video clips found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return read_video_clip_rgb(self.paths[index], num_frames=self.frames_per_clip)


def cosine_ema_schedule(step: int, total_steps: int, m_start: float = 0.996, m_end: float = 0.9995) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing unlabeled ROI .mp4 clips")
    parser.add_argument("--output-dir", default="checkpoints/vjepa2_ssl", help="Where to save adapted HF checkpoint")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="HF model ID or local path")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Sampled frames per clip (default: 16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5, help="Gentle LR for domain adaptation (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    return parser.parse_args(argv)


def train_ssl(args: argparse.Namespace) -> None:
    from accelerate import Accelerator

    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.grad_accum)
    processor = AutoVideoProcessor.from_pretrained(args.model)
    ssl_cfg = VJEPA2SSLConfig(
        model_name_or_path=args.model,
        frames_per_clip=args.frames_per_clip,
        crop_size=processor.crop_size["height"],
    )
    model = VJEPA2SSLModule.from_pretrained(args.model, ssl_config=ssl_cfg)

    dataset = UnlabeledClipDataset(args.data_dir, frames_per_clip=args.frames_per_clip)

    def collate_clips(batch_np: list[np.ndarray]) -> torch.Tensor:
        return processor(batch_np, return_tensors="pt")["pixel_values_videos"]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_clips,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(loader) * args.epochs) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for videos in loader:
            with accelerator.accumulate(model):
                out: VJEPA2SSLOutput = model(videos)
                accelerator.backward(out.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                    m_start, m_end = ssl_cfg.ema_momentum_range
                    ema_m = cosine_ema_schedule(global_step, total_steps, m_start=m_start, m_end=m_end)
                    accelerator.unwrap_model(model).update_target_encoder(momentum=ema_m)

                epoch_loss += float(out.loss.detach().item())

        mean_loss = epoch_loss / max(len(loader), 1)
        if accelerator.is_main_process:
            print(
                f"epoch {epoch}/{args.epochs} step {global_step}/{total_steps} "
                f"loss={mean_loss:.4f} ctx={out.context_ratio:.2f} tgt={out.target_ratio:.2f}"
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped: VJEPA2SSLModule = accelerator.unwrap_model(model)
        unwrapped.save_backbone_pretrained(args.output_dir, use_target_encoder=True)
        processor.save_pretrained(args.output_dir)
        print(f"saved adapted V-JEPA 2 checkpoint -> {args.output_dir}")


if __name__ == "__main__":
    train_ssl(parse_args())
