#!/usr/bin/env python3
"""Self-supervised V-JEPA 2 domain adaptation built with PyTorch Lightning (`L.LightningModule`).

Architecture (Assran et al., V-JEPA / V-JEPA 2):
- `context_encoder`: encodes ONLY unmasked 3D space-time context tokens via 3D-RoPE `position_mask`.
- `predictor`: predicts target latent representations at masked 3D positions.
- `target_encoder`: stop-gradient EMA copy of `context_encoder` updated automatically in `optimizer_step`.

Example CLI usage (single GPU or multi-GPU via Lightning `Trainer`):

    uv run python -m models.vjepa2_ssl --data-dir data/unlabeled/clips --output-dir checkpoints/vjepa2_bay
    uv run python -m models.vjepa2_ssl --data-dir data/unlabeled/clips --devices 2 --precision bf16-mixed
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
from einops import repeat
import lightning as L
import numpy as np
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoVideoProcessor, VJEPA2Config, VJEPA2Model
from transformers.models.vjepa2.modeling_vjepa2 import apply_masks

from utils.masking import MultiBlockMaskGenerator3D

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
LORA_ATTENTION_TARGET_REGEX = r".*attention\.(query|key|value|proj)$"


@dataclass
class VJEPA2SSLConfig:
    """Configuration for V-JEPA 2 self-supervised domain adaptation, LoRA, and optimization."""

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
    lr: float = 1e-4
    weight_decay: float = 0.04
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True


@dataclass
class VJEPA2SSLOutput:
    """Structured output returned by `VJEPA2SSL.forward`."""

    loss: torch.Tensor
    predicted_latents: torch.Tensor
    target_latents: torch.Tensor
    context_ratio: float
    target_ratio: float


def cosine_ema_schedule(step: int, total_steps: int, m_start: float = 0.996, m_end: float = 0.9995) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))


class VJEPA2SSL(L.LightningModule):
    """LightningModule encapsulating V-JEPA 2 Context Encoder, Predictor, EMA Target Encoder, and optimization."""

    def __init__(
        self,
        backbone: VJEPA2Model,
        ssl_config: VJEPA2SSLConfig | None = None,
    ) -> None:
        super().__init__()
        self.ssl_config = ssl_config or VJEPA2SSLConfig()
        self.config: VJEPA2Config = backbone.config

        if self.ssl_config.gradient_checkpointing:
            backbone.gradient_checkpointing_enable()

        # Copy clean unadapted encoder for the stop-gradient EMA teacher before injecting LoRA
        self.target_encoder = copy.deepcopy(backbone.encoder)
        self.target_encoder.gradient_checkpointing = False
        for layer_mod in self.target_encoder.layer:
            layer_mod.gradient_checkpointing = False
        self.target_encoder.eval()
        self.target_encoder.requires_grad_(False)

        self.context_encoder = backbone.encoder
        self.predictor = backbone.predictor
        # HF from_pretrained() initializes modules in eval() mode, which Lightning preserves by default
        # and which silently disables HF GradientCheckpointingLayer (`if self.gradient_checkpointing and self.training`).
        self.context_encoder.train()
        self.predictor.train()

        if self.ssl_config.lora_rank > 0:
            lora_cfg = LoraConfig(
                r=self.ssl_config.lora_rank,
                lora_alpha=self.ssl_config.lora_alpha,
                lora_dropout=self.ssl_config.lora_dropout,
                target_modules=LORA_ATTENTION_TARGET_REGEX,
            )
            inject_adapter_in_model(lora_cfg, self.context_encoder)

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
    ) -> VJEPA2SSL:
        """Load encoder + predictor weights via Hugging Face (`HF_ENDPOINT` / local checkpoint)."""
        cfg = ssl_config or VJEPA2SSLConfig(model_name_or_path=model_name_or_path)
        backbone = VJEPA2Model.from_pretrained(model_name_or_path, dtype=torch_dtype, **hf_kwargs)
        return cls(backbone=backbone, ssl_config=cfg)

    def train(self, mode: bool = True) -> VJEPA2SSL:
        """Keep the stop-gradient EMA `target_encoder` pinned in eval mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

    def token_grid_shape(self, pixel_values_videos: torch.Tensor) -> tuple[int, int, int]:
        """Return `(grid_t, grid_h, grid_w)` patch-token dimensions for `pixel_values_videos` `(B, T, C, H, W)`."""
        _, num_frames, _, height, width = pixel_values_videos.shape
        eff_frames = max(num_frames, self.config.tubelet_size)
        return (
            eff_frames // self.config.tubelet_size,
            height // self.config.patch_size,
            width // self.config.patch_size,
        )

    def encode_context(self, pixel_values_videos: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """Encode ONLY visible `context_idx` `(B, N_ctx)` patches while preserving 3D-RoPE positions."""
        hidden_states = apply_masks(self.context_encoder.embeddings(pixel_values_videos), [context_idx])
        if self.training and self.ssl_config.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states = hidden_states.requires_grad_(True)
        for layer_module in self.context_encoder.layer:
            hidden_states = layer_module(hidden_states, position_mask=context_idx)[0]
        return self.context_encoder.layernorm(hidden_states)

    def predict_from_masked_context(
        self,
        context_hidden_states: torch.Tensor,
        context_idx: torch.Tensor,
        target_mask: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict all `M` target blocks from `context_hidden_states` `(B, N_ctx, D)`."""
        num_targets = len(target_mask)
        batch_size, n_ctx, _ = context_hidden_states.shape
        context_proj = self.predictor.embeddings.predictor_embeddings(context_hidden_states)

        n_mask_tokens = self.predictor.embeddings.num_mask_tokens
        target_proj = torch.cat(
            [
                repeat(
                    self.predictor.embeddings.mask_tokens[
                        (m_idx + 1) % n_mask_tokens if num_targets == 1 else m_idx % n_mask_tokens
                    ],
                    "1 1 d -> b k d",
                    b=batch_size,
                    k=t_mask.size(1),
                )
                for m_idx, t_mask in enumerate(target_mask)
            ],
            dim=0,
        )

        hidden_states = torch.cat([repeat(context_proj, "b n d -> (m b) n d", m=num_targets), target_proj], dim=1)
        position_masks = torch.cat([repeat(context_idx, "b n -> (m b) n", m=num_targets), torch.cat(target_mask, dim=0)], dim=1)

        argsort = torch.argsort(position_masks, dim=1)
        hidden_states, position_masks = self.predictor.sort_tokens(hidden_states, position_masks, argsort)
        for layer_module in self.predictor.layer:
            hidden_states = layer_module(hidden_states, position_masks)[0]
        hidden_states = self.predictor.unsort_tokens(self.predictor.layernorm(hidden_states), argsort)
        return self.predictor.proj(hidden_states[:, n_ctx:])

    @torch.no_grad()
    def encode_target(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """Encode full unmasked video with the stop-gradient EMA `target_encoder`."""
        target_out = self.target_encoder(pixel_values_videos=pixel_values_videos).last_hidden_state
        if self.ssl_config.normalize_targets:
            target_out = F.layer_norm(target_out, (target_out.size(-1),))
        return target_out

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.998) -> None:
        """Exponential Moving Average update (`θ_target <- m * θ_target + (1 - m) * θ_context_effective`)."""
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"EMA momentum must be in [0, 1], got {momentum}")
        if self.ssl_config.lora_rank > 0:
            tgt_modules = dict(self.target_encoder.named_modules())
            for name, mod in self.context_encoder.named_modules():
                if isinstance(mod, LoraLayer):
                    eff_w = mod.base_layer.weight.detach() + mod.get_delta_weight("default").detach()
                    tgt_w = tgt_modules[name].weight.data
                    tgt_w.mul_(momentum).add_(eff_w.to(tgt_w.dtype), alpha=1.0 - momentum)
            return

        for param_ctx, param_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_tgt.data.mul_(momentum).add_(param_ctx.detach().data, alpha=1.0 - momentum)

    @torch.no_grad()
    def extract_features(self, pixel_values_videos: torch.Tensor, use_target_encoder: bool = True) -> torch.Tensor:
        """Extract dense space-time features `(B, N_tokens, D)` for downstream Attentive Probe / retrieval."""
        encoder = self.target_encoder if use_target_encoder else self.context_encoder
        return encoder(pixel_values_videos=pixel_values_videos).last_hidden_state

    def save_backbone_pretrained(self, save_directory: str | Path, use_target_encoder: bool = True) -> None:
        """Export a clean HF-compatible `VJEPA2Model` directory ready for `VJEPA2ForVideoClassification`."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_model = VJEPA2Model(self.config)
        if use_target_encoder or self.ssl_config.lora_rank <= 0:
            chosen_encoder = self.target_encoder if use_target_encoder else self.context_encoder
            export_model.encoder.load_state_dict(chosen_encoder.state_dict())
        else:
            merged_ctx = copy.deepcopy(self.context_encoder)
            for mod in merged_ctx.modules():
                if isinstance(mod, LoraLayer):
                    mod.merge()
            # Remap base_layer keys back to standard HF VJEPA2Encoder parameter names
            export_sd = export_model.encoder.state_dict()
            merged_modules = dict(merged_ctx.named_modules())
            for name, mod in merged_modules.items():
                if isinstance(mod, LoraLayer):
                    export_sd[f"{name}.weight"] = mod.base_layer.weight.detach().cpu()
                    if mod.base_layer.bias is not None:
                        export_sd[f"{name}.bias"] = mod.base_layer.bias.detach().cpu()
            for k, v in merged_ctx.state_dict().items():
                if k in export_sd and "lora_" not in k:
                    export_sd[k] = v.detach().cpu()
            export_model.encoder.load_state_dict(export_sd)
        export_model.predictor.load_state_dict(self.predictor.state_dict())
        export_model.save_pretrained(out_dir)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        context_mask: list[torch.Tensor] | None = None,
        target_mask: list[torch.Tensor] | None = None,
    ) -> VJEPA2SSLOutput:
        """Compute V-JEPA 2 masked latent prediction loss on `pixel_values_videos` `(B, T, C, H, W)`."""
        grid_t, grid_h, grid_w = self.token_grid_shape(pixel_values_videos)
        total_tokens = max(grid_t * grid_h * grid_w, 1)

        if context_mask is None or target_mask is None:
            context_mask, target_mask = self.mask_generator(
                batch_size=pixel_values_videos.size(0),
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                device=pixel_values_videos.device,
            )

        with torch.no_grad():
            target_latents = apply_masks(self.encode_target(pixel_values_videos), target_mask)

        context_hidden = self.encode_context(pixel_values_videos, context_mask[0])
        predicted_latents = self.predict_from_masked_context(context_hidden, context_mask[0], target_mask)
        if self.ssl_config.normalize_targets:
            predicted_latents = F.layer_norm(predicted_latents, (predicted_latents.size(-1),))

        loss = F.smooth_l1_loss(predicted_latents, target_latents, beta=self.ssl_config.loss_beta)
        return VJEPA2SSLOutput(
            loss=loss,
            predicted_latents=predicted_latents,
            target_latents=target_latents,
            context_ratio=float(context_mask[0].size(1)) / total_tokens,
            target_ratio=float(target_mask[0].size(1) * len(target_mask)) / total_tokens,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        self.log("train/loss", out.loss, prog_bar=True, sync_dist=True)
        self.log("train/ctx_ratio", out.context_ratio)
        self.log("train/tgt_ratio", out.target_ratio)
        return out.loss

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer, optimizer_closure=None) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        m_start, m_end = self.ssl_config.ema_momentum_range
        ema_m = cosine_ema_schedule(self.global_step, total_steps, m_start=m_start, m_end=m_end)
        self.update_target_encoder(momentum=ema_m)
        self.log("train/ema_momentum", ema_m)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.ssl_config.lr,
            weight_decay=self.ssl_config.weight_decay,
        )
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.ssl_config.lr * 0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


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


class UnlabeledClipDataModule(L.LightningDataModule):
    """LightningDataModule wrapping `UnlabeledClipDataset` and an HF `AutoVideoProcessor`."""

    def __init__(
        self,
        data_dir: str | Path,
        processor,
        frames_per_clip: int = 16,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.frames_per_clip = frames_per_clip
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset: UnlabeledClipDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = UnlabeledClipDataset(self.data_dir, frames_per_clip=self.frames_per_clip)

    def _collate(self, batch_np: list[np.ndarray]) -> torch.Tensor:
        return self.processor(batch_np, return_tensors="pt")["pixel_values_videos"]

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing unlabeled ROI .mp4 clips")
    parser.add_argument("--output-dir", default="checkpoints/vjepa2_ssl", help="Where to save adapted HF checkpoint")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="HF model ID or local path")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Sampled frames per clip (default: 16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 with LoRA)")
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank on context_encoder attention Q/K/V/proj (0 = full encoder fine-tuning; default: 16)",
    )
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling alpha (default: 32)")
    parser.add_argument(
        "--ema-range",
        type=float,
        nargs=2,
        default=[0.996, 0.9995],
        metavar=("M_START", "M_END"),
        help="Cosine EMA momentum range (default: 0.996 0.9995; use e.g. 0.98 0.996 for <500 steps)",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing (uses more VRAM)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--devices", default="auto", help="Devices count (e.g. '1', '2', 'auto') or comma list ('0,')")
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Lightning training precision (default: bf16-mixed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")
    processor = AutoVideoProcessor.from_pretrained(args.model)
    ssl_cfg = VJEPA2SSLConfig(
        model_name_or_path=args.model,
        frames_per_clip=args.frames_per_clip,
        crop_size=processor.crop_size["height"],
        ema_momentum_range=(float(args.ema_range[0]), float(args.ema_range[1])),
        lr=args.lr,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=not args.no_grad_checkpoint,
    )
    backbone_dtype = (
        torch.bfloat16
        if args.precision == "bf16-mixed"
        else (torch.float16 if args.precision == "16-mixed" else torch.float32)
    )
    model = VJEPA2SSL.from_pretrained(args.model, ssl_config=ssl_cfg, torch_dtype=backbone_dtype)
    datamodule = UnlabeledClipDataModule(
        data_dir=args.data_dir,
        processor=processor,
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    devices = int(args.devices) if str(args.devices).isdigit() else args.devices
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=devices,
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        default_root_dir=args.output_dir,
    )
    trainer.fit(model, datamodule=datamodule)

    if trainer.is_global_zero:
        model.save_backbone_pretrained(args.output_dir, use_target_encoder=True)
        processor.save_pretrained(args.output_dir)
        print(f"saved adapted V-JEPA 2 checkpoint -> {args.output_dir}")


if __name__ == "__main__":
    main()
