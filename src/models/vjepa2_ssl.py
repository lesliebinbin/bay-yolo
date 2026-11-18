#!/usr/bin/env python3
"""Self-supervised V-JEPA 2 domain adaptation built with PyTorch Lightning (`L.LightningModule`).

Architecture (Assran et al., V-JEPA / V-JEPA 2):
- `context_encoder`: encodes ONLY unmasked 3D space-time context tokens via 3D-RoPE `position_mask`.
- `predictor`: predicts target latent representations at masked 3D positions.
- `target_encoder`: stop-gradient EMA copy of `context_encoder` updated automatically in `optimizer_step`.
"""

from __future__ import annotations

import copy
from pathlib import Path

from einops import repeat
import lightning as L
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
import torch
import torch.nn.functional as F
from transformers import VJEPA2Config, VJEPA2Model
from transformers.models.vjepa2.modeling_vjepa2 import apply_masks

from config.vjepa2_config import VJEPA2SSLConfig, VJEPA2SSLOutput
from utils.masking import MultiBlockMaskGenerator3D
from utils.training_utils import cosine_ema_schedule

LORA_ATTENTION_TARGET_REGEX = r".*attention\.(query|key|value|proj)$"


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

        # Copy clean unadapted encoder for the stop-gradient EMA teacher before enabling checkpointing or LoRA
        self.target_encoder = copy.deepcopy(backbone.encoder)
        self.target_encoder.eval()
        self.target_encoder.requires_grad_(False)

        if self.ssl_config.gradient_checkpointing:
            if hasattr(backbone, "activation_checkpointing_enable"):
                backbone.activation_checkpointing_enable(
                    activation_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        self.context_encoder = backbone.encoder
        self.predictor = backbone.predictor
        # HF from_pretrained() returns modules in eval() mode, which disables HF GradientCheckpointingLayer
        # (`if self.gradient_checkpointing and self.training`) until explicitly placed in train() mode.
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
            export_sd = export_model.encoder.state_dict()
            for name, mod in merged_ctx.named_modules():
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
