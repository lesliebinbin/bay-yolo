"""Vectorized 3D space-time multi-block mask sampler for V-JEPA 2."""

from __future__ import annotations

import math
from dataclasses import dataclass

from einops import rearrange
import torch


@dataclass(frozen=True)
class MultiBlockMaskGenerator3D:
    """Stateless 3D space-time block mask sampler producing `(context_mask, target_mask)` index tensors."""

    num_targets: int = 4
    spatial_scale: tuple[float, float] = (0.15, 0.35)
    temporal_scale: tuple[float, float] = (0.50, 1.00)
    aspect_ratio: tuple[float, float] = (0.75, 1.50)

    def _sample_block_size(
        self,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        generator: torch.Generator | None = None,
    ) -> tuple[int, int, int]:
        u_t, u_s, u_ar = torch.rand(3, generator=generator).tolist()
        t_scale = self.temporal_scale[0] + u_t * (self.temporal_scale[1] - self.temporal_scale[0])
        s_scale = self.spatial_scale[0] + u_s * (self.spatial_scale[1] - self.spatial_scale[0])
        ar = math.exp(
            math.log(self.aspect_ratio[0]) + u_ar * (math.log(self.aspect_ratio[1]) - math.log(self.aspect_ratio[0]))
        )

        bt = max(1, min(grid_t, int(round(grid_t * t_scale))))
        spatial_tokens = grid_h * grid_w * s_scale
        bh = max(1, min(grid_h - 1 if grid_h > 1 else 1, int(round(math.sqrt(spatial_tokens * ar)))))
        bw = max(1, min(grid_w - 1 if grid_w > 1 else 1, int(round(math.sqrt(spatial_tokens / ar)))))
        return bt, bh, bw

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Vectorized sampling of `M` 3D target blocks + remaining visible context indices across `batch_size`."""
        bt, bh, bw = self._sample_block_size(grid_t, grid_h, grid_w, generator=generator)
        m = self.num_targets

        # Sample (B, M) top-left corners and add local (bt, bh, bw) block offsets via broadcasting
        t0 = torch.randint(0, grid_t - bt + 1, (batch_size, m, 1, 1, 1), generator=generator)
        h0 = torch.randint(0, grid_h - bh + 1, (batch_size, m, 1, 1, 1), generator=generator)
        w0 = torch.randint(0, grid_w - bw + 1, (batch_size, m, 1, 1, 1), generator=generator)

        dt = torch.arange(bt).view(1, 1, bt, 1, 1)
        dh = torch.arange(bh).view(1, 1, 1, bh, 1)
        dw = torch.arange(bw).view(1, 1, 1, 1, bw)

        flat_targets = (t0 + dt) * (grid_h * grid_w) + (h0 + dh) * grid_w + (w0 + dw)
        flat_targets = rearrange(flat_targets, "b m bt bh bw -> m b (bt bh bw)")

        # Mark occupied target tokens per sample to gather remaining visible context indices
        occupied = torch.zeros((batch_size, grid_t * grid_h * grid_w), dtype=torch.bool)
        occupied.scatter_(1, rearrange(flat_targets, "m b k -> b (m k)"), True)

        unmasked = [row.nonzero(as_tuple=False).squeeze(-1) for row in ~occupied]
        unmasked = [u if u.numel() > 0 else torch.zeros(1, dtype=torch.long) for u in unmasked]
        min_ctx = min(int(u.numel()) for u in unmasked)

        ctx_batch = torch.stack([u[:min_ctx] for u in unmasked], dim=0).to(device)
        target_batches = [blk.to(device) for blk in flat_targets.unbind(0)]
        return [ctx_batch], target_batches
