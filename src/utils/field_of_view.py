"""Subject field-of-view (FoV) crop utilities.

A FoV crop keeps a detected subject (e.g. a person) plus a small surrounding margin
(``pad``), optionally upscaled (``upscale``) or resized to a target width
(``target_width``) so that fine-grained items (helmet, goggles, phone in hand)
occupy a large share of the crop's pixels instead of being downsampled away in
a wide scene crop.

Box convention follows the rest of ``utils`` (see ``crops.py``): pixel-space
`(x1, y1, x2, y2)` within a frame of shape `(H, W, ...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FieldOfView:
    """Padded subject crop: box expanded by ``pad`` on every side, clamped to the frame.

    ``square`` (e.g. person crops): after padding, the shorter side is widened to
    match the longer one, centered on the box (shifted inward if that would run
    past the frame), so the crop is always square before any resize.

    Resize semantics (in priority order):
    - ``target_width``: resize so the output width equals this value (aspect kept).
    - ``upscale``: multiply the clamped crop size by this factor (e.g. ``2.0``).
    - neither: keep native resolution (no downscale, no upscale).
    """

    pad: float = 0.10
    upscale: float | None = None
    target_width: int | None = None
    even_dims: bool = True
    square: bool = False
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if self.pad < 0.0:
            raise ValueError(f"pad must be >= 0, got {self.pad!r}")
        if self.upscale is not None and self.upscale <= 0.0:
            raise ValueError(f"upscale must be > 0, got {self.upscale!r}")
        if self.target_width is not None and self.target_width < 2:
            raise ValueError(f"target_width must be >= 2, got {self.target_width!r}")

    def compute_box(self, box, frame_shape: tuple[int, int] | np.ndarray) -> tuple[int, int, int, int]:
        """Expand `box` (pixel xyxy) by ``pad`` of its own size on every side, clamped to the frame."""
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if h < 2 or w < 2:
            raise ValueError(f"invalid frame shape {tuple(frame_shape)}")
        x1, y1, x2, y2 = (float(v) for v in box)
        if not all(np.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate box {tuple(box)!r} (need finite x2>x1, y2>y1)")

        pad_x = 0.5 * (x2 - x1) * self.pad
        pad_y = 0.5 * (y2 - y1) * self.pad
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(w, int(round(x2 + pad_x)))
        cy2 = min(h, int(round(y2 + pad_y)))

        if self.square:
            # Widen the shorter side to the longer one (square person/face crop
            # convention), centered on the box and shifted inward to stay in frame.
            side = min(max(cx2 - cx1, cy2 - cy1), w, h)
            cx = int(round((cx1 + cx2) / 2))
            cy = int(round((cy1 + cy2) / 2))
            cx1 = min(max(cx - side // 2, 0), max(0, w - side))
            cy1 = min(max(cy - side // 2, 0), max(0, h - side))
            cx2 = cx1 + side
            cy2 = cy1 + side

        if self.even_dims:
            # Prefer extending the far edge; if it is clamped to the frame, move
            # the near edge in by 1 so the span becomes even (only if there is room).
            if (cx2 - cx1) % 2 == 1:
                if cx2 < w:
                    cx2 += 1
                elif cx1 > 0:
                    cx1 -= 1
            if (cy2 - cy1) % 2 == 1:
                if cy2 < h:
                    cy2 += 1
                elif cy1 > 0:
                    cy1 -= 1
        if cx2 <= cx1 or cy2 <= cy1:
            raise ValueError(f"box {tuple(box)!r} degenerates after clamping to {w}x{h}")
        return cx1, cy1, cx2, cy2

    def output_size(self, box, frame_shape: tuple[int, int] | np.ndarray) -> tuple[int, int]:
        """Output crop size `(width, height)` after clamping and resize options."""
        cx1, cy1, cx2, cy2 = self.compute_box(box, frame_shape)
        w, h = cx2 - cx1, cy2 - cy1
        if self.target_width is not None:
            out_w = float(self.target_width)
        elif self.upscale is not None:
            out_w = w * self.upscale
        else:
            return w, h
        out_w = max(2.0, out_w)
        out_h = max(2.0, out_w * (h / w))
        if self.even_dims:
            out_w = max(2.0, 2.0 * round(out_w / 2.0))
            out_h = max(2.0, 2.0 * round(out_h / 2.0))
        return int(out_w), int(out_h)

    def crop(self, frame: np.ndarray, box) -> np.ndarray:
        """Cut the subject FoV from `frame` (HxWx3) and resize per this configuration."""
        h, w = frame.shape[:2]
        cx1, cy1, cx2, cy2 = self.compute_box(box, (h, w))
        region = frame[cy1:cy2, cx1:cx2]
        out_w, out_h = self.output_size(box, (h, w))
        if (out_w, out_h) != (region.shape[1], region.shape[0]):
            interpolation = cv2.INTER_CUBIC if out_w > region.shape[1] else cv2.INTER_AREA
            region = cv2.resize(region, (out_w, out_h), interpolation=interpolation)
        return region

    def save_crop(self, frame: np.ndarray, box, out_path: str | Path) -> Path:
        """Write the subject FoV crop as JPEG and return its path."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), self.crop(frame, box), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return out


# Typical person FoV preset: small margin + square crop (shorter side widened
# to the longer one) + 2x upscale so fine-grained items (helmet, goggles,
# phone/food in hand) stay large and centered in the crop.
PERSON_FOV = FieldOfView(pad=0.10, upscale=2.0, square=True)

__all__ = ["FieldOfView", "PERSON_FOV"]
