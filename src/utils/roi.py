"""Shared YOLO normalized-ROI geometry utilities.

Bounding boxes follow Ultralytics YOLO normalized `(x_center, y_center, width, height)` in `[0, 1]`.
Used by `preprocessing.roi_clip_slicer` (clip slicing) and `tasks.roi_extract` (full-duration ROI crops).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class YoloBox:
    """Normalized Ultralytics YOLO bounding box `(x_center, y_center, width, height)` in `[0, 1]`."""

    x_center: float
    y_center: float
    width: float
    height: float

    def validate(self) -> YoloBox:
        for field_name, val in (
            ("x_center", self.x_center),
            ("y_center", self.y_center),
            ("width", self.width),
            ("height", self.height),
        ):
            if not (math.isfinite(val) and 0.0 <= val <= 1.0):
                raise ValueError(f"YOLO ROI {field_name}={val!r} must be within [0, 1]")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError(f"YOLO ROI width and height must be > 0, got ({self.width}, {self.height})")
        x1 = self.x_center - 0.5 * self.width
        y1 = self.y_center - 0.5 * self.height
        x2 = self.x_center + 0.5 * self.width
        y2 = self.y_center + 0.5 * self.height
        eps = 1e-6
        if x1 < -eps or y1 < -eps or x2 > 1.0 + eps or y2 > 1.0 + eps:
            raise ValueError(
                f"YOLO ROI ({self.x_center}, {self.y_center}, {self.width}, {self.height}) "
                f"extends outside [0, 1] -> [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]"
            )
        return self


# Bottom-left quadrant (1/2 width x 1/2 height): x in [0.0, 0.5], y in [0.5, 1.0]
DEFAULT_YOLO_ROI = YoloBox(x_center=0.25, y_center=0.75, width=0.50, height=0.50)


def parse_yolo_roi(values: list[float] | tuple[float, float, float, float] | str) -> YoloBox:
    """Parse 4 normalized floats `(x_center, y_center, width, height)` or a YOLO label line."""
    if isinstance(values, str):
        tokens = values.replace(",", " ").split()
        # Support optional leading class_id (`0 0.25 0.75 0.5 0.5`) as well as (`0.25 0.75 0.5 0.5`)
        if len(tokens) == 5 and tokens[0].isdigit():
            tokens = tokens[1:]
        if len(tokens) != 4:
            raise ValueError(f"expected 4 YOLO values (x y w h), got {values!r}")
        nums = [float(t) for t in tokens]
    else:
        if len(values) != 4:
            raise ValueError(f"expected 4 YOLO values (x y w h), got {values!r}")
        nums = [float(v) for v in values]
    return YoloBox(*nums).validate()


def yolo_roi_to_xyxy(roi: YoloBox, frame_width: int, frame_height: int, even_dims: bool = True) -> tuple[int, int, int, int]:
    """Convert normalized `YoloBox` into pixel `(x1, y1, x2, y2)` bounds."""
    roi.validate()
    if frame_width <= 1 or frame_height <= 1:
        raise ValueError(f"invalid frame dimensions: {frame_width}x{frame_height}")

    x1 = max(0, int(round((roi.x_center - 0.5 * roi.width) * frame_width)))
    y1 = max(0, int(round((roi.y_center - 0.5 * roi.height) * frame_height)))
    x2 = min(frame_width, int(round((roi.x_center + 0.5 * roi.width) * frame_width)))
    y2 = min(frame_height, int(round((roi.y_center + 0.5 * roi.height) * frame_height)))

    if even_dims:
        if (x2 - x1) % 2 == 1:
            if x2 < frame_width:
                x2 += 1
            else:
                x1 = max(0, x1 - 1)
        if (y2 - y1) % 2 == 1:
            if y2 < frame_height:
                y2 += 1
            else:
                y1 = max(0, y1 - 1)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate pixel ROI {(x1, y1, x2, y2)} for {frame_width}x{frame_height}")
    return x1, y1, x2, y2


__all__ = [
    "DEFAULT_YOLO_ROI",
    "YoloBox",
    "parse_yolo_roi",
    "yolo_roi_to_xyxy",
]
