"""Video ROI preprocessing and clip slicing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from utils.roi import (  # noqa: F401  (canonical shared ROI geometry, re-exported for convenience)
    DEFAULT_YOLO_ROI,
    YoloBox,
    parse_yolo_roi,
    yolo_roi_to_xyxy,
)

if TYPE_CHECKING:
    from preprocessing.roi_clip_slicer import SplitSpec, slice_video_roi_clips

__all__ = [
    "DEFAULT_YOLO_ROI",
    "SplitSpec",
    "YoloBox",
    "parse_yolo_roi",
    "slice_video_roi_clips",
    "yolo_roi_to_xyxy",
]


def __getattr__(name: str):
    if name == "SplitSpec" or name == "slice_video_roi_clips":
        from preprocessing import roi_clip_slicer

        return getattr(roi_clip_slicer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
