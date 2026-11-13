"""Video ROI preprocessing and clip slicing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from preprocessing.roi_clip_slicer import (
        DEFAULT_YOLO_ROI,
        SplitSpec,
        YoloBox,
        parse_yolo_roi,
        slice_video_roi_clips,
        yolo_roi_to_xyxy,
    )

__all__ = [
    "DEFAULT_YOLO_ROI",
    "SplitSpec",
    "YoloBox",
    "parse_yolo_roi",
    "slice_video_roi_clips",
    "yolo_roi_to_xyxy",
]


def __getattr__(name: str):
    if name in __all__:
        from preprocessing import roi_clip_slicer

        return getattr(roi_clip_slicer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
