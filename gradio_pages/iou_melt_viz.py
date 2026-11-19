#!/usr/bin/env python3
"""Standalone Gradio visualizer for melting overlapping person detections.

Run from the project root:

    uv run gradio_pages/iou_melt_viz.py
    uv run gradio_pages/iou_melt_viz.py --port 7860

The app runs YOLO person detection on an uploaded clip, then groups detections
whose square-expanded influence zones overlap.  The core geometry is delegated
to :func:`utils.melt_by_iou.melt_boxes`:

1. Every ``[cx, cy, w, h]`` box (Ultralytics normalized ``xywhn``) becomes a
   square with side ``max(w, h)`` centered on the box.
2. Two boxes are connected when the IoU of their expanded squares exceeds the
   threshold; connected components form a cluster.
3. Each cluster is replaced by the tight envelope of its *original* boxes.

Three synchronized views make the process easy to inspect:

1. Original person detections.
2. Cluster assignment (colors group detections that were melted together).
3. Melted envelopes (thick) drawn over the original boxes, with the expanded
   influence squares shown as dashed outlines.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from utils.melt_by_fov import melt_by_fov
from utils.melt_by_iou import melt_boxes
from utils.time_span import TimeSpan

DEFAULT_MODEL = "yolo26n.pt"
ALGO_IOU = "Expanded-square IoU"
ALGO_FOV = "FOV expansion"
_MAX_DISPLAY_WIDTH = 960
_MAX_DISPLAY_HEIGHT = 720
_HEADER_HEIGHT = 58

# High-contrast RGB palette.  Cluster IDs index into this cyclically.
_PALETTE: list[tuple[int, int, int]] = [
    (255, 82, 82),      # red
    (64, 196, 255),     # blue
    (105, 240, 105),    # green
    (255, 196, 64),     # amber
    (206, 118, 255),    # violet
    (255, 118, 206),    # pink
    (0, 220, 180),      # teal
    (255, 140, 40),     # orange
    (130, 160, 255),    # periwinkle
    (200, 230, 60),     # lime
    (255, 90, 140),     # rose
    (90, 230, 255),     # cyan
]

_MODEL_CACHE: dict[tuple[str, str], YOLO] = {}


# ---------------------------------------------------------------------------
# Model and media helpers
# ---------------------------------------------------------------------------
def load_model_cached(model_name_or_path: str, device_str: str) -> YOLO:
    """Load (and cache) a YOLO model for the given weights/device pair."""
    key = (model_name_or_path.strip(), device_str)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = YOLO(key[0])
    return _MODEL_CACHE[key]


def _model_names(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def resolve_class_ids(model_names: dict[int, str], requested: list[str]) -> list[int]:
    """Map requested class names (case-insensitive) to model class IDs."""
    name_to_id = {str(name).lower(): int(cid) for cid, name in model_names.items()}
    missing = [name for name in requested if name.lower() not in name_to_id]
    if missing:
        raise gr.Error(f"Model does not have class(es) {missing!r}.")
    return [name_to_id[name.lower()] for name in requested]


def _temp_mp4_path() -> str:
    """Reserve a temporary ``.mp4`` path and return it."""
    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = handle.name
    handle.close()
    return path


class BrowserVideoWriter:
    """Stream RGB frames to a browser-friendly H.264 ``.mp4`` file.

    Frames are piped directly to ``ffmpeg`` so long clips do not accumulate in
    RAM.  If ``ffmpeg`` is unavailable we fall back to OpenCV's ``mp4v`` codec.
    """

    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.closed = False

        self._proc: subprocess.Popen[bytes] | None = None
        self._writer: cv2.VideoWriter | None = None

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is not None:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{self.width}x{self.height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-loglevel",
                "error",
                self.path,
            ]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self._writer = cv2.VideoWriter(
                self.path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height),
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    "Could not open an OpenCV video writer and ffmpeg is unavailable."
                )

    def __enter__(self) -> BrowserVideoWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
            frame_rgb = cv2.resize(
                frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.write(frame_rgb.tobytes())
        elif self._writer is not None:
            self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> str:
        if self.closed:
            return self.path
        self.closed = True
        if self._proc is not None:
            assert self._proc.stdin is not None
            self._proc.stdin.close()
            self._proc.wait()
            if self._proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {self._proc.returncode}")
        if self._writer is not None:
            self._writer.release()
        return self.path


# ---------------------------------------------------------------------------
# Geometry helpers (torch -> numpy only at the drawing boundary)
# ---------------------------------------------------------------------------
@dataclass
class MeltFrame:
    """Per-frame geometry used by the three visualization views."""

    orig_xyxy: np.ndarray          # (N, 4) pixel coords
    confs: np.ndarray              # (N,)
    expanded_xyxy: np.ndarray      # (N, 4) pixel coords of influence squares
    melted_xyxy: np.ndarray        # (K, 4) pixel coords of cluster envelopes
    clusters: list[list[int]]      # K lists of original detection indices
    cluster_of: np.ndarray         # (N,) cluster index for each detection

    @property
    def num_detections(self) -> int:
        return int(self.orig_xyxy.shape[0])

    @property
    def num_envelopes(self) -> int:
        return int(self.melted_xyxy.shape[0])


def _empty_melt_frame() -> MeltFrame:
    empty = np.zeros((0, 4), dtype=np.float32)
    return MeltFrame(
        orig_xyxy=empty,
        confs=np.zeros((0,), dtype=np.float32),
        expanded_xyxy=empty,
        melted_xyxy=empty,
        clusters=[],
        cluster_of=np.zeros((0,), dtype=np.int64),
    )


def _boxes_xywhn_to_xyxy_px(
    boxes_xywhn: torch.Tensor, width: int, height: int
) -> np.ndarray:
    """Convert normalized ``[cx, cy, w, h]`` boxes to pixel ``[x1, y1, x2, y2]``."""
    if boxes_xywhn.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes_xywhn.unbind(-1)
    xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    scale = torch.tensor([width, height, width, height], dtype=xyxy.dtype)
    return (xyxy * scale).cpu().numpy()


def _expanded_squares_xyxy_px(
    boxes_xywhn: torch.Tensor, width: int, height: int, expansion: float = 1.0
) -> np.ndarray:
    """Return the pixel-space ``expansion * max(w, h)`` influence squares."""
    if boxes_xywhn.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes_xywhn.unbind(-1)
    side = float(expansion) * torch.maximum(w, h)
    half = side / 2
    xyxy = torch.stack([cx - half, cy - half, cx + half, cy + half], dim=-1)
    scale = torch.tensor([width, height, width, height], dtype=xyxy.dtype)
    return (xyxy * scale).cpu().numpy()


def _build_melt_frame(
    boxes: object,
    threshold: float,
    max_iterations: int,
    width: int,
    height: int,
    algorithm: str = ALGO_IOU,
    expansion: float = 1.0,
) -> MeltFrame:
    """Run YOLO ``xywhn`` boxes through the selected melting algorithm."""
    if boxes is None or len(boxes) == 0:  # type: ignore[arg-type]
        return _empty_melt_frame()

    xywhn = boxes.xywhn.detach().cpu().float()  # type: ignore[attr-defined]
    confs = boxes.conf.detach().cpu().float()  # type: ignore[attr-defined]

    # Drop degenerate boxes: they cannot form meaningful envelopes.
    valid = (xywhn[:, 2] > 1e-6) & (xywhn[:, 3] > 1e-6)
    xywhn = xywhn[valid]
    confs = confs[valid]
    if xywhn.numel() == 0:
        return _empty_melt_frame()

    if algorithm == ALGO_FOV:
        melted_xywhn, clusters = melt_by_fov(
            xywhn,
            a=float(expansion),
            return_clusters=True,
        )
    else:
        melted_xywhn, clusters = melt_boxes(
            xywhn,
            threshold=float(threshold),
            max_iterations=int(max_iterations),
            return_clusters=True,
        )

    cluster_of = np.full(int(xywhn.shape[0]), -1, dtype=np.int64)
    for cid, members in enumerate(clusters):
        for idx in members:
            cluster_of[idx] = cid

    return MeltFrame(
        orig_xyxy=_boxes_xywhn_to_xyxy_px(xywhn, width, height),
        confs=confs.numpy(),
        expanded_xyxy=_expanded_squares_xyxy_px(
            xywhn, width, height, expansion=expansion
        ),
        melted_xyxy=_boxes_xywhn_to_xyxy_px(melted_xywhn, width, height),
        clusters=clusters,
        cluster_of=cluster_of,
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _palette_color(index: int) -> tuple[int, int, int]:
    return _PALETTE[int(index) % len(_PALETTE)]


def _dim_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(c * factor)))) for c in color)


def _clip_xyxy(
    xyxy: np.ndarray, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    x1 = max(0.0, min(x1, width - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    x2 = max(0.0, min(x2, width - 1.0))
    y2 = max(0.0, min(y2, height - 1.0))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _draw_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.50,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    img_h, img_w = img.shape[:2]

    x = int(org[0])
    y = int(org[1])
    x = max(0, min(x, max(0, img_w - tw - 4)))
    y = max(th + 4, min(y, img_h - baseline - 2))

    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _draw_box(
    img: np.ndarray,
    xyxy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> tuple[int, int, int, int] | None:
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(xyxy, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


def _draw_dashed_line(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: float = 8.0,
    gap: float = 5.0,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        a = (int(round(x1 + dx * t)), int(round(y1 + dy * t)))
        b = (int(round(x1 + dx * t2)), int(round(y1 + dy * t2)))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        t += dash + gap


def _draw_dashed_rect(
    img: np.ndarray,
    xyxy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(xyxy, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        return
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
    _draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
    _draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)


def _fit_text_scale(
    text: str, max_scale: float, max_width: int, thickness: int = 1
) -> float:
    """Shrink a font scale until ``text`` fits into ``max_width`` pixels."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max_scale
    while scale > 0.32:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_width:
            return scale
        scale -= 0.02
    return max(0.32, scale)


def _draw_panel_header(img: np.ndarray, title: str, subtitle: str | None = None) -> None:
    img_h, img_w = img.shape[:2]
    bar_h = min(_HEADER_HEIGHT if subtitle else 40, img_h)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, bar_h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.62, img, 0.38, 0)

    max_width = max(40, img_w - 24)
    title_scale = _fit_text_scale(title, 0.68, max_width, thickness=2)
    cv2.putText(
        img,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if subtitle:
        subtitle_scale = _fit_text_scale(subtitle, 0.46, max_width, thickness=1)
        cv2.putText(
            img,
            subtitle,
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            subtitle_scale,
            (215, 215, 215),
            1,
            cv2.LINE_AA,
        )


def _draw_empty_message(img: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.80
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    img_h, img_w = img.shape[:2]
    org = (max(0, (img_w - tw) // 2), min(img_h - 8, (img_h + th) // 2))
    cv2.putText(img, text, org, font, scale, (235, 235, 235), thickness, cv2.LINE_AA)


def _box_center(xyxy: np.ndarray) -> tuple[int, int]:
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


# ---------------------------------------------------------------------------
# Per-view renderers
# ---------------------------------------------------------------------------
def _render_original_view(
    img: np.ndarray, melt: MeltFrame, conf_threshold: float
) -> np.ndarray:
    if melt.num_detections == 0:
        _draw_empty_message(img, "No person detections")
    else:
        color = (0, 220, 255)
        for i in range(melt.num_detections):
            box = _draw_box(img, melt.orig_xyxy[i], color, 2)
            if box is not None:
                x1, y1, _, _ = box
                _draw_label(
                    img,
                    f"P{i} {float(melt.confs[i]):.2f}",
                    (x1 + 3, max(y1 + 16, _HEADER_HEIGHT + 14)),
                    color,
                )

    box_word = "box" if melt.num_detections == 1 else "boxes"
    _draw_panel_header(
        img,
        "1. Original detections",
        f"{melt.num_detections} {box_word} | conf >= {conf_threshold:.2f}",
    )
    return img


def _render_cluster_view(
    img: np.ndarray, melt: MeltFrame, show_links: bool
) -> np.ndarray:
    if melt.num_detections == 0:
        _draw_empty_message(img, "No person detections")
    else:
        # Link each detection to the center of its merged envelope.
        if show_links:
            for cid, members in enumerate(melt.clusters):
                if len(members) < 2:
                    continue
                color = _palette_color(cid)
                ex, ey = _box_center(melt.melted_xyxy[cid])
                for idx in members:
                    ox, oy = _box_center(melt.orig_xyxy[idx])
                    cv2.line(img, (ox, oy), (ex, ey), color, 1, cv2.LINE_AA)

        # Dashed envelopes preview the group extent.
        for cid, xyxy in enumerate(melt.melted_xyxy):
            _draw_dashed_rect(img, xyxy, _palette_color(cid), 1)

        # Original boxes colored by cluster.
        for i in range(melt.num_detections):
            cid = int(melt.cluster_of[i])
            color = _palette_color(cid)
            box = _draw_box(img, melt.orig_xyxy[i], color, 2)
            if box is not None:
                x1, y1, x2, y2 = box
                _draw_label(
                    img,
                    f"C{cid}",
                    (x1 + 3, max(y1 + 16, _HEADER_HEIGHT + 14)),
                    color,
                )
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)

    cluster_word = "cluster" if melt.num_envelopes == 1 else "clusters"
    subtitle = (
        f"{melt.num_detections} detections -> {melt.num_envelopes} {cluster_word} "
        f"| color = merged group"
    )
    _draw_panel_header(img, "2. Cluster assignment", subtitle)
    return img


def _render_melted_view(
    img: np.ndarray,
    melt: MeltFrame,
    show_squares: bool,
    square_label: str = "max(w, h)",
) -> np.ndarray:
    if melt.num_detections == 0:
        _draw_empty_message(img, "No person detections")
    else:
        # Expanded influence squares show *why* boxes were considered close.
        if show_squares:
            for i in range(melt.num_detections):
                cid = int(melt.cluster_of[i])
                color = _dim_color(_palette_color(cid), 0.55)
                _draw_dashed_rect(img, melt.expanded_xyxy[i], color, 1)

        # Original boxes stay visible as thin grey references.
        for xyxy in melt.orig_xyxy:
            _draw_box(img, xyxy, (170, 170, 170), 1)

        if melt.num_envelopes:
            # Translucent fill makes the envelopes easy to see over the footage.
            overlay = img.copy()
            img_h, img_w = img.shape[:2]
            for cid, xyxy in enumerate(melt.melted_xyxy):
                x1, y1, x2, y2 = _clip_xyxy(xyxy, img_w, img_h)
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), _palette_color(cid), -1)
            img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)

            for cid, xyxy in enumerate(melt.melted_xyxy):
                color = _palette_color(cid)
                box = _draw_box(img, xyxy, color, 3)
                if box is not None:
                    x1, y1, _, _ = box
                    _draw_label(
                        img,
                        f"E{cid} (n={len(melt.clusters[cid])})",
                        (x1 + 3, max(y1 + 16, _HEADER_HEIGHT + 14)),
                        color,
                        scale=0.52,
                    )

    envelope_word = "envelope" if melt.num_envelopes == 1 else "envelopes"
    subtitle = (
        f"{melt.num_detections} detections -> {melt.num_envelopes} {envelope_word}"
        + (f" | dashed = expanded {square_label} squares" if show_squares else "")
    )
    _draw_panel_header(img, "3. Melted envelopes", subtitle)
    return img


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------
def _parse_timecode(value: object, field_name: str) -> float:
    """Accept ``HH:MM:SS`` from the UI, or seconds from an API caller."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip()
    try:
        return TimeSpan.parse_hms(text)
    except Exception as exc:
        raise gr.Error(f"{field_name} must be HH:MM:SS, got {text!r}.") from exc


def run_melt_visualization(
    uploaded_video: str | None,
    model_id: str,
    device_choice: str,
    imgsz: int,
    conf: float,
    algorithm: str,
    threshold: float,
    expansion: float,
    max_iterations: float,
    start_sec: str | float,
    duration_sec: str | float,
    target_fps: float,
    show_squares: bool,
    show_links: bool,
    progress=gr.Progress(),
) -> tuple[str, str, str, str]:
    """Detect persons, melt their boxes, and render the three comparison views."""
    if not uploaded_video or not Path(uploaded_video).is_file():
        raise gr.Error("Upload a video clip first.")
    if not str(model_id).strip():
        raise gr.Error("Enter a YOLO weights path or model name.")

    video_path = Path(uploaded_video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")

    try:
        use_fov = algorithm == ALGO_FOV
        expansion_a = float(expansion or 1.0)
        square_label = (
            f"{expansion_a:.2f} x max(w, h)" if use_fov else "max(w, h)"
        )

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if src_w <= 0 or src_h <= 0:
            raise gr.Error("The uploaded video has invalid dimensions.")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Keep output small enough for fast H.264 encoding and Gradio playback.
        scale = min(
            1.0,
            _MAX_DISPLAY_WIDTH / float(src_w),
            _MAX_DISPLAY_HEIGHT / float(src_h),
        )
        disp_w = max(2, int(round(src_w * scale)) // 2 * 2)
        disp_h = max(2, int(round(src_h * scale)) // 2 * 2)

        target_fps = float(target_fps or src_fps or 1.0)
        target_fps = min(max(target_fps, 1.0), max(src_fps, 1.0))
        step = max(1, int(round(src_fps / target_fps))) if src_fps > 0 else 1
        effective_fps = src_fps / step if src_fps > 0 else target_fps
        playback_fps = min(max(effective_fps, 1.0), 60.0)

        start_sec = _parse_timecode(start_sec, "Start Time")
        duration_sec = _parse_timecode(duration_sec, "Duration")
        video_duration = (
            total_frames / src_fps if total_frames > 0 and src_fps > 0 else None
        )
        if video_duration is not None and start_sec >= video_duration:
            raise gr.Error(
                f"Start time {start_sec:.1f}s is at or beyond the video duration "
                f"({video_duration:.1f}s)."
            )

        start_frame = int(round(start_sec * src_fps)) if src_fps > 0 else 0
        if total_frames > 0:
            start_frame = min(start_frame, max(0, total_frames - 1))
        if start_frame > 0:
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

        if duration_sec > 0 and src_fps > 0:
            frame_budget = max(1, int(round(duration_sec * src_fps)))
        else:
            # No duration: process from the start position to the end.
            frame_budget = total_frames if total_frames > 0 else 10**9

        remaining_frames = (
            max(0, total_frames - start_frame) if total_frames > 0 else frame_budget
        )
        source_budget = min(frame_budget, remaining_frames)
        expected_frames = (
            int(math.ceil(source_budget / step)) if source_budget > 0 else 0
        )

        model = load_model_cached(str(model_id).strip(), device_choice)
        class_ids = resolve_class_ids(_model_names(model), ["person"])

        frame_stats: list[tuple[int, int]] = []
        frame_idx = 0

        with ExitStack() as stack:
            writers = [
                stack.enter_context(
                    BrowserVideoWriter(_temp_mp4_path(), disp_w, disp_h, playback_fps)
                )
                for _ in range(3)
            ]

            source_reads = 0
            while source_reads < source_budget:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if source_reads % step == 0:
                    # Detect on the native frame so small people are not lost to
                    # an extra downscale; xywhn stays valid for the display resize.
                    result = model.predict(
                        source=frame_bgr,
                        classes=class_ids,
                        conf=float(conf),
                        imgsz=int(imgsz),
                        device=device_choice,
                        verbose=False,
                    )[0]

                    display_bgr = cv2.resize(
                        frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                    )
                    melt = _build_melt_frame(
                        result.boxes,
                        threshold=float(threshold),
                        max_iterations=int(max_iterations),
                        width=disp_w,
                        height=disp_h,
                        algorithm=algorithm,
                        expansion=(expansion_a if use_fov else 1.0),
                    )
                    frame_stats.append((melt.num_detections, melt.num_envelopes))

                    frame_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                    writers[0].write(
                        _render_original_view(frame_rgb.copy(), melt, float(conf))
                    )
                    writers[1].write(
                        _render_cluster_view(frame_rgb.copy(), melt, bool(show_links))
                    )
                    writers[2].write(
                        _render_melted_view(
                            frame_rgb.copy(),
                            melt,
                            bool(show_squares),
                            square_label=square_label,
                        )
                    )

                    frame_idx += 1
                    if expected_frames > 0:
                        progress(
                            min(frame_idx / float(expected_frames), 1.0),
                            desc=f"Frame {frame_idx}/{expected_frames}",
                        )

                source_reads += 1

        if frame_idx == 0:
            raise gr.Error("Could not read any frames from the uploaded video.")

        total_dets = sum(n for n, _ in frame_stats)
        total_envs = sum(k for _, k in frame_stats)
        avg_dets = total_dets / len(frame_stats)
        avg_envs = total_envs / len(frame_stats)
        reduction = 100.0 * (1.0 - total_envs / total_dets) if total_dets > 0 else 0.0
        frames_with_merges = sum(1 for n, k in frame_stats if k < n)
        max_envs = max(k for _, k in frame_stats)

        start_label = TimeSpan.format_hms(start_sec)
        duration_label = (
            TimeSpan.format_hms(duration_sec) if duration_sec > 0 else "to end"
        )
        sample_label = "every frame" if step == 1 else f"every {step} frames"
        if use_fov:
            algo_label = f"FOV expansion | a={expansion_a:.2f}"
        else:
            algo_label = (
                f"Expanded-square IoU | threshold={float(threshold):.2f} | "
                f"max iterations={int(max_iterations)}"
            )
        metrics_md = (
            f"**Frames:** `{len(frame_stats)}` &nbsp;|&nbsp; "
            f"**Avg persons/frame:** `{avg_dets:.2f}` &nbsp;|&nbsp; "
            f"**Avg envelopes/frame:** `{avg_envs:.2f}` &nbsp;|&nbsp; "
            f"**Box reduction:** `{reduction:.1f}%` &nbsp;|&nbsp; "
            f"**Frames with merges:** `{frames_with_merges}/{len(frame_stats)}` &nbsp;|&nbsp; "
            f"**Max envelopes:** `{max_envs}`\n\n"
            f"Start `{start_label}` &nbsp;|&nbsp; "
            f"Duration `{duration_label}` &nbsp;|&nbsp; "
            f"Algorithm `{algo_label}` &nbsp;|&nbsp; "
            f"Source `{src_fps:.1f}` fps -> `{playback_fps:.1f}` fps "
            f"({sample_label}) &nbsp;|&nbsp; "
            f"Output `{disp_w}x{disp_h}`"
        )

        return (
            writers[0].path,
            writers[1].path,
            writers[2].path,
            metrics_md,
        )
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
def build_app(default_model: str) -> gr.Blocks:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_choices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]

    with gr.Blocks(title="YOLO Person Box Melting Visualizer") as demo:
        gr.Markdown(
            "## YOLO Person Box Melting Visualizer\n"
            "Detects **person** boxes with YOLO and melts nearby people into tight envelopes.\n\n"
            "**Algorithms:**\n"
            "- **Expanded-square IoU:** expand each box to a square of side `max(w, h)` and "
            "cluster boxes whose expanded squares overlap above the IoU threshold.\n"
            "- **FOV expansion:** grow a box's field of view by `a × max(w, h)` and merge any "
            "box whose center falls inside it; append the new envelope and repeat until stable.\n\n"
            "Expansion is computed in normalized `xywhn` space, so on non-square frames the "
            "dashed influence zones look rectangular.\n\n"
            "**Views:** 1) raw detections, 2) cluster membership, 3) melted envelopes with the "
            "influence squares shown as dashed outlines.\n\n"
            "Use **Start Time** and **Duration** in `HH:MM:SS` format to process a specific "
            "segment. A duration of `00:00:00` processes from the start time to the end. "
            "**Target Processing FPS** samples the segment; lower values are faster."
        )

        with gr.Row():
            model_box = gr.Textbox(
                value=default_model, label="YOLO Weights", scale=2
            )
            device_dd = gr.Dropdown(
                choices=device_choices, value=default_device, label="Device", scale=1
            )
            imgsz_dd = gr.Dropdown(
                choices=[480, 640, 800, 960, 1280],
                value=960,
                label="Inference Size",
                scale=1,
            )
            algo_dd = gr.Dropdown(
                choices=[ALGO_IOU, ALGO_FOV],
                value=ALGO_IOU,
                label="Melting Algorithm",
                scale=2,
            )

        upload_vid = gr.Video(
            label="Upload Clip (.mp4)", sources=["upload"], height=240
        )

        with gr.Row():
            conf_slider = gr.Slider(
                0.05, 0.90, value=0.25, step=0.05, label="Person Confidence"
            )
            threshold_slider = gr.Slider(
                0.0,
                1.0,
                value=0.40,
                step=0.05,
                label="Expanded-Square IoU Threshold",
            )
            iter_slider = gr.Slider(
                -1, 5, value=-1, step=1, label="Max Melt Iterations (-1 = until stable)"
            )
            expansion_slider = gr.Slider(
                1.0,
                3.0,
                value=1.5,
                step=0.1,
                label="FOV Expansion Factor (a)",
                visible=False,
            )

        with gr.Row():
            start_text = gr.Textbox(
                value="00:00:00",
                label="Start Time (HH:MM:SS)",
                placeholder="00:00:00",
            )
            duration_text = gr.Textbox(
                value="00:00:00",
                label="Duration (HH:MM:SS; 00:00:00 = to end)",
                placeholder="00:00:00",
            )
            target_fps_slider = gr.Slider(
                1, 30, value=5, step=1, label="Target Processing FPS"
            )

        with gr.Row():
            show_squares_cb = gr.Checkbox(
                value=True, label="Show expanded influence squares"
            )
            show_links_cb = gr.Checkbox(
                value=True, label="Show cluster connection lines"
            )
            run_btn = gr.Button(
                "Run Box Melting Visualization", variant="primary", interactive=False
            )

        metrics_out = gr.Markdown(
            "*Upload a video clip above to enable **Run Box Melting Visualization**.*"
        )

        with gr.Row():
            vid_original = gr.Video(
                label="1. Original person detections", autoplay=True, loop=True
            )
            vid_clusters = gr.Video(
                label="2. Cluster assignment (color = merged group)",
                autoplay=True,
                loop=True,
            )

        vid_melted = gr.Video(
            label="3. Melted envelopes over original boxes",
            autoplay=True,
            loop=True,
        )

        def _toggle_run_btn(video_file: str | None):
            has_video = bool(video_file and Path(video_file).is_file())
            if not has_video:
                return (
                    gr.update(interactive=False),
                    "*Upload a video clip above to enable **Run Box Melting Visualization**.*",
                )

            info = ""
            try:
                cap = cv2.VideoCapture(str(video_file))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                if fps > 0 and total > 0:
                    info = (
                        f" Video: `{width}x{height}`, `{fps:.2f}` fps, "
                        f"duration `{TimeSpan.format_hms(total / fps)}`."
                    )
            except Exception:
                info = ""

            hint = (
                "*Click **Run Box Melting Visualization** to detect, cluster, and melt "
                f"person boxes.{info}*"
            )
            return gr.update(interactive=True), hint

        def _toggle_algo(algorithm: str):
            is_iou = algorithm == ALGO_IOU
            return (
                gr.update(visible=is_iou),
                gr.update(visible=is_iou),
                gr.update(visible=not is_iou),
            )

        algo_dd.change(
            _toggle_algo,
            inputs=[algo_dd],
            outputs=[threshold_slider, iter_slider, expansion_slider],
        )
        upload_vid.change(
            _toggle_run_btn, inputs=[upload_vid], outputs=[run_btn, metrics_out]
        )
        run_btn.click(
            run_melt_visualization,
            inputs=[
                upload_vid,
                model_box,
                device_dd,
                imgsz_dd,
                conf_slider,
                algo_dd,
                threshold_slider,
                expansion_slider,
                iter_slider,
                start_text,
                duration_text,
                target_fps_slider,
                show_squares_cb,
                show_links_cb,
            ],
            outputs=[vid_original, vid_clusters, vid_melted, metrics_out],
        )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="YOLO weights path or model name"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Gradio server bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7861, help="Gradio server port (default: 7861)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = build_app(default_model=args.model)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
