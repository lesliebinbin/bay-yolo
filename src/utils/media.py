"""Media source classification, OpenCV capture/image loading, and interactive display."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def configure_opencv_ffmpeg() -> None:
    """Force RTSP over TCP in OpenCV FFMPEG backend to prevent UDP packet loss / torn frames."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


configure_opencv_ffmpeg()

import cv2  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


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



@dataclass(frozen=True)
class MediaSource:
    """Represents a validated input `--source` (RTSP/HTTP stream, webcam, video file, image, or image folder)."""

    source: str
    kind: str

    @property
    def is_local_video(self) -> bool:
        return self.kind in {"video", "file"}

    @property
    def is_image_source(self) -> bool:
        return self.kind in {"image_dir", "image"}

    @staticmethod
    def natural_sort_key(path: Path) -> list[int | str]:
        """Sort filenames with embedded frame/instance integers in human numeric order."""
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

    @classmethod
    def classify(cls, source: str, allow_images: bool = True) -> str:
        """Classify `source` into `'stream'`, `'webcam'`, `'image_dir'`, `'image'`, `'video'`, or `'file'`."""
        if "://" in source:
            return "stream"
        path = Path(source)
        if allow_images:
            if source.isdigit() and not path.exists():
                return "webcam"
            if path.is_dir():
                return "image_dir"
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                return "image"
            return "video"
        if source.isdigit() and not path.is_file():
            return "webcam"
        return "file"

    @classmethod
    def from_cli(
        cls,
        parser: argparse.ArgumentParser,
        source: str,
        start: float | None = None,
        duration: float | None = None,
        allow_images: bool = True,
    ) -> MediaSource:
        """Classify `source` and enforce existence & `--start`/`--duration` constraints on `parser`."""
        kind = cls.classify(source, allow_images=allow_images)
        if allow_images:
            if kind in {"image_dir", "image", "video"} and not Path(source).exists():
                parser.error(f"argument --source: path not found: {source!r}")
            if kind != "video" and (start is not None or duration is not None):
                parser.error("--start and --duration are only supported when --source is a local video file")
        else:
            if kind == "file" and not Path(source).is_file():
                parser.error(f"argument --source: local video file not found: {source!r}")
            if kind != "file" and (start is not None or duration is not None):
                parser.error("--start and --duration are only supported when --source is a local video file")
        return cls(source=source, kind=kind)

    @staticmethod
    def validate_save_dir(
        parser: argparse.ArgumentParser,
        save: str | None,
        extra_forbidden_exts: set[str] | None = None,
    ) -> Path | None:
        """Validate `--save DIR` ensures a directory target rather than a file name."""
        if save is None:
            return None
        save_path = Path(save)
        if save_path.exists() and not save_path.is_dir():
            parser.error(f"argument --save: expected a directory path, got file {save!r}")
        forbidden = VIDEO_EXTENSIONS | (extra_forbidden_exts or set())
        if save_path.suffix.lower() in forbidden:
            parser.error(f"argument --save: expected a directory name, got {save!r}")
        return save_path

    def collect_images(self) -> list[Path]:
        """Return ordered image paths for `'image'` or `'image_dir'` sources."""
        if self.kind == "image":
            return [Path(self.source)]
        if self.kind == "image_dir":
            images = sorted(
                (p for p in Path(self.source).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                key=self.natural_sort_key,
            )
            if not images:
                raise SystemExit(f"no images found in {self.source}")
            return images
        raise ValueError(f"collect_images() requires an image source, got {self.kind!r}")

    def open_capture(self) -> cv2.VideoCapture:
        """Open a `cv2.VideoCapture` for webcam, RTSP/HTTP stream, or local video file."""
        if self.kind == "webcam":
            cap = cv2.VideoCapture(int(self.source))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.kind == "stream":
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source: {self.source}")
        return cap


class FrameDisplay:
    """Stateful OpenCV window manager supporting pause/resume (`space`) and quit (`q`)."""

    def __init__(self, window_name: str, enabled: bool = True, allow_pause: bool = True) -> None:
        self.window_name = window_name
        self.enabled = enabled
        self.allow_pause = allow_pause
        self.paused = False
        self._shown = False

    def show(self, image: np.ndarray, delay_ms: int = 1) -> bool:
        """Render `image` and return `True` if the user requested quit (`q`)."""
        if not self.enabled:
            return False
        cv2.imshow(self.window_name, image)
        self._shown = True
        wait = 0 if (self.allow_pause and self.paused) else max(1, delay_ms)
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            return True
        if self.allow_pause and key == ord(" "):
            self.paused = not self.paused
        return False

    def close(self) -> None:
        """Destroy OpenCV windows if this display was active."""
        if self.enabled or self._shown:
            cv2.destroyAllWindows()
