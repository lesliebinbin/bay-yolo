"""Timestamp parsing, formatting, and video clip span boundary utilities."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cv2


@dataclass(frozen=True)
class TimeSpan:
    """Represents an optional `[start_sec, start_sec + duration_sec)` clip window."""

    start_sec: float | None = None
    duration_sec: float | None = None

    @property
    def effective_start(self) -> float:
        return self.start_sec or 0.0

    @property
    def end_msec(self) -> float | None:
        if self.duration_sec is None:
            return None
        return (self.effective_start + self.duration_sec) * 1000.0

    @staticmethod
    def parse_hms(value: str) -> float:
        """Parse an `HH:MM:SS` (or `HH:MM:SS.sss`) timestamp into seconds."""
        parts = value.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"expected HH:MM:SS format, got {value!r}")
        hh_str, mm_str, ss_str = parts
        if not (hh_str.isdigit() and mm_str.isdigit()):
            raise argparse.ArgumentTypeError(f"expected HH:MM:SS format, got {value!r}")
        try:
            hh = int(hh_str)
            mm = int(mm_str)
            ss = float(ss_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected HH:MM:SS format, got {value!r}") from exc
        if mm >= 60 or ss < 0 or ss >= 60 or not math.isfinite(ss):
            raise argparse.ArgumentTypeError(f"out-of-range HH:MM:SS timestamp: {value!r}")
        return hh * 3600 + mm * 60 + ss

    @classmethod
    def parse_duration_hms(cls, value: str) -> float:
        """Parse a positive `HH:MM:SS` duration into seconds."""
        seconds = cls.parse_hms(value)
        if seconds <= 0:
            raise argparse.ArgumentTypeError("duration must be greater than 00:00:00")
        return seconds

    @staticmethod
    def format_hms(seconds: float) -> str:
        """Format seconds as `HH:MM:SS` (or `HH:MM:SS.mmm` when fractional)."""
        total_ms = round(max(0.0, seconds) * 1000)
        hh, rem_ms = divmod(total_ms, 3_600_000)
        mm, rem_ms = divmod(rem_ms, 60_000)
        ss, ms = divmod(rem_ms, 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}" if ms else f"{hh:02d}:{mm:02d}:{ss:02d}"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, default_start: str | None = None) -> None:
        """Register `--start` and `--duration` arguments on `parser`."""
        parser.add_argument(
            "--start",
            type=cls.parse_hms,
            default=default_start,
            metavar="HH:MM:SS",
            help="Start timestamp in HH:MM:SS (local video file only; default: 00:00:00)",
        )
        parser.add_argument(
            "--duration",
            type=cls.parse_duration_hms,
            default=None,
            metavar="HH:MM:SS",
            help="Clip duration in HH:MM:SS (local video file only; default: whole clip)",
        )

    def validate_and_seek(self, cap: cv2.VideoCapture, fps: float, total_frames: int) -> None:
        """Validate `start_sec` against video duration and seek `cap` if needed."""
        import cv2 as _cv2

        start = self.effective_start
        if total_frames > 0 and start >= total_frames / fps:
            cap.release()
            raise SystemExit(
                f"--start {self.format_hms(start)} exceeds video duration {self.format_hms(total_frames / fps)}"
            )
        if start > 0:
            cap.set(_cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    def reached_end(self, pos_msec: float, processed_frames: int, fps: float) -> bool:
        """Return `True` once playback reaches the configured `duration_sec`."""
        end_ms = self.end_msec
        if end_ms is None or processed_frames <= 0:
            return False
        half_frame_ms = 500.0 / fps
        if pos_msec > 0 and pos_msec >= end_ms - half_frame_ms:
            return True
        if pos_msec <= 0 and self.duration_sec is not None and (processed_frames / fps) >= self.duration_sec - (0.5 / fps):
            return True
        return False

    def describe(self) -> str:
        """Format optional `(start=... duration=...)` suffix for CLI status output."""
        parts: list[str] = []
        if self.start_sec is not None:
            parts.append(f"start={self.format_hms(self.effective_start)}")
        if self.duration_sec is not None:
            parts.append(f"duration={self.format_hms(self.duration_sec)}")
        return f" ({' '.join(parts)})" if parts else ""
