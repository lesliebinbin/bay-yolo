#!/usr/bin/env python3
"""Slice long CCTV videos into fixed-ROI short `.mp4` clips using YOLO normalized coordinates.

Bounding boxes follow Ultralytics YOLO normalized `(x_center, y_center, width, height)` in `[0, 1]`.
Default ROI is the bottom-left quadrant (`1/2` width x `1/2` height):
    x_center=0.25, y_center=0.75, width=0.50, height=0.50

Run from the project root:

    uv run python -m preprocessing.roi_clip_slicer --source videos/sample01.mp4 --save data/unlabeled
    uv run python -m preprocessing.roi_clip_slicer --source videos/sample01.mp4 --start 00:10:00 --duration 00:05:00 --save data/unlabeled
    uv run python -m preprocessing.roi_clip_slicer --source videos/sample01.mp4 --roi 0.25 0.75 0.5 0.5 --clip-sec 5 --stride-sec 5 --save data/unlabeled
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from utils.time_span import TimeSpan


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


parse_hms = TimeSpan.parse_hms
parse_duration_hms = TimeSpan.parse_duration_hms
format_hms = TimeSpan.format_hms


def slice_video_roi_clips(
    source: str | Path,
    save_dir: str | Path,
    roi: YoloBox = DEFAULT_YOLO_ROI,
    clip_sec: float = 5.0,
    stride_sec: float | None = None,
    target_fps: float | None = None,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    resize: int | None = None,
    min_clip_ratio: float = 0.8,
) -> list[Path]:
    """Slice `source` video into ROI-cropped `.mp4` clips + `manifest.csv` + `roi_preview.jpg`."""
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"video file not found: {source_path}")
    if clip_sec <= 0:
        raise ValueError("clip_sec must be > 0")
    stride_sec = clip_sec if stride_sec is None else stride_sec
    if stride_sec <= 0:
        raise ValueError("stride_sec must be > 0")

    out_dir = Path(save_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {source_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0 and start_sec >= total_frames / src_fps:
        cap.release()
        raise ValueError(f"start_sec={start_sec} exceeds video duration {total_frames / src_fps:.2f}s")

    x1, y1, x2, y2 = yolo_roi_to_xyxy(roi, src_w, src_h, even_dims=True)
    out_w = resize if resize is not None else (x2 - x1)
    out_h = resize if resize is not None else (y2 - y1)

    frame_step = max(1, int(round(src_fps / target_fps))) if (target_fps and target_fps < src_fps) else 1
    out_fps = src_fps / frame_step
    clip_src_frames = max(1, int(round(clip_sec * src_fps)))
    stride_src_frames = max(1, int(round(stride_sec * src_fps)))
    expected_out_frames = max(1, (clip_src_frames + frame_step - 1) // frame_step)
    min_out_frames = max(1, int(math.floor(expected_out_frames * min_clip_ratio)))

    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    end_msec = (start_sec + duration_sec) * 1000.0 if duration_sec is not None else None
    half_frame_ms = 500.0 / src_fps

    # Write Ultralytics-format ROI definition for reproducibility
    (out_dir / "roi_yolo.txt").write_text(
        f"0 {roi.x_center:.6f} {roi.y_center:.6f} {roi.width:.6f} {roi.height:.6f}\n",
        encoding="utf-8",
    )

    stem = source_path.stem
    active_clips: list[dict[str, object]] = []
    saved_paths: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    preview_saved = False
    src_idx = 0

    def finalize_clip(slot: dict[str, object], end_s: float) -> None:
        frames_buf: list = slot["frames"]  # type: ignore[assignment]
        if len(frames_buf) < min_out_frames:
            return
        c_idx = len(saved_paths)
        s_sec = float(slot["start_sec"])  # type: ignore[arg-type]
        clip_name = (
            f"{stem}_{format_hms(s_sec).replace(':', '-')}"
            f"__to__{format_hms(end_s).replace(':', '-')}_clip{c_idx:05d}.mp4"
        )
        clip_path = clips_dir / clip_name
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
        try:
            for crop_frame in frames_buf:
                writer.write(crop_frame)
        finally:
            writer.release()
        saved_paths.append(clip_path)
        manifest_rows.append(
            {
                "clip_path": str(clip_path.relative_to(out_dir)),
                "source": source_path.name,
                "start_sec": round(s_sec, 3),
                "end_sec": round(end_s, 3),
                "num_frames": len(frames_buf),
                "fps": round(out_fps, 3),
                "roi_x": roi.x_center,
                "roi_y": roi.y_center,
                "roi_w": roi.width,
                "roi_h": roi.height,
            }
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            now_sec = (pos_msec / 1000.0) if pos_msec > 0 else (start_sec + src_idx / src_fps)
            if end_msec is not None and src_idx > 0 and pos_msec > 0 and pos_msec >= end_msec - half_frame_ms:
                break
            if end_msec is not None and pos_msec <= 0 and (src_idx / src_fps) >= (duration_sec or 0.0) - (0.5 / src_fps):
                break

            if not preview_saved:
                preview = frame.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(
                    preview,
                    f"YOLO ROI=({roi.x_center:.2f}, {roi.y_center:.2f}, {roi.width:.2f}, {roi.height:.2f}) [{x2-x1}x{y2-y1}]",
                    (x1 + 10, max(32, y1 + 36)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(out_dir / "roi_preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
                preview_saved = True

            if src_idx % stride_src_frames == 0:
                active_clips.append({"start_idx": src_idx, "start_sec": now_sec, "frames": []})

            crop = frame[y1:y2, x1:x2]
            if resize is not None:
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                crop = crop.copy()

            still_active: list[dict[str, object]] = []
            for slot in active_clips:
                rel_idx = src_idx - int(slot["start_idx"])  # type: ignore[arg-type]
                if rel_idx % frame_step == 0:
                    slot["frames"].append(crop)  # type: ignore[union-attr]
                if rel_idx + 1 >= clip_src_frames:
                    finalize_clip(slot, now_sec + (1.0 / src_fps))
                else:
                    still_active.append(slot)
            active_clips = still_active
            src_idx += 1

        # Flush any trailing partial clip meeting min_clip_ratio
        final_sec = start_sec + src_idx / src_fps
        for slot in active_clips:
            finalize_clip(slot, final_sec)
    finally:
        cap.release()

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_path",
                "source",
                "start_sec",
                "end_sec",
                "num_frames",
                "fps",
                "roi_x",
                "roi_y",
                "roi_w",
                "roi_h",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return saved_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Local input video path (e.g. videos/sample01.mp4)")
    parser.add_argument("--save", required=True, metavar="DIR", help="Output directory for clips/, manifest.csv, roi_preview.jpg")
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        default=[
            DEFAULT_YOLO_ROI.x_center,
            DEFAULT_YOLO_ROI.y_center,
            DEFAULT_YOLO_ROI.width,
            DEFAULT_YOLO_ROI.height,
        ],
        metavar=("X", "Y", "W", "H"),
        help="Normalized YOLO ROI (x_center y_center width height) in [0, 1]; default: 0.25 0.75 0.5 0.5 (bottom-left 1/2x1/2)",
    )
    parser.add_argument("--start", type=parse_hms, default="00:00:00", metavar="HH:MM:SS", help="Start timestamp")
    parser.add_argument("--duration", type=parse_duration_hms, default=None, metavar="HH:MM:SS", help="Duration to slice")
    parser.add_argument("--clip-sec", type=float, default=5.0, help="Seconds per output clip (default: 5.0)")
    parser.add_argument("--stride-sec", type=float, default=None, help="Sliding stride in seconds (default: clip-sec)")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional downsampled clip FPS (default: keep source FPS)")
    parser.add_argument("--resize", type=int, default=None, help="Optional square output resolution (e.g. 256 or 384)")
    args = parser.parse_args(argv)
    try:
        args.roi_box = parse_yolo_roi(args.roi)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()
    clips = slice_video_roi_clips(
        source=args.source,
        save_dir=args.save,
        roi=args.roi_box,
        clip_sec=args.clip_sec,
        stride_sec=args.stride_sec,
        target_fps=args.target_fps,
        start_sec=args.start,
        duration_sec=args.duration,
        resize=args.resize,
    )
    print(
        f"saved {len(clips)} ROI clip(s) -> {Path(args.save) / 'clips'} "
        f"(YOLO ROI={args.roi_box.x_center:.2f},{args.roi_box.y_center:.2f},{args.roi_box.width:.2f},{args.roi_box.height:.2f})"
    )


if __name__ == "__main__":
    main()
