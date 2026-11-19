#!/usr/bin/env python3
"""Slice long CCTV videos into fixed-ROI short `.mp4` clips using YOLO normalized coordinates.

Bounding boxes follow Ultralytics YOLO normalized `(x_center, y_center, width, height)` in `[0, 1]`.
Default ROI is the bottom-left quadrant (`1/2` width x `1/2` height):
    x_center=0.25, y_center=0.75, width=0.50, height=0.50

Run from the project root:

    uv run python -m preprocessing.roi_clip_slicer --source videos/sample01.mp4 --save data/unlabeled
    uv run python -m preprocessing.roi_clip_slicer --source videos/sample01.mp4 --split 3:1:1 --seed 42 --save data/unlabeled
    uv run python -m preprocessing.roi_clip_slicer --source videos/sample01.mp4 --roi 0.25 0.75 0.5 0.5 --clip-sec 4 --stride-sec 2 --target-fps 5 --resize 256 --split 3:2 --seed 42 --save data/unlabeled
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2

from utils.roi import DEFAULT_YOLO_ROI, YoloBox, parse_yolo_roi, yolo_roi_to_xyxy  # noqa: F401  (re-exported for backward compatibility)
from utils.time_span import TimeSpan

__all__ = [
    "DEFAULT_YOLO_ROI",
    "SplitSpec",
    "YoloBox",
    "parse_yolo_roi",
    "slice_video_roi_clips",
    "yolo_roi_to_xyxy",
]


parse_hms = TimeSpan.parse_hms
parse_duration_hms = TimeSpan.parse_duration_hms
format_hms = TimeSpan.format_hms


@dataclass(frozen=True)
class SplitSpec:
    """Weighted probabilistic split configuration (e.g. `'3:1:1'` -> `train/val/test`)."""

    names: tuple[str, ...]
    weights: tuple[float, ...]

    @classmethod
    def default_names(cls, count: int) -> tuple[str, ...]:
        if count == 2:
            return ("train", "val")
        if count == 3:
            return ("train", "val", "test")
        return tuple(f"split_{i}" for i in range(count))

    @classmethod
    def parse(cls, spec: str | list[float] | tuple[float, ...], names: list[str] | tuple[str, ...] | None = None) -> SplitSpec:
        parts = spec.split(":") if isinstance(spec, str) else [str(v) for v in spec]
        if len(parts) < 2:
            raise ValueError(f"split requires at least 2 colon-separated weights (e.g. '3:2' or '3:1:1'), got {spec!r}")
        try:
            weights = tuple(float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"invalid numeric weights in split {spec!r}") from exc
        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError(f"all split weights must be > 0, got {weights}")

        resolved_names = tuple(names) if names is not None else cls.default_names(len(weights))
        if len(resolved_names) != len(weights):
            raise ValueError(f"split_names count ({len(resolved_names)}) must match split weights ({len(weights)})")
        if len(set(resolved_names)) != len(resolved_names) or any(not n or "/" in n or "\\" in n for n in resolved_names):
            raise ValueError(f"invalid or duplicate split names: {resolved_names!r}")
        return cls(names=resolved_names, weights=weights)

    def choose(self, rng: random.Random) -> str:
        return rng.choices(self.names, weights=self.weights, k=1)[0]


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
    split: SplitSpec | str | None = None,
    split_names: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
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

    split_spec = (
        split
        if isinstance(split, SplitSpec) or split is None
        else SplitSpec.parse(split, names=split_names)
    )
    if split_spec is not None and split_names is not None and tuple(split_names) != split_spec.names:
        split_spec = SplitSpec.parse(split_spec.weights, names=split_names)
    rng = random.Random(seed)

    out_dir = Path(save_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    if split_spec is not None:
        for s_name in split_spec.names:
            (clips_dir / s_name).mkdir(parents=True, exist_ok=True)

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
        chosen_split = split_spec.choose(rng) if split_spec is not None else ""
        clip_path = (clips_dir / chosen_split / clip_name) if chosen_split else (clips_dir / clip_name)
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
                "split": chosen_split,
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
                "split",
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
    parser.add_argument(
        "--split",
        default=None,
        metavar="RATIOS",
        help="Optional colon-separated split weights, e.g. '3:2' (train/val), '3:1:1' (train/val/test), or '4:1:1:1'",
    )
    parser.add_argument(
        "--split-names",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Optional custom subfolder names matching --split weights (e.g. --split 3:1:1 --split-names ssl probe test)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible --split assignment")
    args = parser.parse_args(argv)
    try:
        args.roi_box = parse_yolo_roi(args.roi)
        if args.split is None and args.split_names is not None:
            raise ValueError("--split-names requires --split")
        args.split_spec = SplitSpec.parse(args.split, names=args.split_names) if args.split is not None else None
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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
        split=args.split_spec,
        seed=args.seed,
    )
    split_summary = ""
    if args.split_spec is not None:
        counts = Counter(p.parent.name for p in clips)
        breakdown = ", ".join(f"{name}={counts.get(name, 0)}" for name in args.split_spec.names)
        split_summary = f" [{breakdown}]"
    print(
        f"saved {len(clips)} ROI clip(s){split_summary} -> {Path(args.save) / 'clips'} "
        f"(YOLO ROI={args.roi_box.x_center:.2f},{args.roi_box.y_center:.2f},{args.roi_box.width:.2f},{args.roi_box.height:.2f})"
    )


if __name__ == "__main__":
    main()
