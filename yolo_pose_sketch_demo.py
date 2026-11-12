#!/usr/bin/env python3
"""YOLO-Pose blank-canvas keypoint sketch demo.

Draws only pose keypoints and skeleton lines on a black canvas (no photo, no boxes)
so you can evaluate whether body pose alone is sufficient to judge an action.

Run from the project root:

    uv run yolo_pose_sketch_demo.py --show
    uv run yolo_pose_sketch_demo.py --source person_crops/crops --save person_pose_sketch
    uv run yolo_pose_sketch_demo.py --source person_crops/crops --show --fps 8
    uv run yolo_pose_sketch_demo.py --source videos/sample01.mp4 --start 00:12:00 --duration 00:00:15 --show
    uv run yolo_pose_sketch_demo.py --show --side-by-side   # compare original vs sketch
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = "person_crops/crops"
DEFAULT_MODEL = "yolo26s-pose.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_hms(value: str) -> float:
    """Parse an HH:MM:SS (or HH:MM:SS.sss) timestamp into seconds."""
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


def parse_duration_hms(value: str) -> float:
    seconds = parse_hms(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than 00:00:00")
    return seconds


def format_hms(seconds: float) -> str:
    total_ms = round(max(0.0, seconds) * 1000)
    hh, rem_ms = divmod(total_ms, 3_600_000)
    mm, rem_ms = divmod(rem_ms, 60_000)
    ss, ms = divmod(rem_ms, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}" if ms else f"{hh:02d}:{mm:02d}:{ss:02d}"


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def classify_source(source: str) -> str:
    """Return 'stream', 'webcam', 'image_dir', 'image', or 'video'."""
    if "://" in source:
        return "stream"
    path = Path(source)
    if source.isdigit() and not path.exists():
        return "webcam"
    if path.is_dir():
        return "image_dir"
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"
    return "video"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Image folder / image file / video file / RTSP URL / webcam index (default: person_crops/crops)",
    )
    parser.add_argument(
        "--start",
        type=parse_hms,
        default=None,
        metavar="HH:MM:SS",
        help="Start timestamp in HH:MM:SS (local video file only; default: 00:00:00)",
    )
    parser.add_argument(
        "--duration",
        type=parse_duration_hms,
        default=None,
        metavar="HH:MM:SS",
        help="Clip duration in HH:MM:SS (local video file only; default: whole clip)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO pose weights (default: yolo26s-pose.pt)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Person/pose confidence threshold")
    parser.add_argument("--device", default="0", help="'0' = first CUDA GPU, 'cpu' = CPU")
    parser.add_argument("--kpt-radius", type=int, default=5, help="Keypoint circle radius in pixels")
    parser.add_argument("--line-width", type=int, default=2, help="Skeleton limb line width in pixels")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames/images (0 = all)")
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback FPS when using --show on an image directory (default: 10; space pauses, q quits)",
    )
    parser.add_argument("--show", action="store_true", help="Display sketch in a window (space pauses, q quits)")
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Place original image next to the blank-canvas sketch",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="DIR",
        help="Output directory (saves *.jpg for image sources, or sketch.mp4 for video/stream sources)",
    )
    args = parser.parse_args(argv)
    args.source_kind = classify_source(args.source)

    if args.source_kind in {"image_dir", "image", "video"} and not Path(args.source).exists():
        parser.error(f"argument --source: path not found: {args.source!r}")
    if args.source_kind != "video" and (args.start is not None or args.duration is not None):
        parser.error("--start and --duration are only supported when --source is a local video file")
    if args.save is not None:
        save_path = Path(args.save)
        if save_path.exists() and not save_path.is_dir():
            parser.error(f"argument --save: expected a directory path, got file {args.save!r}")
        if save_path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            parser.error(f"argument --save: expected a directory name (not a video file), got {args.save!r}")
    return args


def render_sketch(result, kpt_radius: int = 5, line_width: int = 2, side_by_side: bool = False) -> np.ndarray:
    """Render pose keypoints and skeleton onto a pure black canvas."""
    canvas = np.zeros_like(result.orig_img)
    sketch = result.plot(
        img=canvas,
        boxes=False,
        labels=False,
        conf=False,
        kpt_radius=kpt_radius,
        line_width=line_width,
    )
    if side_by_side:
        return np.hstack([result.orig_img, sketch])
    return sketch


def show_frame(window: str, image: np.ndarray, delay_ms: int, paused: bool) -> tuple[bool, bool]:
    """Display frame and return (should_quit, paused)."""
    cv2.imshow(window, image)
    wait = 0 if paused else max(1, delay_ms)
    key = cv2.waitKey(wait) & 0xFF
    if key == ord("q"):
        return True, paused
    if key == ord(" "):
        return False, not paused
    return False, paused


def run_on_images(args: argparse.Namespace, model: YOLO, image_paths: list[Path], save_dir: Path | None) -> int:
    delay_ms = 0 if len(image_paths) == 1 else int(round(1000.0 / max(args.fps, 0.1)))
    paused = False
    processed = 0

    for path in image_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        sketch = render_sketch(result, args.kpt_radius, args.line_width, args.side_by_side)
        processed += 1

        if save_dir is not None:
            cv2.imwrite(str(save_dir / path.name), sketch, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if args.show:
            quit_now, paused = show_frame("yolo_pose_sketch", sketch, delay_ms, paused)
            if quit_now:
                break

        if args.max_frames and processed >= args.max_frames:
            break

    return processed


def run_on_video(args: argparse.Namespace, model: YOLO, save_dir: Path | None) -> int:
    is_file = args.source_kind == "video"
    start_sec = args.start or 0.0

    if args.source_kind == "webcam":
        cap = cv2.VideoCapture(int(args.source))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif args.source_kind == "stream":
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if is_file and total_frames > 0 and start_sec >= total_frames / fps:
        cap.release()
        raise SystemExit(f"--start {format_hms(start_sec)} exceeds video duration {format_hms(total_frames / fps)}")
    if is_file and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    out_width = width * 2 if args.side_by_side else width
    writer = (
        cv2.VideoWriter(str(save_dir / "sketch.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, height))
        if save_dir is not None
        else None
    )
    end_msec = (start_sec + args.duration) * 1000.0 if (is_file and args.duration is not None) else None
    half_frame_ms = 500.0 / fps

    paused = False
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if end_msec is not None and processed > 0:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec > 0 and pos_msec >= end_msec - half_frame_ms:
                    break
                if pos_msec <= 0 and (processed / fps) >= args.duration - (0.5 / fps):
                    break

            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            sketch = render_sketch(result, args.kpt_radius, args.line_width, args.side_by_side)
            processed += 1

            if writer is not None:
                writer.write(sketch)
            if args.show:
                quit_now, paused = show_frame("yolo_pose_sketch", sketch, 1, paused)
                if quit_now:
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return processed


def main() -> None:
    args = parse_args()
    if not args.show and args.save is None:
        raise SystemExit("nothing to do: pass --show and/or --save DIR")

    save_dir = Path(args.save) if args.save else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"model={args.model} source={args.source} ({args.source_kind}) device={args.device}")

    started = time.time()
    try:
        if args.source_kind == "image_dir":
            images = sorted(
                (p for p in Path(args.source).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                key=natural_sort_key,
            )
            if not images:
                raise SystemExit(f"no images found in {args.source}")
            count = run_on_images(args, model, images, save_dir)
        elif args.source_kind == "image":
            count = run_on_images(args, model, [Path(args.source)], save_dir)
        else:
            count = run_on_video(args, model, save_dir)
    except KeyboardInterrupt:
        print("\ninterrupted")
        count = 0
    finally:
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    if count:
        msg = f"processed {count} item(s) in {elapsed:.1f}s ({count / elapsed:.1f} fps)"
        if save_dir is not None:
            msg += f" -> {save_dir}"
        print(msg)


if __name__ == "__main__":
    main()
