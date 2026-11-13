#!/usr/bin/env python3
"""YOLO-Pose blank-canvas keypoint sketch CLI (`c_yolo_pose`).

Draws only pose keypoints and skeleton lines on a black canvas (no photo, no boxes)
so you can evaluate whether body pose alone is sufficient to judge an action.

Examples:

    c_yolo_pose --show
    c_yolo_pose --source person_crops/crops --save person_pose_sketch
    c_yolo_pose --source person_crops/crops --show --fps 8
    c_yolo_pose --source videos/sample01.mp4 --start 00:12:00 --duration 00:00:15 --show
    c_yolo_pose --show --side-by-side   # compare original vs sketch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from utils.media import FrameDisplay, MediaSource
from utils.time_span import TimeSpan

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = "person_crops/crops"
DEFAULT_MODEL = "yolo26s-pose.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="c_yolo_pose",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Image folder / image file / video file / RTSP URL / webcam index (default: person_crops/crops)",
    )
    TimeSpan.add_cli_args(parser)
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
    args.save_path = MediaSource.validate_save_dir(parser, args.save)
    args.media = MediaSource.from_cli(parser, args.source, args.start, args.duration, allow_images=True)
    args.source_kind = args.media.kind
    args.span = TimeSpan(start_sec=args.start, duration_sec=args.duration)
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


def run_on_images(
    args: argparse.Namespace,
    model: YOLO,
    image_paths: list[Path],
    save_dir: Path | None,
    display: FrameDisplay,
) -> int:
    delay_ms = 0 if len(image_paths) == 1 else int(round(1000.0 / max(args.fps, 0.1)))
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

        if display.show(sketch, delay_ms=delay_ms):
            break

        if args.max_frames and processed >= args.max_frames:
            break

    return processed


def run_on_video(
    args: argparse.Namespace,
    model: YOLO,
    save_dir: Path | None,
    display: FrameDisplay,
) -> int:
    is_file = args.media.is_local_video
    cap = args.media.open_capture()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if is_file:
        args.span.validate_and_seek(cap, fps, total_frames)

    out_width = width * 2 if args.side_by_side else width
    writer = (
        cv2.VideoWriter(str(save_dir / "sketch.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, height))
        if save_dir is not None
        else None
    )

    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_file and args.span.reached_end(cap.get(cv2.CAP_PROP_POS_MSEC), processed, fps):
                break

            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
            sketch = render_sketch(result, args.kpt_radius, args.line_width, args.side_by_side)
            processed += 1

            if writer is not None:
                writer.write(sketch)
            if display.show(sketch, delay_ms=1):
                break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return processed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.show and args.save_path is None:
        raise SystemExit("nothing to do: pass --show and/or --save DIR")

    save_dir: Path | None = args.save_path
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"model={args.model} source={args.source} ({args.source_kind}) device={args.device}")

    display = FrameDisplay("yolo_pose_sketch", enabled=args.show)
    started = time.time()
    try:
        if args.media.is_image_source:
            count = run_on_images(args, model, args.media.collect_images(), save_dir, display)
        else:
            count = run_on_video(args, model, save_dir, display)
    except KeyboardInterrupt:
        print("\ninterrupted")
        count = 0
    finally:
        display.close()

    elapsed = time.time() - started
    if count:
        msg = f"processed {count} item(s) in {elapsed:.1f}s ({count / elapsed:.1f} fps)"
        if save_dir is not None:
            msg += f" -> {save_dir}"
        print(msg)


if __name__ == "__main__":
    main()
