#!/usr/bin/env python3
"""YOLOE open-vocabulary inference CLI (`c_yoloe`) on RTSP/HTTP streams, local videos, or webcams.

Examples:

    c_yoloe
    c_yoloe --prompts worker truck --max-frames 100
    c_yoloe --source videos/sample01.mp4 --show bbox
    c_yoloe --source videos/sample01.mp4 --start 00:05:00 --duration 00:00:10 --save output_run
    c_yoloe --show segment      # masks only
    c_yoloe --show both         # boxes + masks

Notes:
- YOLOE text prompts need the `clip` package (Ultralytics fork) plus the
  MobileCLIP2 text encoder `mobileclip2_b.ts` in the working directory.
- Prompt wording matters: descriptive captions (e.g. "a large red semi truck")
  usually score higher than single words ("truck").
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from utils.crops import BBoxCropper
from utils.media import FrameDisplay, MediaSource
from utils.time_span import TimeSpan

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = "rtsp://100.76.133.114:8554/test"
DEFAULT_MODEL = "yoloe-26s-seg.pt"
DEFAULT_PROMPTS = ["worker", "truck"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="c_yoloe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="RTSP / HTTP / video file / webcam index")
    TimeSpan.add_cli_args(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLOE weights")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="Open-vocabulary class names")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--device", default="0", help="'0' = first CUDA GPU, 'cpu' = CPU")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = until end / Ctrl-C)")
    parser.add_argument("--print-every", type=int, default=25, help="Print detection summary every N frames")
    parser.add_argument(
        "--show",
        nargs="?",
        const="both",
        choices=["bbox", "segment", "both"],
        default=None,
        help="Show annotated frames in a window: 'bbox', 'segment' or 'both' (bare --show = both); q quits",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="DIR",
        help="Output directory for annotated.mp4 and crops/{name}_{frame_index}_{instance_index}.jpg",
    )
    parser.add_argument(
        "--crop-pad",
        type=float,
        default=0.35,
        help="Fractional bbox padding around saved crops (default: 0.35)",
    )
    args = parser.parse_args(argv)
    if args.crop_pad < 0:
        parser.error("argument --crop-pad: must be >= 0")

    args.save_path = MediaSource.validate_save_dir(parser, args.save)
    args.media = MediaSource.from_cli(parser, args.source, args.start, args.duration, allow_images=False)
    args.source_kind = args.media.kind
    args.span = TimeSpan(start_sec=args.start, duration_sec=args.duration)
    return args


def render(result, mode: str):
    """Draw bbox, segment masks, or both on a result frame."""
    if result.masks is None:
        return result.plot()
    if mode == "bbox":
        return result.plot(masks=False)
    if mode == "segment":
        return result.plot(boxes=False)
    return result.plot()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    is_file = args.media.is_local_video

    # For local files, open & validate timestamps before loading the model; for live
    # streams/webcams, load the model first so buffered frames don't go stale.
    cap: cv2.VideoCapture | None = None
    if is_file:
        cap = args.media.open_capture()
        fps_probe = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        args.span.validate_and_seek(cap, fps_probe, total_frames)

    model = YOLO(args.model)
    model.set_classes(args.prompts, model.get_text_pe(args.prompts))
    print(f"model={args.model} prompts={args.prompts} device={args.device}")

    if cap is None:
        cap = args.media.open_capture()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"source={args.source} {width}x{height} @ {fps:.0f} fps{args.span.describe()}")

    save_dir: Path | None = args.save_path
    crops_dir: Path | None = None
    writer: cv2.VideoWriter | None = None
    if save_dir is not None:
        crops_dir = save_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(save_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    cropper = BBoxCropper(crop_pad=args.crop_pad)
    display = FrameDisplay("yoloe", enabled=args.show is not None, allow_pause=False)

    frames = 0
    crops_saved = 0
    started = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("stream ended or read failed")
                break

            if is_file and args.span.reached_end(cap.get(cv2.CAP_PROP_POS_MSEC), frames, fps):
                break

            result = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
            )[0]
            frames += 1

            boxes = result.boxes
            if boxes is not None and len(boxes):
                if frames % args.print_every == 0:
                    labels = ", ".join(
                        f"{result.names[int(c)]} {float(q):.2f}" for c, q in zip(boxes.cls, boxes.conf)
                    )
                    print(f"frame {frames}: {len(boxes)} det -> {labels}")
                if crops_dir is not None:
                    crops_saved += cropper.save_crops(frame, boxes, result.names, crops_dir, frames)

            if writer is not None or args.show is not None:
                annotated = render(result, args.show or "both")
                if writer is not None:
                    writer.write(annotated)
                if display.show(annotated, delay_ms=1):
                    break

            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        display.close()
        elapsed = time.time() - started
        if frames:
            summary = f"processed {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps)"
            if save_dir is not None:
                summary += f" -> {save_dir / 'annotated.mp4'} + {crops_saved} crops in {crops_dir}"
            print(summary)


if __name__ == "__main__":
    main()
