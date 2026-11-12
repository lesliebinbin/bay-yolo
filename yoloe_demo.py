#!/usr/bin/env python3
"""YOLOE open-vocabulary inference demo on RTSP/HTTP streams, local videos, or webcams.

Run from the project root:

    uv run yoloe_demo.py
    uv run yoloe_demo.py --prompts worker truck --max-frames 100
    uv run yoloe_demo.py --source videos/sample01.mp4 --show bbox
    uv run yoloe_demo.py --source videos/sample01.mp4 --start 00:05:00 --duration 00:00:10 --save output_run
    uv run yoloe_demo.py --show segment      # masks only
    uv run yoloe_demo.py --show both         # boxes + masks

Notes:
- YOLOE text prompts need the `clip` package (Ultralytics fork) plus the
  MobileCLIP2 text encoder `mobileclip2_b.ts`. The .ts file is resolved relative
  to the current working directory, so keep it next to this script (otherwise
  ultralytics tries to download it).
- Prompt wording matters: descriptive captions (e.g. "a large red semi truck")
  usually score higher than single words ("truck").
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path

# RTSP over TCP avoids UDP packet loss / torn frames on a loaded network.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = "rtsp://100.76.133.114:8554/test"
DEFAULT_MODEL = "yoloe-26s-seg.pt"
DEFAULT_PROMPTS = ["worker", "truck"]


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


def classify_source(source: str) -> str:
    """Return 'stream', 'webcam', or 'file' for a --source value."""
    if "://" in source:
        return "stream"
    if source.isdigit() and not Path(source).is_file():
        return "webcam"
    return "file"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="RTSP / HTTP / video file / webcam index")
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
    if args.save is not None:
        save_path = Path(args.save)
        if save_path.exists() and not save_path.is_dir():
            parser.error(f"argument --save: expected a directory path, got file {args.save!r}")
        if save_path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
            parser.error(f"argument --save: expected a directory name (not a video file), got {args.save!r}")
    args.source_kind = classify_source(args.source)
    if args.source_kind == "file" and not Path(args.source).is_file():
        parser.error(f"argument --source: local video file not found: {args.source!r}")
    if args.source_kind != "file" and (args.start is not None or args.duration is not None):
        parser.error("--start and --duration are only supported when --source is a local video file")
    return args


def render(result, mode: str):
    """Draw bbox, segment masks, or both on a result frame."""
    if result.masks is None:  # detect-only weights: no masks to draw
        return result.plot()
    if mode == "bbox":
        return result.plot(masks=False)
    if mode == "segment":
        return result.plot(boxes=False)
    return result.plot()


def save_detection_crops(
    frame,
    boxes,
    names: dict[int, str],
    crops_dir: Path,
    frame_index: int,
    crop_pad: float,
) -> int:
    """Save padded native-resolution JPG crops as {name}_{frame_index}_{instance_index}.jpg."""
    h, w = frame.shape[:2]
    per_class_idx: dict[str, int] = {}
    saved = 0
    for cls_t, xyxy_t in zip(boxes.cls, boxes.xyxy):
        x1, y1, x2, y2 = (float(v) for v in xyxy_t.tolist())
        pad_x = (x2 - x1) * crop_pad
        pad_y = (y2 - y1) * crop_pad
        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y)))
        cx2 = min(w, int(round(x2 + pad_x)))
        cy2 = min(h, int(round(y2 + pad_y)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        slug = re.sub(r"[^a-zA-Z0-9-]+", "_", names[int(cls_t)]).strip("_") or "det"
        inst_idx = per_class_idx.get(slug, 0)
        per_class_idx[slug] = inst_idx + 1
        out_path = crops_dir / f"{slug}_{frame_index}_{inst_idx}.jpg"
        cv2.imwrite(str(out_path), frame[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    return saved


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
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
    return cap


def main() -> None:
    args = parse_args()
    is_file = args.source_kind == "file"
    start_sec = args.start or 0.0

    # For local files, open & validate timestamps before loading the model; for live
    # streams/webcams, load the model first so buffered frames don't go stale.
    cap: cv2.VideoCapture | None = None
    if is_file:
        cap = open_capture(args)
        fps_probe = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and start_sec >= total_frames / fps_probe:
            cap.release()
            raise SystemExit(
                f"--start {format_hms(start_sec)} exceeds video duration {format_hms(total_frames / fps_probe)}"
            )
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    model = YOLO(args.model)
    model.set_classes(args.prompts, model.get_text_pe(args.prompts))
    print(f"model={args.model} prompts={args.prompts} device={args.device}")

    if cap is None:
        cap = open_capture(args)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    span_parts = []
    if args.start is not None:
        span_parts.append(f"start={format_hms(start_sec)}")
    if args.duration is not None:
        span_parts.append(f"duration={format_hms(args.duration)}")
    span_suffix = f" ({' '.join(span_parts)})" if span_parts else ""
    print(f"source={args.source} {width}x{height} @ {fps:.0f} fps{span_suffix}")

    save_dir = Path(args.save) if args.save else None
    crops_dir: Path | None = None
    writer: cv2.VideoWriter | None = None
    if save_dir is not None:
        crops_dir = save_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(save_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
    end_msec = (start_sec + args.duration) * 1000.0 if (is_file and args.duration is not None) else None
    half_frame_ms = 500.0 / fps

    frames = 0
    crops_saved = 0
    started = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("stream ended or read failed")
                break

            if end_msec is not None and frames > 0:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_msec > 0 and pos_msec >= end_msec - half_frame_ms:
                    break
                if pos_msec <= 0 and (frames / fps) >= args.duration - (0.5 / fps):
                    break

            result = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
            )[0]
            frames += 1

            boxes = result.boxes
            if boxes is not None and len(boxes):
                if frames % args.print_every == 0:
                    labels = ", ".join(f"{result.names[int(c)]} {float(q):.2f}" for c, q in zip(boxes.cls, boxes.conf))
                    print(f"frame {frames}: {len(boxes)} det -> {labels}")
                if crops_dir is not None:
                    crops_saved += save_detection_crops(frame, boxes, result.names, crops_dir, frames, args.crop_pad)

            if writer is not None or args.show is not None:
                annotated = render(result, args.show or "both")
                if writer is not None:
                    writer.write(annotated)
                if args.show is not None:
                    cv2.imshow("yoloe", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        elapsed = time.time() - started
        if frames:
            summary = f"processed {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps)"
            if save_dir is not None:
                summary += f" -> {save_dir / 'annotated.mp4'} + {crops_saved} crops in {crops_dir}"
            print(summary)


if __name__ == "__main__":
    main()
