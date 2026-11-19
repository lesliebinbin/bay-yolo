#!/usr/bin/env python3
"""Extract per-track subject field-of-view (FoV) crops from a video span.

Runs YOLO detection (`yolo26s.pt` by default) + ByteTrack over `--start`/`--duration`
of a local video, crops every tracked object with a square `FieldOfView`
(pad + square + upscale), and saves each distinct object under its own track id:

    {out-dir}/{track_id:03d}_{label}/{frame:05d}.jpg   # --output frames
    {out-dir}/{track_id:03d}_{label}.mp4                # --output video
    {out-dir}/manifest.csv

Video output note: per-frame crop size varies as the subject moves, so each track
video is letterboxed onto a per-track canonical canvas chosen from the median crop
size of its first `--warmup` frames (subject stays centered; source fps kept).
Tracks with fewer than `--min-frames` detections are dropped (overlap flicker).

Examples:

    uv run python -m tasks.fov_extract --source video_sources/sample01.mp4 --start 00:05:00 --duration 00:00:20
    uv run python -m tasks.fov_extract --source video_sources/sample01.mp4 --start 00:05:00 --duration 00:00:30 --output both
    uv run python -m tasks.fov_extract --source video_sources/sample01.mp4 --duration 00:01:00 --fov-target-width 512
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from utils.field_of_view import FieldOfView
from utils.media import MediaSource
from utils.time_span import TimeSpan

_REPO_TRACKER = Path(__file__).resolve().parents[2] / "config" / "trackers" / "bytetrack_tight.yaml"

DEFAULT_MODEL = "yolo26s.pt"
DEFAULT_CLASSES = ["person"]
DEFAULT_TRACKER = str(_REPO_TRACKER) if _REPO_TRACKER.is_file() else "bytetrack.yaml"
DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.25
DEFAULT_MIN_FRAMES = 30


def fit_to_canvas(img: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Scale `img` to fit `canvas` (W, H) preserving aspect, centered on black bars."""
    W, H = canvas
    h, w = img.shape[:2]
    if (w, h) == (W, H):
        return img
    scale = min(W / w, H / h)
    nw, nh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    img = cv2.resize(img, (nw, nh), interpolation=interpolation)
    if (nw, nh) == (W, H):
        return img
    out = np.zeros((H, W, 3), dtype=np.uint8)
    y0 = max(0, (H - nh) // 2)
    x0 = max(0, (W - nw) // 2)
    hh, ww = min(nh, H), min(nw, W)
    out[y0 : y0 + hh, x0 : x0 + ww] = img[:hh, :ww]
    return out


def even_size(w: float, h: float) -> tuple[int, int]:
    """Round a canvas size to even integers (encoder-safe), min 2."""
    ew = max(2, 2 * int(round(max(2.0, w) / 2)))
    eh = max(2, 2 * int(round(max(2.0, h) / 2)))
    return ew, eh


def resolve_class_ids(model_names: dict[int, str], requested: list[str]) -> list[int]:
    """Map requested class names (case-insensitive) to model class IDs."""
    name_to_id = {str(name).lower(): int(cid) for cid, name in model_names.items()}
    missing = [name for name in requested if name.lower() not in name_to_id]
    if missing:
        raise SystemExit(f"model does not have class(es) {missing!r}")
    return [name_to_id[name.lower()] for name in requested]


@dataclass
class TrackState:
    track_id: int
    label: str
    first_frame: int | None = None
    last_frame: int | None = None
    count: int = 0
    confs: list[float] = field(default_factory=list)


class TrackWriter:
    """Collects per-track FoV crops and materializes them as frames and/or one video.

    The video canvas is fixed from the median crop size of the first `warmup`
    frames (avoids a second inference pass); later frames are letterboxed onto it.
    """

    def __init__(
        self,
        state: TrackState,
        out_dir: Path,
        fps: float,
        output: str,
        min_frames: int = DEFAULT_MIN_FRAMES,
        warmup: int = 10,
        jpeg_quality: int = 95,
    ) -> None:
        self.state = state
        self.out_dir = Path(out_dir)
        self.fps = max(1.0, float(fps))
        self.output = output
        self.min_frames = max(1, min_frames)
        self.warmup = max(1, warmup)
        self.jpeg_quality = jpeg_quality

        slug = re.sub(r"[^a-zA-Z0-9-]+", "_", state.label).strip("_") or "det"
        self.track_dir = self.out_dir / f"{state.track_id:03d}_{slug}"
        self.video_path = self.out_dir / f"{self.track_dir.name}.mp4"

        self.canvas: tuple[int, int] | None = None
        self.writer: cv2.VideoWriter | None = None
        self.buffer: list[np.ndarray] = []
        self.saved_frames: list[Path] = []

    def add(self, frame_idx: int, crop: np.ndarray, conf: float) -> None:
        st = self.state
        st.first_frame = frame_idx if st.first_frame is None else min(st.first_frame, frame_idx)
        st.last_frame = frame_idx if st.last_frame is None else max(st.last_frame, frame_idx)
        st.count += 1
        st.confs.append(float(conf))

        if "frames" in self.output:
            self.track_dir.mkdir(parents=True, exist_ok=True)
            path = self.track_dir / f"{frame_idx:05d}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            self.saved_frames.append(path)

        if "video" in self.output:
            if self.canvas is None:
                self.buffer.append(crop)
                if len(self.buffer) >= self.warmup:
                    self._open_writer()
            else:
                self.writer.write(fit_to_canvas(crop, self.canvas))

    def _open_writer(self) -> None:
        widths = [c.shape[1] for c in self.buffer]
        heights = [c.shape[0] for c in self.buffer]
        self.canvas = even_size(float(np.median(widths)), float(np.median(heights)))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.canvas
        )
        for buffered in self.buffer:
            self.writer.write(fit_to_canvas(buffered, self.canvas))
        self.buffer = []

    def _cleanup(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            if self.video_path.exists():
                self.video_path.unlink()
        for path in self.saved_frames:
            if path.exists():
                path.unlink()
        if self.track_dir.exists() and not any(self.track_dir.iterdir()):
            self.track_dir.rmdir()
        self.saved_frames = []
        self.buffer = []

    def finalize(self) -> dict[str, object] | None:
        """Close outputs; return a manifest row, or `None` if the track was dropped."""
        st = self.state
        if st.count < self.min_frames:
            self._cleanup()
            return None
        if "video" in self.output:
            if self.writer is None and self.buffer:
                self._open_writer()
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        return {
            "track_id": st.track_id,
            "label": st.label,
            "kept": True,
            "num_frames": st.count,
            "first_frame": st.first_frame,
            "last_frame": st.last_frame,
            "mean_conf": round(float(np.mean(st.confs)), 4) if st.confs else "",
            "video": str(self.video_path.relative_to(self.out_dir)) if "video" in self.output else "",
            "frames_dir": str(self.track_dir.relative_to(self.out_dir)) if "frames" in self.output else "",
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Local input video file (.mp4)")
    TimeSpan.add_cli_args(parser, default_start="00:00:00")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO detection weights (default: %(default)s)")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Target class names (default: person)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="ByteTrack .yaml config (default: %(default)s)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference size (default: %(default)s)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Detection confidence threshold (default: %(default)s)")
    parser.add_argument("--device", default="0", help="'0' = first CUDA GPU, 'cpu' = CPU")
    parser.add_argument(
        "--output",
        choices=["frames", "video", "both"],
        default="video",
        help="Save per-track crops as frames, one video per track, or both (default: video)",
    )
    parser.add_argument("--fov-pad", type=float, default=0.10, help="FoV padding fraction of the box (default: 0.10)")
    parser.add_argument("--fov-upscale", type=float, default=2.0, help="FoV upscale factor (default: 2.0; 0 = native)")
    parser.add_argument("--fov-target-width", type=int, default=None, help="FoV output width (overrides --fov-upscale)")
    parser.add_argument(
        "--fov-square",
        dest="fov_square",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Widen the shorter crop side to the longer one so crops are square before resize (default: on)",
    )
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES, help="Drop tracks with fewer detections (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=10, help="Frames buffered before fixing each track video canvas (default: 10)")
    parser.add_argument("--max-tracks", type=int, default=0, help="Keep at most N tracks (0 = all)")
    parser.add_argument("--print-every", type=int, default=25, help="Print status every N frames")
    parser.add_argument("--out-dir", default="fov_out", metavar="DIR", help="Output directory (default: fov_out)")
    args = parser.parse_args(argv)
    if args.fov_pad < 0:
        parser.error("--fov-pad must be >= 0")
    if args.fov_upscale < 0:
        parser.error("--fov-upscale must be >= 0 (0 = native resolution)")
    if args.fov_target_width is not None and args.fov_target_width < 2:
        parser.error("--fov-target-width must be >= 2")
    if not args.tracker.lower().endswith((".yaml", ".yml")):
        args.tracker = f"{args.tracker}.yaml"
    args.span = TimeSpan(start_sec=args.start, duration_sec=args.duration)
    args.out_path = MediaSource.validate_save_dir(parser, args.out_dir)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    media = MediaSource.from_cli(parser=argparse.ArgumentParser(), source=args.source, start=None, duration=None, allow_images=False)
    if not media.is_local_video:
        raise SystemExit(f"--source must be a local video file, got {args.source!r}")
    cap = media.open_capture()
    fps_probe = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    args.span.validate_and_seek(cap, fps_probe, total_frames)

    from ultralytics import YOLO

    model = YOLO(args.model)
    track_classes = resolve_class_ids(model.names, args.classes)
    fov = FieldOfView(
        pad=args.fov_pad,
        upscale=(args.fov_upscale if args.fov_upscale > 0 else None),
        target_width=args.fov_target_width,
        square=args.fov_square,
    )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"model={args.model} classes={args.classes} imgsz={args.imgsz} tracker={args.tracker} device={args.device}")
    print(f"source={args.source} {width}x{height} @ {fps:.0f} fps{args.span.describe()}")
    print(f"output={args.output} -> {args.out_path}")

    writers: dict[int, TrackWriter] = {}
    frames = 0
    crops = 0
    crop_warned = False
    id_warned = False
    started = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.span.reached_end(cap.get(cv2.CAP_PROP_POS_MSEC), frames, fps):
                break

            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=track_classes,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            frames += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            tids = boxes.id
            if tids is None:
                if not id_warned:
                    print("warning: frame has detections without track IDs (tracker rejected all); skipping")
                    id_warned = True
                continue

            # Single GPU->CPU transfer per frame instead of per-box scalar syncs.
            tids_np = tids.int().cpu().numpy()
            xyxy_np = boxes.xyxy.cpu().numpy()
            cls_np = boxes.cls.int().cpu().numpy()
            conf_np = boxes.conf.cpu().numpy()

            for tid, xyxy, cid, conf in zip(tids_np, xyxy_np, cls_np, conf_np):
                tid_i = int(tid)
                if tid_i < 0 or (args.max_tracks and tid_i >= args.max_tracks):
                    continue
                if tid_i not in writers:
                    writers[tid_i] = TrackWriter(
                        TrackState(track_id=tid_i, label=result.names[int(cid)]),
                        args.out_path,
                        fps,
                        args.output,
                        min_frames=args.min_frames,
                        warmup=args.warmup,
                    )
                try:
                    writers[tid_i].add(frames, fov.crop(frame, xyxy), float(conf))
                except ValueError as e:
                    if not crop_warned:
                        print(f"warning: skipping degenerate box ({e}); later degenerate boxes will be silent")
                        crop_warned = True
                    continue
                crops += 1
            if frames % args.print_every == 0:
                print(f"frame {frames}: {len(boxes)} det, {len(writers)} active track(s)")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()

    rows: list[dict[str, object]] = []
    for tid in sorted(writers):
        row = writers[tid].finalize()
        if row is not None:
            rows.append(row)
    if rows:
        with (args.out_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["track_id", "label", "kept", "num_frames", "first_frame", "last_frame", "mean_conf", "video", "frames_dir"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    elapsed = time.time() - started
    kept = len(rows)
    print(f"processed {frames} frames in {elapsed:.1f}s -> {kept} track(s), {crops} crop(s) -> {args.out_path}")
    for row in rows:
        print(f"  track {int(row['track_id']):03d} {row['label']}: {row['num_frames']} frames (frames {row['first_frame']}..{row['last_frame']})")


if __name__ == "__main__":
    main()
