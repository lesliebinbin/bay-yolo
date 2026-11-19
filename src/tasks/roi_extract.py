"""Extract full-duration ROI-cropped videos for every item in `item_rois.yaml`.

Rules
-----
- No FPS downsampling: output keeps the source fps exactly.
- No resizing: output resolution is exactly the ROI pixel size (no downscale).
- Angle matching: `A` (overhead) -> `sample*` videos, `B` (side) -> non-sample videos (e.g. F110).
- Output layout under `--out-dir` (default: `video_rois/`):
    {item}/{video-stem}.mp4          for items with a single ROI
    {item}_{index}/{video-stem}.mp4  for items with multiple ROIs (index = 0-based position in config)
- A `manifest.csv` summarizing every extracted file is written to the output dir.

Run from the project root:

    uv run python -m tasks.roi_extract --dry-run
    uv run python -m tasks.roi_extract
    uv run python -m tasks.roi_extract --source-dir video_sources --out-dir video_rois --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

from utils.roi import parse_yolo_roi, yolo_roi_to_xyxy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "src" / "config" / "item_rois.yaml"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "video_sources"
DEFAULT_OUT_DIR = PROJECT_ROOT / "video_rois"


@dataclass(frozen=True)
class RoiTask:
    item: str
    angle_type: str
    folder: str
    roi_x: float
    roi_y: float
    roi_w: float
    roi_h: float
    src: Path
    out: Path
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int


def angle_for(video: Path) -> str:
    """`sample*` files are overhead (A); everything else is side (B)."""
    return "A" if video.name.startswith("sample") else "B"


def ffprobe_video(path: Path) -> tuple[int, int, str]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json",
        str(path),
    ]
    data = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"]), stream.get("r_frame_rate", "")


def detect_encoder() -> list[str]:
    """Prefer NVIDIA NVENC when available, otherwise libx264 (fast preset)."""
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=True,
    ).stdout
    if "h264_nvenc" in encoders:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-spatial_aq", "1"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]


def build_tasks(config_path: Path, source_dir: Path, out_dir: Path) -> list[RoiTask]:
    items = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    videos = sorted(p for p in source_dir.glob("*.mp4") if p.is_file())
    if not videos:
        raise FileNotFoundError(f"no .mp4 videos found in {source_dir}")

    tasks: list[RoiTask] = []
    for item, entries in items.items():
        entries = [e for e in entries if isinstance(e, dict) and "roi" in e]
        if not entries:
            continue
        multiple = len(entries) > 1
        for idx, entry in enumerate(entries):
            angle_type = str(entry["angle_type"])
            roi = parse_yolo_roi(entry["roi"])
            folder = f"{item}_{idx}" if multiple else str(item)
            for video in videos:
                if angle_for(video) != angle_type:
                    continue
                src_w, src_h, _ = ffprobe_video(video)
                x1, y1, x2, y2 = yolo_roi_to_xyxy(roi, src_w, src_h, even_dims=True)
                out_path = out_dir / folder / f"{video.stem}.mp4"
                tasks.append(
                    RoiTask(
                        item=str(item),
                        angle_type=angle_type,
                        folder=folder,
                        roi_x=roi.x_center, roi_y=roi.y_center, roi_w=roi.width, roi_h=roi.height,
                        src=video,
                        out=out_path,
                        crop_x=x1, crop_y=y1, crop_w=x2 - x1, crop_h=y2 - y1,
                    )
                )
    return tasks


def run_task(task: RoiTask, encoder: list[str], overwrite: bool) -> str:
    if task.out.exists() and not overwrite:
        return f"skip {task.out} (exists)"
    task.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
        "-i", str(task.src),
        "-vf", f"crop={task.crop_w}:{task.crop_h}:{task.crop_x}:{task.crop_y}",
        "-map", "0:v:0", "-map", "0:a?",
        *encoder,
        "-c:a", "aac", "-b:a", "128k",
        str(task.out),
    ]
    subprocess.run(cmd, check=True)
    size_mb = task.out.stat().st_size / 1e6
    return f"ok   {task.out} ({task.crop_w}x{task.crop_h}, {size_mb:.0f} MB)"


def write_manifest(tasks: list[RoiTask], out_dir: Path) -> None:
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "angle_type", "folder", "roi_x", "roi_y", "roi_w", "roi_h", "source", "output", "crop_w", "crop_h"])
        for t in tasks:
            writer.writerow([t.item, t.angle_type, t.folder, t.roi_x, t.roi_y, t.roi_w, t.roi_h, t.src.name, str(t.out), t.crop_w, t.crop_h])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="item_rois.yaml path")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Directory containing source videos")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory (video_rois/)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel ffmpeg jobs (default: 4)")
    parser.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto", help="Video encoder to use (default: auto)")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode even if the output file already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned jobs without running ffmpeg")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    if args.encoder == "nvenc":
        if not ("h264_nvenc" in subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout):
            print("h264_nvenc not available", file=sys.stderr)
            return 2
        encoder = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-spatial_aq", "1"]
    elif args.encoder == "x264":
        encoder = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]
    else:
        encoder = detect_encoder()

    tasks = build_tasks(args.config, args.source_dir, args.out_dir)
    if not tasks:
        print("no ROI tasks matched (check angle mapping / video names)")
        return 1

    if args.dry_run:
        print(f"encoder: {' '.join(encoder)}")
        for t in tasks:
            print(f"  [{t.angle_type}] {t.item:<22} -> {t.out.relative_to(args.out_dir)}  crop={t.crop_w}x{t.crop_h} @({t.crop_x},{t.crop_y})")
        print(f"\n{len(tasks)} job(s) planned -> {args.out_dir}")
        return 0

    print(f"running {len(tasks)} job(s) with {args.workers} worker(s), encoder={' '.join(encoder)}")
    failures: list[tuple[RoiTask, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_task, t, encoder, args.overwrite): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
                print(f"[{i}/{len(tasks)}] {result}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append((task, str(exc)))
                print(f"[{i}/{len(tasks)}] FAIL {task.out}: {exc}", file=sys.stderr, flush=True)

    write_manifest(tasks, args.out_dir)
    print(f"manifest written -> {args.out_dir / 'manifest.csv'}")
    if failures:
        print(f"{len(failures)} job(s) FAILED", file=sys.stderr)
        return 1
    print("all done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
