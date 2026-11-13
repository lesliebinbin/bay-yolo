#!/usr/bin/env python3
"""YOLO-Pose keypoint feature extractor and adaptive wrist/hand cropper.

Extracts fixed 17-keypoint normalized features (ready for XGBoost / Random Forest)
and crops native-resolution hand patches projected along the forearm vector.

Run from the project root:

    uv run yolo_keypoints_extract.py --source videos/sample01.mp4 --start 00:12:00 --duration 00:00:15 --show
    uv run yolo_keypoints_extract.py --source videos/sample01.mp4 --start 00:12:00 --duration 00:00:15 --save output_keypoints
    uv run yolo_keypoints_extract.py --source person_crops/crops --save output_crop_keypoints
    uv run yolo_keypoints_extract.py --source videos/sample01.mp4 --start 00:12:00 --duration 00:01:30 --raised-only --show

When --save DIR is provided, writes:
- <DIR>/keypoints.csv        (fixed columns: normalized keypoints + angles/distances)
- <DIR>/hand_crops/*.jpg     ({side}_hand_{frame_index}_{person_index}.jpg)
- <DIR>/annotated.mp4        (or <DIR>/annotated/*.jpg for image sources)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = "videos/sample01.mp4"
DEFAULT_MODEL = "yolo26s-pose.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


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
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video / image / folder / RTSP / webcam")
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
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280)")
    parser.add_argument("--conf", type=float, default=0.45, help="Person confidence threshold (default: 0.45)")
    parser.add_argument("--wrist-conf", type=float, default=0.40, help="Minimum wrist keypoint conf to crop")
    parser.add_argument(
        "--hand-shift",
        type=float,
        default=0.35,
        help="Shift crop center past wrist along (wrist - elbow) vector (default: 0.35)",
    )
    parser.add_argument(
        "--hand-scale",
        type=float,
        default=1.4,
        help="Hand crop box size relative to forearm length (default: 1.4)",
    )
    parser.add_argument("--min-crop-size", type=int, default=96, help="Minimum hand crop side in pixels (default: 96)")
    parser.add_argument(
        "--raised-only",
        action="store_true",
        help="Only crop hands when forearm is bent (<125 deg) or wrist is raised near/above elbow",
    )
    parser.add_argument("--device", default="0", help="'0' = first CUDA GPU, 'cpu' = CPU")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames/images (0 = all)")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS on image folders with --show")
    parser.add_argument("--show", action="store_true", help="Show skeleton + hand crop boxes (space pauses, q quits)")
    parser.add_argument(
        "--save",
        default=None,
        metavar="DIR",
        help="Output directory for keypoints.csv, hand_crops/, and annotated video/images",
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
        if save_path.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".csv"}:
            parser.error(f"argument --save: expected a directory name, got {args.save!r}")
    return args


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees."""
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-6:
        return float("nan")
    cos_val = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def csv_columns() -> list[str]:
    cols = [
        "source_item",
        "frame_index",
        "timestamp_sec",
        "person_index",
        "person_conf",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "back_facing",
        "left_elbow_deg",
        "right_elbow_deg",
        "left_wrist_to_ear",
        "right_wrist_to_ear",
        "left_wrist_to_chest",
        "right_wrist_to_chest",
    ]
    for name in COCO_KEYPOINTS:
        cols.extend([f"{name}_nx", f"{name}_ny", f"{name}_conf", f"{name}_x", f"{name}_y"])
    return cols


def compute_person_features(
    xy: np.ndarray,
    conf: np.ndarray,
    box_xyxy: np.ndarray,
    person_conf: float,
    source_item: str,
    frame_index: int,
    person_index: int,
    timestamp_sec: float | None,
) -> dict[str, float | int | str]:
    """Compute camera-position-invariant features + normalized 17 keypoints for one person."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    person_h = max(y2 - y1, 1.0)

    # Anchor at mid-shoulder if visible, otherwise bbox upper-third center
    if conf[5] > 0.3 and conf[6] > 0.3:
        chest = 0.5 * (xy[5] + xy[6])
    elif conf[5] > 0.3:
        chest = xy[5]
    elif conf[6] > 0.3:
        chest = xy[6]
    else:
        chest = np.array([0.5 * (x1 + x2), y1 + 0.3 * person_h], dtype=np.float32)

    # Torso scale for zoom invariance
    if conf[11] > 0.3 and conf[12] > 0.3:
        mid_hip = 0.5 * (xy[11] + xy[12])
        scale = max(float(np.linalg.norm(chest - mid_hip)), 0.25 * person_h, 1.0)
    else:
        scale = max(0.35 * person_h, 1.0)

    back_facing = int(conf[5] > 0.3 and conf[6] > 0.3 and float(xy[5, 0]) < float(xy[6, 0]))
    l_deg = joint_angle_deg(xy[5], xy[7], xy[9]) if (conf[5] > 0.3 and conf[7] > 0.3 and conf[9] > 0.3) else float("nan")
    r_deg = (
        joint_angle_deg(xy[6], xy[8], xy[10]) if (conf[6] > 0.3 and conf[8] > 0.3 and conf[10] > 0.3) else float("nan")
    )

    def norm_dist(i: int, target: np.ndarray, target_ok: bool) -> float:
        if conf[i] <= 0.3 or not target_ok:
            return float("nan")
        return round(float(np.linalg.norm(xy[i] - target)) / scale, 4)

    # Nearest visible ear for each wrist
    ears = [xy[e] for e in (3, 4) if conf[e] > 0.3]
    l_ear_d = round(min(float(np.linalg.norm(xy[9] - e)) for e in ears) / scale, 4) if (conf[9] > 0.3 and ears) else float("nan")
    r_ear_d = (
        round(min(float(np.linalg.norm(xy[10] - e)) for e in ears) / scale, 4)
        if (conf[10] > 0.3 and ears)
        else float("nan")
    )

    row: dict[str, float | int | str] = {
        "source_item": source_item,
        "frame_index": frame_index,
        "timestamp_sec": "" if timestamp_sec is None else round(timestamp_sec, 3),
        "person_index": person_index,
        "person_conf": round(float(person_conf), 4),
        "box_x1": round(x1, 1),
        "box_y1": round(y1, 1),
        "box_x2": round(x2, 1),
        "box_y2": round(y2, 1),
        "back_facing": back_facing,
        "left_elbow_deg": "" if math.isnan(l_deg) else round(l_deg, 2),
        "right_elbow_deg": "" if math.isnan(r_deg) else round(r_deg, 2),
        "left_wrist_to_ear": "" if math.isnan(l_ear_d) else l_ear_d,
        "right_wrist_to_ear": "" if math.isnan(r_ear_d) else r_ear_d,
        "left_wrist_to_chest": "" if conf[9] <= 0.3 else norm_dist(9, chest, True),
        "right_wrist_to_chest": "" if conf[10] <= 0.3 else norm_dist(10, chest, True),
    }

    for idx, name in enumerate(COCO_KEYPOINTS):
        c = float(conf[idx])
        px, py = float(xy[idx, 0]), float(xy[idx, 1])
        nx = round((px - float(chest[0])) / scale, 4) if c > 0 else ""
        ny = round((py - float(chest[1])) / scale, 4) if c > 0 else ""
        row[f"{name}_nx"] = nx
        row[f"{name}_ny"] = ny
        row[f"{name}_conf"] = round(c, 4)
        row[f"{name}_x"] = round(px, 1)
        row[f"{name}_y"] = round(py, 1)
    return row


def compute_hand_boxes(
    xy: np.ndarray,
    conf: np.ndarray,
    box_xyxy: np.ndarray,
    frame_shape: tuple[int, int],
    wrist_conf: float = 0.40,
    hand_shift: float = 0.35,
    hand_scale: float = 1.4,
    min_crop_size: int = 96,
    raised_only: bool = False,
) -> list[dict[str, object]]:
    """Compute adaptive native-resolution square hand crop regions for one person."""
    h, w = frame_shape[:2]
    _, y1, _, y2 = (float(v) for v in box_xyxy)
    person_h = max(y2 - y1, 1.0)

    if conf[5] > 0.3 and conf[6] > 0.3 and conf[11] > 0.3 and conf[12] > 0.3:
        torso_len = float(np.linalg.norm(0.5 * (xy[5] + xy[6]) - 0.5 * (xy[11] + xy[12])))
    else:
        torso_len = 0.35 * person_h

    hands: list[dict[str, object]] = []
    for side_name, sh_i, el_i, wr_i in (("left", 5, 7, 9), ("right", 6, 8, 10)):
        if float(conf[wr_i]) < wrist_conf:
            continue
        wrist = xy[wr_i].astype(np.float32)
        has_elbow = float(conf[el_i]) >= 0.3
        elbow = xy[el_i].astype(np.float32) if has_elbow else wrist

        if raised_only:
            deg = joint_angle_deg(xy[sh_i], elbow, wrist) if (float(conf[sh_i]) >= 0.3 and has_elbow) else 180.0
            wrist_raised = has_elbow and float(wrist[1]) <= float(elbow[1]) + 20.0
            if not (deg < 125.0 or wrist_raised):
                continue

        forearm_vec = (wrist - elbow) if has_elbow else np.zeros(2, dtype=np.float32)
        forearm_len = float(np.linalg.norm(forearm_vec))
        center = wrist + hand_shift * forearm_vec
        side = max(hand_scale * forearm_len, 0.45 * torso_len, 0.22 * person_h, float(min_crop_size))
        half = 0.5 * side

        cx1 = max(0, int(round(float(center[0]) - half)))
        cy1 = max(0, int(round(float(center[1]) - half)))
        cx2 = min(w, int(round(float(center[0]) + half)))
        cy2 = min(h, int(round(float(center[1]) + half)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        hands.append(
            {
                "side": side_name,
                "bbox": (cx1, cy1, cx2, cy2),
                "center": (int(round(float(center[0]))), int(round(float(center[1])))),
                "wrist": (int(round(float(wrist[0]))), int(round(float(wrist[1])))),
                "conf": float(conf[wr_i]),
            }
        )
    return hands


def process_result(
    frame: np.ndarray,
    result,
    args: argparse.Namespace,
    source_item: str,
    frame_index: int,
    timestamp_sec: float | None,
    csv_writer: csv.DictWriter | None,
    hand_crops_dir: Path | None,
    need_annotated: bool,
) -> tuple[np.ndarray | None, int, int]:
    """Extract CSV rows + hand crops + optional annotated image for one frame."""
    annotated = (
        result.plot(boxes=False, labels=False, conf=False) if need_annotated else None
    )
    people_count = 0
    crops_saved = 0

    if result.keypoints is None or len(result.keypoints) == 0 or result.keypoints.conf is None:
        return annotated, 0, 0

    xy_all = result.keypoints.xy.cpu().numpy()
    cf_all = result.keypoints.conf.cpu().numpy()
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_conf = result.boxes.conf.cpu().numpy()

    for p_idx, (xy, cf, b_xyxy, p_conf) in enumerate(zip(xy_all, cf_all, boxes_xyxy, boxes_conf)):
        people_count += 1
        if csv_writer is not None:
            row = compute_person_features(xy, cf, b_xyxy, float(p_conf), source_item, frame_index, p_idx, timestamp_sec)
            csv_writer.writerow(row)

        hands = compute_hand_boxes(
            xy,
            cf,
            b_xyxy,
            frame.shape[:2],
            wrist_conf=args.wrist_conf,
            hand_shift=args.hand_shift,
            hand_scale=args.hand_scale,
            min_crop_size=args.min_crop_size,
            raised_only=args.raised_only,
        )
        for hand in hands:
            cx1, cy1, cx2, cy2 = hand["bbox"]
            side_name = hand["side"]
            if hand_crops_dir is not None:
                crop_name = f"{side_name}_hand_{frame_index}_{p_idx}.jpg"
                cv2.imwrite(str(hand_crops_dir / crop_name), frame[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 95])
                crops_saved += 1
            if annotated is not None:
                color = (0, 255, 255) if side_name == "left" else (255, 200, 0)
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 2)
                cv2.arrowedLine(annotated, hand["wrist"], hand["center"], color, 2, tipLength=0.25)
                cv2.putText(
                    annotated,
                    f"{side_name[0].upper()}-hand {hand['conf']:.2f}",
                    (cx1, max(18, cy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    return annotated, people_count, crops_saved


def show_frame(window: str, image: np.ndarray, delay_ms: int, paused: bool) -> tuple[bool, bool]:
    cv2.imshow(window, image)
    wait = 0 if paused else max(1, delay_ms)
    key = cv2.waitKey(wait) & 0xFF
    if key == ord("q"):
        return True, paused
    if key == ord(" "):
        return False, not paused
    return False, paused


def main() -> None:
    args = parse_args()
    if not args.show and args.save is None:
        raise SystemExit("nothing to do: pass --show and/or --save DIR")

    save_dir = Path(args.save) if args.save else None
    hand_crops_dir: Path | None = None
    ann_img_dir: Path | None = None
    csv_file = None
    csv_writer: csv.DictWriter | None = None

    if save_dir is not None:
        hand_crops_dir = save_dir / "hand_crops"
        hand_crops_dir.mkdir(parents=True, exist_ok=True)
        if args.source_kind in {"image_dir", "image"}:
            ann_img_dir = save_dir / "annotated"
            ann_img_dir.mkdir(parents=True, exist_ok=True)
        csv_file = (save_dir / "keypoints.csv").open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns())
        csv_writer.writeheader()

    model = YOLO(args.model)
    print(f"model={args.model} source={args.source} ({args.source_kind}) imgsz={args.imgsz} conf={args.conf}")

    need_annotated = bool(args.show or save_dir is not None)
    frames = 0
    total_people = 0
    total_hand_crops = 0
    started = time.time()

    try:
        if args.source_kind in {"image_dir", "image"}:
            images = (
                sorted(
                    (p for p in Path(args.source).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=natural_sort_key,
                )
                if args.source_kind == "image_dir"
                else [Path(args.source)]
            )
            if not images:
                raise SystemExit(f"no images found in {args.source}")

            delay_ms = 0 if len(images) == 1 else int(round(1000.0 / max(args.fps, 0.1)))
            paused = False
            for path in images:
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                frames += 1
                annotated, n_people, n_crops = process_result(
                    frame, result, args, path.name, frames, None, csv_writer, hand_crops_dir, need_annotated
                )
                total_people += n_people
                total_hand_crops += n_crops

                if ann_img_dir is not None and annotated is not None:
                    cv2.imwrite(str(ann_img_dir / path.name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if args.show and annotated is not None:
                    quit_now, paused = show_frame("yolo_keypoints_extract", annotated, delay_ms, paused)
                    if quit_now:
                        break
                if args.max_frames and frames >= args.max_frames:
                    break
        else:
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
                raise SystemExit(
                    f"--start {format_hms(start_sec)} exceeds video duration {format_hms(total_frames / fps)}"
                )
            if is_file and start_sec > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

            writer = (
                cv2.VideoWriter(
                    str(save_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if save_dir is not None
                else None
            )
            end_msec = (start_sec + args.duration) * 1000.0 if (is_file and args.duration is not None) else None
            half_frame_ms = 500.0 / fps
            paused = False
            source_label = Path(args.source).name if is_file else args.source

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) if is_file else 0.0
                    if end_msec is not None and frames > 0:
                        if pos_msec > 0 and pos_msec >= end_msec - half_frame_ms:
                            break
                        if pos_msec <= 0 and (frames / fps) >= args.duration - (0.5 / fps):
                            break

                    result = model.predict(
                        frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
                    )[0]
                    frames += 1
                    ts_sec = (
                        (pos_msec / 1000.0)
                        if (is_file and pos_msec > 0)
                        else (start_sec + (frames - 1) / fps if is_file else None)
                    )
                    annotated, n_people, n_crops = process_result(
                        frame, result, args, source_label, frames, ts_sec, csv_writer, hand_crops_dir, need_annotated
                    )
                    total_people += n_people
                    total_hand_crops += n_crops

                    if writer is not None and annotated is not None:
                        writer.write(annotated)
                    if args.show and annotated is not None:
                        quit_now, paused = show_frame("yolo_keypoints_extract", annotated, 1, paused)
                        if quit_now:
                            break
                    if args.max_frames and frames >= args.max_frames:
                        break
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if csv_file is not None:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    if frames:
        summary = f"processed {frames} frame(s) in {elapsed:.1f}s ({frames / elapsed:.1f} fps), {total_people} pose(s)"
        if save_dir is not None:
            summary += f" -> {save_dir / 'keypoints.csv'} + {total_hand_crops} hand crop(s) in {hand_crops_dir}"
        print(summary)


if __name__ == "__main__":
    main()
