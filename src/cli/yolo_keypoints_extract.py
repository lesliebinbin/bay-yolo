#!/usr/bin/env python3
"""YOLO-Pose keypoint feature extractor and adaptive wrist/hand cropper CLI (`c_yolo_keypoints_extract`).

Extracts fixed 17-keypoint normalized features (ready for XGBoost / Random Forest)
and crops native-resolution hand patches projected along the forearm vector.

Examples:

    c_yolo_keypoints_extract --source videos/sample01.mp4 --start 00:12:00 --duration 00:00:15 --show
    c_yolo_keypoints_extract --source videos/sample01.mp4 --start 00:12:00 --duration 00:00:15 --save output_keypoints
    c_yolo_keypoints_extract --source person_crops/crops --save output_crop_keypoints
    c_yolo_keypoints_extract --source videos/sample01.mp4 --start 00:12:00 --duration 00:01:30 --raised-only --show

When --save DIR is provided, writes:
- <DIR>/keypoints.csv        (fixed columns: normalized keypoints + angles/distances)
- <DIR>/hand_crops/*.jpg     ({side}_hand_{frame_index}_{person_index}.jpg)
- <DIR>/annotated.mp4        (or <DIR>/annotated/*.jpg for image sources)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from utils.crops import HandCropper
from utils.media import FrameDisplay, MediaSource
from utils.pose_features import PoseFeatureExtractor
from utils.time_span import TimeSpan

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_SOURCE = "videos/sample01.mp4"
DEFAULT_MODEL = "yolo26s-pose.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="c_yolo_keypoints_extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video / image / folder / RTSP / webcam")
    TimeSpan.add_cli_args(parser)
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
    args.save_path = MediaSource.validate_save_dir(parser, args.save, extra_forbidden_exts={".csv"})
    args.media = MediaSource.from_cli(parser, args.source, args.start, args.duration, allow_images=True)
    args.source_kind = args.media.kind
    args.span = TimeSpan(start_sec=args.start, duration_sec=args.duration)
    return args


def process_result(
    frame: np.ndarray,
    result,
    feature_extractor: PoseFeatureExtractor,
    hand_cropper: HandCropper,
    source_item: str,
    frame_index: int,
    timestamp_sec: float | None,
    csv_writer: csv.DictWriter | None,
    hand_crops_dir: Path | None,
    need_annotated: bool,
) -> tuple[np.ndarray | None, int, int]:
    """Extract CSV rows + hand crops + optional annotated image for one frame."""
    annotated = result.plot(boxes=False, labels=False, conf=False) if need_annotated else None
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
            row = feature_extractor.extract(
                xy, cf, b_xyxy, float(p_conf), source_item, frame_index, p_idx, timestamp_sec
            )
            csv_writer.writerow(row)

        hands = hand_cropper.compute_boxes(xy, cf, b_xyxy, frame.shape[:2])
        crops_saved += hand_cropper.save_and_annotate(
            frame, annotated, hands, hand_crops_dir, frame_index, p_idx
        )

    return annotated, people_count, crops_saved


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.show and args.save_path is None:
        raise SystemExit("nothing to do: pass --show and/or --save DIR")

    save_dir: Path | None = args.save_path
    hand_crops_dir: Path | None = None
    ann_img_dir: Path | None = None
    csv_file = None
    csv_writer: csv.DictWriter | None = None
    feature_extractor = PoseFeatureExtractor()
    hand_cropper = HandCropper(
        wrist_conf=args.wrist_conf,
        hand_shift=args.hand_shift,
        hand_scale=args.hand_scale,
        min_crop_size=args.min_crop_size,
        raised_only=args.raised_only,
    )

    if save_dir is not None:
        hand_crops_dir = save_dir / "hand_crops"
        hand_crops_dir.mkdir(parents=True, exist_ok=True)
        if args.media.is_image_source:
            ann_img_dir = save_dir / "annotated"
            ann_img_dir.mkdir(parents=True, exist_ok=True)
        csv_file = (save_dir / "keypoints.csv").open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=feature_extractor.columns())
        csv_writer.writeheader()

    model = YOLO(args.model)
    print(f"model={args.model} source={args.source} ({args.source_kind}) imgsz={args.imgsz} conf={args.conf}")

    need_annotated = bool(args.show or save_dir is not None)
    display = FrameDisplay("yolo_keypoints_extract", enabled=args.show)
    frames = 0
    total_people = 0
    total_hand_crops = 0
    started = time.time()

    try:
        if args.media.is_image_source:
            images = args.media.collect_images()
            delay_ms = 0 if len(images) == 1 else int(round(1000.0 / max(args.fps, 0.1)))
            for path in images:
                frame = cv2.imread(str(path))
                if frame is None:
                    continue
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                frames += 1
                annotated, n_people, n_crops = process_result(
                    frame,
                    result,
                    feature_extractor,
                    hand_cropper,
                    path.name,
                    frames,
                    None,
                    csv_writer,
                    hand_crops_dir,
                    need_annotated,
                )
                total_people += n_people
                total_hand_crops += n_crops

                if ann_img_dir is not None and annotated is not None:
                    cv2.imwrite(str(ann_img_dir / path.name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if annotated is not None and display.show(annotated, delay_ms=delay_ms):
                    break
                if args.max_frames and frames >= args.max_frames:
                    break
        else:
            is_file = args.media.is_local_video
            start_sec = args.span.effective_start
            cap = args.media.open_capture()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if is_file:
                args.span.validate_and_seek(cap, fps, total_frames)

            writer = (
                cv2.VideoWriter(
                    str(save_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if save_dir is not None
                else None
            )
            source_label = Path(args.source).name if is_file else args.source

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) if is_file else 0.0
                    if is_file and args.span.reached_end(pos_msec, frames, fps):
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
                        frame,
                        result,
                        feature_extractor,
                        hand_cropper,
                        source_label,
                        frames,
                        ts_sec,
                        csv_writer,
                        hand_crops_dir,
                        need_annotated,
                    )
                    total_people += n_people
                    total_hand_crops += n_crops

                    if writer is not None and annotated is not None:
                        writer.write(annotated)
                    if annotated is not None and display.show(annotated, delay_ms=1):
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
        display.close()

    elapsed = time.time() - started
    if frames:
        summary = f"processed {frames} frame(s) in {elapsed:.1f}s ({frames / elapsed:.1f} fps), {total_people} pose(s)"
        if save_dir is not None:
            summary += f" -> {save_dir / 'keypoints.csv'} + {total_hand_crops} hand crop(s) in {hand_crops_dir}"
        print(summary)


if __name__ == "__main__":
    main()
