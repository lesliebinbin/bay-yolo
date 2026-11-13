"""Padded bounding-box and forearm-projected adaptive wrist/hand crop utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.pose_features import joint_angle_deg


@dataclass
class BBoxCropper:
    """Saves padded native-resolution detection crops `{slug}_{frame_index}_{instance_index}.jpg`."""

    crop_pad: float = 0.35
    jpeg_quality: int = 95

    def save_crops(
        self,
        frame: np.ndarray,
        boxes,
        names: dict[int, str],
        crops_dir: Path,
        frame_index: int,
    ) -> int:
        """Crop and write detections for a single `frame`, returning number of saved files."""
        h, w = frame.shape[:2]
        per_class_idx: dict[str, int] = {}
        saved = 0
        for cls_t, xyxy_t in zip(boxes.cls, boxes.xyxy):
            x1, y1, x2, y2 = (float(v) for v in xyxy_t.tolist())
            pad_x = (x2 - x1) * self.crop_pad
            pad_y = (y2 - y1) * self.crop_pad
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
            cv2.imwrite(str(out_path), frame[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            saved += 1
        return saved


@dataclass
class HandCropper:
    """Computes, saves, and annotates adaptive square hand crops projected past the wrist."""

    wrist_conf: float = 0.40
    hand_shift: float = 0.35
    hand_scale: float = 1.4
    min_crop_size: int = 96
    raised_only: bool = False
    jpeg_quality: int = 95

    def compute_boxes(
        self,
        xy: np.ndarray,
        conf: np.ndarray,
        box_xyxy: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> list[dict[str, object]]:
        """Compute native-resolution square hand crop regions for one person."""
        h, w = frame_shape[:2]
        _, y1, _, y2 = (float(v) for v in box_xyxy)
        person_h = max(y2 - y1, 1.0)

        if conf[5] > 0.3 and conf[6] > 0.3 and conf[11] > 0.3 and conf[12] > 0.3:
            torso_len = float(np.linalg.norm(0.5 * (xy[5] + xy[6]) - 0.5 * (xy[11] + xy[12])))
        else:
            torso_len = 0.35 * person_h

        hands: list[dict[str, object]] = []
        for side_name, sh_i, el_i, wr_i in (("left", 5, 7, 9), ("right", 6, 8, 10)):
            if float(conf[wr_i]) < self.wrist_conf:
                continue
            wrist = xy[wr_i].astype(np.float32)
            has_elbow = float(conf[el_i]) >= 0.3
            elbow = xy[el_i].astype(np.float32) if has_elbow else wrist

            if self.raised_only:
                deg = joint_angle_deg(xy[sh_i], elbow, wrist) if (float(conf[sh_i]) >= 0.3 and has_elbow) else 180.0
                wrist_raised = has_elbow and float(wrist[1]) <= float(elbow[1]) + 20.0
                if not (deg < 125.0 or wrist_raised):
                    continue

            forearm_vec = (wrist - elbow) if has_elbow else np.zeros(2, dtype=np.float32)
            forearm_len = float(np.linalg.norm(forearm_vec))
            center = wrist + self.hand_shift * forearm_vec
            side = max(
                self.hand_scale * forearm_len,
                0.45 * torso_len,
                0.22 * person_h,
                float(self.min_crop_size),
            )
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

    def save_and_annotate(
        self,
        frame: np.ndarray,
        annotated: np.ndarray | None,
        hands: list[dict[str, object]],
        hand_crops_dir: Path | None,
        frame_index: int,
        person_index: int,
    ) -> int:
        """Write hand crop images to `hand_crops_dir` and draw vectors/boxes onto `annotated`."""
        crops_saved = 0
        for hand in hands:
            cx1, cy1, cx2, cy2 = hand["bbox"]  # type: ignore[misc]
            side_name = str(hand["side"])
            if hand_crops_dir is not None:
                crop_name = f"{side_name}_hand_{frame_index}_{person_index}.jpg"
                cv2.imwrite(
                    str(hand_crops_dir / crop_name),
                    frame[cy1:cy2, cx1:cx2],
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                crops_saved += 1
            if annotated is not None:
                color = (0, 255, 255) if side_name == "left" else (255, 200, 0)
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), color, 2)
                cv2.arrowedLine(annotated, hand["wrist"], hand["center"], color, 2, tipLength=0.25)  # type: ignore[arg-type]
                cv2.putText(
                    annotated,
                    f"{side_name[0].upper()}-hand {float(hand['conf']):.2f}",  # type: ignore[arg-type]
                    (cx1, max(18, cy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return crops_saved
