"""COCO 17-keypoint geometric angle and scale-invariant pose feature extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return interior angle ABC in degrees."""
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-6:
        return float("nan")
    cos_val = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


@dataclass(frozen=True)
class PoseFeatureExtractor:
    """Extracts camera-position- and zoom-invariant 17-keypoint tabular features per person."""

    min_vis_conf: float = 0.3

    @staticmethod
    def columns() -> list[str]:
        """Ordered CSV column names produced by `extract`."""
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

    def extract(
        self,
        xy: np.ndarray,
        conf: np.ndarray,
        box_xyxy: np.ndarray,
        person_conf: float,
        source_item: str,
        frame_index: int,
        person_index: int,
        timestamp_sec: float | None,
    ) -> dict[str, float | int | str]:
        """Compute normalized keypoints, joint angles, and relative wrist distances for one person."""
        vis = self.min_vis_conf
        x1, y1, x2, y2 = (float(v) for v in box_xyxy)
        person_h = max(y2 - y1, 1.0)

        # Anchor at mid-shoulder if visible, otherwise bbox upper-third center
        if conf[5] > vis and conf[6] > vis:
            chest = 0.5 * (xy[5] + xy[6])
        elif conf[5] > vis:
            chest = xy[5]
        elif conf[6] > vis:
            chest = xy[6]
        else:
            chest = np.array([0.5 * (x1 + x2), y1 + 0.3 * person_h], dtype=np.float32)

        # Torso scale for zoom invariance
        if conf[11] > vis and conf[12] > vis:
            mid_hip = 0.5 * (xy[11] + xy[12])
            scale = max(float(np.linalg.norm(chest - mid_hip)), 0.25 * person_h, 1.0)
        else:
            scale = max(0.35 * person_h, 1.0)

        back_facing = int(conf[5] > vis and conf[6] > vis and float(xy[5, 0]) < float(xy[6, 0]))
        l_deg = (
            joint_angle_deg(xy[5], xy[7], xy[9])
            if (conf[5] > vis and conf[7] > vis and conf[9] > vis)
            else float("nan")
        )
        r_deg = (
            joint_angle_deg(xy[6], xy[8], xy[10])
            if (conf[6] > vis and conf[8] > vis and conf[10] > vis)
            else float("nan")
        )

        def norm_dist(i: int, target: np.ndarray, target_ok: bool) -> float:
            if conf[i] <= vis or not target_ok:
                return float("nan")
            return round(float(np.linalg.norm(xy[i] - target)) / scale, 4)

        ears = [xy[e] for e in (3, 4) if conf[e] > vis]
        l_ear_d = (
            round(min(float(np.linalg.norm(xy[9] - e)) for e in ears) / scale, 4)
            if (conf[9] > vis and ears)
            else float("nan")
        )
        r_ear_d = (
            round(min(float(np.linalg.norm(xy[10] - e)) for e in ears) / scale, 4)
            if (conf[10] > vis and ears)
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
            "left_wrist_to_chest": "" if conf[9] <= vis else norm_dist(9, chest, True),
            "right_wrist_to_chest": "" if conf[10] <= vis else norm_dist(10, chest, True),
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
