"""Shared media, timestamp, cropping, ROI, FoV, masking, and pose feature utilities."""

from utils.crops import BBoxCropper, HandCropper
from utils.field_of_view import PERSON_FOV, FieldOfView
from utils.masking import MultiBlockMaskGenerator3D
from utils.media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, FrameDisplay, MediaSource, configure_opencv_ffmpeg
from utils.pose_features import COCO_KEYPOINTS, PoseFeatureExtractor, joint_angle_deg
from utils.roi import DEFAULT_YOLO_ROI, YoloBox, parse_yolo_roi, yolo_roi_to_xyxy
from utils.time_span import TimeSpan

__all__ = [
    "BBoxCropper",
    "COCO_KEYPOINTS",
    "DEFAULT_YOLO_ROI",
    "FieldOfView",
    "FrameDisplay",
    "HandCropper",
    "IMAGE_EXTENSIONS",
    "MediaSource",
    "MultiBlockMaskGenerator3D",
    "PERSON_FOV",
    "PoseFeatureExtractor",
    "TimeSpan",
    "VIDEO_EXTENSIONS",
    "YoloBox",
    "configure_opencv_ffmpeg",
    "joint_angle_deg",
    "parse_yolo_roi",
    "yolo_roi_to_xyxy",
]
