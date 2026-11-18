import argparse
from pathlib import Path
import shutil
import subprocess

import pytest

from cli import yolo_keypoints_extract, yolo_pose, yoloe
import my_python_project
from utils.crops import BBoxCropper, HandCropper
from utils.media import MediaSource
from utils.pose_features import PoseFeatureExtractor
from utils.time_span import TimeSpan


def test_version():
    assert my_python_project.__version__ == "0.1.0"


def test_time_span_parsing_and_boundaries():
    assert TimeSpan.parse_hms("00:00:00") == 0.0
    assert TimeSpan.parse_hms("01:02:03") == 3723.0
    assert TimeSpan.parse_hms("00:00:01.5") == 1.5
    assert TimeSpan.format_hms(3723.5) == "01:02:03.500"
    with pytest.raises(argparse.ArgumentTypeError):
        TimeSpan.parse_hms("12:34")
    with pytest.raises(argparse.ArgumentTypeError):
        TimeSpan.parse_hms("00:60:00")
    with pytest.raises(argparse.ArgumentTypeError):
        TimeSpan.parse_duration_hms("00:00:00")

    span = TimeSpan(start_sec=10.0, duration_sec=2.0)
    assert span.describe() == " (start=00:00:10 duration=00:00:02)"
    assert not span.reached_end(11_000.0, processed_frames=25, fps=25.0)
    assert span.reached_end(12_000.0, processed_frames=50, fps=25.0)
    assert span.reached_end(0.0, processed_frames=50, fps=25.0)


def test_parse_args_source_kinds(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")

    args_default = yoloe.parse_args([])
    assert args_default.source_kind == "stream"
    assert args_default.start is None
    assert args_default.duration is None

    args_file_whole = yoloe.parse_args(["--source", str(clip)])
    assert args_file_whole.source_kind == "file"
    assert args_file_whole.start is None
    assert args_file_whole.duration is None

    args_file_span = yoloe.parse_args(
        ["--source", str(clip), "--start", "00:01:00", "--duration", "00:00:10"]
    )
    assert args_file_span.start == 60.0
    assert args_file_span.duration == 10.0

    args_webcam = yoloe.parse_args(["--source", "0"])
    assert args_webcam.source_kind == "webcam"

    with pytest.raises(SystemExit):
        yoloe.parse_args(["--start", "00:00:05"])

    with pytest.raises(SystemExit):
        yoloe.parse_args(["--source", "0", "--duration", "00:00:05"])

    with pytest.raises(SystemExit):
        yoloe.parse_args(["--source", str(tmp_path / "missing.mp4")])


def test_bbox_cropper_and_save_validation(tmp_path):
    import numpy as np
    import torch

    with pytest.raises(SystemExit):
        yoloe.parse_args(["--save", "out.mp4"])

    existing_file = tmp_path / "already_a_file"
    existing_file.write_text("x")
    with pytest.raises(SystemExit):
        yoloe.parse_args(["--save", str(existing_file)])

    with pytest.raises(SystemExit):
        yolo_keypoints_extract.parse_args(["--save", "features.csv"])

    class DummyBoxes:
        cls = torch.tensor([0.0, 0.0, 1.0])
        xyxy = torch.tensor(
            [
                [100.0, 200.0, 200.0, 400.0],
                [300.0, 200.0, 400.0, 400.0],
                [50.0, 50.0, 150.0, 150.0],
            ]
        )

    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    crops_dir = tmp_path / "run_out" / "crops"
    crops_dir.mkdir(parents=True)
    saved = BBoxCropper(crop_pad=0.35).save_crops(
        frame, DummyBoxes(), {0: "an engineer with clear visible PPE", 1: "truck"}, crops_dir, 212
    )
    assert saved == 3
    names = sorted(p.name for p in crops_dir.glob("*.jpg"))
    assert names == [
        "an_engineer_with_clear_visible_PPE_212_0.jpg",
        "an_engineer_with_clear_visible_PPE_212_1.jpg",
        "truck_212_0.jpg",
    ]


def test_pose_sketch_helpers(tmp_path):
    import numpy as np

    paths = [Path("crop_299_0.jpg"), Path("crop_2_0.jpg"), Path("crop_300_0.jpg")]
    assert [p.name for p in sorted(paths, key=MediaSource.natural_sort_key)] == [
        "crop_2_0.jpg",
        "crop_299_0.jpg",
        "crop_300_0.jpg",
    ]

    img_dir = tmp_path / "crops"
    img_dir.mkdir()
    args = yolo_pose.parse_args(["--source", str(img_dir), "--show"])
    assert args.source_kind == "image_dir"

    with pytest.raises(SystemExit):
        yolo_pose.parse_args(["--source", str(img_dir), "--start", "00:00:05", "--show"])

    class FakeResult:
        orig_img = np.full((100, 60, 3), 255, dtype=np.uint8)

        def plot(self, img, **kwargs):
            assert int(img.max()) == 0
            assert kwargs["boxes"] is False and kwargs["labels"] is False
            out = img.copy()
            out[20, 30] = [0, 255, 0]
            return out

    sketch = yolo_pose.render_sketch(FakeResult())
    assert sketch.shape == (100, 60, 3)
    assert int(sketch[0, 0].max()) == 0
    assert list(sketch[20, 30]) == [0, 255, 0]

    sbs = yolo_pose.render_sketch(FakeResult(), side_by_side=True)
    assert sbs.shape == (100, 120, 3)


def test_pose_feature_extractor_and_hand_cropper():
    import numpy as np

    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    # Back-facing person: left_shoulder (5) x=100 < right_shoulder (6) x=140
    xy[5], conf[5] = [100.0, 200.0], 0.9
    xy[6], conf[6] = [140.0, 200.0], 0.9
    xy[7], conf[7] = [90.0, 260.0], 0.9   # left_elbow
    xy[9], conf[9] = [60.0, 240.0], 0.95  # left_wrist (bent & raised above elbow)
    xy[11], conf[11] = [105.0, 320.0], 0.8
    xy[12], conf[12] = [135.0, 320.0], 0.8
    box = np.array([70.0, 160.0, 170.0, 480.0], dtype=np.float32)

    extractor = PoseFeatureExtractor()
    row = extractor.extract(xy, conf, box, 0.75, "sample01.mp4", 212, 0, 728.4)
    assert set(row.keys()) == set(extractor.columns())
    assert row["back_facing"] == 1
    assert 60.0 < float(row["left_elbow_deg"]) < 90.0
    assert row["left_wrist_nx"] < 0  # left of chest center

    cropper = HandCropper(wrist_conf=0.4, hand_shift=0.35, raised_only=True)
    hands = cropper.compute_boxes(xy, conf, box, (1440, 2560))
    assert len(hands) == 1
    assert hands[0]["side"] == "left"
    # Center shifted past wrist (x < 60, y < 240) along elbow->wrist vector (-30, -20)
    assert hands[0]["center"][0] < 60
    assert hands[0]["center"][1] < 240


def test_console_scripts_installed():
    import sys

    for cmd in ("c_yoloe", "c_yolo_pose", "c_yolo_keypoints_extract"):
        candidate = Path(sys.executable).with_name(cmd)
        exe = shutil.which(cmd) or (str(candidate) if candidate.is_file() else None)
        assert exe is not None, f"{cmd} not found in venv/bin or PATH"
        proc = subprocess.run([exe, "--help"], capture_output=True, text=True, check=False)
        assert proc.returncode == 0
        assert cmd in proc.stdout


def test_roi_clip_slicer_yolo_box(tmp_path):
    import cv2
    import numpy as np
    from preprocessing.roi_clip_slicer import (
        DEFAULT_YOLO_ROI,
        SplitSpec,
        parse_args as parse_slicer_args,
        parse_yolo_roi,
        slice_video_roi_clips,
        yolo_roi_to_xyxy,
    )

    # Default ROI is bottom-left 1/2 x 1/2: x_center=0.25, y_center=0.75, w=0.5, h=0.5
    assert (DEFAULT_YOLO_ROI.x_center, DEFAULT_YOLO_ROI.y_center, DEFAULT_YOLO_ROI.width, DEFAULT_YOLO_ROI.height) == (
        0.25,
        0.75,
        0.5,
        0.5,
    )
    assert yolo_roi_to_xyxy(DEFAULT_YOLO_ROI, 2560, 1440) == (0, 720, 1280, 1440)
    assert parse_yolo_roi("0 0.25 0.75 0.5 0.5") == DEFAULT_YOLO_ROI

    with pytest.raises(ValueError):
        parse_yolo_roi([0.1, 0.5, 0.5, 0.5])  # extends < 0 on x

    # Write a synthetic 2-second 10fps video (20 frames of 200x100) and slice into two 1s ROI clips
    src_mp4 = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(src_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (200, 100))
    for i in range(20):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[50:100, 0:100] = (0, 200, 100)  # paint bottom-left quadrant green
        writer.write(frame)
    writer.release()

    out_dir = tmp_path / "sliced"
    clips = slice_video_roi_clips(src_mp4, out_dir, clip_sec=1.0, stride_sec=1.0)
    assert len(clips) == 2
    assert (out_dir / "manifest.csv").is_file()
    assert (out_dir / "roi_preview.jpg").is_file()
    assert (out_dir / "roi_yolo.txt").is_file()

    cap = cv2.VideoCapture(str(clips[0]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 100
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 50
    cap.release()

    # Verify SplitSpec smart naming (2 -> train/val, 3 -> train/val/test, 4 -> split_0..3) & deterministic --seed
    assert SplitSpec.parse("3:2").names == ("train", "val")
    assert SplitSpec.parse("3:1:1").names == ("train", "val", "test")
    assert SplitSpec.parse("4:1:1:1").names == ("split_0", "split_1", "split_2", "split_3")
    assert SplitSpec.parse("3:1", names=["ssl", "holdout"]).names == ("ssl", "holdout")
    with pytest.raises(ValueError):
        SplitSpec.parse("3")
    with pytest.raises(SystemExit):
        parse_slicer_args(["--source", str(src_mp4), "--save", str(out_dir), "--split-names", "a", "b"])

    split_dir_a = tmp_path / "split_a"
    split_dir_b = tmp_path / "split_b"
    clips_a = slice_video_roi_clips(src_mp4, split_dir_a, clip_sec=0.2, stride_sec=0.2, split="3:1:1", seed=123)
    clips_b = slice_video_roi_clips(src_mp4, split_dir_b, clip_sec=0.2, stride_sec=0.2, split="3:1:1", seed=123)
    assert len(clips_a) == 10
    assert [p.relative_to(split_dir_a) for p in clips_a] == [p.relative_to(split_dir_b) for p in clips_b]
    assert all(p.parent.name in {"train", "val", "test"} for p in clips_a)


def test_vjepa2_ssl_lightning_forward_and_trainer_fit(tmp_path):
    import lightning as L
    import torch
    from torch.utils.data import DataLoader
    from transformers import VJEPA2Config, VJEPA2ForVideoClassification, VJEPA2Model
    from transformers.models.vjepa2.modeling_vjepa2 import apply_masks
    from models.vjepa2_ssl import VJEPA2SSL, VJEPA2SSLConfig
    from utils.masking import MultiBlockMaskGenerator3D

    # Verify MultiBlockMaskGenerator3D is a plain non-nn.Module dataclass and produces disjoint context/targets
    assert not isinstance(MultiBlockMaskGenerator3D(), torch.nn.Module)
    gen = torch.Generator().manual_seed(42)
    ctx_masks, tgt_masks = MultiBlockMaskGenerator3D(num_targets=3)(2, 4, 4, 4, generator=gen)
    assert len(ctx_masks) == 1 and len(tgt_masks) == 3
    for b in range(2):
        ctx_set = set(ctx_masks[0][b].tolist())
        tgt_union = set().union(*(t[b].tolist() for t in tgt_masks))
        assert ctx_set.isdisjoint(tgt_union)

    tiny_cfg = VJEPA2Config(
        crop_size=64,
        image_size=64,
        patch_size=16,
        frames_per_clip=8,
        tubelet_size=2,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        pred_hidden_size=32,
        pred_num_attention_heads=4,
        pred_num_hidden_layers=2,
        pred_num_mask_tokens=4,
    )
    ssl_model = VJEPA2SSL(
        backbone=VJEPA2Model(tiny_cfg),
        ssl_config=VJEPA2SSLConfig(
            frames_per_clip=8,
            crop_size=64,
            num_targets=3,
            lr=1e-3,
            ema_momentum_range=(0.5, 0.5),
        ),
    )

    # Verify target_encoder is frozen, context_encoder base weights are frozen (only LoRA trainable), & predictor trainable
    ssl_model.train()
    assert not ssl_model.target_encoder.training
    assert all(not p.requires_grad for p in ssl_model.target_encoder.parameters())
    ctx_trainable = sum(p.numel() for p in ssl_model.context_encoder.parameters() if p.requires_grad)
    ctx_total = sum(p.numel() for p in ssl_model.context_encoder.parameters())
    assert 0 < ctx_trainable < ctx_total
    assert any(p.requires_grad for p in ssl_model.predictor.parameters())

    videos = torch.randn(2, 8, 3, 64, 64)
    out = ssl_model(videos)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.predicted_latents.shape == out.target_latents.shape
    assert 0.0 < out.context_ratio < 1.0

    # Verify single-target predictor matches HF native VJEPA2Predictor output bit-for-bit!
    single_tgt = [torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])]
    full_ctx = torch.arange(8, 64).unsqueeze(0).repeat(2, 1)
    ssl_model.eval()
    with torch.no_grad():
        full_enc = ssl_model.context_encoder(pixel_values_videos=videos).last_hidden_state
        hf_pred = ssl_model.predictor(full_enc, [full_ctx], single_tgt).last_hidden_state
        our_pred = ssl_model.predict_from_masked_context(
            apply_masks(full_enc, [full_ctx]), full_ctx, single_tgt
        )
        assert torch.allclose(hf_pred, our_pred, atol=1e-5)

    # End-to-end Lightning Trainer.fit() step (exercises LoRA + grad_checkpoint + EMA delta fold into target_encoder)
    before_tgt_q = ssl_model.target_encoder.layer[0].attention.query.weight.clone()
    loader = DataLoader([torch.randn(8, 3, 64, 64) for _ in range(4)], batch_size=2)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(ssl_model, train_dataloaders=loader)

    after_tgt_q = ssl_model.target_encoder.layer[0].attention.query.weight
    assert not torch.allclose(before_tgt_q, after_tgt_q)

    ckpt_dir = tmp_path / "adapted_vjepa2"
    ssl_model.save_backbone_pretrained(ckpt_dir)
    probe_model = VJEPA2ForVideoClassification.from_pretrained(ckpt_dir, num_labels=4)
    with torch.no_grad():
        logits = probe_model(pixel_values_videos=videos).logits
    assert logits.shape == (2, 4)
