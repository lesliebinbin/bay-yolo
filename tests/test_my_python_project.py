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

    # Verify target_encoder is frozen, context_encoder base weights are frozen (only LoRA trainable),
    # gradient checkpointing is active on context_encoder & predictor (and disabled on target_encoder), & predictor trainable
    ssl_model.train()
    assert not ssl_model.target_encoder.training
    assert not ssl_model.target_encoder.layer[0].gradient_checkpointing
    assert ssl_model.context_encoder.layer[0].gradient_checkpointing
    assert ssl_model.predictor.layer[0].gradient_checkpointing
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

    # Also verify exporting merged LoRA context_encoder directly & tasks.train CLI flags
    ckpt_ctx_dir = tmp_path / "adapted_vjepa2_ctx"
    ssl_model.save_backbone_pretrained(ckpt_ctx_dir, use_target_encoder=False)
    VJEPA2ForVideoClassification.from_pretrained(ckpt_ctx_dir, num_labels=2)

    from tasks.train import parse_args as parse_train_args

    t_args = parse_train_args(["--data-dir", str(tmp_path)])
    assert t_args.gradient_checkpointing is True and t_args.lora_rank == 16
    assert parse_train_args(["--data-dir", str(tmp_path), "--no-gradient-checkpointing"]).gradient_checkpointing is False
    assert parse_train_args(["--data-dir", str(tmp_path), "--no-grad-checkpoint"]).gradient_checkpointing is False


def test_field_of_view_crops(tmp_path):
    import cv2
    import numpy as np

    from utils.field_of_view import PERSON_FOV, FieldOfView

    # 144x256 frame, subject box 140x72 -> pad 0.10 adds 7px x / 3.6px y per side
    frame = np.zeros((144, 256, 3), dtype=np.uint8)
    box = (58, 36, 198, 108)
    frame[40:100, 60:190] = 255  # painted region fully inside the box

    fov = FieldOfView(pad=0.10)
    assert fov.compute_box(box, frame.shape[:2]) == (51, 32, 205, 112)
    assert fov.output_size(box, frame.shape[:2]) == (154, 80)  # native, no resize

    crop = fov.crop(frame, box)
    assert crop.shape == (80, 154, 3)
    assert int(crop.max()) == 255  # subject content preserved

    # upscale=2 -> 308x160 (INTER_CUBIC upsize)
    fov_x2 = FieldOfView(pad=0.10, upscale=2.0)
    assert fov_x2.output_size(box, frame.shape[:2]) == (308, 160)
    assert fov_x2.crop(frame, box).shape == (160, 308, 3)

    # target_width wins and keeps aspect (154:80 -> 256:~133 -> even 132)
    fov_tw = FieldOfView(pad=0.10, target_width=256)
    assert fov_tw.output_size(box, frame.shape[:2]) == (256, 132)

    # clamping at right/bottom edges (pad runs past the frame)
    assert fov.compute_box((250, 140, 260, 150), (144, 256)) == (250, 140, 256, 144)

    # Square crops: shorter side widened to the longer one, centered, in frame
    fov_sq = FieldOfView(pad=0.10, square=True)
    # padded 154x80 -> side 144 (frame height) -> (56,0,200,144)
    assert fov_sq.compute_box(box, frame.shape[:2]) == (56, 0, 200, 144)
    # not clamped: padded 66x110 -> side 110, centered -> (75,45,185,155)
    assert fov_sq.compute_box((100, 50, 160, 150), (200, 300)) == (75, 45, 185, 155)

    # Regression: box clamped to the bottom edge with an odd span used to
    # degenerate (the even-dims fix assigned cy1-1 to cy2). Real case from a
    # 2560x1440 CCTV frame: person at the very bottom of the image.
    edge_box = (0.0, 1282.3790283203125, 155.18719482421875, 1436.7440185546875)
    assert fov.compute_box(edge_box, (1440, 2560)) == (0, 1274, 164, 1440)  # 164x166, even
    # general bottom-edge case: span 21 odd -> cy1 moves up by 1
    assert fov.compute_box((10, 80, 60, 99), (100, 200)) == (8, 78, 62, 100)

    # Graceful degradation: whole-frame box on an odd-sized frame cannot be
    # made even; it must be returned as-is, not raised.
    fov_odd = FieldOfView(pad=0.10)
    assert fov_odd.compute_box((0, 0, 201, 101), (101, 201)) == (0, 0, 201, 101)

    # degenerate inputs are rejected
    with pytest.raises(ValueError):
        fov.compute_box((100, 100, 100, 200), (144, 256))
    with pytest.raises(ValueError):
        FieldOfView(pad=-0.1)
    with pytest.raises(ValueError):
        FieldOfView(upscale=0.0)

    # PERSON_FOV preset: pad 0.1 + square + 2x upscale; save_crop writes a valid JPEG.
    # 154x80 padded -> square 144x144 -> 288x288 output
    assert (PERSON_FOV.pad, PERSON_FOV.upscale, PERSON_FOV.square) == (0.10, 2.0, True)
    out_path = tmp_path / "fov" / "person_0.jpg"
    PERSON_FOV.save_crop(frame, box, out_path)
    assert out_path.is_file()
    loaded = cv2.imread(str(out_path))
    assert loaded is not None and loaded.shape == (288, 288, 3)


def test_fov_extract_track_writer(tmp_path):
    import cv2
    import numpy as np

    from tasks.fov_extract import TrackState, TrackWriter, even_size, fit_to_canvas

    assert even_size(154, 80) == (154, 80)
    assert even_size(155, 81) == (156, 80)
    assert even_size(1, 1) == (2, 2)

    # fit_to_canvas: 60x40 into (200, 100) -> scale 2.5 -> 150x100 centered, black side bars
    img = np.full((40, 60, 3), 200, dtype=np.uint8)
    out = fit_to_canvas(img, (200, 100))
    assert out.shape == (100, 200, 3)
    assert int(out[50, 100].max()) == 200     # center filled
    assert int(out[50, 5].max()) == 0         # left bar black
    assert int(out[50, 195].max()) == 0       # right bar black
    assert fit_to_canvas(np.full((100, 200, 3), 7, dtype=np.uint8), (200, 100)).shape == (100, 200, 3)

    def crop(w: int, h: int, value: int) -> np.ndarray:
        return np.full((h, w, 3), value, dtype=np.uint8)

    # frames mode: 6 crops, track 3 "worker"
    fw = TrackWriter(TrackState(track_id=3, label="worker"), tmp_path / "out_f", fps=25.0, output="frames", min_frames=5, warmup=3)
    for i in range(1, 7):
        fw.add(i, crop(120, 200, i * 20), 0.9)
    row = fw.finalize()
    assert row is not None and row["kept"] is True and row["num_frames"] == 6
    track_dir = tmp_path / "out_f" / "003_worker"
    assert sorted(p.name for p in track_dir.glob("*.jpg")) == [f"{i:05d}.jpg" for i in range(1, 7)]

    # video mode: canvas from median of buffered sizes; 6 even-sized frames written
    vw = TrackWriter(TrackState(track_id=7, label="truck"), tmp_path / "out_v", fps=25.0, output="video", min_frames=5, warmup=3)
    for i in range(1, 7):
        vw.add(i, crop(120 + i, 200, i * 20), 0.8)
    row = vw.finalize()
    assert row is not None and row["video"] == "007_truck.mp4"
    cap = cv2.VideoCapture(str(tmp_path / "out_v" / "007_truck.mp4"))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert w % 2 == 0 and h % 2 == 0
    cap.release()

    # short track below min_frames is dropped: no files remain
    sw = TrackWriter(TrackState(track_id=9, label="worker"), tmp_path / "out_s", fps=25.0, output="both", min_frames=5, warmup=2)
    for i in range(1, 4):
        sw.add(i, crop(100, 100, 20), 0.9)
    assert sw.finalize() is None
    assert not (tmp_path / "out_s" / "009_worker").exists()
    assert not (tmp_path / "out_s" / "009_worker.mp4").exists()
