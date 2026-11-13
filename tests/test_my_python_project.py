import argparse
import pytest

import my_python_project
import yolo_keypoints_extract
import yolo_pose_sketch_demo
import yoloe_demo


def test_version():
    assert my_python_project.__version__ == "0.1.0"


def test_parse_hms():
    assert yoloe_demo.parse_hms("00:00:00") == 0.0
    assert yoloe_demo.parse_hms("01:02:03") == 3723.0
    assert yoloe_demo.parse_hms("00:00:01.5") == 1.5
    with pytest.raises(argparse.ArgumentTypeError):
        yoloe_demo.parse_hms("12:34")
    with pytest.raises(argparse.ArgumentTypeError):
        yoloe_demo.parse_hms("00:60:00")
    with pytest.raises(argparse.ArgumentTypeError):
        yoloe_demo.parse_duration_hms("00:00:00")


def test_parse_args_source_kinds(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake")

    args_default = yoloe_demo.parse_args([])
    assert args_default.source_kind == "stream"
    assert args_default.start is None
    assert args_default.duration is None

    args_file_whole = yoloe_demo.parse_args(["--source", str(clip)])
    assert args_file_whole.source_kind == "file"
    assert args_file_whole.start is None
    assert args_file_whole.duration is None

    args_file_span = yoloe_demo.parse_args(
        ["--source", str(clip), "--start", "00:01:00", "--duration", "00:00:10"]
    )
    assert args_file_span.start == 60.0
    assert args_file_span.duration == 10.0

    args_webcam = yoloe_demo.parse_args(["--source", "0"])
    assert args_webcam.source_kind == "webcam"

    with pytest.raises(SystemExit):
        yoloe_demo.parse_args(["--start", "00:00:05"])

    with pytest.raises(SystemExit):
        yoloe_demo.parse_args(["--source", "0", "--duration", "00:00:05"])

    with pytest.raises(SystemExit):
        yoloe_demo.parse_args(["--source", str(tmp_path / "missing.mp4")])


def test_save_detection_crops_and_save_validation(tmp_path):
    import numpy as np
    import torch

    with pytest.raises(SystemExit):
        yoloe_demo.parse_args(["--save", "out.mp4"])

    existing_file = tmp_path / "already_a_file"
    existing_file.write_text("x")
    with pytest.raises(SystemExit):
        yoloe_demo.parse_args(["--save", str(existing_file)])

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
    saved = yoloe_demo.save_detection_crops(
        frame, DummyBoxes(), {0: "an engineer with clear visible PPE", 1: "truck"}, crops_dir, 212, 0.35
    )
    assert saved == 3
    names = sorted(p.name for p in crops_dir.glob("*.jpg"))
    assert names == [
        "an_engineer_with_clear_visible_PPE_212_0.jpg",
        "an_engineer_with_clear_visible_PPE_212_1.jpg",
        "truck_212_0.jpg",
    ]


def test_pose_sketch_demo_helpers(tmp_path):
    from pathlib import Path
    import numpy as np

    paths = [Path("crop_299_0.jpg"), Path("crop_2_0.jpg"), Path("crop_300_0.jpg")]
    assert [p.name for p in sorted(paths, key=yolo_pose_sketch_demo.natural_sort_key)] == [
        "crop_2_0.jpg",
        "crop_299_0.jpg",
        "crop_300_0.jpg",
    ]

    img_dir = tmp_path / "crops"
    img_dir.mkdir()
    args = yolo_pose_sketch_demo.parse_args(["--source", str(img_dir), "--show"])
    assert args.source_kind == "image_dir"

    with pytest.raises(SystemExit):
        yolo_pose_sketch_demo.parse_args(["--source", str(img_dir), "--start", "00:00:05", "--show"])

    class FakeResult:
        orig_img = np.full((100, 60, 3), 255, dtype=np.uint8)

        def plot(self, img, **kwargs):
            assert int(img.max()) == 0
            assert kwargs["boxes"] is False and kwargs["labels"] is False
            out = img.copy()
            out[20, 30] = [0, 255, 0]
            return out

    sketch = yolo_pose_sketch_demo.render_sketch(FakeResult())
    assert sketch.shape == (100, 60, 3)
    assert int(sketch[0, 0].max()) == 0
    assert list(sketch[20, 30]) == [0, 255, 0]

    sbs = yolo_pose_sketch_demo.render_sketch(FakeResult(), side_by_side=True)
    assert sbs.shape == (100, 120, 3)


def test_keypoints_extract_features_and_hand_boxes():
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

    row = yolo_keypoints_extract.compute_person_features(xy, conf, box, 0.75, "sample01.mp4", 212, 0, 728.4)
    assert row["back_facing"] == 1
    assert 60.0 < float(row["left_elbow_deg"]) < 90.0
    assert row["left_wrist_nx"] < 0  # left of chest center

    hands = yolo_keypoints_extract.compute_hand_boxes(
        xy, conf, box, (1440, 2560), wrist_conf=0.4, hand_shift=0.35, raised_only=True
    )
    assert len(hands) == 1
    assert hands[0]["side"] == "left"
    # Center shifted past wrist (x < 60, y < 240) along elbow->wrist vector (-30, -20)
    assert hands[0]["center"][0] < 60
    assert hands[0]["center"][1] < 240


def test_roi_clip_slicer_yolo_box(tmp_path):
    import cv2
    import numpy as np
    from preprocessing.roi_clip_slicer import (
        DEFAULT_YOLO_ROI,
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


def test_vjepa2_ssl_module_forward_and_ema(tmp_path):
    import torch
    from transformers import VJEPA2Config, VJEPA2Model, VJEPA2ForVideoClassification
    from transformers.models.vjepa2.modeling_vjepa2 import apply_masks
    from models.vjepa2_ssl import VJEPA2SSLConfig, VJEPA2SSLModule

    # Tiny V-JEPA 2 config to test exact Predictor + masked 3D-RoPE context encoder + EMA
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
    ssl_module = VJEPA2SSLModule(
        backbone=VJEPA2Model(tiny_cfg),
        ssl_config=VJEPA2SSLConfig(frames_per_clip=8, crop_size=64, num_targets=3),
    )

    # Verify target_encoder is frozen while context_encoder & predictor require grad
    assert all(not p.requires_grad for p in ssl_module.target_encoder.parameters())
    assert any(p.requires_grad for p in ssl_module.context_encoder.parameters())
    assert any(p.requires_grad for p in ssl_module.predictor.parameters())

    videos = torch.randn(2, 8, 3, 64, 64)
    out = ssl_module(videos)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.predicted_latents.shape == out.target_latents.shape
    assert 0.0 < out.context_ratio < 1.0

    # Verify single-target predictor matches HF native VJEPA2Predictor output bit-for-bit!
    single_tgt = [torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])]
    full_ctx = torch.arange(8, 64).unsqueeze(0).repeat(2, 1)
    ssl_module.eval()
    with torch.no_grad():
        full_enc = ssl_module.context_encoder(pixel_values_videos=videos).last_hidden_state
        hf_pred = ssl_module.predictor(full_enc, [full_ctx], single_tgt).last_hidden_state
        our_pred = ssl_module.predict_from_masked_context(
            apply_masks(full_enc, [full_ctx]), full_ctx, single_tgt
        )
        assert torch.allclose(hf_pred, our_pred, atol=1e-5)

    # Backward step + EMA update + HF save_backbone_pretrained roundtrip into VJEPA2ForVideoClassification
    ssl_module.train()
    opt = torch.optim.AdamW([p for p in ssl_module.parameters() if p.requires_grad], lr=1e-3)
    before_tgt = next(ssl_module.target_encoder.parameters()).clone()
    out.loss.backward()
    opt.step()
    ssl_module.update_target_encoder(momentum=0.5)
    after_tgt = next(ssl_module.target_encoder.parameters())
    assert not torch.allclose(before_tgt, after_tgt)

    ckpt_dir = tmp_path / "adapted_vjepa2"
    ssl_module.save_backbone_pretrained(ckpt_dir)
    probe_model = VJEPA2ForVideoClassification.from_pretrained(ckpt_dir, num_labels=4)
    with torch.no_grad():
        logits = probe_model(pixel_values_videos=videos).logits
    assert logits.shape == (2, 4)
