import argparse
import pytest

import my_python_project
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
