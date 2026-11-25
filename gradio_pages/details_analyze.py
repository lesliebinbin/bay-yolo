"""Inspect an ROI clip as a synchronized 2×2 video. No person tracking required.

Run: uv run python gradio_pages/details_analyze.py
"""

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

from cli.yolo_pose import DEFAULT_MODEL, render_sketch
from utils.crops import HandCropper
from utils.field_of_view import FieldOfView
from utils.roi import parse_yolo_roi, yolo_roi_to_xyxy
from utils.time_span import TimeSpan

WIDTH, HEIGHT = 640, 384  # one panel, including its header
DEFAULT_ROI = '0.25 0.75 0.5 0.5'
POSE_FOV = FieldOfView(pad=0.20)
DEFAULT_FOV_SCALE = 1.5


def fit(tile, image):
    """Letterbox an image into an existing black tile without stretching it."""
    h, w = tile.shape[:2]
    ih, iw = image.shape[:2]
    if min(w, h, iw, ih) < 1:
        return
    scale = min(w / iw, h / ih)
    rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
    x, y = (w - rw) // 2, (h - rh) // 2
    tile[y:y + rh, x:x + rw] = cv2.resize(image, (rw, rh), interpolation=(
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    ))


def caption(tile, text, small=False):
    cv2.putText(tile, text, (5, 15 if small else 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45 if small else 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def fit_person_fov(tile, frame, box, scale):
    """Map a person-centered pixel square into the tile, padding beyond ROI edges."""
    x1, y1, x2, y2 = map(float, box)
    side = scale * max(x2 - x1, y2 - y1)
    h, w = tile.shape[:2]
    size = min(h, w)
    if size < 1 or side <= 0:
        return
    zoom = size / side
    transform = np.float32([[zoom, 0, (size - zoom * (x1 + x2)) / 2],
                            [0, zoom, (size - zoom * (y1 + y2)) / 2]])
    x, y = (w - size) // 2, (h - size) // 2
    tile[y:y + size, x:x + size] = cv2.warpAffine(frame, transform, (size, size))


def compose(frame, result, cropper, fov_scale=DEFAULT_FOV_SCALE):
    canvas = np.zeros((2 * HEIGHT, 2 * WIDTH, 3), dtype=np.uint8)
    panels = []
    for i, title in enumerate(('ROI', 'Pose', 'Hands: left | right', 'Person field of view')):
        x, y = i % 2 * WIDTH, i // 2 * HEIGHT
        panel = canvas[y:y + HEIGHT, x:x + WIDTH]
        panel[:28] = (50, 40, 32)
        caption(panel, title)
        panels.append(panel[28:])
    fit(panels[0], frame)
    if result.boxes is None or not len(result.boxes) or result.keypoints is None:
        for panel in panels[1:]:
            caption(panel, 'No person detected')
        return canvas

    result = result.cpu()
    boxes = result.boxes.xyxy.numpy()
    xy = result.keypoints.xy.numpy()
    confidence = result.keypoints.conf.numpy()
    cols = math.ceil(math.sqrt(len(boxes)))
    rows = math.ceil(len(boxes) / cols)
    for rank, idx in enumerate(np.argsort((boxes[:, 0] + boxes[:, 2]) / 2, kind='stable')):
        col, row = rank % cols, rank // cols
        h, w = panels[1].shape[:2]
        bounds = np.s_[row*h//rows:(row+1)*h//rows, col*w//cols:(col+1)*w//cols]
        pose, hands, person = [panel[bounds] for panel in panels[1:]]
        for tile in (pose, hands, person):
            caption(tile, f'P{rank+1}', small=True)

        fit_person_fov(person[18:], frame, boxes[idx], fov_scale)
        x1, y1, x2, y2 = POSE_FOV.compute_box(boxes[idx], frame.shape[:2])
        # Draw just this person's keypoints: overlapping neighbours must not appear
        # in their skeleton tile.
        sketch = render_sketch(result[int(idx)])
        fit(pose[18:], sketch[y1:y2, x1:x2])
        patches = {hand['side']: hand['bbox'] for hand in cropper.compute_boxes(
            xy[idx], confidence[idx], boxes[idx], frame.shape[:2]
        )}
        for side_index, side in enumerate(('left', 'right')):
            half = hands[:, side_index*hands.shape[1]//2:(side_index+1)*hands.shape[1]//2]
            caption(half[17:], side, small=True)
            if side in patches:
                hx, hy, ex, ey = patches[side]
                fit(half[35:], frame[hy:ey, hx:ex])
    return canvas


def upload_properties(source):
    hidden = gr.update(value=None, visible=False, interactive=False)
    if not source:
        return gr.update(value='', visible=False), hidden
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            return gr.update(value='Could not read video frame rate.', visible=True), hidden
        w, h = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info = f'{w} × {h} · {fps:.6g} FPS'
        if frames > 0:
            info += f' · {TimeSpan.format_hms(frames / fps)} · {frames} frames'
        return gr.update(value=info, visible=True), gr.update(
            value=min(2.0, fps), maximum=fps, visible=True, interactive=True)
    finally:
        cap.release()


def render_video(source, destination, model, device, imgsz, conf, wrist_conf, progress, span=TimeSpan(),
                 roi=parse_yolo_roi(DEFAULT_ROI), target_fps=2.0, fov_scale=DEFAULT_FOV_SCALE):
    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Could not read the ROI video or its frame rate.')
        if target_fps is None or not math.isfinite(target_fps) or not 0 < target_fps <= fps:
            raise ValueError(f'Target FPS must be positive and no greater than the source rate ({fps:.6g} FPS).')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Include frames whose source timestamps fall in [start, end).
        first = math.ceil(span.effective_start * fps - 1e-9)
        stop = total if total > 0 else None
        if total > 0 and first >= total:
            raise ValueError('Start time is outside the uploaded video.')
        if span.duration_sec is not None:
            end = math.ceil((span.effective_start + span.duration_sec) * fps - 1e-9)
            stop = min(stop, end) if stop is not None else end
        if first and not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError('Could not seek to the requested start time.')
        cropper = HandCropper(wrist_conf=wrist_conf)
        cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-pix_fmt', 'bgr24', '-s', f'{2*WIDTH}x{2*HEIGHT}', '-r', str(target_fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
        count = 0
        source_index = next_sample = first
        bounds = None
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            while stop is None or source_index < stop:
                if not cap.grab():
                    break
                source_index += 1
                if source_index - 1 < next_sample:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if bounds is None:
                    bounds = yolo_roi_to_xyxy(roi, frame.shape[1], frame.shape[0], even_dims=False)
                x1, y1, x2, y2 = bounds
                frame = frame[y1:y2, x1:x2]
                result = model.predict(frame, device=device, imgsz=int(imgsz), conf=conf,
                                       classes=[0], verbose=False)[0]
                encoder.stdin.write(compose(frame, result, cropper, fov_scale).tobytes())
                count += 1
                next_sample = first + math.floor(count * fps / target_fps + 1e-9)
                if count == 1 or count % 10 == 0:
                    progress((source_index - first) / (stop - first) if stop is not None else None,
                             desc=f'Processing frame {count}')
        if encoder.returncode or not count:
            raise ValueError('No playable output: check the input clip and FFmpeg H.264 encoder.')
        return count, target_fps
    finally:
        cap.release()


def build_app(weights=DEFAULT_MODEL):
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] + ['cpu']
    with gr.Blocks(title='ROI detail analysis', delete_cache=(86400, 86400)) as app:
        gr.Markdown('## ROI detail analysis\nUpload a video and select an ROI, then process it to inspect '
                    'all four views on one timeline. P1, P2, … are ordered left to right on each '
                    'frame, not tracked identities. A blank hand tile means no confident wrist.')
        source = gr.File(label='Source video', file_types=['video'], type='filepath')
        with gr.Row():
            properties = gr.Textbox(label='Source video properties', interactive=False, visible=False)
            target = gr.Number(label='Target processing FPS', visible=False, interactive=False,
                               minimum=0, info='Only sampled frames are sent to the pose model.')
        source.change(upload_properties, source, [properties, target], queue=False)
        roi_input = gr.Textbox(value=DEFAULT_ROI, label='YOLO ROI: x_center y_center width height',
                               info='Defaults to the bottom-left quarter. Enter 0.5 0.5 1 1 for the full frame.')
        with gr.Row():
            start = gr.Textbox(value='00:00:00', label='Start time (HH:MM:SS)')
            duration = gr.Textbox(value='00:00:00', label='Duration (HH:MM:SS; zero = to end)')
        with gr.Row():
            device = gr.Dropdown(devices, value=devices[0], label='Device')
            imgsz = gr.Dropdown([640, 960, 1280], value=960, label='Pose inference size')
            conf = gr.Slider(0.05, 0.95, value=0.25, label='Person confidence')
            wrist = gr.Slider(0.05, 0.95, value=0.40, label='Wrist confidence')
        scale = gr.Slider(1.0, 3.0, value=DEFAULT_FOV_SCALE, step=0.1, label='Person FoV expansion',
                          info='Bottom-right panel: square side = scale × longer side of the person box.')
        run = gr.Button('Process ROI clip', variant='primary')
        output = gr.Video(label='Synchronized detail views', interactive=False)
        status = gr.Markdown()

        def analyze(source, device, imgsz, conf, wrist, start, duration, roi_input, target, scale, progress=gr.Progress()):
            if not source:
                raise gr.Error('Upload an ROI video first.')
            try:
                roi = parse_yolo_roi(roi_input)
                span = TimeSpan(TimeSpan.parse_hms(start.strip()), TimeSpan.parse_hms(duration.strip()) or None)
            except (ValueError, argparse.ArgumentTypeError) as exc:
                raise gr.Error(str(exc)) from exc
            progress(0, desc='Loading pose weights')
            model = YOLO(weights)
            if model.task != 'pose':
                raise gr.Error('Choose YOLO pose weights with --model.')
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'roi-details.mp4'
                try:
                    count, fps = render_video(source, path, model, device, imgsz, conf, wrist, progress,
                                              span, roi, target, fov_scale=scale)
                except ValueError as exc:
                    raise gr.Error(str(exc)) from exc
                video = output.move_resource_to_block_cache(str(path))
            return video, f'Processed {count} frames at {fps:.2f} fps.'

        run.click(analyze, [source, device, imgsz, conf, wrist, start, duration, roi_input, target, scale],
                  [output, status], concurrency_limit=1)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='YOLO pose weights')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7862)
    args = parser.parse_args()
    build_app(args.model).launch(server_name=args.host, server_port=args.port)
