#!/usr/bin/env python3
"""Standalone Gradio visualizer for V-JEPA 2 3D masking & latent prediction.

Run from the project root:

    uv run gradio_pages/vjepa2_viz.py
    uv run gradio_pages/vjepa2_viz.py --port 7860

Displays 3 synchronized views for an uploaded clip (no downstream head required):
1. Original Video Clip
2. Masked Context Input (what `context_encoder` sees; 3D target blocks hidden)
3. Predictor Output in Latent Space (3D PCA semantic map with predicted blocks spliced in
   + per-patch cosine similarity heatmap against `target_encoder`).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from einops import rearrange, repeat
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoVideoProcessor

from models.vjepa2_ssl import VJEPA2SSL, VJEPA2SSLConfig, read_video_clip_rgb
from utils.masking import MultiBlockMaskGenerator3D

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"
_MODEL_CACHE: dict[tuple[str, str], tuple[VJEPA2SSL, object]] = {}


def load_model_cached(model_name_or_path: str, device_str: str) -> tuple[VJEPA2SSL, object]:
    key = (model_name_or_path.strip(), device_str)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        dtype = torch.bfloat16 if device_str.startswith("cuda") else torch.float32
        processor = AutoVideoProcessor.from_pretrained(key[0])
        cfg = VJEPA2SSLConfig(
            model_name_or_path=key[0],
            crop_size=processor.crop_size["height"],
            lora_rank=0,
            gradient_checkpointing=False,
        )
        model = VJEPA2SSL.from_pretrained(key[0], ssl_config=cfg, torch_dtype=dtype)
        model.eval().to(device_str)
        _MODEL_CACHE[key] = (model, processor)
    return _MODEL_CACHE[key]


def write_browser_mp4(frames_rgb: np.ndarray, fps: float = 5.0) -> str:
    """Encode `(T, H, W, 3)` RGB uint8 array into an H.264 `.mp4` playable in all browsers."""
    t, h, w, _ = frames_rgb.shape
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-loglevel",
            "error",
            out_path,
        ]
        subprocess.run(cmd, input=frames_rgb.tobytes(), check=True)
        return out_path

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for idx in range(t):
            writer.write(cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return out_path


def pca_latents_to_rgb(
    full_target_tokens: torch.Tensor,
    pred_hybrid_tokens: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 3D PCA on `full_target_tokens` `(N, D)` and project both target & hybrid tokens to `[0, 255]` RGB."""
    x_tgt = full_target_tokens.float()
    x_hyb = pred_hybrid_tokens.float()
    mean = x_tgt.mean(dim=0, keepdim=True)
    centered_tgt = x_tgt - mean
    _, _, v = torch.pca_lowrank(centered_tgt, q=3, center=False)

    proj_tgt = centered_tgt @ v[:, :3]
    proj_hyb = (x_hyb - mean) @ v[:, :3]

    lo = torch.quantile(proj_tgt, 0.02, dim=0, keepdim=True)
    hi = torch.quantile(proj_tgt, 0.98, dim=0, keepdim=True)
    scale = (hi - lo).clamp_min(1e-6)

    rgb_tgt = ((proj_tgt - lo) / scale).clamp(0.0, 1.0)
    rgb_hyb = ((proj_hyb - lo) / scale).clamp(0.0, 1.0)
    return (rgb_tgt.cpu().numpy() * 255).astype(np.uint8), (rgb_hyb.cpu().numpy() * 255).astype(np.uint8)


def render_Masked_and_prediction_views(
    clip_rgb: np.ndarray,
    occupied_thw: np.ndarray,
    rgb_tgt_thw: np.ndarray,
    rgb_hyb_thw: np.ndarray,
    cos_sim_thw: np.ndarray,
    tubelet_size: int,
    crop_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(T, H, W, 3)` Masked Context clip and `(T, H, 2*W, 3)` Latent Prediction clip."""
    num_frames = clip_rgb.shape[0]
    resized_rgb = np.stack(
        [cv2.resize(f, (crop_size, crop_size), interpolation=cv2.INTER_AREA) for f in clip_rgb],
        axis=0,
    )

    # Expand tubelet grid_t -> frame T and patch (grid_h, grid_w) -> pixel (H, W)
    occ_frames = repeat(occupied_thw, "gt gh gw -> (gt ts) gh gw", ts=tubelet_size)[:num_frames]
    tgt_pca_frames = repeat(rgb_tgt_thw, "gt gh gw c -> (gt ts) gh gw c", ts=tubelet_size)[:num_frames]
    hyb_pca_frames = repeat(rgb_hyb_thw, "gt gh gw c -> (gt ts) gh gw c", ts=tubelet_size)[:num_frames]
    sim_frames = repeat(cos_sim_thw, "gt gh gw -> (gt ts) gh gw", ts=tubelet_size)[:num_frames]

    masked_frames = resized_rgb.copy()
    pred_panel_frames = []

    for t in range(num_frames):
        occ_px = cv2.resize(occ_frames[t].astype(np.uint8), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST) > 0

        # Panel 2: Darken masked patches to 12% brightness + draw crisp cyan border around masked region
        frame_masked = masked_frames[t]
        frame_masked[occ_px] = (frame_masked[occ_px].astype(np.float32) * 0.12).astype(np.uint8)
        contours, _ = cv2.findContours(occ_px.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_masked, contours, -1, (0, 220, 255), 2)
        masked_frames[t] = frame_masked

        # Panel 3 Left: Predicted Latent PCA spliced into unmasked Context PCA (with outline on predicted blocks)
        hyb_up = cv2.resize(hyb_pca_frames[t], (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(hyb_up, contours, -1, (255, 255, 255), 2)
        cv2.putText(hyb_up, "Pred PCA", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 3 Right: Ground-truth Target PCA vs Cosine Similarity Heatmap on masked blocks
        tgt_up = cv2.resize(tgt_pca_frames[t], (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        sim_norm = np.clip((sim_frames[t] + 1.0) * 0.5, 0.0, 1.0)
        sim_u8 = cv2.resize((sim_norm * 255).astype(np.uint8), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
        heat_bgr = cv2.applyColorMap(sim_u8, cv2.COLORMAP_TURBO)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        overlay = (resized_rgb[t].astype(np.float32) * 0.35).astype(np.uint8)
        overlay[occ_px] = (0.30 * resized_rgb[t][occ_px] + 0.70 * heat_rgb[occ_px]).astype(np.uint8)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        cv2.putText(overlay, "Target PCA", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "CosSim", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Side-by-side: [Target PCA | Pred Hybrid PCA | Masked Cosine-Sim Heatmap]
        tgt_labeled = tgt_up.copy()
        cv2.drawContours(tgt_labeled, contours, -1, (255, 255, 255), 1)
        cv2.putText(tgt_labeled, "Teacher PCA", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Clear overlay label
        overlay_clean = (resized_rgb[t].astype(np.float32) * 0.35).astype(np.uint8)
        overlay_clean[occ_px] = (0.25 * resized_rgb[t][occ_px] + 0.75 * heat_rgb[occ_px]).astype(np.uint8)
        cv2.drawContours(overlay_clean, contours, -1, (255, 255, 255), 2)
        cv2.putText(overlay_clean, "CosSim Map", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        pred_panel_frames.append(np.concatenate([tgt_labeled, hyb_up, overlay_clean], axis=1))

    return resized_rgb, masked_frames, np.stack(pred_panel_frames, axis=0)


@torch.no_grad()
def run_vjepa2_visualization(
    uploaded_video: str | None,
    model_id: str,
    frames_per_clip: int,
    num_targets: int,
    spatial_max: float,
    seed: int,
    device_choice: str,
) -> tuple[str, str, str, str]:
    if not uploaded_video or not Path(uploaded_video).is_file():
        raise gr.Error("Upload a video clip first.")
    video_path = uploaded_video

    cap = cv2.VideoCapture(str(video_path))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 5.0)
    cap.release()
    playback_fps = min(max(src_fps, 3.0), 8.0)

    clip_rgb = read_video_clip_rgb(Path(video_path), num_frames=int(frames_per_clip))
    model, processor = load_model_cached(model_id, device_choice)

    inputs = processor([clip_rgb], return_tensors="pt")
    videos = inputs["pixel_values_videos"].to(device=device_choice, dtype=next(model.parameters()).dtype)

    grid_t, grid_h, grid_w = model.token_grid_shape(videos)
    total_tokens = grid_t * grid_h * grid_w

    mask_gen = MultiBlockMaskGenerator3D(
        num_targets=int(num_targets),
        spatial_scale=(0.15, float(spatial_max)),
        temporal_scale=(0.50, 1.00),
    )
    gen = torch.Generator().manual_seed(int(seed))
    context_mask, target_mask = mask_gen(1, grid_t, grid_h, grid_w, device=device_choice, generator=gen)

    out = model(videos, context_mask=context_mask, target_mask=target_mask)
    full_targets = model.encode_target(videos)[0]  # (N_tokens, D)

    # Splice predictor's masked latent predictions back into the 3D token grid
    hybrid_latents = full_targets.clone()
    cos_sim_flat = torch.ones(total_tokens, device=device_choice, dtype=torch.float32)
    occupied_flat = torch.zeros(total_tokens, device=device_choice, dtype=torch.bool)

    preds_per_block = out.predicted_latents.chunk(len(target_mask), dim=0)
    tgts_per_block = out.target_latents.chunk(len(target_mask), dim=0)
    block_sims = []
    for t_mask, p_blk, g_blk in zip(target_mask, preds_per_block, tgts_per_block):
        idx = t_mask[0]  # (K,)
        occupied_flat[idx] = True
        hybrid_latents[idx] = p_blk[0].to(hybrid_latents.dtype)
        sim = F.cosine_similarity(p_blk[0].float(), g_blk[0].float(), dim=-1)
        cos_sim_flat[idx] = sim
        block_sims.append(float(sim.mean().item()))

    rgb_tgt, rgb_hyb = pca_latents_to_rgb(full_targets, hybrid_latents)
    orig_vis, masked_vis, pred_vis = render_Masked_and_prediction_views(
        clip_rgb=clip_rgb,
        occupied_thw=rearrange(occupied_flat.cpu().numpy(), "(gt gh gw) -> gt gh gw", gt=grid_t, gh=grid_h, gw=grid_w),
        rgb_tgt_thw=rearrange(rgb_tgt, "(gt gh gw) c -> gt gh gw c", gt=grid_t, gh=grid_h, gw=grid_w),
        rgb_hyb_thw=rearrange(rgb_hyb, "(gt gh gw) c -> gt gh gw c", gt=grid_t, gh=grid_h, gw=grid_w),
        cos_sim_thw=rearrange(cos_sim_flat.cpu().numpy(), "(gt gh gw) -> gt gh gw", gt=grid_t, gh=grid_h, gw=grid_w),
        tubelet_size=model.config.tubelet_size,
        crop_size=model.ssl_config.crop_size,
    )

    mean_cos = float(np.mean(block_sims))
    metrics_md = (
        f"**Smooth-L1 Loss:** `{float(out.loss.item()):.4f}` &nbsp;|&nbsp; "
        f"**Mean Masked Cosine Sim:** `{mean_cos:.4f}` &nbsp;|&nbsp; "
        f"**3D Grid:** `{grid_t}×{grid_h}×{grid_w}` (`{total_tokens}` tokens) &nbsp;|&nbsp; "
        f"**Visible Context:** `{out.context_ratio * 100:.1f}%` &nbsp;|&nbsp; "
        f"**Masked Target:** `{int(occupied_flat.sum().item()) / total_tokens * 100:.1f}%`"
    )

    return (
        write_browser_mp4(orig_vis, fps=playback_fps),
        write_browser_mp4(masked_vis, fps=playback_fps),
        write_browser_mp4(pred_vis, fps=playback_fps),
        metrics_md,
    )


def build_app(default_model: str) -> gr.Blocks:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_choices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]

    with gr.Blocks(title="V-JEPA 2 Masked Latent Visualizer") as demo:
        gr.Markdown(
            "## V-JEPA 2 Self-Supervised 3D Masking & Latent Prediction Visualizer\n"
            "Upload a clip to inspect what `context_encoder` sees and how well `predictor` reconstructs hidden 3D latent blocks "
            "(compare pretrained `facebook/vjepa2-vitl-fpc64-256` vs. your domain-adapted `checkpoints/vjepa2_bay`)."
        )

        with gr.Row():
            model_box = gr.Textbox(value=default_model, label="HF Model ID or Adapted Checkpoint Path", scale=4)
            device_dd = gr.Dropdown(choices=device_choices, value=default_device, label="Device", scale=1)

        upload_vid = gr.Video(label="Upload Clip (.mp4)", sources=["upload"], height=220)

        with gr.Row():
            fpc_slider = gr.Slider(8, 32, value=16, step=2, label="Frames per Clip")
            targets_slider = gr.Slider(1, 6, value=4, step=1, label="Num Target Blocks")
            spatial_slider = gr.Slider(0.20, 0.55, value=0.35, step=0.05, label="Max Block Spatial Scale")
            seed_num = gr.Number(value=42, precision=0, label="Mask Seed")
            run_btn = gr.Button("Run V-JEPA 2 Visualization", variant="primary", interactive=False)

        metrics_out = gr.Markdown("*Upload a video clip above to enable **Run V-JEPA 2 Visualization**.*")

        with gr.Row():
            vid_orig = gr.Video(label="1. Original ROI Clip (256×256)", autoplay=True, loop=True)
            vid_masked = gr.Video(label="2. Masked Context Input (seen by context_encoder)", autoplay=True, loop=True)

        with gr.Row():
            vid_pred = gr.Video(
                label="3. Latent Prediction: [Teacher 3D-PCA | Predictor Spliced 3D-PCA | Masked Cosine-Similarity Heatmap]",
                autoplay=True,
                loop=True,
            )

        def _toggle_run_btn(video_file: str | None):
            has_video = bool(video_file and Path(video_file).is_file())
            hint = (
                "*Click **Run V-JEPA 2 Visualization** to encode clip & predict masked latents.*"
                if has_video
                else "*Upload a video clip above to enable **Run V-JEPA 2 Visualization**.*"
            )
            return gr.update(interactive=has_video), hint

        upload_vid.change(_toggle_run_btn, inputs=[upload_vid], outputs=[run_btn, metrics_out])
        run_btn.click(
            run_vjepa2_visualization,
            inputs=[upload_vid, model_box, fpc_slider, targets_slider, spatial_slider, seed_num, device_dd],
            outputs=[vid_orig, vid_masked, vid_pred, metrics_out],
        )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local adapted checkpoint directory")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = build_app(default_model=args.model)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
