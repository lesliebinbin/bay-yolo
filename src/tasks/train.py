"""Task for training V-JEPA 2 self-supervised domain adaptation."""

from __future__ import annotations

import argparse

import lightning as L
import torch
from transformers import AutoVideoProcessor

from config.vjepa2_config import VJEPA2SSLConfig
from dataset.vjepa2_datamodule import UnlabeledClipDataModule
from models.vjepa2_ssl import VJEPA2SSL


def _parse_devices(raw: str) -> int | str | list[int]:
    val = str(raw).strip()
    if val.isdigit():
        return int(val)
    if "," in val:
        parts = [p.strip() for p in val.split(",") if p.strip()]
        if parts and all(p.isdigit() for p in parts):
            return [int(p) for p in parts]
    return val


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 Domain Adaptation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing unlabeled ROI .mp4 clips")
    parser.add_argument("--output-dir", default="checkpoints/vjepa2_ssl", help="Where to save adapted HF checkpoint")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="HF model ID or local path")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Sampled frames per clip (default: 16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4 with LoRA)")
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank on context_encoder attention Q/K/V/proj (0 = full encoder fine-tuning; default: 16)",
    )
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling alpha (default: 32)")
    parser.add_argument(
        "--ema-range",
        type=float,
        nargs=2,
        default=[0.996, 0.9995],
        metavar=("M_START", "M_END"),
        help="Cosine EMA momentum range (default: 0.996 0.9995; use e.g. 0.98 0.996 for <500 steps)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable activation checkpointing on context_encoder & predictor (default: enabled)",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        dest="gradient_checkpointing",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--devices",
        default="auto",
        help="Devices count (e.g. '1', '2', 'auto') or comma list (e.g. '0,' or '0,1')",
    )
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Lightning training precision (default: bf16-mixed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_float32_matmul_precision("high")

    # 1. Load Processor
    processor = AutoVideoProcessor.from_pretrained(args.model)

    # 2. Setup Config
    ssl_cfg = VJEPA2SSLConfig(
        model_name_or_path=args.model,
        frames_per_clip=args.frames_per_clip,
        crop_size=processor.crop_size["height"],
        ema_momentum_range=(float(args.ema_range[0]), float(args.ema_range[1])),
        lr=args.lr,
        weight_decay=args.weight_decay,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # 3. Initialize Model (keep FP32 master weights so EMA & AdamW never underflow; Lightning autocasts forward)
    model = VJEPA2SSL.from_pretrained(args.model, ssl_config=ssl_cfg)

    # 4. Initialize DataModule
    datamodule = UnlabeledClipDataModule(
        data_dir=args.data_dir,
        processor=processor,
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 5. Setup Trainer
    devices = _parse_devices(args.devices)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=devices,
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        default_root_dir=args.output_dir,
    )

    # 6. Run Training
    trainer.fit(model, datamodule=datamodule)

    # 7. Save Final Checkpoint
    if trainer.is_global_zero:
        model.save_backbone_pretrained(args.output_dir, use_target_encoder=True)
        processor.save_pretrained(args.output_dir)
        print(f"saved adapted V-JEPA 2 checkpoint -> {args.output_dir}")


if __name__ == "__main__":
    main()
