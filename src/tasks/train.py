"""Task for training V-JEPA 2 self-supervised domain adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from transformers import AutoVideoProcessor

from config.vjepa2_config import VJEPA2SSLConfig
from dataset.vjepa2_datamodule import UnlabeledClipDataModule
from models.vjepa2_ssl import VJEPA2SSL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 Domain Adaptation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing unlabeled ROI .mp4 clips")
    parser.add_argument("--output-dir", default="checkpoints/vjepa2_ssl", help="Where to save adapted HF checkpoint")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="HF model ID or local path")
    parser.add_argument("--frames-per-clip", type=int, default=16, help="Sampled frames per clip (default: 16)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5, help="Gentle LR for domain adaptation (default: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--devices", default="auto", help="Devices per node, e.g. 'auto', '1', '2'")
    parser.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
        help="Lightning training precision (default: bf16-mixed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    
    # 1. Load Processor
    processor = AutoVideoProcessor.from_pretrained(args.model)
    
    # 2. Setup Config
    ssl_cfg = VJEPA2SSLConfig(
        model_name_or_path=args.model,
        frames_per_clip=args.frames_per_clip,
        crop_size=processor.crop_size["height"],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 3. Initialize Model
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
    devices = int(args.devices) if str(args.devices).isdigit() else args.devices
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=devices,
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=1.0,
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
