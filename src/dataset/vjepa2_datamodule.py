"""LightningDataModule for V-JEPA 2 training."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset.vjepa2_dataset import UnlabeledClipDataset


class UnlabeledClipDataModule(L.LightningDataModule):
    """LightningDataModule wrapping `UnlabeledClipDataset` and an HF `AutoVideoProcessor`."""

    def __init__(
        self,
        data_dir: str | Path,
        processor,
        frames_per_clip: int = 16,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.frames_per_clip = frames_per_clip
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset: UnlabeledClipDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            self.dataset = UnlabeledClipDataset(self.data_dir, frames_per_clip=self.frames_per_clip)

    def _collate(self, batch_np: list) -> torch.Tensor:
        # The processor is usually an AutoVideoProcessor from transformers
        return self.processor(batch_np, return_tensors="pt")["pixel_values_videos"]

    def train_dataloader(self) -> DataLoader:
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate,
        )
