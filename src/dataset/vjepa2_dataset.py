"""Dataset for unlabeled video clips for V-JEPA 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.media import VIDEO_EXTENSIONS, read_video_clip_rgb


class UnlabeledClipDataset(Dataset):
    """Unlabeled video clip dataset scanning any folder recursively (no class subfolders needed)."""

    def __init__(self, root_dir: str | Path, frames_per_clip: int = 16) -> None:
        self.root_dir = Path(root_dir)
        self.frames_per_clip = frames_per_clip
        self.paths = sorted(
            p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"no video clips found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return read_video_clip_rgb(self.paths[index], num_frames=self.frames_per_clip)
