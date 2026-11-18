"""Self-supervised and probe models built on V-JEPA 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.vjepa2_config import VJEPA2SSLConfig, VJEPA2SSLOutput
    from dataset.vjepa2_datamodule import UnlabeledClipDataModule
    from dataset.vjepa2_dataset import UnlabeledClipDataset
    from models.vjepa2_ssl import VJEPA2SSL
    from utils.masking import MultiBlockMaskGenerator3D

__all__ = [
    "MultiBlockMaskGenerator3D",
    "UnlabeledClipDataModule",
    "UnlabeledClipDataset",
    "VJEPA2SSL",
    "VJEPA2SSLConfig",
    "VJEPA2SSLOutput",
]


def __getattr__(name: str):
    if name in {"VJEPA2SSLConfig", "VJEPA2SSLOutput"}:
        from config import vjepa2_config

        return getattr(vjepa2_config, name)
    if name == "UnlabeledClipDataModule":
        from dataset.vjepa2_datamodule import UnlabeledClipDataModule

        return UnlabeledClipDataModule
    if name == "UnlabeledClipDataset":
        from dataset.vjepa2_dataset import UnlabeledClipDataset

        return UnlabeledClipDataset
    if name == "MultiBlockMaskGenerator3D":
        from utils.masking import MultiBlockMaskGenerator3D

        return MultiBlockMaskGenerator3D
    if name == "VJEPA2SSL":
        from models.vjepa2_ssl import VJEPA2SSL

        return VJEPA2SSL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
