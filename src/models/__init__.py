"""Self-supervised and probe models built on V-JEPA 2."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.vjepa2_ssl import (
        MultiBlockMaskGenerator3D,
        UnlabeledClipDataModule,
        UnlabeledClipDataset,
        VJEPA2SSL,
        VJEPA2SSLConfig,
        VJEPA2SSLOutput,
    )

__all__ = [
    "MultiBlockMaskGenerator3D",
    "UnlabeledClipDataModule",
    "UnlabeledClipDataset",
    "VJEPA2SSL",
    "VJEPA2SSLConfig",
    "VJEPA2SSLOutput",
]


def __getattr__(name: str):
    if name in __all__:
        from models import vjepa2_ssl

        return getattr(vjepa2_ssl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
