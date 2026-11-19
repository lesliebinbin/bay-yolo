"""Melt person boxes by repeatedly increasing each box's field of view.

This is an alternative to :mod:`utils.melt_by_iou`. Instead of clustering by
IoU on fixed-size expanded squares, it grows a box's field of view by a
user-controlled factor ``a`` and merges any box whose center falls inside that
expanded square. The two boxes are replaced by their tight envelope, the
envelope is appended to the worklist, and the process repeats until a full pass
produces no merge.

Box convention matches Ultralytics YOLO ``xywhn``: center-based
``[cx, cy, w, h]`` in normalized 0-1.0 coordinates.
"""

from __future__ import annotations

import math

import torch


def _expand_square(box: torch.Tensor, a: float) -> torch.Tensor:
    """Return the ``a * max(w, h)`` field-of-view square as ``[x1, y1, x2, y2]``."""
    cx, cy, w, h = box.unbind(-1)
    side = a * torch.maximum(w, h)
    half = side / 2
    return torch.stack([cx - half, cy - half, cx + half, cy + half])


def _center_inside(box: torch.Tensor, square: torch.Tensor) -> bool:
    """Return whether the center of ``box`` lies inside ``square`` (inclusive)."""
    cx, cy = box[0], box[1]
    return bool(
        (square[0] <= cx)
        & (cx <= square[2])
        & (square[1] <= cy)
        & (cy <= square[3])
    )


def _envelope(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """Return the tight ``[cx, cy, w, h]`` envelope of one or more boxes."""
    cx, cy, w, h = boxes_xywh.unbind(-1)
    x1 = (cx - w / 2).min()
    y1 = (cy - h / 2).min()
    x2 = (cx + w / 2).max()
    y2 = (cy + h / 2).max()
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def melt_by_fov(
    boxes: torch.Tensor | list[list[float]],
    a: float = 1.5,
    return_clusters: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
    """Melt boxes by growing each box's field of view by ``a``.

    Algorithm
    ---------
    1. Scan the worklist from the beginning.
    2. For the first box with a match, expand it to a square of side
       ``a * max(w, h)`` centered on the box.
    3. If another box's center lies inside that square, remove both boxes,
       replace them with their tight envelope, and append the envelope to the
       end of the worklist.
    4. Restart the scan from the beginning so the new envelope can absorb more
       boxes.
    5. Stop when a full scan finds no merge.

    Parameters
    ----------
    boxes:
        ``(N, 4)`` tensor (or nested list) of ``[cx, cy, w, h]`` boxes in
        normalized 0-1.0 coordinates (Ultralytics ``xywhn``).
    a:
        Field-of-view expansion factor. Must be ``>= 1.0``. ``a=1.0`` uses a
        square of side ``max(w, h)``; larger values grow the influence zone.
    return_clusters:
        If ``True``, also return a list mapping each output box to the original
        input indices it contains.

    Returns
    -------
    torch.Tensor
        ``(M, 4)`` melted boxes in ``[cx, cy, w, h]`` format.
    tuple[torch.Tensor, list[list[int]]]
        When ``return_clusters=True``: the melted boxes plus the provenance
        mapping.
    """
    if not math.isfinite(float(a)) or float(a) < 1.0:
        raise ValueError(f"a must be a finite value >= 1.0, got {a!r}")

    boxes = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu()
    if boxes.numel() == 0:
        return (boxes, []) if return_clusters else boxes

    work: list[tuple[torch.Tensor, list[int]]] = [
        (boxes[i].clone(), [i]) for i in range(boxes.size(0))
    ]

    while True:
        merged_any = False
        i = 0

        while i < len(work):
            box, indices = work[i]
            square = _expand_square(box, float(a))

            match = None
            for j, (other, _) in enumerate(work):
                if j != i and _center_inside(other, square):
                    match = j
                    break

            if match is None:
                i += 1
                continue

            other, other_indices = work[match]
            merged = _envelope(torch.stack([box, other]))
            work = [
                entry for k, entry in enumerate(work) if k != i and k != match
            ]
            work.append((merged, indices + other_indices))
            merged_any = True
            break

        if not merged_any:
            break

    melted = torch.stack([box for box, _ in work])
    clusters = [sorted(indices) for _, indices in work]
    if return_clusters:
        return melted, clusters
    return melted


if __name__ == "__main__":
    test_boxes = torch.tensor(
        [
            [0.30, 0.50, 0.10, 0.20],
            [0.35, 0.50, 0.10, 0.20],
            [0.80, 0.50, 0.10, 0.20],
        ]
    )
    melted, clusters = melt_by_fov(test_boxes, a=1.5, return_clusters=True)
    print("Original:\n", test_boxes)
    print("\nMelted (a=1.5):\n", melted)
    print("Clusters (original indices):\n", clusters)

    melted_none, clusters_none = melt_by_fov(test_boxes, a=1.0, return_clusters=True)
    print("\nMelted (a=1.0):\n", melted_none)
    print("Clusters (original indices):\n", clusters_none)
