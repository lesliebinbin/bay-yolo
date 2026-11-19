"""Melt overlapping person boxes into cluster envelopes.

This module is intentionally standalone and has no dependency on
``field_of_view.py``: the FoV crop logic and the clustering logic answer
different questions and are consumed independently by downstream agents/models.

Given a list/tensor of YOLO-style boxes ``[cx, cy, w, h]``, we:

1. Expand each box into a square "influence zone" with side
   ``m = max(w, h)``, centered on the box. This gives a proximity buffer so
   that two people standing side-by-side (tight boxes, low IoU) still merge
   when their bodies/arms would overlap.
2. Build a graph where an edge exists if the *expanded* squares have IoU
   above ``threshold``, then find connected components (DSU).
3. Replace each component with the enveloping ``[cx, cy, w, h]`` of the
   *original* boxes (not the expanded squares) so the merged box stays tight.
4. Repeat until stable (``max_iterations == -1``) or until the requested
   number of passes.

Box convention aligns with Ultralytics YOLO: center-based ``[cx, cy, w, h]``
in **normalized 0-1.0 coordinates** (i.e. ``results.boxes.xywhn``). The
operations are scale-invariant, so pixel-space boxes work identically, but
the normalized convention is the intended interface.
"""

from __future__ import annotations

import torch
from einops import rearrange


def _expand_to_squares(boxes: torch.Tensor) -> torch.Tensor:
    """Expand ``[cx, cy, w, h]`` boxes into ``[x1, y1, x2, y2]`` squares.

    The square side is ``max(w, h)`` and it is centered on the original box.
    """
    cx, cy, w, h = boxes.unbind(-1)
    side = torch.maximum(w, h)
    half = side / 2
    return torch.stack(
        [cx - half, cy - half, cx + half, cy + half],
        dim=-1,
    )


def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, N)`` IoU matrix for ``[x1, y1, x2, y2]`` boxes."""
    b1 = rearrange(boxes, 'n c -> n 1 c')
    b2 = rearrange(boxes, 'n c -> 1 n c')

    inter_min = torch.maximum(b1[..., :2], b2[..., :2])
    inter_max = torch.minimum(b1[..., 2:], b2[..., 2:])
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter_area
    return inter_area / union.clamp(min=1e-7)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    half_w, half_h = w / 2, h / 2
    return torch.stack(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
        dim=-1,
    )


def _connected_components(adj: torch.Tensor) -> list[list[int]]:
    """Group node indices by connected components of a boolean adjacency matrix."""
    n = adj.size(0)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_i] = root_j

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _envelope(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """Return the tight ``[cx, cy, w, h]`` envelope of a set of boxes."""
    xyxy = _xywh_to_xyxy(boxes_xywh)
    x1, y1 = xyxy[..., 0].min(), xyxy[..., 1].min()
    x2, y2 = xyxy[..., 2].max(), xyxy[..., 3].max()
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
    )


def melt_boxes(
    boxes: torch.Tensor | list[list[float]],
    threshold: float = 0.4,
    max_iterations: int = -1,
    return_clusters: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[list[int]]]:
    """Melt overlapping boxes based on expanded square influence zones.

    Parameters
    ----------
    boxes:
        ``(N, 4)`` tensor (or nested list) of ``[cx, cy, w, h]`` boxes in
        normalized 0-1.0 coordinates (Ultralytics ``xywhn`` convention).
        Pixel-space boxes are also accepted because all geometry is
        scale-invariant.
    threshold:
        IoU threshold on the *expanded* squares required to merge two boxes.
    max_iterations:
        Maximum number of melting passes. ``-1`` means melt until stable
        (no merges possible) or until one box remains.
    return_clusters:
        If ``True``, also return the cluster mapping: a list where each entry
        contains the indices of the original ``boxes`` that were merged into
        the corresponding output box. This is useful for handing the agent a
        "cluster information view".

    Returns
    -------
    torch.Tensor
        ``(M, 4)`` melted boxes in ``[cx, cy, w, h]`` format.
    tuple[torch.Tensor, list[list[int]]]
        When ``return_clusters=True``: the melted boxes plus the mapping from
        each melted box to the original input indices it contains.
    """
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.numel() == 0:
        return (boxes, []) if return_clusters else boxes

    current_boxes = boxes.clone()
    # Provenance: original index(es) represented by each current box.
    current_indices: list[list[int]] = [[i] for i in range(current_boxes.size(0))]

    iteration = 0
    while max_iterations == -1 or iteration < max_iterations:
        if current_boxes.size(0) <= 1:
            break

        squares = _expand_to_squares(current_boxes)
        iou = _pairwise_iou(squares)
        # Keep the graph on CPU: the DSU loop indexes it one entry at a time,
        # so leaving it on GPU would force a sync per lookup.
        adj = (iou > threshold).cpu()

        clusters = _connected_components(adj)
        if len(clusters) == current_boxes.size(0):
            break  # Nothing merged this pass; stable.

        new_boxes = []
        new_indices = []
        for cluster in clusters:
            if len(cluster) == 1:
                idx = cluster[0]
                new_boxes.append(current_boxes[idx])
                new_indices.append(current_indices[idx])
            else:
                new_boxes.append(_envelope(current_boxes[cluster]))
                merged = [i for idx in cluster for i in current_indices[idx]]
                new_indices.append(merged)

        current_boxes = torch.stack(new_boxes)
        current_indices = new_indices
        iteration += 1

    if return_clusters:
        return current_boxes, current_indices
    return current_boxes


if __name__ == "__main__":
    test_boxes = torch.tensor(
        [
            [10.0, 10.0, 2.0, 2.0],
            [11.5, 10.0, 2.0, 2.0],
            [50.0, 50.0, 1.0, 1.0],
        ]
    )

    melted, clusters = melt_boxes(
        test_boxes, threshold=0.01, max_iterations=-1, return_clusters=True
    )
    print("Original:\n", test_boxes)
    print("\nMelted (threshold=0.01):\n", melted)
    print("Clusters (original indices):\n", clusters)

    melted_none = melt_boxes(test_boxes, threshold=0.9, max_iterations=-1)
    print("\nMelted (threshold=0.9):\n", melted_none)
