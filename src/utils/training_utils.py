"""Training utilities for domain adaptation and SSL."""

from __future__ import annotations

import math


def cosine_ema_schedule(step: int, total_steps: int, m_start: float = 0.996, m_end: float = 0.9995) -> float:
    """Cosine schedule for EMA momentum."""
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return m_end - (m_end - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))
