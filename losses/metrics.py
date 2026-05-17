"""Metrics used by evaluation and search."""

from __future__ import annotations

import numpy as np


def top_percent_mean_energy(y_pred: np.ndarray, mask: np.ndarray, top_percent: float = 0.05) -> float:
    e_sq = y_pred[0] ** 2 + y_pred[1] ** 2 + y_pred[2] ** 2 + y_pred[3] ** 2
    valid = mask[0] > 0
    vals = e_sq[valid]
    if vals.size == 0:
        return 0.0
    k = max(1, int(vals.size * top_percent))
    topk = np.partition(vals.ravel(), -k)[-k:]
    return float(np.mean(topk))
