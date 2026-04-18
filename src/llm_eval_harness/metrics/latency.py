"""Latency statistics computed over a list of millisecond values."""

from __future__ import annotations

import statistics


def compute_latency_stats(values_ms: list[float]) -> dict:
    """Return p50, p95, mean, min, max over a list of latency values in ms."""
    if not values_ms:
        return {}
    sorted_vals = sorted(values_ms)
    n = len(sorted_vals)
    return {
        "p50_ms": sorted_vals[n // 2],
        "p95_ms": sorted_vals[int(n * 0.95)],
        "mean_ms": statistics.mean(sorted_vals),
        "min_ms": sorted_vals[0],
        "max_ms": sorted_vals[-1],
    }
