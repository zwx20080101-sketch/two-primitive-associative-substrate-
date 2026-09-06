"""Read-side observation tools.  Pure functions: they never touch the network.

Attention is NOT part of Synapse-Net.  The bottom layer always returns the
full activation map; this module decides which nodes enter the current
cognitive content (focus at high threshold, daydream / dream at low one).
"""

from __future__ import annotations


def attention_filter(
    activations: dict[str, float],
    threshold: float | None = None,
    top_k: int | None = None,
) -> dict[str, float]:
    """Filter an activation map by absolute threshold and/or top-k rank.

    threshold: keep nodes with activation >= threshold (scale 0..1).
    top_k:     keep the k strongest nodes (ties broken by label order).
    Both may be combined: filter first, then rank.
    """
    if threshold is not None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        activations = {k: v for k, v in activations.items() if v >= threshold}
    if top_k is not None:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        ranked = sorted(activations.items(), key=lambda kv: (-kv[1], kv[0]))
        activations = dict(ranked[:top_k])
    return activations
