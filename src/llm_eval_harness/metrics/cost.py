"""Token-based cost computation."""

from __future__ import annotations


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate_per_mtok: float,
    output_rate_per_mtok: float,
    cached_input_tokens: int = 0,
    cached_input_rate_per_mtok: float = 0.0,
) -> float:
    """Compute estimated USD cost given token counts and per-million-token rates.

    Args:
        input_tokens: Total input tokens (including cached).
        output_tokens: Output tokens.
        input_rate_per_mtok: Cost per million non-cached input tokens.
        output_rate_per_mtok: Cost per million output tokens.
        cached_input_tokens: Tokens served from cache (billed at reduced rate).
        cached_input_rate_per_mtok: Cost per million cached input tokens.

    Returns:
        Estimated cost in USD.
    """
    regular = input_tokens - cached_input_tokens
    return (
        (regular / 1_000_000) * input_rate_per_mtok
        + (cached_input_tokens / 1_000_000) * cached_input_rate_per_mtok
        + (output_tokens / 1_000_000) * output_rate_per_mtok
    )
