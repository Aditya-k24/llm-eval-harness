"""Async experiment runner — fans out calls across examples × models."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ..adapters.base import ModelAdapter
from ..datasets.loaders import EvalExample
from ..prompts.renderer import render
from ..storage.jsonl_store import JSONLStore
from ..metrics.cost import compute_cost


async def run_experiment(
    examples: list[EvalExample],
    adapters: list[ModelAdapter],
    run_id: str | None = None,
    prompts_dir: str = "prompts",
    store: JSONLStore | None = None,
    concurrency: int = 5,
) -> list[dict]:
    """Run all (example, adapter) pairs with bounded concurrency.

    Returns a list of result dicts, one per (example, adapter) pair.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    sem = asyncio.Semaphore(concurrency)

    async def process_one(example: EvalExample, adapter: ModelAdapter) -> dict:
        async with sem:
            system, user = render(example.task, example, prompts_dir)
            result = await adapter.generate(system, user)
            cost = adapter.estimate_cost_usd(
                result.input_tokens,
                result.output_tokens,
                result.cached_input_tokens,
            )
            row: dict[str, Any] = {
                "run_id": run_id,
                "example_id": example.id,
                "task": example.task,
                "model_id": result.model_id,
                "provider": result.provider,
                "raw_text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cached_input_tokens": result.cached_input_tokens,
                "estimated_cost_usd": cost,
                "end_to_end_ms": result.timing.end_to_end_ms,
                "api_round_trip_ms": result.timing.api_round_trip_ms,
                "attempt_count": result.attempt_count,
                "error": result.error,
            }
            if store:
                store.write(row)
            return row

    tasks = [
        process_one(ex, adapter)
        for ex in examples
        for adapter in adapters
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)
