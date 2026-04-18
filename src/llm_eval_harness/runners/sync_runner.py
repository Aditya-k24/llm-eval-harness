"""Synchronous wrapper around the async runner."""

from __future__ import annotations

import asyncio
from typing import Any

from ..adapters.base import ModelAdapter
from ..datasets.loaders import EvalExample
from ..storage.jsonl_store import JSONLStore
from .async_runner import run_experiment


def run_experiment_sync(
    examples: list[EvalExample],
    adapters: list[ModelAdapter],
    run_id: str | None = None,
    prompts_dir: str = "prompts",
    store: JSONLStore | None = None,
    concurrency: int = 5,
) -> list[dict]:
    """Blocking wrapper that drives the async runner via asyncio.run().

    Useful for scripts and notebook cells that don't have a running event loop.
    """
    return asyncio.run(
        run_experiment(
            examples=examples,
            adapters=adapters,
            run_id=run_id,
            prompts_dir=prompts_dir,
            store=store,
            concurrency=concurrency,
        )
    )
