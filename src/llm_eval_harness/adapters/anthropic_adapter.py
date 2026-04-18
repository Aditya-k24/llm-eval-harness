"""Anthropic Claude adapter with async support and tenacity retry."""

import os
import time

from anthropic import AsyncAnthropic, APIStatusError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from .base import ModelResult, TimingInfo

RETRY_STATUS = {429, 500, 502, 503, 504}


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRY_STATUS
    return isinstance(exc, APIConnectionError)


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(
        self,
        model_id: str,
        max_output_tokens: int = 256,
        temperature: float = 0,
        input_cost_per_mtok: float = 3.0,
        output_cost_per_mtok: float = 15.0,
        max_attempts: int = 5,
    ):
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.input_cost_per_mtok = input_cost_per_mtok
        self.output_cost_per_mtok = output_cost_per_mtok
        self.max_attempts = max_attempts
        self._client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def generate(self, system: str, user: str) -> ModelResult:
        timing = TimingInfo(t0=time.perf_counter())
        attempt_count = 0

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(min=1, max=60),
            retry=retry_if_exception(_should_retry),
            reraise=True,
        )
        async def _call():
            nonlocal attempt_count
            attempt_count += 1
            timing.t1 = time.perf_counter()
            resp = await self._client.messages.create(
                model=self.model_id,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            timing.t2 = time.perf_counter()
            return resp

        try:
            resp = await _call()
            timing.t3 = time.perf_counter()
            text = "".join(
                block.text for block in resp.content if block.type == "text"
            )
            usage = resp.usage
            return ModelResult(
                model_id=self.model_id,
                provider=self.provider,
                text=text,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                timing=timing,
                attempt_count=attempt_count,
                raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
            )
        except Exception as exc:
            timing.t3 = time.perf_counter()
            return ModelResult(
                model_id=self.model_id,
                provider=self.provider,
                text="",
                input_tokens=0,
                output_tokens=0,
                timing=timing,
                attempt_count=attempt_count,
                error=str(exc),
            )

    def estimate_cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        return (input_tokens / 1_000_000) * self.input_cost_per_mtok + (
            output_tokens / 1_000_000
        ) * self.output_cost_per_mtok
