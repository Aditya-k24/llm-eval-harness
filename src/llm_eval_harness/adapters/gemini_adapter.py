"""Google Gemini adapter with async support via run_in_executor."""

import os
import time
import asyncio

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from .base import ModelResult, TimingInfo

RETRY_STATUS = {429, 500, 502, 503, 504}


def _should_retry(exc: BaseException) -> bool:
    exc_str = str(type(exc).__name__).lower()
    exc_msg = str(exc).lower()
    # Gemini SDK may raise various errors; catch rate-limit and server errors
    if "429" in exc_msg or "rate" in exc_msg:
        return True
    if any(str(code) in exc_msg for code in [500, 502, 503, 504]):
        return True
    if "resourceexhausted" in exc_str or "serviceunavailable" in exc_str:
        return True
    return False


class GeminiAdapter:
    provider = "gemini"

    def __init__(
        self,
        model_id: str,
        max_output_tokens: int = 256,
        temperature: float = 0,
        input_cost_per_mtok: float = 1.25,
        output_cost_per_mtok: float = 10.0,
        max_attempts: int = 5,
    ):
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.input_cost_per_mtok = input_cost_per_mtok
        self.output_cost_per_mtok = output_cost_per_mtok
        self.max_attempts = max_attempts
        # Lazy import to avoid requiring google-genai at import time
        from google import genai

        self._client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self._genai = genai

    async def generate(self, system: str, user: str) -> ModelResult:
        timing = TimingInfo(t0=time.perf_counter())
        attempt_count = 0
        genai = self._genai

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
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model_id,
                    contents=f"{system}\n\n{user}",
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                    ),
                ),
            )
            timing.t2 = time.perf_counter()
            return resp

        try:
            resp = await _call()
            timing.t3 = time.perf_counter()
            text = resp.text or ""
            usage = resp.usage_metadata
            return ModelResult(
                model_id=self.model_id,
                provider=self.provider,
                text=text,
                input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                timing=timing,
                attempt_count=attempt_count,
                raw=None,
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
