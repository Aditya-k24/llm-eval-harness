"""Base adapter interface and shared data classes."""

from dataclasses import dataclass, field
from typing import Protocol, Any, runtime_checkable
import time


@dataclass
class TimingInfo:
    t0: float = 0.0  # before prompt assembly
    t1: float = 0.0  # after request sent
    t2: float = 0.0  # first token received
    t3: float = 0.0  # final response + parse complete

    @property
    def end_to_end_ms(self) -> float:
        return (self.t3 - self.t0) * 1000

    @property
    def api_round_trip_ms(self) -> float:
        return (self.t3 - self.t1) * 1000


@dataclass
class ModelResult:
    model_id: str
    provider: str
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    timing: TimingInfo = field(default_factory=TimingInfo)
    attempt_count: int = 1
    raw: Any = None
    error: str | None = None


@runtime_checkable
class ModelAdapter(Protocol):
    model_id: str
    provider: str

    async def generate(self, system: str, user: str) -> ModelResult: ...

    def estimate_cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float: ...
