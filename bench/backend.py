"""Backend abstraction for the streaming benchmark client.

Adapted from vLLM's ``benchmarks/backend_request_func.py``
(https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py).
The ``RequestFuncInput`` / ``RequestFuncOutput`` / ``ASYNC_REQUEST_FUNCS``
pattern decouples engine-specific HTTP handling from orchestration logic.
Both vLLM and Friendli expose OpenAI-compatible streaming chat completion
endpoints, so they share a single backend function (``openai-chat``);
the registry exists so additional engines can be added without coupling
the runner to any particular HTTP shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class RequestFuncInput:
    prompt: str
    prompt_id: int
    prompt_tokens: int
    api_url: str
    model: str
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    ignore_eos: bool = True


@dataclass
class RequestFuncOutput:
    engine: str
    prompt_id: int
    prompt_tokens: int
    output_tokens: int = 0
    arrival_time: float = 0.0
    send_time: float = 0.0
    first_token_time: float | None = None
    last_token_time: float | None = None
    itl: list[float] = field(default_factory=list)
    success: bool = False
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.send_time

    @property
    def tpot(self) -> float | None:
        # GenAI-Perf convention: ITL/TPOT excludes the first token.
        # tpot = (last_token_time - first_token_time) / (output_tokens - 1)
        # Source: https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
        if (
            self.first_token_time is None
            or self.last_token_time is None
            or self.output_tokens <= 1
        ):
            return None
        return (self.last_token_time - self.first_token_time) / (self.output_tokens - 1)

    @property
    def e2e_latency(self) -> float | None:
        if self.last_token_time is None:
            return None
        return self.last_token_time - self.send_time

    @property
    def queue_delay(self) -> float:
        return self.send_time - self.arrival_time


BackendFunc = Callable[..., Awaitable[RequestFuncOutput]]

ASYNC_REQUEST_FUNCS: dict[str, BackendFunc] = {}


def register_backend(name: str) -> Callable[[BackendFunc], BackendFunc]:
    """Register an async backend function under ``name`` in ``ASYNC_REQUEST_FUNCS``."""

    def decorator(fn: BackendFunc) -> BackendFunc:
        ASYNC_REQUEST_FUNCS[name] = fn
        return fn

    return decorator
