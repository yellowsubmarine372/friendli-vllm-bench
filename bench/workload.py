"""Prompt sampling and open-loop Poisson dispatcher.

Adapted from vLLM ``benchmark_serving.py``'s ``get_request()`` async
generator (https://github.com/vllm-project/vllm/blob/main/benchmarks/).
Open-loop dispatch is the critical correctness property: new requests
arrive on the Poisson schedule regardless of in-flight count, so server
saturation is observable rather than masked by client-side gating.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from bench.backend import BackendFunc, RequestFuncInput, RequestFuncOutput

PromptTuple = tuple[int, str, int]  # (id, text, approx_tokens)


def generate_arrivals(rate: float, duration: float, seed: int) -> list[float]:
    """Return sorted arrival times for a Poisson process with mean ``rate`` rps.

    Inter-arrival times are exponential with mean ``1/rate``. Adapted from
    vLLM ``benchmark_serving.py``'s ``get_request()``.
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    rng = np.random.default_rng(seed)
    arrivals: list[float] = []
    t = 0.0
    while t < duration:
        t += float(rng.exponential(1.0 / rate))
        if t < duration:
            arrivals.append(t)
    return arrivals


def load_prompts(path: Path) -> list[PromptTuple]:
    """Load ``prompts.jsonl`` into a list of ``(id, text, approx_tokens)`` tuples.

    Order in the file is preserved — both engines consume the same sequence
    so vLLM and Friendli (or modeled Friendli) operate on identical inputs.
    """
    items: list[PromptTuple] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(
                (int(obj["id"]), str(obj["text"]), int(obj["approx_tokens"]))
            )
    return items


def cycle_prompts(prompts: list[PromptTuple]) -> Iterator[PromptTuple]:
    """Infinite cycle of prompts in their loaded order."""
    return itertools.cycle(prompts)


async def run_open_loop(
    backend_fn: BackendFunc,
    prompts: Iterator[PromptTuple],
    rate: float,
    duration: float,
    seed: int,
    base_input_template: dict[str, Any],
    engine_name: str,
) -> list[RequestFuncOutput]:
    """Open-loop Poisson dispatcher.

    Schedule arrivals once via ``generate_arrivals``; for each arrival,
    sleep until its scheduled time relative to ``start = perf_counter()``
    then call ``backend_fn(req, arrival_time=..., engine=...)`` and append
    the resulting task. The function awaits all tasks via ``asyncio.gather``
    only after the dispatch loop finishes.

    Critical (SPEC §6.3): dispatch does NOT gate on completions. Tasks pile
    up if the server saturates — that is the signal we want to measure.

    ``base_input_template`` supplies the ``RequestFuncInput`` fields not
    derived from the prompt iterator (``api_url``, ``model``, ``max_tokens``,
    ``temperature``, ``top_p``, ``seed``, ``ignore_eos``).
    """
    arrivals = generate_arrivals(rate, duration, seed)
    start = time.perf_counter()
    tasks: list[asyncio.Task[RequestFuncOutput]] = []
    for arrival_t in arrivals:
        now = time.perf_counter() - start
        if arrival_t > now:
            await asyncio.sleep(arrival_t - now)
        prompt_id, prompt_text, approx_tokens = next(prompts)
        req = RequestFuncInput(
            prompt=prompt_text,
            prompt_id=prompt_id,
            prompt_tokens=approx_tokens,
            **base_input_template,
        )
        tasks.append(
            asyncio.create_task(
                backend_fn(req, arrival_time=start + arrival_t, engine=engine_name)
            )
        )
    return await asyncio.gather(*tasks, return_exceptions=False)
