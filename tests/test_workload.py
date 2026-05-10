"""Tests for Poisson generator, prompt iteration, and open-loop dispatch (§6.3)."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pytest

from bench.backend import RequestFuncOutput
from bench.workload import (
    cycle_prompts,
    generate_arrivals,
    load_prompts,
    run_open_loop,
)

# --- generate_arrivals ---


def test_poisson_count_within_2pct():
    arrivals = generate_arrivals(rate=10.0, duration=600.0, seed=42)
    expected = 6000
    assert abs(len(arrivals) - expected) / expected < 0.02


def test_poisson_mean_inter_arrival_within_2pct():
    arrivals = generate_arrivals(rate=10.0, duration=600.0, seed=42)
    inter = np.diff([0.0, *arrivals])
    mean = inter.mean()
    assert abs(mean - 0.1) / 0.1 < 0.02


def test_arrivals_sorted():
    arrivals = generate_arrivals(rate=5.0, duration=100.0, seed=42)
    assert arrivals == sorted(arrivals)


def test_arrivals_strictly_within_duration():
    duration = 50.0
    arrivals = generate_arrivals(rate=10.0, duration=duration, seed=42)
    assert all(0.0 < t < duration for t in arrivals)


def test_determinism_same_seed():
    a1 = generate_arrivals(rate=5.0, duration=100.0, seed=42)
    a2 = generate_arrivals(rate=5.0, duration=100.0, seed=42)
    assert a1 == a2


def test_different_seeds_yield_different_arrivals():
    a1 = generate_arrivals(rate=5.0, duration=100.0, seed=42)
    a2 = generate_arrivals(rate=5.0, duration=100.0, seed=43)
    assert a1 != a2


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        generate_arrivals(rate=0.0, duration=10.0, seed=0)
    with pytest.raises(ValueError):
        generate_arrivals(rate=-1.0, duration=10.0, seed=0)


def test_invalid_duration_raises():
    with pytest.raises(ValueError):
        generate_arrivals(rate=1.0, duration=0.0, seed=0)


# --- load_prompts / cycle_prompts ---


def test_load_prompts(tmp_path: Path):
    p = tmp_path / "p.jsonl"
    p.write_text(
        '{"id": 0, "text": "hi", "approx_tokens": 2}\n'
        '\n'
        '{"id": 1, "text": "bye", "approx_tokens": 3}\n',
        encoding="utf-8",
    )
    items = load_prompts(p)
    assert items == [(0, "hi", 2), (1, "bye", 3)]


def test_load_prompts_real_file_has_200_in_order():
    items = load_prompts(Path("prompts/prompts.jsonl"))
    assert len(items) == 200
    assert [it[0] for it in items] == list(range(200))


def test_cycle_prompts_repeats_in_order():
    items = [(0, "a", 1), (1, "b", 2)]
    it = cycle_prompts(items)
    seen = [next(it) for _ in range(5)]
    assert seen == [(0, "a", 1), (1, "b", 2), (0, "a", 1), (1, "b", 2), (0, "a", 1)]


# --- run_open_loop ---


def _base_template() -> dict:
    return {
        "api_url": "http://x",
        "model": "test-model",
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "ignore_eos": True,
    }


def _make_fast_backend():
    async def fn(req, *, arrival_time: float, engine: str) -> RequestFuncOutput:
        return RequestFuncOutput(
            engine=engine,
            prompt_id=req.prompt_id,
            prompt_tokens=req.prompt_tokens,
            output_tokens=4,
            arrival_time=arrival_time,
            send_time=arrival_time,
            first_token_time=arrival_time + 0.01,
            last_token_time=arrival_time + 0.04,
            itl=[0.01, 0.01, 0.01],
            success=True,
        )
    return fn


def test_run_open_loop_returns_one_output_per_arrival():
    items = [(0, "a", 1), (1, "b", 2)]
    expected = generate_arrivals(rate=20.0, duration=0.3, seed=42)

    async def go():
        return await run_open_loop(
            backend_fn=_make_fast_backend(),
            prompts=cycle_prompts(items),
            rate=20.0,
            duration=0.3,
            seed=42,
            base_input_template=_base_template(),
            engine_name="vllm",
        )

    results = asyncio.run(go())
    assert len(results) == len(expected)
    assert all(r.success for r in results)
    assert all(r.engine == "vllm" for r in results)


def test_run_open_loop_dispatch_does_not_gate_on_completions():
    """SPEC §6.3 critical: dispatch must NOT wait for in-flight tasks to finish."""
    backend_delay = 0.4
    # rate=20, duration=0.3, seed=7 deterministically yields 5 arrivals.
    rate, duration, seed = 20.0, 0.3, 7

    async def slow_backend(req, *, arrival_time: float, engine: str) -> RequestFuncOutput:
        await asyncio.sleep(backend_delay)
        return RequestFuncOutput(
            engine=engine,
            prompt_id=req.prompt_id,
            prompt_tokens=req.prompt_tokens,
            output_tokens=1,
            arrival_time=arrival_time,
            send_time=arrival_time,
            success=True,
        )

    async def go():
        return await run_open_loop(
            backend_fn=slow_backend,
            prompts=cycle_prompts([(0, "a", 1)]),
            rate=rate,
            duration=duration,
            seed=seed,
            base_input_template=_base_template(),
            engine_name="t",
        )

    start = time.perf_counter()
    results = asyncio.run(go())
    elapsed = time.perf_counter() - start

    n = len(results)
    assert n >= 3, f"setup precondition: expected multiple arrivals, got {n}"
    # Open-loop expectation: elapsed ≈ duration + backend_delay (one trailing wait).
    # Gated expectation:    elapsed ≈ N * backend_delay, far larger for N ≥ 3.
    assert elapsed < duration + backend_delay + 0.3, (
        f"elapsed={elapsed:.3f}s with N={n} arrivals suggests gated dispatch"
    )


def test_run_open_loop_propagates_prompt_metadata_to_input():
    captured_prompt_ids: list[int] = []

    async def recorder(req, *, arrival_time: float, engine: str) -> RequestFuncOutput:
        captured_prompt_ids.append(req.prompt_id)
        # Propagate input fields onto output for assertion.
        return RequestFuncOutput(
            engine=engine,
            prompt_id=req.prompt_id,
            prompt_tokens=req.prompt_tokens,
            output_tokens=1,
            arrival_time=arrival_time,
            send_time=arrival_time,
            success=True,
        )

    items = [(7, "seven", 11), (3, "three", 9)]

    async def go():
        return await run_open_loop(
            backend_fn=recorder,
            prompts=cycle_prompts(items),
            rate=20.0,
            duration=0.2,
            seed=1,
            base_input_template=_base_template(),
            engine_name="t",
        )

    results = asyncio.run(go())
    # First two prompts should be 7, 3; subsequent reuse the cycle.
    assert captured_prompt_ids[:2] == [7, 3]
    assert all(r.prompt_id in {7, 3} for r in results)
    # Output prompt_tokens should match the input metadata
    by_id = {7: 11, 3: 9}
    for r in results:
        assert r.prompt_tokens == by_id[r.prompt_id]
