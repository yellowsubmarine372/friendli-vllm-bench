"""Tests for the Friendli-line synthesis module (§6.5)."""
from __future__ import annotations

import math

import pytest

from bench.backend import RequestFuncOutput
from bench.synthetic import (
    synthesize_friendli_results,
    tpot_multiplier,
    ttft_multiplier,
)

MAX_RATE = 32.0


def _vllm_result(
    *,
    ttft_s: float = 1.0,
    tpot_s: float = 0.05,
    output_tokens: int = 11,
    prompt_id: int = 7,
    prompt_tokens: int = 42,
    success: bool = True,
    itl: list[float] | None = None,
) -> RequestFuncOutput:
    """Build a synthetic vLLM result with the given TTFT and TPOT."""
    send = 100.0
    first = send + ttft_s if success else None
    if success and output_tokens > 1:
        last: float | None = first + tpot_s * (output_tokens - 1)  # type: ignore[operator]
    elif success:
        last = first
    else:
        last = None
    return RequestFuncOutput(
        engine="vllm",
        prompt_id=prompt_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        arrival_time=send,
        send_time=send,
        first_token_time=first,
        last_token_time=last,
        itl=(itl if itl is not None else [tpot_s] * max(output_tokens - 1, 0)),
        success=success,
    )


# --- ttft_multiplier ---


def test_ttft_multiplier_at_zero_rate_is_two():
    assert ttft_multiplier(0.0, MAX_RATE) == 2.0


def test_ttft_multiplier_at_max_rate_is_eleven():
    assert ttft_multiplier(MAX_RATE, MAX_RATE) == 11.0


def test_ttft_multiplier_caps_above_max_rate():
    assert ttft_multiplier(2 * MAX_RATE, MAX_RATE) == 11.0
    assert ttft_multiplier(100.0, MAX_RATE) == 11.0


def test_ttft_multiplier_linear_midpoint():
    assert ttft_multiplier(MAX_RATE / 2, MAX_RATE) == pytest.approx(2.0 + 9.0 * 0.5)


def test_ttft_multiplier_negative_rate_returns_low_load_value():
    # Defensive: rate <= 0 → low-load factor 2.0.
    assert ttft_multiplier(-1.0, MAX_RATE) == 2.0


# --- tpot_multiplier ---


def test_tpot_multiplier_at_zero_rate_is_one_point_three():
    assert tpot_multiplier(0.0, MAX_RATE) == 1.3


def test_tpot_multiplier_at_max_rate_is_two_point_five():
    assert tpot_multiplier(MAX_RATE, MAX_RATE) == pytest.approx(2.5)


def test_tpot_multiplier_caps_above_max_rate():
    assert tpot_multiplier(2 * MAX_RATE, MAX_RATE) == pytest.approx(2.5)


def test_tpot_multiplier_linear_midpoint():
    assert tpot_multiplier(MAX_RATE / 2, MAX_RATE) == pytest.approx(1.3 + 1.2 * 0.5)


# --- synthesize_friendli_results ---


def test_synthesize_preserves_prompt_id_prompt_tokens_output_tokens():
    r = _vllm_result(prompt_id=17, prompt_tokens=99, output_tokens=21)
    [s] = synthesize_friendli_results([r], target_rate=10.0, max_rate=MAX_RATE)
    assert s.prompt_id == 17
    assert s.prompt_tokens == 99
    assert s.output_tokens == 21
    assert s.engine == "friendli_modeled"


def test_synthesize_reduces_ttft_by_expected_factor_at_max_rate():
    r = _vllm_result(ttft_s=1.0, tpot_s=0.05, output_tokens=11)
    [s] = synthesize_friendli_results([r], target_rate=MAX_RATE, max_rate=MAX_RATE)
    expected_ttft_s = 1.0 / 11.0
    assert s.ttft == pytest.approx(expected_ttft_s, rel=1e-12)


def test_synthesize_reduces_tpot_by_expected_factor_at_max_rate():
    r = _vllm_result(ttft_s=0.5, tpot_s=0.05, output_tokens=11)
    [s] = synthesize_friendli_results([r], target_rate=MAX_RATE, max_rate=MAX_RATE)
    expected_tpot_s = 0.05 / 2.5
    assert s.tpot == pytest.approx(expected_tpot_s, rel=1e-12)


def test_synthesize_uses_load_dependent_factors():
    """Verify that lower load → smaller multiplier → smaller speedup."""
    r = _vllm_result(ttft_s=1.0, tpot_s=0.05, output_tokens=11)
    [s_low] = synthesize_friendli_results([r], target_rate=0.0, max_rate=MAX_RATE)
    [s_high] = synthesize_friendli_results([r], target_rate=MAX_RATE, max_rate=MAX_RATE)
    # High-load Friendli should be FASTER (smaller TTFT) than low-load Friendli.
    assert s_high.ttft < s_low.ttft
    assert s_high.tpot < s_low.tpot


def test_synthesize_failed_input_passes_through_with_only_engine_relabelled():
    r = RequestFuncOutput(
        engine="vllm",
        prompt_id=3,
        prompt_tokens=5,
        output_tokens=0,
        success=False,
        error="HTTP 500",
    )
    [s] = synthesize_friendli_results([r], target_rate=10.0, max_rate=MAX_RATE)
    assert s.success is False
    assert s.engine == "friendli_modeled"
    assert s.error == "HTTP 500"
    assert s.first_token_time is None
    assert s.last_token_time is None
    assert s.output_tokens == 0


def test_synthesize_does_not_invent_success_when_ttft_unavailable():
    """Successful flag but missing first_token_time → must not run synthesis math."""
    r = RequestFuncOutput(
        engine="vllm",
        prompt_id=0,
        prompt_tokens=10,
        output_tokens=5,
        first_token_time=None,
        last_token_time=None,
        success=True,
    )
    [s] = synthesize_friendli_results([r], target_rate=10.0, max_rate=MAX_RATE)
    assert s.engine == "friendli_modeled"
    assert s.first_token_time is None
    assert s.last_token_time is None


def test_synthesize_itl_list_scaled_by_tpot_multiplier():
    itl = [0.05, 0.06, 0.04, 0.05]
    r = _vllm_result(ttft_s=0.5, tpot_s=0.05, output_tokens=5, itl=itl)
    [s] = synthesize_friendli_results([r], target_rate=MAX_RATE, max_rate=MAX_RATE)
    f = tpot_multiplier(MAX_RATE, MAX_RATE)
    for actual, original in zip(s.itl, itl):
        assert actual == pytest.approx(original / f, rel=1e-12)


def test_synthesize_first_and_last_token_time_are_consistent_with_new_latencies():
    r = _vllm_result(ttft_s=1.0, tpot_s=0.05, output_tokens=11)
    [s] = synthesize_friendli_results([r], target_rate=MAX_RATE, max_rate=MAX_RATE)
    f_ttft = ttft_multiplier(MAX_RATE, MAX_RATE)
    f_tpot = tpot_multiplier(MAX_RATE, MAX_RATE)
    assert s.send_time == r.send_time  # send_time must not be touched
    expected_first = r.send_time + (1.0 / f_ttft)
    expected_last = expected_first + (0.05 / f_tpot) * 10
    assert s.first_token_time == pytest.approx(expected_first, rel=1e-12)
    assert s.last_token_time == pytest.approx(expected_last, rel=1e-12)


def test_synthesize_engine_label_is_friendli_modeled_for_all_results():
    items = [
        _vllm_result(prompt_id=0),
        _vllm_result(prompt_id=1, success=False),
        RequestFuncOutput(engine="vllm", prompt_id=2, prompt_tokens=1, success=False),
    ]
    results = synthesize_friendli_results(items, target_rate=10.0, max_rate=MAX_RATE)
    assert all(r.engine == "friendli_modeled" for r in results)


def test_synthesize_pure_and_deterministic():
    """Same inputs → same outputs (no global state)."""
    r = _vllm_result(prompt_id=0, ttft_s=0.7, tpot_s=0.04, output_tokens=8)
    out1 = synthesize_friendli_results([r], target_rate=12.0, max_rate=MAX_RATE)
    out2 = synthesize_friendli_results([r], target_rate=12.0, max_rate=MAX_RATE)
    assert len(out1) == len(out2) == 1
    assert out1[0].first_token_time == out2[0].first_token_time
    assert out1[0].last_token_time == out2[0].last_token_time
    assert out1[0].itl == out2[0].itl


def test_synthesize_does_not_mutate_input_list():
    original_ttft_s = 1.0
    r = _vllm_result(ttft_s=original_ttft_s, tpot_s=0.05, output_tokens=11)
    original_first = r.first_token_time
    original_engine = r.engine
    synthesize_friendli_results([r], target_rate=MAX_RATE, max_rate=MAX_RATE)
    assert r.first_token_time == original_first
    assert r.engine == original_engine


def test_synthesize_output_tokens_per_request_unchanged():
    items = [_vllm_result(output_tokens=k) for k in (5, 11, 21)]
    results = synthesize_friendli_results(items, target_rate=10.0, max_rate=MAX_RATE)
    assert [r.output_tokens for r in results] == [5, 11, 21]
