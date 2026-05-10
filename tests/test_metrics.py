"""Tests for RequestFuncOutput property methods (§6.1)
and SLO + aggregation logic (§6.4)."""
from __future__ import annotations

import numpy as np
import pytest

from bench.backend import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
    register_backend,
)
from bench.metrics import SLO, RunSummary, aggregate, is_slo_compliant


def make_output(**overrides) -> RequestFuncOutput:
    base = dict(
        engine="vllm",
        prompt_id=0,
        prompt_tokens=10,
        output_tokens=11,
        arrival_time=10.0,
        send_time=10.05,
        first_token_time=10.20,
        last_token_time=11.20,
        itl=[0.1] * 10,
        success=True,
        error=None,
    )
    base.update(overrides)
    return RequestFuncOutput(**base)


def test_ttft_basic():
    r = make_output()
    assert r.ttft == pytest.approx(0.15)


def test_ttft_none_when_no_first_token():
    r = make_output(first_token_time=None)
    assert r.ttft is None


def test_tpot_basic():
    # (last - first) / (output_tokens - 1) = (11.20 - 10.20) / 10 = 0.1
    r = make_output()
    assert r.tpot == pytest.approx(0.1)


def test_tpot_returns_none_when_output_tokens_one():
    r = make_output(output_tokens=1)
    assert r.tpot is None


def test_tpot_returns_none_when_output_tokens_zero():
    r = make_output(output_tokens=0)
    assert r.tpot is None


def test_tpot_returns_none_when_first_token_missing():
    r = make_output(first_token_time=None)
    assert r.tpot is None


def test_tpot_returns_none_when_last_token_missing():
    r = make_output(last_token_time=None)
    assert r.tpot is None


def test_e2e_latency_basic():
    r = make_output()
    assert r.e2e_latency == pytest.approx(1.15)


def test_e2e_latency_none_when_no_last_token():
    r = make_output(last_token_time=None)
    assert r.e2e_latency is None


def test_queue_delay_basic():
    r = make_output()
    assert r.queue_delay == pytest.approx(0.05)


def test_queue_delay_zero_when_send_equals_arrival():
    r = make_output(arrival_time=10.0, send_time=10.0)
    assert r.queue_delay == 0.0


def test_request_func_input_defaults():
    inp = RequestFuncInput(
        prompt="hi", prompt_id=0, prompt_tokens=5,
        api_url="http://x", model="m", max_tokens=10,
    )
    assert inp.temperature == 0.0
    assert inp.top_p == 1.0
    assert inp.seed == 42
    assert inp.ignore_eos is True


def test_register_backend_adds_to_registry():
    sentinel_name = "test-backend-xyz"
    assert sentinel_name not in ASYNC_REQUEST_FUNCS

    @register_backend(sentinel_name)
    async def fn(*args, **kwargs):  # pragma: no cover - body unused in this test
        return None  # type: ignore[return-value]

    try:
        assert ASYNC_REQUEST_FUNCS[sentinel_name] is fn
    finally:
        del ASYNC_REQUEST_FUNCS[sentinel_name]


# --- SLO judgment (§6.4) ---


def _slo() -> SLO:
    return SLO(ttft_ms=500.0, tpot_ms=50.0)


def _make_request(*, ttft_s: float, tpot_s: float, output_tokens: int = 11,
                  success: bool = True) -> RequestFuncOutput:
    """Build a RequestFuncOutput with given TTFT and TPOT in seconds."""
    send = 100.0
    first = send + ttft_s
    last = first + tpot_s * (output_tokens - 1) if output_tokens > 1 else first
    return RequestFuncOutput(
        engine="test",
        prompt_id=0,
        prompt_tokens=10,
        output_tokens=output_tokens,
        arrival_time=send,
        send_time=send,
        first_token_time=first,
        last_token_time=last,
        itl=[tpot_s] * (output_tokens - 1),
        success=success,
    )


def test_slo_compliant_within_threshold():
    r = _make_request(ttft_s=0.4, tpot_s=0.04)  # 400ms, 40ms
    assert is_slo_compliant(r, _slo()) is True


def test_slo_compliant_exactly_at_threshold():
    r = _make_request(ttft_s=0.5, tpot_s=0.05)  # 500ms, 50ms exactly
    assert is_slo_compliant(r, _slo()) is True


def test_slo_violated_when_ttft_beyond_threshold():
    r = _make_request(ttft_s=0.501, tpot_s=0.04)
    assert is_slo_compliant(r, _slo()) is False


def test_slo_violated_when_tpot_beyond_threshold():
    r = _make_request(ttft_s=0.4, tpot_s=0.0501)
    assert is_slo_compliant(r, _slo()) is False


def test_slo_violated_when_both_beyond():
    r = _make_request(ttft_s=0.6, tpot_s=0.06)
    assert is_slo_compliant(r, _slo()) is False


def test_slo_violated_when_request_failed():
    r = _make_request(ttft_s=0.1, tpot_s=0.01, success=False)
    assert is_slo_compliant(r, _slo()) is False


def test_slo_violated_when_first_token_missing():
    r = _make_request(ttft_s=0.1, tpot_s=0.01)
    r.first_token_time = None
    assert is_slo_compliant(r, _slo()) is False


def test_slo_violated_when_tpot_undefined_due_to_one_token():
    r = _make_request(ttft_s=0.1, tpot_s=0.01, output_tokens=1)
    assert r.tpot is None
    assert is_slo_compliant(r, _slo()) is False


# --- Percentile / aggregation (§6.4) ---


def test_aggregate_percentiles_match_numpy_on_100_values():
    """SPEC §8: 100-value list, p50/p95/p99 match numpy.percentile."""
    # Build 100 requests with ttft_ms = 1..100 (in seconds: 0.001..0.100).
    results = [_make_request(ttft_s=i / 1000.0, tpot_s=0.01, output_tokens=11)
               for i in range(1, 101)]
    summary = aggregate("vllm", target_rate=1.0, results=results, slo=_slo())

    expected_ttft_ms = np.array([float(i) for i in range(1, 101)])
    assert summary.ttft_p50_ms == pytest.approx(
        float(np.percentile(expected_ttft_ms, 50, method="linear")))
    assert summary.ttft_p95_ms == pytest.approx(
        float(np.percentile(expected_ttft_ms, 95, method="linear")))
    assert summary.ttft_p99_ms == pytest.approx(
        float(np.percentile(expected_ttft_ms, 99, method="linear")))


def test_aggregate_counts_and_rates_basic():
    # 3 successful (compliant), 1 successful but SLO-violated, 1 failed.
    results = [
        _make_request(ttft_s=0.10, tpot_s=0.01),  # compliant
        _make_request(ttft_s=0.20, tpot_s=0.02),  # compliant
        _make_request(ttft_s=0.30, tpot_s=0.03),  # compliant
        _make_request(ttft_s=0.60, tpot_s=0.01),  # SLO-violated (ttft=600ms > 500)
        _make_request(ttft_s=0.10, tpot_s=0.01, success=False),
    ]
    summary = aggregate("vllm", target_rate=8.0, results=results, slo=_slo())
    assert summary.actual_request_count == 5
    assert summary.successful_count == 4  # all with timing data
    assert summary.slo_compliant_count == 3
    assert summary.measurement_window > 0
    # goodput ≤ raw throughput by construction
    assert summary.goodput_rps <= summary.raw_throughput_rps


def test_aggregate_handles_zero_successful():
    results = [_make_request(ttft_s=0.1, tpot_s=0.01, success=False) for _ in range(3)]
    summary = aggregate("vllm", target_rate=1.0, results=results, slo=_slo())
    assert summary.actual_request_count == 3
    assert summary.successful_count == 0
    assert summary.slo_compliant_count == 0
    assert summary.goodput_rps == 0.0
    assert summary.raw_throughput_rps == 0.0
    assert summary.ttft_p50_ms == 0.0


def test_aggregate_output_tokens_per_sec():
    # Two requests, output_tokens = 11 each → 22 tokens total over the window.
    r1 = _make_request(ttft_s=0.1, tpot_s=0.01, output_tokens=11)
    r2 = _make_request(ttft_s=0.1, tpot_s=0.01, output_tokens=11)
    # Force a non-trivial window: shift r2 forward in time.
    r2.send_time += 1.0
    r2.first_token_time = r2.send_time + 0.1
    r2.last_token_time = r2.first_token_time + 0.01 * 10  # 10 inter-token gaps
    summary = aggregate("vllm", target_rate=1.0, results=[r1, r2], slo=_slo())
    expected_window = r2.last_token_time - r1.send_time
    assert summary.measurement_window == pytest.approx(expected_window)
    assert summary.output_tokens_per_sec == pytest.approx(22.0 / expected_window)


def test_aggregate_itl_jitter_p95_uses_per_request_stddev():
    # Build a request with itl having a known stddev.
    r = _make_request(ttft_s=0.1, tpot_s=0.01, output_tokens=5)
    r.itl = [0.01, 0.02, 0.01, 0.03]  # stddev ≈ 0.0083 → 8.3 ms
    summary = aggregate("vllm", target_rate=1.0, results=[r], slo=_slo())
    expected_jitter_ms = float(np.std(r.itl) * 1000.0)
    assert summary.itl_jitter_p95_ms == pytest.approx(expected_jitter_ms)


def test_run_summary_to_dict_round_trip():
    r = _make_request(ttft_s=0.1, tpot_s=0.01)
    summary = aggregate("vllm", target_rate=2.0, results=[r], slo=_slo())
    d = summary.to_dict()
    assert isinstance(d, dict)
    assert d["engine"] == "vllm"
    assert d["target_rate"] == 2.0
    # All RunSummary fields are present
    assert set(d.keys()) == {f for f in RunSummary.__dataclass_fields__}
