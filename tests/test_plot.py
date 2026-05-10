"""Smoke tests for the plot generator (§6.8).

The graph itself is hard to assert on — these tests verify file I/O,
saturation math, and the synthesized-vs-measured branch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bench.metrics import SLO, RunSummary
from bench.plot import plot_goodput, saturation_rate


def _summary(engine: str, rate: float, goodput: float) -> RunSummary:
    return RunSummary(
        engine=engine,
        target_rate=rate,
        actual_request_count=int(rate * 60),
        successful_count=int(rate * 60),
        slo_compliant_count=int(goodput * 60),
        measurement_window=60.0,
        goodput_rps=goodput,
        raw_throughput_rps=rate,
        output_tokens_per_sec=rate * 256.0,
        ttft_p50_ms=100.0,
        ttft_p95_ms=200.0,
        ttft_p99_ms=300.0,
        tpot_p50_ms=10.0,
        tpot_p95_ms=20.0,
        tpot_p99_ms=30.0,
        e2e_p50_ms=500.0,
        e2e_p95_ms=800.0,
        itl_jitter_p95_ms=2.0,
    )


# --- saturation_rate ---


def test_saturation_rate_returns_first_diverging_rate():
    rates = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    # at rate=8, gp=7.5 → divergence (8-7.5)/8 = 6.25%, below 10% — not saturated.
    # at rate=16, gp=13 → (16-13)/16 = 18.75% — saturated.
    goodputs = [1.0, 2.0, 4.0, 7.5, 13.0, 16.0]
    assert saturation_rate(rates, goodputs) == 16.0


def test_saturation_rate_none_when_perfect_goodput():
    rates = [1.0, 2.0, 4.0, 8.0]
    goodputs = [1.0, 2.0, 4.0, 8.0]
    assert saturation_rate(rates, goodputs) is None


def test_saturation_rate_threshold_is_strict_greater_than():
    rates = [1.0, 2.0]
    # exactly 10% divergence at rate=2: (2-1.8)/2 = 0.10
    goodputs = [1.0, 1.8]
    assert saturation_rate(rates, goodputs) is None  # strict >, not ≥


def test_saturation_rate_handles_unsorted_input():
    rates = [16.0, 4.0, 8.0, 2.0, 1.0]
    goodputs = [13.0, 4.0, 7.5, 2.0, 1.0]
    assert saturation_rate(rates, goodputs) == 16.0


# --- plot_goodput end-to-end ---


def test_plot_goodput_writes_png_and_pdf_synthesized(tmp_path: Path):
    summaries = [
        _summary("vllm", 1.0, 1.0),
        _summary("vllm", 8.0, 7.0),
        _summary("vllm", 32.0, 12.0),
        _summary("friendli_modeled", 1.0, 1.0),
        _summary("friendli_modeled", 8.0, 8.0),
        _summary("friendli_modeled", 32.0, 28.0),
    ]
    out = tmp_path / "goodput.png"
    plot_goodput(
        summaries,
        slo=SLO(ttft_ms=500.0, tpot_ms=50.0),
        output_path=out,
        synthesized=True,
        git_sha="abcdef0123456",
    )
    assert out.exists()
    assert out.stat().st_size > 0
    pdf = tmp_path / "goodput.pdf"
    assert pdf.exists()
    assert pdf.stat().st_size > 0


def test_plot_goodput_writes_png_and_pdf_measurement(tmp_path: Path):
    summaries = [
        _summary("vllm", 1.0, 1.0),
        _summary("vllm", 8.0, 7.0),
        _summary("friendli", 1.0, 1.0),
        _summary("friendli", 8.0, 7.5),
    ]
    out = tmp_path / "goodput.png"
    plot_goodput(
        summaries,
        slo=SLO(ttft_ms=500.0, tpot_ms=50.0),
        output_path=out,
        synthesized=False,
        git_sha="0000000",
    )
    assert out.exists()
    assert (tmp_path / "goodput.pdf").exists()


def test_plot_goodput_creates_parent_dirs(tmp_path: Path):
    summaries = [_summary("vllm", 1.0, 1.0), _summary("vllm", 8.0, 8.0)]
    nested = tmp_path / "a" / "b" / "graph.png"
    plot_goodput(
        summaries,
        slo=SLO(ttft_ms=500.0, tpot_ms=50.0),
        output_path=nested,
        synthesized=True,
        git_sha="abc1234",
    )
    assert nested.exists()


def test_plot_goodput_rejects_empty_summaries(tmp_path: Path):
    with pytest.raises(ValueError, match="no summaries"):
        plot_goodput(
            summaries=[],
            slo=SLO(500.0, 50.0),
            output_path=tmp_path / "g.png",
            synthesized=False,
            git_sha="x",
        )


def test_plot_goodput_handles_missing_git_sha(tmp_path: Path):
    summaries = [_summary("vllm", 1.0, 1.0), _summary("vllm", 8.0, 7.0)]
    out = tmp_path / "g.png"
    plot_goodput(
        summaries,
        slo=SLO(500.0, 50.0),
        output_path=out,
        synthesized=True,
        git_sha="",
    )
    assert out.exists()
