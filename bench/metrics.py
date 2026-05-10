"""SLO judgment and per-rate aggregation.

Goodput here uses a DistServe-style SLO definition (OSDI '24): a request
is compliant only if both TTFT and TPOT satisfy their thresholds.
DistServe follows from the paper's framing of TTFT and TPOT as joint latency requirements
("maximum request rate that can be served within both TTFT and TPOT constraints").

Percentiles use ``numpy.percentile`` with ``method="linear"`` per SPEC §6.4.
Metric definitions (TTFT, ITL/TPOT, e2e_latency) follow NVIDIA GenAI-Perf:
https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from bench.backend import RequestFuncOutput


@dataclass
class SLO:
    ttft_ms: float
    tpot_ms: float


def is_slo_compliant(r: RequestFuncOutput, slo: SLO) -> bool:
    """Return True iff ``r`` satisfies both the per-request TTFT and TPOT SLOs.

    Failed requests, requests missing TTFT, and requests missing TPOT all
    count as non-compliant. Threshold is inclusive: a request whose TTFT
    equals the threshold passes; a request whose TTFT exceeds it fails.
    """
    if not r.success:
        return False
    if r.ttft is None or r.tpot is None:
        return False
    if r.ttft * 1000.0 > slo.ttft_ms:
        return False
    if r.tpot * 1000.0 > slo.tpot_ms:
        return False
    return True


@dataclass
class RunSummary:
    engine: str
    target_rate: float
    actual_request_count: int
    successful_count: int
    slo_compliant_count: int
    measurement_window: float
    goodput_rps: float
    raw_throughput_rps: float
    output_tokens_per_sec: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    itl_jitter_p95_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q, method="linear"))


def aggregate(
    engine: str,
    target_rate: float,
    results: Iterable[RequestFuncOutput],
    slo: SLO,
) -> RunSummary:
    """Aggregate per-request outputs into a ``RunSummary``.

    Measurement window is from the earliest ``send_time`` to the latest
    ``last_token_time`` across requests that produced at least one token.
    Per SPEC §6.7, callers pass post-warmup results only; warmup exclusion
    is the runner's responsibility.
    """
    results_list = list(results)
    actual = len(results_list)

    timed = [
        r for r in results_list
        if r.success and r.first_token_time is not None and r.last_token_time is not None
    ]
    successful = len(timed)

    if timed:
        window_start = min(r.send_time for r in timed)
        window_end = max(r.last_token_time for r in timed if r.last_token_time is not None)
        window = max(window_end - window_start, 1e-9)
    else:
        window = 1e-9

    compliant = [r for r in results_list if is_slo_compliant(r, slo)]

    output_tokens_total = sum(r.output_tokens for r in timed)
    ttfts_ms = [r.ttft * 1000.0 for r in timed if r.ttft is not None]
    tpots_ms = [r.tpot * 1000.0 for r in timed if r.tpot is not None]
    e2es_ms = [r.e2e_latency * 1000.0 for r in timed if r.e2e_latency is not None]
    jitters_ms = [float(np.std(r.itl) * 1000.0) for r in timed if len(r.itl) >= 1]

    return RunSummary(
        engine=engine,
        target_rate=target_rate,
        actual_request_count=actual,
        successful_count=successful,
        slo_compliant_count=len(compliant),
        measurement_window=window,
        goodput_rps=len(compliant) / window,
        raw_throughput_rps=successful / window,
        output_tokens_per_sec=output_tokens_total / window,
        ttft_p50_ms=_percentile(ttfts_ms, 50),
        ttft_p95_ms=_percentile(ttfts_ms, 95),
        ttft_p99_ms=_percentile(ttfts_ms, 99),
        tpot_p50_ms=_percentile(tpots_ms, 50),
        tpot_p95_ms=_percentile(tpots_ms, 95),
        tpot_p99_ms=_percentile(tpots_ms, 99),
        e2e_p50_ms=_percentile(e2es_ms, 50),
        e2e_p95_ms=_percentile(e2es_ms, 95),
        itl_jitter_p95_ms=_percentile(jitters_ms, 95),
    )
