"""Smoke tests for runner orchestration (§6.7).

The fake backend returns deterministic timings so the end-to-end sweep
can be exercised without an HTTP endpoint. The tests verify file output,
engine alternation, synthesis mode, and warmup behaviour.
"""
from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pytest

from bench.backend import RequestFuncOutput
from bench.metrics import SLO
from bench.runner import RunConfig, run_sweep
from bench.workload import load_prompts


def _make_fake_backend(latency_ms: float = 30.0, dispatch_log: list[str] | None = None,
                      warmup_log: list[str] | None = None):
    """Backend that returns success with constant TTFT/TPOT.

    ``dispatch_log`` records ``"<engine>:<prompt_id>"`` for each call so a test
    can verify ordering. ``warmup_log`` is a separate list; warmup calls are
    tagged with arrival_time==perf_counter (closed-loop) so they look the same
    as measurement calls. We distinguish them via the engine label only — the
    runner does not emit a "warmup" tag.
    """
    async def fn(req, *, arrival_time: float, engine: str) -> RequestFuncOutput:
        if dispatch_log is not None:
            dispatch_log.append(f"{engine}:{req.prompt_id}")
        # Simulate a quick stream: TTFT 10ms, then 4 ITLs of 5ms each (5 tokens).
        send = arrival_time
        first = send + 0.01
        last = first + 0.005 * 4
        return RequestFuncOutput(
            engine=engine,
            prompt_id=req.prompt_id,
            prompt_tokens=req.prompt_tokens,
            output_tokens=5,
            arrival_time=arrival_time,
            send_time=send,
            first_token_time=first,
            last_token_time=last,
            itl=[0.005] * 4,
            success=True,
        )
    return fn


def _make_config(tmp_path: Path, backend, prompts, **overrides) -> RunConfig:
    base = dict(
        model="m",
        backend_fn=backend,
        prompts=prompts,
        max_output_tokens=8,
        slo=SLO(ttft_ms=500.0, tpot_ms=50.0),
        seed=7,  # produces multiple arrivals at low rate/duration
        output_dir=tmp_path,
        cli_args={"sample": "args"},
    )
    base.update(overrides)
    return RunConfig(**base)


def _real_prompts():
    return load_prompts(Path("prompts/prompts.jsonl"))


# --- end-to-end output files ---


def test_run_sweep_synthesis_mode_writes_all_outputs(tmp_path: Path):
    backend = _make_fake_backend()
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    summaries = asyncio.run(run_sweep(
        engines={"vllm": "http://test", "friendli": None},
        rates=[20.0, 30.0],
        duration_per_rate=0.3,
        warmup_requests=2,
        cooldown_seconds=0.0,
        synthesize_friendli=True,
        config=cfg,
    ))

    # Two rates × two engines (vllm + synthesized friendli) = 4 summaries.
    assert len(summaries) == 4
    engines_seen = {s.engine for s in summaries}
    assert engines_seen == {"vllm", "friendli_modeled"}

    # summary.json
    summary_json = json.loads((tmp_path / "summary.json").read_text())
    assert "summaries" in summary_json
    assert len(summary_json["summaries"]) == 4

    # summary.csv
    with (tmp_path / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    csv_engines = {row["engine"] for row in rows}
    assert csv_engines == {"vllm", "friendli_modeled"}

    # run_metadata.json
    md = json.loads((tmp_path / "run_metadata.json").read_text())
    assert md["synthesize_friendli"] is True
    assert md["rates"] == [20.0, 30.0]
    assert md["cli_args"] == {"sample": "args"}
    assert "git_sha" in md and md["git_sha"]  # captured (likely the current SHA)
    assert "library_versions" in md

    # raw/ directory
    raw_files = sorted((tmp_path / "raw").glob("*.jsonl"))
    # 2 rates × 2 engines (vllm + friendli_modeled) = 4 files
    assert len(raw_files) == 4
    by_engine = {}
    for p in raw_files:
        engine = p.name.split("_")[0]
        # friendli_modeled has an underscore in the engine name
        if engine == "friendli":
            engine = "friendli_modeled"
        by_engine.setdefault(engine, []).append(p)
    assert "vllm" in by_engine and len(by_engine["vllm"]) == 2
    assert "friendli_modeled" in by_engine and len(by_engine["friendli_modeled"]) == 2


def test_run_sweep_measurement_mode_alternates_engines_within_each_rate(tmp_path: Path):
    """SPEC §6.7: vllm@1, friendli@1, vllm@2, friendli@2, ..."""
    dispatch_log: list[str] = []
    backend = _make_fake_backend(dispatch_log=dispatch_log)
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    summaries = asyncio.run(run_sweep(
        engines={"vllm": "http://v", "friendli": "http://f"},
        rates=[20.0, 30.0],
        duration_per_rate=0.3,
        warmup_requests=2,
        cooldown_seconds=0.0,
        synthesize_friendli=False,
        config=cfg,
    ))

    # 2 rates × 2 engines = 4 summaries
    assert len(summaries) == 4
    # Order in summary list reflects dispatch order
    seen_engines = [s.engine for s in summaries]
    seen_rates = [s.target_rate for s in summaries]
    assert seen_engines == ["vllm", "friendli", "vllm", "friendli"]
    assert seen_rates == [20.0, 20.0, 30.0, 30.0]

    # Confirm vLLM dispatches happen before Friendli at each rate
    # (i.e., the first vllm call is before the first friendli call).
    first_vllm = next(i for i, x in enumerate(dispatch_log) if x.startswith("vllm:"))
    first_friendli = next(i for i, x in enumerate(dispatch_log) if x.startswith("friendli:"))
    assert first_vllm < first_friendli


def test_run_sweep_warmup_runs_for_each_measured_engine(tmp_path: Path):
    """Both engines warm up before any measurement begins."""
    dispatch_log: list[str] = []
    backend = _make_fake_backend(dispatch_log=dispatch_log)
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    asyncio.run(run_sweep(
        engines={"vllm": "http://v", "friendli": "http://f"},
        rates=[20.0],
        duration_per_rate=0.2,
        warmup_requests=3,
        cooldown_seconds=0.0,
        synthesize_friendli=False,
        config=cfg,
    ))

    # Per-engine warmup count: at least 3 calls per engine before measurement
    # begins. The total dispatch count = warmup calls + measurement arrivals.
    vllm_calls = [x for x in dispatch_log if x.startswith("vllm:")]
    friendli_calls = [x for x in dispatch_log if x.startswith("friendli:")]
    assert len(vllm_calls) >= 3
    assert len(friendli_calls) >= 3


def test_run_sweep_synthesis_mode_does_not_dispatch_friendli(tmp_path: Path):
    dispatch_log: list[str] = []
    backend = _make_fake_backend(dispatch_log=dispatch_log)
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    asyncio.run(run_sweep(
        engines={"vllm": "http://v", "friendli": None},
        rates=[20.0],
        duration_per_rate=0.2,
        warmup_requests=2,
        cooldown_seconds=0.0,
        synthesize_friendli=True,
        config=cfg,
    ))

    # Ensure the backend was never asked to talk to Friendli.
    assert all(x.startswith("vllm:") for x in dispatch_log)


def test_run_sweep_rates_sorted_ascending(tmp_path: Path):
    backend = _make_fake_backend()
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    summaries = asyncio.run(run_sweep(
        engines={"vllm": "http://v", "friendli": None},
        rates=[30.0, 20.0],  # purposely out of order
        duration_per_rate=0.2,
        warmup_requests=1,
        cooldown_seconds=0.0,
        synthesize_friendli=True,
        config=cfg,
    ))

    # Order in summaries reflects ascending rate, both engines per rate.
    rates_seen = [s.target_rate for s in summaries]
    assert rates_seen == [20.0, 20.0, 30.0, 30.0]


def test_run_sweep_rejects_empty_rates(tmp_path: Path):
    backend = _make_fake_backend()
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    async def go():
        return await run_sweep(
            engines={"vllm": "http://v", "friendli": None},
            rates=[],
            duration_per_rate=0.1,
            warmup_requests=1,
            cooldown_seconds=0.0,
            synthesize_friendli=True,
            config=cfg,
        )

    with pytest.raises(ValueError, match="rates"):
        asyncio.run(go())


def test_run_sweep_summary_csv_has_all_runsummary_columns(tmp_path: Path):
    backend = _make_fake_backend()
    prompts = _real_prompts()
    cfg = _make_config(tmp_path, backend, prompts)

    asyncio.run(run_sweep(
        engines={"vllm": "http://v", "friendli": None},
        rates=[20.0],
        duration_per_rate=0.2,
        warmup_requests=1,
        cooldown_seconds=0.0,
        synthesize_friendli=True,
        config=cfg,
    ))

    from bench.metrics import RunSummary
    expected_cols = list(RunSummary.__dataclass_fields__)
    with (tmp_path / "summary.csv").open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == expected_cols
