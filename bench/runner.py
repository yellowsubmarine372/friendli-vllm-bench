"""Sweep orchestration: warmup → ascending-rate measurement → aggregation → output.

Implements SPEC §6.7. In measurement mode, both engines run back-to-back at
each rate (``vllm@1, friendli@1, vllm@2, friendli@2, …``) so thermal/system
drift cannot accumulate across rates. In synthesis mode, only vLLM is
measured directly; Friendli timings are derived immediately afterward via
``bench.synthetic`` against the same vLLM raw data.

Tabulated console output and JSON/CSV summaries follow vLLM's
``benchmark_serving.py`` convention.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tabulate import tabulate

from bench.backend import BackendFunc, RequestFuncInput, RequestFuncOutput
from bench.metrics import SLO, RunSummary, aggregate
from bench.synthetic import synthesize_friendli_results
from bench.utils import capture_environment, jsonl_dump
from bench.workload import PromptTuple, cycle_prompts, run_open_loop


@dataclass
class RunConfig:
    model: str
    backend_fn: BackendFunc          # bound with client + tokenizer at the CLI layer
    prompts: list[PromptTuple]
    max_output_tokens: int
    slo: SLO
    seed: int
    output_dir: Path
    cli_args: dict[str, Any] = field(default_factory=dict)


def _base_input_template(api_url: str, config: RunConfig) -> dict[str, Any]:
    return {
        "api_url": api_url,
        "model": config.model,
        "max_tokens": config.max_output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": config.seed,
        "ignore_eos": True,
    }


async def _warmup(
    engine_name: str,
    api_url: str,
    config: RunConfig,
    n_requests: int,
) -> None:
    """Closed-loop warmup: send ``n_requests`` sequentially, discard results."""
    if n_requests <= 0:
        return
    print(f"[warmup] {engine_name} ({n_requests} requests sequential)…", flush=True)
    for i in range(n_requests):
        prompt_id, text, approx = config.prompts[i % len(config.prompts)]
        req = RequestFuncInput(
            prompt=text,
            prompt_id=prompt_id,
            prompt_tokens=approx,
            **_base_input_template(api_url, config),
        )
        await config.backend_fn(
            req, arrival_time=time.perf_counter(), engine=engine_name
        )


async def _measure_one(
    engine_name: str,
    api_url: str,
    rate: float,
    duration: float,
    config: RunConfig,
) -> list[RequestFuncOutput]:
    """Run the open-loop dispatcher once for ``(engine, rate)``."""
    return await run_open_loop(
        backend_fn=config.backend_fn,
        prompts=cycle_prompts(config.prompts),
        rate=rate,
        duration=duration,
        seed=config.seed,
        base_input_template=_base_input_template(api_url, config),
        engine_name=engine_name,
    )


def _dump_raw(
    results: list[RequestFuncOutput],
    engine_name: str,
    rate: float,
    output_dir: Path,
    timestamp: str,
) -> Path:
    rate_token = f"{rate:g}".replace(".", "p")
    filename = f"{engine_name}_{rate_token}rps_{timestamp}.jsonl"
    path = output_dir / "raw" / filename
    jsonl_dump([asdict(r) for r in results], path)
    return path


def _print_table(summaries: list[RunSummary]) -> None:
    headers = [
        "engine", "rate", "n", "ok", "slo_ok", "good_rps",
        "raw_rps", "tok/s", "ttft p95", "tpot p95", "e2e p95",
    ]
    rows = []
    for s in summaries:
        rows.append([
            s.engine,
            f"{s.target_rate:g}",
            s.actual_request_count,
            s.successful_count,
            s.slo_compliant_count,
            f"{s.goodput_rps:.2f}",
            f"{s.raw_throughput_rps:.2f}",
            f"{s.output_tokens_per_sec:.1f}",
            f"{s.ttft_p95_ms:.0f}",
            f"{s.tpot_p95_ms:.1f}",
            f"{s.e2e_p95_ms:.0f}",
        ])
    print()
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


def _write_summary_json(summaries: list[RunSummary], output_dir: Path) -> Path:
    path = output_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summaries": [s.to_dict() for s in summaries]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _write_summary_csv(summaries: list[RunSummary], output_dir: Path) -> Path:
    path = output_dir / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RunSummary.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s.to_dict())
    return path


def _write_run_metadata(
    config: RunConfig,
    rates: list[float],
    summaries: list[RunSummary],
    synthesize_friendli: bool,
    output_dir: Path,
) -> Path:
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cli_args": config.cli_args,
        "rates": rates,
        "synthesize_friendli": synthesize_friendli,
        "n_summaries": len(summaries),
        **capture_environment(),
    }
    path = output_dir / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


async def run_sweep(
    engines: dict[str, str | None],
    rates: list[float],
    duration_per_rate: float,
    warmup_requests: int,
    cooldown_seconds: float,
    synthesize_friendli: bool,
    config: RunConfig,
) -> list[RunSummary]:
    """Drive the full engine × rate sweep.

    ``engines`` maps engine name → API URL (or ``None`` to skip). In
    synthesis mode only ``"vllm"`` is dispatched; Friendli timings are
    derived per rate from the vLLM raw data.
    """
    rates = sorted(rates)
    if not rates:
        raise ValueError("rates must be non-empty")
    max_rate = float(max(rates))

    if synthesize_friendli:
        measured_engines = ["vllm"]
    else:
        measured_engines = [e for e in ("vllm", "friendli") if engines.get(e)]
        if not measured_engines:
            raise ValueError("no engines to measure (synthesis mode is off)")

    summaries: list[RunSummary] = []

    # 1. Warmup every engine that will be measured directly.
    for engine_name in measured_engines:
        url = engines[engine_name]
        if url is None:
            continue
        await _warmup(engine_name, url, config, warmup_requests)

    # 2. Ascending-rate sweep.
    for rate_idx, rate in enumerate(rates):
        # In synthesis mode, only vLLM runs at each rate. In measurement mode,
        # both engines run back-to-back so drift cannot accumulate.
        for engine_name in measured_engines:
            url = engines[engine_name]
            if url is None:
                continue
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            print(
                f"[measure] {engine_name} @ {rate:g} RPS for {duration_per_rate}s "
                f"(prompt seed={config.seed})",
                flush=True,
            )
            results = await _measure_one(
                engine_name, url, rate, duration_per_rate, config
            )
            _dump_raw(results, engine_name, rate, config.output_dir, timestamp)
            summary = aggregate(engine_name, rate, results, config.slo)
            summaries.append(summary)
            print(
                f"   → goodput={summary.goodput_rps:.2f} RPS, "
                f"raw={summary.raw_throughput_rps:.2f} RPS, "
                f"ttft p95={summary.ttft_p95_ms:.0f}ms, "
                f"tpot p95={summary.tpot_p95_ms:.1f}ms, "
                f"ok={summary.successful_count}/{summary.actual_request_count}",
                flush=True,
            )

            # Synthesis mode: derive Friendli for this rate immediately.
            if synthesize_friendli and engine_name == "vllm":
                fr_results = synthesize_friendli_results(results, rate, max_rate)
                _dump_raw(
                    fr_results, "friendli_modeled", rate, config.output_dir, timestamp
                )
                fr_summary = aggregate(
                    "friendli_modeled", rate, fr_results, config.slo
                )
                summaries.append(fr_summary)
                print(
                    f"   → friendli_modeled goodput={fr_summary.goodput_rps:.2f} RPS "
                    f"(modeled from vLLM, see README §Data Sources)",
                    flush=True,
                )

        # Cooldown (skip after the last rate).
        if rate_idx < len(rates) - 1 and cooldown_seconds > 0:
            print(f"[cooldown] {cooldown_seconds:g}s …", flush=True)
            await asyncio.sleep(cooldown_seconds)

    # 3. Tabulated summary to console.
    _print_table(summaries)

    # 4. Structured outputs.
    _write_summary_json(summaries, config.output_dir)
    _write_summary_csv(summaries, config.output_dir)

    # 5. Run metadata (env + git SHA + nvidia-smi + CLI args).
    _write_run_metadata(
        config, rates, summaries, synthesize_friendli, config.output_dir
    )

    return summaries
