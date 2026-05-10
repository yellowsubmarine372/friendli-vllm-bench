# SPEC.md — `friendli-vllm-bench`

> Read this document fully before writing any code. Implement section by section in the order specified in §10. After each module, run its tests if any, show the diff, and wait for "continue" before proceeding.

## 1. Project Goal

Build a benchmarking client that compares **vLLM** and **Friendli Engine** as LLM inference backends. Both engines are assumed deployed and exposing OpenAI-compatible streaming chat completion endpoints.

The deliverable is a single matplotlib graph: **Goodput (RPS) vs Request Rate (RPS)**, with one line per engine. The graph visualizes which engine sustains higher SLO-compliant throughput as load increases.

The script supports two modes:
- **Measurement mode**: hits both endpoints and measures both directly.
- **Synthesis mode** (`--synthesize-friendli`): measures only vLLM directly; derives the Friendli line from vLLM measurements using transformations modeled from FriendliAI's published benchmarks. This mode exists because the author cannot deploy Friendli locally; it must be transparent in code, citations, and the graph itself.

The deliverable optimizes for **measurement correctness** and **reproducibility**, not absolute numbers. Per the task issuer: "only the trend matters, not the absolute value."

## 2. Design References

This implementation deliberately follows established patterns from the LLM inference benchmarking community. README must cite these.

- **vLLM `benchmark_serving.py`** (https://github.com/vllm-project/vllm/blob/main/benchmarks/) — primary structural reference. The backend plug-in pattern (`RequestFuncInput` / `RequestFuncOutput` / `ASYNC_REQUEST_FUNCS`), the open-loop Poisson dispatcher, the synthetic prompt generation, and the output format (JSON summary + per-request raw) are directly adapted from this code.
- **NVIDIA GenAI-Perf** (https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html) — authoritative source for metric definitions. ITL = (E2E − TTFT) / (output_tokens − 1), with first token excluded. Empty-chunk skipping convention.
- **Anyscale LLMPerf** (https://github.com/ray-project/llmperf) — referenced in README footnote for awareness of the historical TTFT-in-ITL discrepancy.
- **DistServe (OSDI '24)** — source for goodput as a per-request SLO-compliance metric.
- **FriendliAI public benchmarks** — sources for synthetic Friendli line modeling (TCache blog, AWQ blog, Mixtral benchmark, Qwen3 benchmark). Cited in `bench/synthetic.py` docstrings.

## 3. Repository Layout

```
friendli-vllm-bench/
├── README.md
├── SPEC.md                        # this file (committed for transparency)
├── pyproject.toml                 # uv-managed
├── uv.lock
├── .python-version                # 3.11
├── .gitignore
├── benchmark.py                   # entry point
├── bench/
│   ├── __init__.py
│   ├── backend.py                 # RequestFuncInput/Output + backend registry
│   ├── client.py                  # async streaming client (registered as a backend)
│   ├── workload.py                # prompt sampler + Poisson scheduler
│   ├── metrics.py                 # SLO judgment + percentile + goodput aggregation
│   ├── synthetic.py               # vLLM measurements → modeled Friendli timings
│   ├── runner.py                  # orchestrates engine × rate sweep
│   ├── plot.py                    # single-graph generator
│   └── utils.py                   # TimeCollector, tabulate helpers
├── prompts/
│   └── prompts.jsonl              # 200 fixed prompts, varied input lengths
├── scripts/
│   └── build_prompts.py           # generates prompts.jsonl deterministically
├── results/
│   ├── raw/                       # per-request JSONL, gitignored except validation
│   ├── summary.json               # aggregated metrics
│   ├── summary.csv                # one row per (engine, rate) for spreadsheet use
│   └── goodput_comparison.png
└── tests/
    ├── test_metrics.py
    ├── test_synthetic.py
    ├── test_workload.py
    └── test_backend.py
```

## 4. Dependencies (`pyproject.toml`)

```toml
[project]
name = "friendli-vllm-bench"
version = "0.1.0"
requires-python = "==3.11.*"
dependencies = [
    "httpx==0.27.2",
    "numpy==2.1.3",
    "matplotlib==3.9.2",
    "transformers==4.46.0",
    "tqdm==4.66.5",
    "pydantic==2.9.2",
    "tabulate==0.9.0",
]

[dependency-groups]
dev = [
    "pytest==8.3.3",
    "ruff==0.7.0",
]
```

Lock with `uv lock` and commit `uv.lock`.

## 5. CLI Contract (`benchmark.py`)

Single entry point. One invocation runs the full sweep and produces the graph.

```
uv run python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url http://localhost:8001 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --rates 1,2,4,8,12,16,24,32 \
    --duration-per-rate 60 \
    --warmup-requests 20 \
    --max-output-tokens 256 \
    --slo-ttft-ms 500 \
    --slo-tpot-ms 50 \
    --seed 42 \
    --output-dir results/
```

### Required flags
- `--vllm-url`: HTTP URL of vLLM endpoint
- `--friendli-url`: HTTP URL of Friendli endpoint (ignored in synthesis mode)
- `--model`: model identifier passed in API requests
- `--tokenizer`: HuggingFace tokenizer ID for output token counting

### Optional flags with defaults
- `--rates`: comma-separated RPS values (default `1,2,4,8,12,16,24,32`)
- `--duration-per-rate`: seconds per rate point (default `60`)
- `--warmup-requests`: pre-measurement requests (default `20`)
- `--max-output-tokens`: hard cap on output length (default `256`)
- `--slo-ttft-ms`: per-request TTFT SLO threshold in ms (default `500`, source: Rafay Token Factory)
- `--slo-tpot-ms`: per-request TPOT SLO threshold in ms (default `50`, source: Rafay Token Factory)
- `--seed`: master seed for prompt order, Poisson arrivals, sampling (default `42`)
- `--output-dir`: results directory (default `results/`)
- `--synthesize-friendli`: if set, do not call Friendli URL; derive Friendli line from vLLM measurements via `bench/synthetic.py`. Adds `(modeled)` annotation to the Friendli line in the graph.
- `--cooldown-seconds`: pause between rates (default `5`)
- `--vllm-backend`, `--friendli-backend`: backend name for the registry (default `openai-chat` for both, allows future extension).

### Validation contract
The script must work with `--vllm-url` and `--friendli-url` pointing to the same address. README §10 references this for end-to-end validation.

## 6. Module Contracts

### 6.1 `bench/backend.py` — Backend abstraction

Adapted from vLLM `benchmark_serving.py`'s `RequestFuncInput` / `RequestFuncOutput` / `ASYNC_REQUEST_FUNCS` pattern. This decouples engine-specific HTTP handling from orchestration logic. Both vLLM and Friendli are OpenAI-compatible, so they share the same backend function but the abstraction allows future engines.

```python
from dataclasses import dataclass, field

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
    arrival_time: float = 0.0          # scheduled arrival (perf_counter)
    send_time: float = 0.0             # actual send time
    first_token_time: float | None = None
    last_token_time: float | None = None
    itl: list[float] = field(default_factory=list)   # per-token gaps after first token
    success: bool = False
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.send_time

    @property
    def tpot(self) -> float | None:
        # GenAI-Perf convention: ITL/TPOT excludes first token
        # = (E2E - TTFT) / (output_tokens - 1)
        # Source: https://docs.nvidia.com/nim/benchmarking/llm/latest/overview.html
        if (self.first_token_time is None or self.last_token_time is None
            or self.output_tokens <= 1):
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


# Backend registry — populated by client.py via decorator
ASYNC_REQUEST_FUNCS: dict[str, callable] = {}

def register_backend(name: str):
    def decorator(fn):
        ASYNC_REQUEST_FUNCS[name] = fn
        return fn
    return decorator
```

This pattern is directly adapted from vLLM's `benchmarks/backend_request_func.py`. README must cite it.

### 6.2 `bench/client.py` — Streaming HTTP client

Implements one async function per supported backend, registered into `ASYNC_REQUEST_FUNCS`. Both vLLM and Friendli use `openai-chat` since both expose OpenAI-compatible chat completions.

#### Streaming parsing rules (must be exact)

1. Use `httpx.AsyncClient` with `timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10)` and `limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)`. The client should be created once at runner level and passed in, not per-request.
2. Parse SSE lines starting with `data: `. Skip empty lines. Terminate on `data: [DONE]`.
3. For each parsed chunk, extract `choices[0].delta.content`. **If empty string or missing, do not update timestamps.** Matches GenAI-Perf and LLMPerf conventions for ignoring empty first chunks.
4. On the first non-empty content: set `first_token_time = time.perf_counter()`. Initialize `last_chunk_time = first_token_time`.
5. On every subsequent non-empty content: `now = time.perf_counter()`; append `now - last_chunk_time` to `itl` list; update `last_chunk_time = now`; `last_token_time = now`.
6. Accumulate all delta content into a single string. After streaming ends, count `output_tokens` using the tokenizer applied to the accumulated string. **Do not** rely on token counts in API response.
7. On exception (timeout, connection error, parse error), set `success=False`, populate `error`, leave timestamps as-is, return.

#### Sampling parameters (request payload)

```python
{
    "model": input.model,
    "messages": [{"role": "user", "content": input.prompt}],
    "stream": True,
    "max_tokens": input.max_tokens,
    "temperature": input.temperature,
    "top_p": input.top_p,
    "seed": input.seed,
    "ignore_eos": input.ignore_eos,
}
```

#### Engine compatibility note

`ignore_eos` is supported by vLLM, not officially by Friendli. The Friendli OpenAI-compatible API may ignore the field silently. Document in README: Friendli outputs may terminate slightly earlier on EOS; this is acceptable for goodput because TTFT and TPOT are length-independent. In synthesis mode, Friendli timings are derived from vLLM, so EOS handling is moot.

### 6.3 `bench/workload.py` — Prompt sampling and Poisson scheduling

#### Prompt source

`prompts/prompts.jsonl` — 200 prompts, each `{"id": int, "text": str, "approx_tokens": int}`. Mix of short (~50 tokens), medium (~200), long (~500). Built deterministically by `scripts/build_prompts.py` using a fixed seed; both files committed.

`scripts/build_prompts.py` produces synthetic prompts (e.g., "Explain the concept of X to a Y in detail" with X drawn from a controlled topic list). This avoids ShareGPT licensing/download dependency and is fully reproducible — the rationale that `benchmark_serving.py` validates ("synthetic with a fixed seed is better — no external dataset download needed").

#### Poisson generator

```python
import numpy as np

def generate_arrivals(rate: float, duration: float, seed: int) -> list[float]:
    """
    Returns sorted arrival times (seconds from t=0) for a Poisson process
    with mean `rate` requests/sec over `duration` seconds.
    Inter-arrival times are exponential with mean 1/rate.

    Adapted from vLLM benchmark_serving.py's get_request() async generator.
    """
    rng = np.random.default_rng(seed)
    arrivals = []
    t = 0.0
    while t < duration:
        t += rng.exponential(1.0 / rate)
        if t < duration:
            arrivals.append(t)
    return arrivals
```

#### Open-loop dispatcher

```python
async def run_open_loop(
    backend_fn,                  # async (RequestFuncInput) -> RequestFuncOutput
    prompts: Iterator[tuple[int, str, int]],  # (id, text, approx_tokens)
    rate: float,
    duration: float,
    seed: int,
    base_input_template: dict,   # api_url, model, max_tokens, etc.
    engine_name: str,
) -> list[RequestFuncOutput]:
    arrivals = generate_arrivals(rate, duration, seed)
    start = time.perf_counter()
    tasks = []
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
        tasks.append(asyncio.create_task(
            backend_fn(req, arrival_time=start + arrival_t, engine=engine_name)
        ))
    return await asyncio.gather(*tasks, return_exceptions=False)
```

**Critical**: do NOT gate dispatch on completions. Open-loop means new requests arrive regardless of in-flight count. If the server saturates, tasks pile up — that is the point.

#### Prompt iteration

Cycle through `prompts.jsonl` in fixed seeded order. Same sequence for all rates and engines so vLLM and Friendli (or modeled Friendli) operate on identical inputs.

### 6.4 `bench/metrics.py` — SLO judgment and aggregation

```python
@dataclass
class SLO:
    ttft_ms: float           # per-request TTFT threshold
    tpot_ms: float           # per-request TPOT threshold

def is_slo_compliant(r: RequestFuncOutput, slo: SLO) -> bool:
    """
    Per-request SLO check using per-request latencies.
    Following DistServe convention (OSDI'24): goodput counts requests
    whose individual latencies satisfy thresholds.
    """
    if not r.success or r.ttft is None or r.tpot is None:
        return False
    if r.ttft * 1000 > slo.ttft_ms:
        return False
    if r.tpot * 1000 > slo.tpot_ms:
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
    goodput_rps: float                 # slo_compliant_count / window
    raw_throughput_rps: float          # successful_count / window
    output_tokens_per_sec: float       # sum(output_tokens) / window
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    itl_jitter_p95_ms: float           # variability of inter-token gaps
```

Measurement window: from earliest `send_time` of post-warmup requests to latest `last_token_time` of post-warmup requests. Warmup excluded.

Use `numpy.percentile` with `method="linear"` for percentile calculation.

`itl_jitter_p95_ms`: stddev of each request's `itl` list, then p95 across requests. Captures streaming smoothness, an axis vLLM benchmarks track.

### 6.5 `bench/synthetic.py` — vLLM measurements → modeled Friendli timings

This module exists to produce a Friendli line without a deployed Friendli endpoint. All transformations are modeled from FriendliAI's published benchmarks. The module must be self-documenting with citations in docstrings.

```python
from dataclasses import replace

def ttft_multiplier(rate: float, max_rate: float) -> float:
    """
    Load-dependent TTFT advantage factor for Friendli over vLLM.

    Modeling: at low load (~0 rate), Friendli ≈ 2x faster; at the maximum
    swept rate, ≈ 11x faster. This is the lower bound of FriendliAI's
    reported range of 11.3-23x at 5N load on Llama-2-70B AWQ across
    4 A100 GPUs.

    Conservative choice: lower bound applied because:
    (a) reference setup is 70B/4-GPU/AWQ; smaller-model single-GPU
        deployments would show smaller absolute gains;
    (b) avoids overstating Friendli's advantage.

    Source: https://friendli.ai/blog/friendli-tcache (TCache benchmark, 1N-5N load)

    Linear interpolation between (rate=0, factor=2.0) and
    (rate=max_rate, factor=11.0). Capped at 11.0 beyond max_rate.

    Returns the divisor: friendli_ttft = vllm_ttft / multiplier.
    """
    if rate <= 0:
        return 2.0
    progress = min(rate / max_rate, 1.0)
    return 2.0 + 9.0 * progress


def tpot_multiplier(rate: float, max_rate: float) -> float:
    """
    Load-dependent TPOT advantage factor for Friendli over vLLM.

    Modeling: at low load, Friendli ≈ 1.3x faster TPOT; at high load,
    ≈ 2.5x faster. Anchored to FriendliAI's reported ~2x TPOT improvement
    on Mixtral-7x8B AWQ on a single A100, with load-dependent shape
    inspired by Orca's iteration batching analysis (low-load parity,
    high-load gains).

    Sources:
      - https://friendli.ai/blog/comparing-friendli-engine-vllm (Mixtral AWQ, ~2x)
      - https://www.usenix.org/system/files/osdi22-yu.pdf (Orca, OSDI'22)

    Linear interpolation between (rate=0, factor=1.3) and
    (rate=max_rate, factor=2.5).

    Returns the divisor: friendli_tpot = vllm_tpot / multiplier.
    """
    if rate <= 0:
        return 1.3
    progress = min(rate / max_rate, 1.0)
    return 1.3 + 1.2 * progress


def synthesize_friendli_results(
    vllm_results: list[RequestFuncOutput],
    target_rate: float,
    max_rate: float,
) -> list[RequestFuncOutput]:
    """
    Produce modeled Friendli RequestFuncOutput list from measured vLLM results.

    For each vLLM result, scale TTFT and TPOT by the load-dependent
    multipliers above and reconstruct first_token_time and last_token_time
    consistent with the new latencies. Failed vLLM requests stay failed
    (modeling does not invent successes from errors).

    Engine field is set to 'friendli_modeled'. The 'itl' list is also
    scaled by tpot_multiplier so per-token jitter computations remain
    self-consistent.

    NOT a measurement. See README §Data Sources.
    """
    f_ttft = ttft_multiplier(target_rate, max_rate)
    f_tpot = tpot_multiplier(target_rate, max_rate)
    out = []
    for r in vllm_results:
        if not r.success or r.ttft is None or r.tpot is None:
            out.append(replace(r, engine="friendli_modeled"))
            continue
        new_ttft = r.ttft / f_ttft
        new_tpot = r.tpot / f_tpot
        new_first = r.send_time + new_ttft
        if r.output_tokens > 1:
            new_last = new_first + new_tpot * (r.output_tokens - 1)
        else:
            new_last = new_first
        new_itl = [v / f_tpot for v in r.itl]
        out.append(replace(
            r,
            engine="friendli_modeled",
            first_token_time=new_first,
            last_token_time=new_last,
            itl=new_itl,
        ))
    return out
```

#### Constraints
- Module must not invent successes from failures.
- Must not change `output_tokens`, `prompt_tokens`, or `prompt_id`.
- Multiplier functions must be pure and deterministic.
- Each multiplier function's docstring must contain its source URL and the conservative-choice justification.
- `max_rate` is the highest rate in the sweep; used to normalize the load-dependent shape.

### 6.6 `bench/utils.py` — Shared utilities

```python
import time
from contextlib import contextmanager

class TimeCollector:
    """
    Adapted from vLLM benchmarks/benchmark_utils.py.
    Collects measurements; supports context manager for timed blocks.
    """
    def __init__(self):
        self.measurements: list[float] = []

    def add(self, value: float) -> None:
        self.measurements.append(value)

    @contextmanager
    def time_block(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(time.perf_counter() - start)

    def avg(self) -> float:
        return sum(self.measurements) / max(len(self.measurements), 1)

    def max(self) -> float:
        return max(self.measurements) if self.measurements else 0.0
```

Plus helpers for `nvidia-smi` output capture, git SHA capture, JSONL dump/load.

### 6.7 `bench/runner.py` — Sweep orchestration

```python
async def run_sweep(
    engines: dict[str, str | None],   # {"vllm": url, "friendli": url_or_None}
    rates: list[float],
    duration_per_rate: float,
    warmup_requests: int,
    cooldown_seconds: float,
    synthesize_friendli: bool,
    config: Config,
) -> list[RunSummary]:
    ...
```

#### Order of operations

1. For each engine that will be measured directly: closed-loop warmup of `warmup_requests` requests sequentially. Discarded.
2. Iterate rates in ascending order. For each rate:
   a. **Measurement mode**: alternate engines per rate (`vllm@1, friendli@1, vllm@2, friendli@2, ...`) to balance thermal/system drift.
   b. **Synthesis mode**: only run vLLM. After each rate's vLLM measurement, immediately call `synthesize_friendli_results(...)` for that rate's data.
   c. Dump raw `RequestFuncOutput` list to `results/raw/<engine>_<rate>rps_<timestamp>.jsonl`.
   d. Compute `RunSummary` and append.
   e. Cooldown for `cooldown_seconds` to let server quiesce.
3. Print tabulated summary to console using `tabulate` (vLLM convention).
4. Write `results/summary.json` (full structured summaries) and `results/summary.csv` (one row per (engine, rate)).
5. Write `results/run_metadata.json` with: git SHA, timestamp, hardware info from `nvidia-smi` if available, library versions, full CLI args.

#### Critical
- Warmup mandatory before measurement.
- Cooldowns prevent leakage of queued requests into the next rate.
- Engine alternation balances drift in measurement mode only; synthesis mode runs vLLM in plain ascending order.
- Console output uses `tabulate` for readability — adapted from vLLM convention.

### 6.8 `bench/plot.py` — Single graph generator

```python
def plot_goodput(
    summaries: list[RunSummary],
    slo: SLO,
    output_path: Path,
    synthesized: bool,
    git_sha: str,
) -> None:
    ...
```

#### Graph specification

- One figure, one axes. `fig, ax = plt.subplots(figsize=(8, 5))`.
- x-axis: target request rate (RPS). Linear scale.
- y-axis: goodput (RPS). Linear scale.
- Two lines:
  - vLLM: `color="#1f77b4"`, `linestyle="--"`, `marker="o"`, label `"vLLM (measured)"`.
  - Friendli: `color="#ff7f0e"`, `linestyle="-"`, `marker="s"`. Label `"Friendli (modeled from public benchmarks)"` if synthesized, else `"Friendli (measured)"`.
- Reference line: `y = x` in light gray dashed (`color="#888"`, `alpha=0.5`), label `"Perfect goodput"`.
- Saturation annotation: for each engine line, identify the rate at which goodput diverges from `y=x` by more than 10%. Vertical dotted line + text `"saturates ≈ X RPS"`.
- Title: `"Goodput under Open-Loop Poisson Load"`.
- Subtitle (set via `ax.text` below title): `"SLO: per-request TTFT ≤ {slo.ttft_ms}ms AND TPOT ≤ {slo.tpot_ms}ms"`.
- Footer (small light gray text in lower-left of figure):
  - If synthesized: `"vLLM: measured. Friendli: modeled from FriendliAI public benchmarks (see README §Data Sources). · git: {sha[:7]} · {timestamp}"`
  - Else: `"Both engines measured. · git: {sha[:7]} · {timestamp}"`
- Legend in upper-left.
- Grid: `ax.grid(True, linestyle=":", alpha=0.4)`.
- Save as PNG at 300 DPI and PDF.

The footer makes the graph self-documenting — anyone seeing the PNG knows what it represents.

## 7. Reproducibility Requirements

Verifiable from the repo:

1. Pinned dependencies via `uv.lock`.
2. Fixed seed propagated everywhere: prompt order, Poisson arrivals, sampling.
3. Deterministic sampling: `temperature=0.0`, `seed=42`.
4. Tokenizer pinning via `--tokenizer`.
5. Hardware/env capture: `nvidia-smi` (if available), `uname -a`, Python version, library versions, full CLI args to `results/run_metadata.json`.
6. Git SHA capture in `run_metadata.json` and graph footer.
7. Raw data dump: every request's full timing to JSONL so summaries can be recomputed.
8. `prompts.jsonl` and `scripts/build_prompts.py` both committed.

## 8. Test Coverage

### `tests/test_metrics.py`
- `is_slo_compliant`: TTFT/TPOT inputs map to expected boolean (within, exactly, beyond threshold; failed; missing timestamps).
- Percentile computation: 100-value list, p50/p95/p99 match `numpy.percentile`.
- `RequestFuncOutput.tpot` property: synthetic timestamps and token count, verify equals expected.
- Edge case: `output_tokens <= 1` returns `None` for tpot.

### `tests/test_workload.py`
- Poisson generator: `rate=10, duration=600, seed=42` → mean inter-arrival within 2% of 0.1, count within 2% of 6000.
- Determinism: same seed → identical arrivals.

### `tests/test_backend.py`
- Mock SSE stream with empty first chunk: `first_token_time` set to first non-empty chunk's timestamp, not the empty one.
- Mock SSE stream with 5 token chunks: `itl` list has length 4 (gaps between 5 tokens).
- Failure path: HTTP 500 → `success=False`, `error` populated.

### `tests/test_synthetic.py`
- `ttft_multiplier(0, max_rate)` returns 2.0.
- `ttft_multiplier(max_rate, max_rate)` returns 11.0.
- `ttft_multiplier(2 * max_rate, max_rate)` returns 11.0 (caps).
- `tpot_multiplier`: 1.3 → 2.5 across the same range.
- `synthesize_friendli_results` preserves `prompt_id`, `prompt_tokens`, `output_tokens`.
- Reduces TTFT and TPOT by expected factors.
- Failed input → failed output (engine field updated, no invented success).
- `itl` list scaled correctly.

## 9. README Structure

Order, target ~600 lines. Terse. No marketing voice.

1. **What this is** — goal, deliverable graph, assumed deployment.
2. **Quick start** — clone, `uv sync`, command, view graph. 5 commands.
3. **The graph** — embed PNG, explain x/y/lines/diagonal/saturation, state the SLO.
4. **Why these metrics** — goodput as headline (cite DistServe, BentoML), TTFT and TPOT as SLO components, why open-loop Poisson over closed-loop (cite tail-latency masking), why fixed output length matters (`ignore_eos`).
5. **Why these SLOs** — cite Rafay Token Factory (p95 TTFT < 500ms for chatbots), AWS conversational AI guidance, BentoML responsiveness recommendations. CLI-configurable.
6. **How fairness is enforced** — controlled variables: model, tokenizer, prompts (fixed file, same seed, same order), sampling params (temp 0, seed, top_p, max_tokens, ignore_eos), arrival schedule (same Poisson seed per rate), warmup, alternating engine order, cooldowns.
7. **Measurement correctness notes** — ITL definition (GenAI-Perf, first token excluded), tokenizer-based output counting, empty-chunk skipping, perf_counter timing. Footnote on LLMPerf's historical TTFT-in-ITL discrepancy.
8. **Design references** — list per §2 of this spec. README acknowledges these as the foundation.
9. **Data Sources** — explicit separation:
   > The graph contains two lines computed by identical aggregation logic but with different raw timing data.
   >
   > **vLLM line**: measured directly using the benchmark client described here. Hardware, model, date logged in `results/run_metadata.json`. Raw per-request timings under `results/raw/vllm_*.jsonl`.
   >
   > **Friendli line**: derived from vLLM measurements via `bench/synthetic.py`, applying load-dependent transformations modeled from FriendliAI's published benchmarks:
   > - TTFT divisor `f_ttft(rate)`: 2.0 at low load → 11.0 at max sweep rate. Source: TCache blog (11.3–23× reported range; conservative lower bound applied because reference setup is 70B/4×A100/AWQ).
   > - TPOT divisor `f_tpot(rate)`: 1.3 at low load → 2.5 at max sweep rate. Source: FriendliAI Mixtral AWQ benchmark (~2× reported); load-dependent shape from Orca paper's high-concurrency analysis.
   >
   > The synthesis code is committed and reproducible from the same vLLM raw data. The Friendli line is a **plausibility model anchored to public benchmarks**, not a direct measurement. To obtain real measurements, replace `--friendli-url` with a deployed Friendli endpoint and remove `--synthesize-friendli`.
10. **Reproducibility** — seeds, pinned deps, raw data dump, env capture, git SHA in graph.
11. **Limitations** — single-node, no multi-tenancy effects, SLO illustrative (chatbot use case), Friendli line is modeled in synthesis mode, vLLM measurement is on a small reference setup (laptop GPU, small model) and may not reflect production GPU behavior.
12. **CLI reference** — full `--help` output.
13. **Reference run details** — hardware, model, prompt source, date, exact command, observations.

## 10. Implementation Order

Build in this order. After each step, run tests, show diff, wait for "continue".

1. `pyproject.toml`, `.python-version`, `.gitignore`, package skeleton.
2. `scripts/build_prompts.py` and `prompts/prompts.jsonl` generation.
3. `bench/backend.py` — `RequestFuncInput`, `RequestFuncOutput`, registry. + `tests/test_metrics.py` for the property methods.
4. `bench/utils.py` — `TimeCollector`, helpers.
5. `bench/workload.py` — Poisson generator + `tests/test_workload.py`.
6. `bench/metrics.py` — SLO judgment, aggregation. Extend `tests/test_metrics.py`.
7. `bench/client.py` — actual streaming HTTP client registered as `openai-chat`. + `tests/test_backend.py` with mock SSE. Hand-test against a real vLLM or OpenAI-compatible endpoint before continuing.
8. `bench/synthetic.py` + `tests/test_synthetic.py`.
9. `bench/runner.py` — start with one engine + one rate, expand to full sweep.
10. `bench/plot.py` — needs real summary data first.
11. `benchmark.py` — wire everything into CLI.
12. README.

## 11. Validation Run (mandatory before submission)

Run the full sweep with both URLs pointing to the same vLLM instance on a small model (`Qwen/Qwen2.5-0.5B-Instruct`). Run twice:

**Validation A (measurement mode, both URLs same vLLM)**:
```
uv run python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url http://localhost:8000 \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --rates 1,2,4,8 \
    --duration-per-rate 30
```
Expected: two lines roughly overlap. Confirms measurement pipeline is fair.

**Validation B (synthesis mode)**:
```
uv run python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url unused \
    --synthesize-friendli \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --rates 1,2,4,8 \
    --duration-per-rate 30
```
Expected: Friendli (modeled) line above vLLM line, gap widening with rate. Confirms synthesis pipeline.

Commit Validation B output as the reference graph. Document in README §13 which validation produced which artifact.
