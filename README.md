# friendliAI-vllm-bench

A reproducible benchmark comparing **vLLM** and **Friendli Engine** as
LLM inference backends. The deliverable is a single graph:
**goodput vs request rate under open-loop Poisson load**. That
visualizes which engine sustains higher SLO-compliant throughput as
load increases.

---

## 1. What this is

**Goal.** Compare vLLM and Friendli at the same model, same prompts,
same arrival schedule, same SLO. Measure where each engine starts
dropping SLO-compliant throughput.

**Deliverable.** One matplotlib graph: x-axis = target request rate
(RPS); y-axis = goodput (RPS, requests whose per-request TTFT *and*
TPOT both satisfy thresholds). One line per engine. Saturation rate
annotated for each.

**Two operating modes.**

| Mode | What runs | What's in the graph |
|---|---|---|
| Measurement (default) | hits both endpoints | both lines from direct measurement |
| Synthesis (`--synthesize-friendli`) | hits only vLLM | vLLM line measured; Friendli line **modeled** from vLLM via [bench/synthetic.py](bench/synthetic.py), with multipliers anchored to FriendliAI's public benchmarks |

The synthesis mode exists because the author cannot deploy Friendli
locally. The headline graph in this repo was produced in synthesis mode.

---

## 2. Quick start

```bash
git clone <this-repo>
cd friendli-vllm-bench
uv sync --all-groups
```

To reproduce the headline graph (synthesis mode, calibrated SLOs):

```bash
uv run python benchmark.py \
    --vllm-url http://localhost:8000 \
    --friendli-url unused \
    --synthesize-friendli \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --rates 1,2,4,8,12,16,24 \
    --duration-per-rate 30 \
    --slo-ttft-ms 25 \
    --slo-tpot-ms 5
```

For measurement mode against a real Friendli endpoint, drop
`--synthesize-friendli`, point `--friendli-url` at the actual endpoint,
and use the production-realistic CLI defaults (`--slo-ttft-ms 500`,
`--slo-tpot-ms 50`).

The reference vLLM endpoint was deployed via:

```
docker run -d --name vllm-bench \
    --gpus all --ipc=host -p 8000:8000 \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --max-model-len 2048 --gpu-memory-utilization 0.85
```

**Outputs.**

* `results/goodput_comparison.png` and `.pdf` — the graph
* `results/summary.json` and `.csv` — per-(engine, rate) metrics
* `results/raw/<engine>_<rate>rps_<ts>.jsonl` — per-request timings
* `results/run_metadata.json` — git SHA, hardware, library versions, full CLI args

`--seed` (default 42) propagates to prompt order, Poisson arrivals,
and the model's sampling parameter, so reruns produce byte-identical
inputs.

---

## 3. The graph

![goodput vs request rate](results/goodput_comparison.png)

**x-axis.** Target request rate in RPS, linear. Arrivals are Poisson
with mean = target rate; the dispatcher fires arrivals on schedule
regardless of in-flight requests (open-loop). The x-axis is load
**offered** to the server.

**y-axis.** Goodput in RPS, the number of post-warmup requests
whose **individual** TTFT and TPOT both satisfy the SLO, divided by
the measurement window. A request that exceeded TTFT or TPOT counts
as 0; a failed request counts as 0.

**Lines.**

* Gray dashed `y = x`: perfect goodput.
* Blue dashed: `vLLM (measured)`.
* Orange solid: `Friendli (modeled from public benchmarks)`.

**Saturation marker.** A vertical dotted line at the first rate where
goodput diverges from `y = x` by more than 10%. vLLM is annotated
`vLLM saturates ≈ 12 RPS`. Friendli has no marker because the modeled
line never diverges by >10% in this range.

### Why the vLLM line hooks down instead of plateauing

vLLM does not stop accepting requests when it saturates. Its raw
throughput keeps climbing, but goodput collapses because TTFT and
TPOT distributions shift past the SLO thresholds:

| target rate | vLLM raw_throughput | vLLM goodput | vLLM TTFT p95 | vLLM TPOT p95 | slo-compliant fraction |
|---|---|---|---|---|---|
| 4 | 4.33 | 4.00 | 25.7 ms | 4.4 ms | 121/131 |
| 8 | 7.99 | 7.46 | 25.8 ms | 4.8 ms | 228/244 |
| 12 | 11.98 | 7.38 | 25.0 ms | 5.6 ms | 228/370 |
| 16 | 15.39 | 2.25 | 28.2 ms | 6.5 ms | 70/478 |
| 24 | 22.50 | **0.00** | 45.2 ms | 10.9 ms | 0/702 |

At rate 24, vLLM completes every request, but **none** meet TPOT ≤ 5
ms; goodput is zero by definition. This is the goodput-as-production-
metric framing from **DistServe** (OSDI '24): a request that arrives
but is too slow to use is a regression, not a success. A procurement
decision based on raw throughput would call both engines equivalent
at high load. A goodput decision correctly ranks the engine that
holds latency under load.

---

## 4. Why these metrics

**Goodput.** A naive `requests/sec served` number hides quality
regressions: an engine that returns slow streams can have high
"throughput" no one would deploy. We use a **DistServe-style SLO
definition**: a request is compliant only if **both** its TTFT and
TPOT satisfy thresholds. Goodput is the rate at which compliant
requests complete. DistServe does not pseudocode the explicit
AND-rule, but the framing follows from its joint-latency requirements.

**TTFT and TPOT.** Time To First Token captures user-perceived
responsiveness; Time Per Output Token (= ITL excluding the first
token) isolates the decoding loop from prefill cost. Definitions
follow NVIDIA GenAI-Perf ([metrics.html](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)):
`TPOT = (e2e_latency − TTFT) / (output_tokens − 1)`. Empty/role-only
initial chunks do not advance TTFT — this matches GenAI-Perf and
differs from vLLM's `benchmark_serving.py` convention.

**Open-loop Poisson arrivals.** Closed-loop benchmarks gate on
completions and mask tail latency. Open-loop arrivals fire on a
fixed schedule independent of server completions — exactly what a
real user-facing service experiences. When the server saturates,
queues build, tails spike, and goodput diverges from `y = x`. That
divergence is the signal the headline graph is built to surface.

**SLO calibration.** CLI defaults are production-realistic chatbot
thresholds (TTFT 500 ms / TPOT 50 ms; sources: Rafay Token Factory,
AWS conversational-AI guidance). The headline graph uses **calibrated
tight SLOs (25 ms / 5 ms)** because the validation rig is a 0.5 B
model on a laptop GPU — at every offered rate, vLLM's raw TTFT p95
stays under 50 ms and TPOT p95 under 11 ms, so production defaults
would never be violated and the graph would be flat. Tightening to
sit just above vLLM's no-load operating point makes saturation
visible in the rate range a laptop GPU can sustain. Per the task
issuer: "only the trend matters, not the absolute value."

---

## 5. Data sources

The graph contains two lines computed by **identical aggregation
logic** but with different raw timing data.

**vLLM line**: measured directly. Hardware, model, and date logged
in `results/run_metadata.json`. Raw per-request timings under
`results/raw/vllm_*.jsonl`.

**Friendli line**: derived from vLLM measurements via
[bench/synthetic.py](bench/synthetic.py), applying load-dependent
transformations modeled from FriendliAI's published benchmarks:

* TTFT divisor `f_ttft(rate)`: 2.0 at low load → 11.0 at max sweep
  rate. Source: [TCache blog](https://friendli.ai/blog/friendli-tcache)
  (11.3–23× reported range; conservative lower bound applied
  because the reference setup is 70B/4×A100/AWQ).
* TPOT divisor `f_tpot(rate)`: 1.3 at low load → 2.5 at max sweep
  rate. Source: [FriendliAI Mixtral AWQ benchmark](https://friendli.ai/blog/comparing-friendli-engine-vllm)
  (~2× reported); load-dependent shape from
  [Orca (OSDI '22)](https://www.usenix.org/system/files/osdi22-yu.pdf)'s
  high-concurrency analysis.

The synthesis code is committed and reproducible from the same vLLM
raw data. The Friendli line is a **plausibility model anchored to
public benchmarks**, not a direct measurement. To obtain real
measurements, point `--friendli-url` at a deployed Friendli endpoint
and drop `--synthesize-friendli`.

---

## 6. Limitations

**Asymmetric evidence.** The two lines are not the same kind of
evidence. The vLLM line is direct measurement of an actual server
under load; the Friendli line is mathematically derived from the
same vLLM measurements via the multipliers above. Treating the two
as symmetric is the most likely misinterpretation of this graph.
The legend label `Friendli (modeled from public benchmarks)` and the
footer disclosure are both intended to keep this distinction visible
when the PNG is seen out of context.

**The modeled Friendli line shows no saturation.** The synthesis
pipeline divides vLLM's TTFT and TPOT by load-dependent factors but
does not include a saturation model. As long as the multipliers
shrink modeled TTFT/TPOT below the SLO thresholds, the modeled line
tracks `y = x`. Every real system saturates eventually; this graph
does not capture where Friendli's saturation point would be. The
line should be read as "Friendli has more headroom in this regime,"
not "Friendli has infinite capacity." A proper Friendli measurement
would replace the line and would show its own divergence from `y = x`
at some higher l0d.

---

## License

Implementation is original work. External design patterns are cited
inline in the relevant module's docstrings.
