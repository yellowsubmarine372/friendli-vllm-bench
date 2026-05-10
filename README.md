# friendli-vllm-bench

A reproducible benchmark comparing **vLLM** and **Friendli Engine** as LLM inference backends. The deliverable is a single graph — **goodput vs request rate under open-loop Poisson load** — that visualizes the engine that sustains higher SLO-compliant throughput as load increases.

> 🚧 **Under construction.** Implementation is in progress against the design specification in [SPEC.md](./SPEC.md). The full README, reference graph, and run instructions will land here once the build is complete.

## Design overview

- **Headline metric**: goodput (requests/sec satisfying per-request TTFT and TPOT SLOs), following the convention from DistServe (OSDI '24).
- **Workload**: open-loop Poisson arrivals over an OpenAI-compatible chat completions endpoint, configurable request-rate sweep.
- **Measurement methodology**: adapted from vLLM's `benchmark_serving.py` (backend plug-in pattern, async dispatcher, raw + summary outputs) and NVIDIA GenAI-Perf's metric definitions (ITL excluding the first token, empty-chunk skipping, tokenizer-based output counting).
- **Friendli line**: in synthesis mode, derived from vLLM measurements via load-dependent transformations modeled from FriendliAI's published benchmarks (see [SPEC.md §6.5](./SPEC.md)). Clearly labeled in the graph footer as modeled, not measured.

See [SPEC.md](./SPEC.md) for the full design.
