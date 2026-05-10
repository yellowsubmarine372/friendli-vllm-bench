"""Modeled Friendli timings derived from measured vLLM timings.

This module exists because the author cannot deploy Friendli locally.
The transformations here are modeled from FriendliAI's published
benchmarks; they are NOT a measurement. README §Data Sources says so
explicitly, the graph footer says so, and ``synthesize_friendli_results``
labels every output with ``engine="friendli_modeled"``.

Each multiplier function carries its source URL and the conservative-choice
justification in its docstring per SPEC §6.5.
"""

from __future__ import annotations

from dataclasses import replace

from bench.backend import RequestFuncOutput


def ttft_multiplier(rate: float, max_rate: float) -> float:
    """Load-dependent TTFT advantage factor for Friendli over vLLM.

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
    """Load-dependent TPOT advantage factor for Friendli over vLLM.

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
    """Produce modeled Friendli ``RequestFuncOutput`` list from measured vLLM results.

    For each vLLM result, scale TTFT and TPOT by the load-dependent
    multipliers above and reconstruct ``first_token_time`` and
    ``last_token_time`` consistent with the new latencies. Failed vLLM
    requests stay failed — modeling does not invent successes from errors.

    Engine field is set to ``"friendli_modeled"``. The ``itl`` list is also
    scaled by ``tpot_multiplier`` so per-token jitter computations remain
    self-consistent.

    NOT a measurement. See README §Data Sources.
    """
    f_ttft = ttft_multiplier(target_rate, max_rate)
    f_tpot = tpot_multiplier(target_rate, max_rate)
    out: list[RequestFuncOutput] = []
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
        out.append(
            replace(
                r,
                engine="friendli_modeled",
                first_token_time=new_first,
                last_token_time=new_last,
                itl=new_itl,
            )
        )
    return out
