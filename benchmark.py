#!/usr/bin/env python3
"""Entry point for the friendli-vllm-bench sweep.

Per SPEC §5: one invocation runs the full engine × rate sweep and
produces the goodput-vs-rate graph. Two operating modes:

* measurement (default): hits both engines directly.
* synthesis (``--synthesize-friendli``): measures only vLLM, derives the
  Friendli line from vLLM measurements via ``bench.synthetic``.

Validation §11: ``--vllm-url`` and ``--friendli-url`` may point to the
same endpoint to verify the measurement pipeline (both lines should
overlap).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from functools import partial
from pathlib import Path

import httpx
from transformers import AutoTokenizer

# Importing bench.client registers the ``openai-chat`` backend.
import bench.client  # noqa: F401
from bench.backend import ASYNC_REQUEST_FUNCS
from bench.metrics import SLO
from bench.plot import plot_goodput
from bench.runner import RunConfig, run_sweep
from bench.utils import capture_git_sha
from bench.workload import load_prompts

DEFAULT_RATES = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]


def _parse_rates(s: str) -> list[float]:
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--rates: bad value {tok!r}: {e}"
            ) from e
    if not out:
        raise argparse.ArgumentTypeError("--rates must contain at least one value")
    return out


def _compose_chat_url(base: str) -> str:
    """Append ``/v1/chat/completions`` if the user passed a base URL."""
    base = base.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    return base + "/v1/chat/completions"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare vLLM and Friendli engines via open-loop goodput sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vllm-url", required=True, help="HTTP URL of vLLM endpoint")
    p.add_argument(
        "--friendli-url",
        required=True,
        help="HTTP URL of Friendli endpoint (ignored in synthesis mode)",
    )
    p.add_argument(
        "--model", required=True, help="model identifier passed in API requests"
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer ID for output token counting",
    )

    p.add_argument(
        "--rates",
        type=_parse_rates,
        default=DEFAULT_RATES,
        help="comma-separated RPS values",
    )
    p.add_argument(
        "--duration-per-rate", type=float, default=60.0,
        help="seconds per rate point",
    )
    p.add_argument(
        "--warmup-requests", type=int, default=20,
        help="pre-measurement requests",
    )
    p.add_argument(
        "--max-output-tokens", type=int, default=256,
        help="hard cap on output length",
    )
    p.add_argument(
        "--slo-ttft-ms", type=float, default=500.0,
        help="per-request TTFT SLO threshold in ms (source: Rafay Token Factory)",
    )
    p.add_argument(
        "--slo-tpot-ms", type=float, default=50.0,
        help="per-request TPOT SLO threshold in ms (source: Rafay Token Factory)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="master seed for prompt order, Poisson arrivals, sampling",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="results directory",
    )
    p.add_argument(
        "--synthesize-friendli", action="store_true",
        help="derive Friendli line from vLLM measurements rather than calling its endpoint",
    )
    p.add_argument(
        "--cooldown-seconds", type=float, default=5.0,
        help="pause between rates",
    )
    p.add_argument(
        "--vllm-backend", default="openai-chat",
        help="backend name from the registry",
    )
    p.add_argument(
        "--friendli-backend", default="openai-chat",
        help="backend name from the registry",
    )
    p.add_argument(
        "--prompts-file", type=Path, default=Path("prompts/prompts.jsonl"),
        help="JSONL file with id/text/approx_tokens prompts",
    )
    return p


def _serialize_cli_args(ns: argparse.Namespace) -> dict:
    """Convert argparse namespace to a JSON-friendly dict for run_metadata.json."""
    out: dict = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


async def _amain(args: argparse.Namespace) -> int:
    # Resolve backend(s) — both must currently use the same registry entry
    # because RunConfig.backend_fn is singular.
    if args.vllm_backend != args.friendli_backend:
        print(
            f"[error] --vllm-backend ({args.vllm_backend}) and "
            f"--friendli-backend ({args.friendli_backend}) must be identical "
            f"in this build (single backend_fn per run).",
            file=sys.stderr,
        )
        return 2
    backend_name = args.vllm_backend
    if backend_name not in ASYNC_REQUEST_FUNCS:
        print(
            f"[error] unknown backend {backend_name!r}. "
            f"Registered: {sorted(ASYNC_REQUEST_FUNCS)}",
            file=sys.stderr,
        )
        return 2

    print(f"[setup] loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"[setup] loading prompts: {args.prompts_file}", flush=True)
    prompts = load_prompts(args.prompts_file)
    print(f"[setup] {len(prompts)} prompts loaded", flush=True)

    timeout = httpx.Timeout(connect=10, read=120, write=10, pool=10)
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=100)

    print(
        f"[setup] mode = "
        f"{'SYNTHESIS (vLLM measured, Friendli modeled)' if args.synthesize_friendli else 'MEASUREMENT (both engines)'}",
        flush=True,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        backend_fn = partial(
            ASYNC_REQUEST_FUNCS[backend_name],
            client=client,
            tokenizer=tokenizer,
        )

        engines = {
            "vllm": _compose_chat_url(args.vllm_url),
            "friendli": (
                None
                if args.synthesize_friendli
                else _compose_chat_url(args.friendli_url)
            ),
        }

        config = RunConfig(
            model=args.model,
            backend_fn=backend_fn,
            prompts=prompts,
            max_output_tokens=args.max_output_tokens,
            slo=SLO(ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms),
            seed=args.seed,
            output_dir=args.output_dir,
            cli_args=_serialize_cli_args(args),
        )

        summaries = await run_sweep(
            engines=engines,
            rates=args.rates,
            duration_per_rate=args.duration_per_rate,
            warmup_requests=args.warmup_requests,
            cooldown_seconds=args.cooldown_seconds,
            synthesize_friendli=args.synthesize_friendli,
            config=config,
        )

    plot_path = args.output_dir / "goodput_comparison.png"
    plot_goodput(
        summaries=summaries,
        slo=SLO(ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms),
        output_path=plot_path,
        synthesized=args.synthesize_friendli,
        git_sha=capture_git_sha(),
    )

    print(
        f"\n[done] graph: {plot_path} and {plot_path.with_suffix('.pdf')}\n"
        f"[done] summary: {args.output_dir / 'summary.json'} / "
        f"{args.output_dir / 'summary.csv'}\n"
        f"[done] raw data: {args.output_dir / 'raw'}\n"
        f"[done] metadata: {args.output_dir / 'run_metadata.json'}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n[interrupt] aborted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
