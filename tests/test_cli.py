"""Tests for the CLI entry point (benchmark.py).

Argument parsing, URL composition, and a subprocess --help smoke check.
The full sweep is integration-tested via tests/test_runner.py with a
fake backend.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark import (
    DEFAULT_RATES,
    _build_arg_parser,
    _compose_chat_url,
    _parse_rates,
    _serialize_cli_args,
)


# --- _parse_rates ---


def test_parse_rates_basic():
    assert _parse_rates("1,2,4,8") == [1.0, 2.0, 4.0, 8.0]


def test_parse_rates_handles_float_values():
    assert _parse_rates("0.5,1.5,2") == [0.5, 1.5, 2.0]


def test_parse_rates_handles_whitespace():
    assert _parse_rates(" 1 , 2 , 4 ") == [1.0, 2.0, 4.0]


def test_parse_rates_rejects_empty():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_rates("")


def test_parse_rates_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_rates("1,not_a_number,4")


# --- _compose_chat_url ---


def test_compose_chat_url_appends_path_to_base():
    assert (
        _compose_chat_url("http://localhost:8000")
        == "http://localhost:8000/v1/chat/completions"
    )


def test_compose_chat_url_strips_trailing_slash():
    assert (
        _compose_chat_url("http://localhost:8000/")
        == "http://localhost:8000/v1/chat/completions"
    )


def test_compose_chat_url_idempotent_when_already_full():
    url = "http://localhost:8000/v1/chat/completions"
    assert _compose_chat_url(url) == url


# --- _build_arg_parser ---


def test_arg_parser_required_flags_missing_raises():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_minimal_invocation_parses():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--vllm-url", "http://x:8000",
        "--friendli-url", "http://y:8001",
        "--model", "m",
        "--tokenizer", "m",
    ])
    assert args.vllm_url == "http://x:8000"
    assert args.rates == DEFAULT_RATES
    assert args.duration_per_rate == 60.0
    assert args.warmup_requests == 20
    assert args.max_output_tokens == 256
    assert args.slo_ttft_ms == 500.0
    assert args.slo_tpot_ms == 50.0
    assert args.seed == 42
    assert args.synthesize_friendli is False
    assert args.cooldown_seconds == 5.0
    assert args.vllm_backend == "openai-chat"
    assert args.friendli_backend == "openai-chat"
    assert args.output_dir == Path("results")
    assert args.prompts_file == Path("prompts/prompts.jsonl")


def test_arg_parser_synthesis_flag():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--vllm-url", "http://x",
        "--friendli-url", "unused",
        "--model", "m",
        "--tokenizer", "m",
        "--synthesize-friendli",
    ])
    assert args.synthesize_friendli is True


def test_arg_parser_custom_rates_and_duration():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--vllm-url", "http://x",
        "--friendli-url", "http://y",
        "--model", "m",
        "--tokenizer", "m",
        "--rates", "1,2,4",
        "--duration-per-rate", "30",
    ])
    assert args.rates == [1.0, 2.0, 4.0]
    assert args.duration_per_rate == 30.0


# --- _serialize_cli_args ---


def test_serialize_cli_args_converts_paths_to_strings():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--vllm-url", "http://x",
        "--friendli-url", "http://y",
        "--model", "m",
        "--tokenizer", "m",
    ])
    out = _serialize_cli_args(args)
    assert out["output_dir"] == "results"  # was Path
    assert out["prompts_file"] == "prompts/prompts.jsonl"
    assert out["rates"] == DEFAULT_RATES
    # Should be JSON-serialisable
    import json
    json.dumps(out)


# --- subprocess --help smoke ---


def test_cli_help_lists_required_flags():
    """Run benchmark.py --help in a subprocess; verify required flags appear."""
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for flag in ("--vllm-url", "--friendli-url", "--model", "--tokenizer",
                 "--rates", "--synthesize-friendli", "--slo-ttft-ms",
                 "--slo-tpot-ms"):
        assert flag in out, f"missing {flag} in --help output"
