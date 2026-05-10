"""Tests for the streaming OpenAI-chat backend (§6.2).

Uses ``httpx.MockTransport`` to deliver canned SSE bodies; no live
endpoint is involved. The hand-test against a real vLLM endpoint
required by SPEC §10 step 7 is deferred to §11 validation.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

import httpx
import pytest

from bench.backend import RequestFuncInput, RequestFuncOutput
from bench.client import async_request_openai_chat


class FakeTokenizer:
    """Whitespace-split tokenizer for deterministic test math."""

    def encode(self, text: str) -> list[int]:
        # Returning fake int IDs keeps the type contract; tests only care
        # about ``len(...)`` of the result.
        return list(range(len(text.split())))


def _input(prompt: str = "hello", max_tokens: int = 8) -> RequestFuncInput:
    return RequestFuncInput(
        prompt=prompt,
        prompt_id=0,
        prompt_tokens=1,
        api_url="http://test/v1/chat/completions",
        model="m",
        max_tokens=max_tokens,
    )


def _sse_chunk(content: str) -> bytes:
    payload = {"choices": [{"delta": {"content": content}}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_role_only() -> bytes:
    """First SSE event many OpenAI-style servers emit: role with no content."""
    payload = {"choices": [{"delta": {"role": "assistant"}}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_usage(completion_tokens: int) -> bytes:
    payload = {"choices": [], "usage": {"completion_tokens": completion_tokens}}
    return f"data: {json.dumps(payload)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"


def _ok_handler(body: bytes) -> Callable[[httpx.Request], Awaitable[httpx.Response]]:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    return handler


def _run(coro):
    return asyncio.run(coro)


async def _call(handler, *, prompt: str = "hello", max_tokens: int = 8) -> RequestFuncOutput:
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        return await async_request_openai_chat(
            _input(prompt=prompt, max_tokens=max_tokens),
            client=client,
            tokenizer=FakeTokenizer(),
            arrival_time=0.0,
            engine="vllm",
        )


# --- §6.2 streaming-rule tests ---


def test_empty_first_chunk_skipped_for_ttft():
    """Role-only first chunk must NOT advance ``first_token_time``."""
    body = _sse_role_only() + _sse_chunk("hello") + _sse_chunk(" world") + _SSE_DONE
    out = _run(_call(_ok_handler(body)))
    assert out.success
    # Two non-empty content chunks → 1 ITL gap.
    assert len(out.itl) == 1
    assert out.first_token_time is not None
    assert out.last_token_time is not None
    assert out.last_token_time >= out.first_token_time


def test_five_content_chunks_yield_four_itls():
    body = (
        _sse_chunk("a")
        + _sse_chunk("b")
        + _sse_chunk("c")
        + _sse_chunk("d")
        + _sse_chunk("e")
        + _SSE_DONE
    )
    out = _run(_call(_ok_handler(body)))
    assert out.success
    assert len(out.itl) == 4


def test_single_content_chunk_sets_last_token_time():
    """1-chunk response should still yield a defined e2e_latency."""
    body = _sse_chunk("a") + _SSE_DONE
    out = _run(_call(_ok_handler(body)))
    assert out.success
    assert out.first_token_time is not None
    assert out.last_token_time == out.first_token_time
    assert out.e2e_latency is not None
    # output_tokens=1 → tpot is None per spec §6.1
    assert out.tpot is None


def test_http_500_marks_failure_with_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"upstream broken")

    out = _run(_call(handler))
    assert out.success is False
    assert out.error and "500" in out.error


def test_connection_error_marks_failure_with_error():
    def handler(request: httpx.Request):
        raise httpx.ConnectError("simulated network failure")

    transport = httpx.MockTransport(handler)

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await async_request_openai_chat(
                _input(),
                client=client,
                tokenizer=FakeTokenizer(),
                arrival_time=0.0,
                engine="vllm",
            )

    out = _run(go())
    assert out.success is False
    assert out.error
    assert "ConnectError" in out.error or "simulated" in out.error


def test_tokenizer_drives_output_tokens_not_api():
    """API-reported usage must be ignored (SPEC §6.2 rule 6)."""
    # Three whitespace-separated tokens after concat: "hello world foo"
    body = (
        _sse_chunk("hello")
        + _sse_chunk(" world")
        + _sse_chunk(" foo")
        + _sse_usage(completion_tokens=999)
        + _SSE_DONE
    )
    out = _run(_call(_ok_handler(body)))
    assert out.success
    # FakeTokenizer is whitespace-split, so output_tokens == 3.
    # The 999 from `usage.completion_tokens` MUST be ignored.
    assert out.output_tokens == 3


def test_sse_comment_lines_skipped():
    body = (
        b": ping\n\n"
        + _sse_chunk("a")
        + b": another ping\n\n"
        + _sse_chunk("b")
        + _SSE_DONE
    )
    out = _run(_call(_ok_handler(body)))
    assert out.success
    assert len(out.itl) == 1


def test_empty_content_chunks_in_middle_skipped():
    """Empty content mid-stream should not advance ``last_token_time`` or itl."""
    payload_with_empty = {"choices": [{"delta": {"content": ""}}]}
    empty_chunk = f"data: {json.dumps(payload_with_empty)}\n\n".encode()
    body = (
        _sse_chunk("a")
        + empty_chunk
        + _sse_chunk("b")
        + empty_chunk
        + _sse_chunk("c")
        + _SSE_DONE
    )
    out = _run(_call(_ok_handler(body)))
    assert out.success
    # Three non-empty chunks → 2 ITL gaps; the two empty chunks contribute nothing.
    assert len(out.itl) == 2


def test_empty_stream_marks_failure():
    body = _SSE_DONE
    out = _run(_call(_ok_handler(body)))
    assert out.success is False
    assert "no content received" in (out.error or "")


def test_payload_contains_spec_required_fields():
    """Verify §6.2 sampling-parameter fields are in the request body."""
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse_chunk("a") + _SSE_DONE,
            headers={"content-type": "text/event-stream"},
        )

    _run(_call(handler))
    body = captured["body"]
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["stream"] is True
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["seed"] == 42
    assert body["ignore_eos"] is True


def test_arrival_and_engine_propagated_to_output():
    body = _sse_chunk("a") + _SSE_DONE
    transport = httpx.MockTransport(_ok_handler(body))

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await async_request_openai_chat(
                _input(),
                client=client,
                tokenizer=FakeTokenizer(),
                arrival_time=12345.678,
                engine="friendli",
            )

    out = _run(go())
    assert out.engine == "friendli"
    assert out.arrival_time == 12345.678
    # send_time is recorded on the actual request; should be a different perf_counter().
    assert out.send_time > 0.0


def test_tokenizer_failure_marks_failure_not_silent_zero():
    class BrokenTokenizer:
        def encode(self, text):
            raise RuntimeError("tokenizer broke")

    body = _sse_chunk("a") + _SSE_DONE
    transport = httpx.MockTransport(_ok_handler(body))

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await async_request_openai_chat(
                _input(),
                client=client,
                tokenizer=BrokenTokenizer(),
                arrival_time=0.0,
                engine="vllm",
            )

    out = _run(go())
    assert out.success is False
    assert "tokenizer" in (out.error or "")


def test_register_backend_registry_entry():
    """``openai-chat`` must be the registered key in ASYNC_REQUEST_FUNCS."""
    from bench.backend import ASYNC_REQUEST_FUNCS
    assert "openai-chat" in ASYNC_REQUEST_FUNCS
    assert ASYNC_REQUEST_FUNCS["openai-chat"] is async_request_openai_chat
