"""Streaming OpenAI-compatible chat completions client.

Adapted from vLLM's ``async_request_openai_chat_completions`` in
``benchmarks/backend_request_func.py``. Three deliberate divergences from
that reference, all per SPEC §6.2:

  - Empty/missing ``delta.content`` is skipped for ALL timestamp updates,
    including TTFT. This matches NVIDIA GenAI-Perf
    (https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html);
    vLLM's openai-chat handler updates TTFT on the first chunk regardless.
  - ``output_tokens`` is computed by tokenizing the accumulated content
    string. We do not trust ``usage.completion_tokens`` from the API,
    because Friendli's reporting may differ from vLLM's and we want
    fairness across engines.
  - Transport is ``httpx`` rather than ``aiohttp`` so the runner can
    create a single shared ``httpx.AsyncClient`` and pass it in.
"""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

import httpx

from bench.backend import RequestFuncInput, RequestFuncOutput, register_backend


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...


def _build_payload(req: RequestFuncInput) -> dict[str, Any]:
    return {
        "model": req.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "stream": True,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "seed": req.seed,
        "ignore_eos": req.ignore_eos,
    }


@register_backend("openai-chat")
async def async_request_openai_chat(
    req: RequestFuncInput,
    *,
    client: httpx.AsyncClient,
    tokenizer: TokenizerLike,
    arrival_time: float,
    engine: str,
) -> RequestFuncOutput:
    """Stream a chat completion and record per-token timing.

    Notes on streaming protocol per SPEC §6.2:

    * Only ``data:``-prefixed lines are parsed. Blank lines and
      ``:``-prefixed SSE comments (often used as keepalive pings) are
      skipped. The stream terminates on ``data: [DONE]``.
    * Chunks where ``choices[0].delta.content`` is missing or empty are
      skipped for *all* timestamp updates. The first non-empty content
      sets ``first_token_time``; subsequent non-empty content append to
      ``itl`` and advance ``last_token_time``. ``last_token_time`` is also
      set on the first chunk so 1-chunk responses still yield a defined
      ``e2e_latency``.
    * ``output_tokens`` is computed via ``tokenizer.encode`` on the
      accumulated text; any ``usage`` field returned by the server is
      ignored.
    * Any exception during the request marks the result failed with
      the exception's text in ``error`` and timestamps left as-is.
    """
    out = RequestFuncOutput(
        engine=engine,
        prompt_id=req.prompt_id,
        prompt_tokens=req.prompt_tokens,
        arrival_time=arrival_time,
    )

    payload = _build_payload(req)
    headers = {"Content-Type": "application/json"}

    accumulated = ""
    last_chunk_time: float = 0.0  # initialised when first content arrives

    out.send_time = time.perf_counter()

    try:
        async with client.stream(
            "POST", req.api_url, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body_bytes = b""
                async for chunk in response.aiter_bytes():
                    body_bytes += chunk
                    if len(body_bytes) > 4096:
                        break
                body_text = body_bytes.decode("utf-8", errors="replace")[:512]
                out.error = f"HTTP {response.status_code}: {body_text}"
                out.success = False
                return out

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].lstrip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue

                now = time.perf_counter()
                if out.first_token_time is None:
                    out.first_token_time = now
                    last_chunk_time = now
                else:
                    out.itl.append(now - last_chunk_time)
                    last_chunk_time = now
                out.last_token_time = now
                accumulated += content
    except Exception as exc:
        out.success = False
        out.error = f"{type(exc).__name__}: {exc}"
        return out

    if not accumulated:
        out.success = False
        out.error = "no content received in stream"
        return out

    try:
        out.output_tokens = len(tokenizer.encode(accumulated))
    except Exception as exc:
        out.success = False
        out.error = f"tokenizer error: {type(exc).__name__}: {exc}"
        return out

    out.success = True
    return out
