"""Shared utilities: timing collector, env capture, JSONL I/O.

``TimeCollector`` is adapted from vLLM's ``benchmarks/benchmark_utils.py``.
The env-capture helpers and JSONL I/O exist to satisfy the reproducibility
contract in SPEC §7: every run must record git SHA, hardware info, library
versions, and the raw per-request timings so summaries can be recomputed.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Iterator


class TimeCollector:
    """Collect timing measurements; supports a ``time_block`` context manager.

    Adapted from vLLM ``benchmarks/benchmark_utils.py``.
    """

    def __init__(self) -> None:
        self.measurements: list[float] = []

    def add(self, value: float) -> None:
        self.measurements.append(value)

    @contextmanager
    def time_block(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(time.perf_counter() - start)

    def avg(self) -> float:
        return sum(self.measurements) / max(len(self.measurements), 1)

    def max(self) -> float:
        return max(self.measurements) if self.measurements else 0.0


# --- Env capture helpers (SPEC §7 reproducibility requirements) ---

_TRACKED_PACKAGES = (
    "httpx",
    "numpy",
    "matplotlib",
    "transformers",
    "tqdm",
    "pydantic",
    "tabulate",
)


def capture_git_sha(cwd: Path | None = None) -> str:
    """Return current git HEAD SHA, or ``"unknown"`` if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cwd) if cwd else None,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def capture_nvidia_smi() -> str | None:
    """Return per-GPU summary from ``nvidia-smi``, or ``None`` if unavailable.

    The output is one CSV row per GPU with columns:
    ``name, driver_version, memory.total``.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def capture_uname() -> str:
    """Return platform identification string (uname -a equivalent)."""
    return platform.platform()


def capture_python_version() -> str:
    """Return the active Python version string."""
    return sys.version.replace("\n", " ")


def capture_library_versions() -> dict[str, str]:
    """Return installed versions of the project's pinned runtime packages."""
    out: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def capture_environment() -> dict[str, Any]:
    """Bundle all reproducibility-relevant environment info into one dict."""
    return {
        "git_sha": capture_git_sha(),
        "uname": capture_uname(),
        "python_version": capture_python_version(),
        "library_versions": capture_library_versions(),
        "nvidia_smi": capture_nvidia_smi(),
    }


# --- JSONL helpers (raw per-request dump/load) ---


def jsonl_dump(items: Iterable[dict[str, Any]], path: Path) -> None:
    """Write ``items`` to ``path`` as JSONL, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def jsonl_load(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
