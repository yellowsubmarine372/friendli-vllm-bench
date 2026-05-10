"""Single-graph generator: Goodput (RPS) vs Request Rate (RPS).

Implements SPEC §6.8. One figure, one axes, no subplots. The footer
explicitly states whether the Friendli line is measured or modeled, so
anyone seeing the PNG can interpret it correctly without the README.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import

import matplotlib.pyplot as plt  # noqa: E402

from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from bench.metrics import SLO, RunSummary  # noqa: E402

VLLM_COLOR = "#1f77b4"
FRIENDLI_COLOR = "#ff7f0e"
DIAGONAL_COLOR = "#888"
SATURATION_THRESHOLD = 0.10  # 10% divergence from y=x


def saturation_rate(
    rates: list[float],
    goodputs: list[float],
    threshold: float = SATURATION_THRESHOLD,
) -> float | None:
    """First rate at which goodput diverges from ``y = x`` by more than ``threshold``.

    Returns ``None`` if the engine never saturates within the swept rates.
    """
    paired = sorted(zip(rates, goodputs))
    for rate, goodput in paired:
        if rate <= 0:
            continue
        if (rate - goodput) / rate > threshold:
            return rate
    return None


def _draw_line(
    ax: plt.Axes,
    summaries: list[RunSummary],
    *,
    color: str,
    linestyle: str,
    marker: str,
    label: str,
    sat_label_y: float,
) -> None:
    if not summaries:
        return
    summaries = sorted(summaries, key=lambda s: s.target_rate)
    rates = [s.target_rate for s in summaries]
    goodputs = [s.goodput_rps for s in summaries]
    ax.plot(
        rates,
        goodputs,
        color=color,
        linestyle=linestyle,
        marker=marker,
        label=label,
        zorder=3,
    )
    sat = saturation_rate(rates, goodputs)
    if sat is not None:
        ax.axvline(sat, color=color, linestyle=":", alpha=0.5, zorder=2)
        ax.text(
            sat,
            sat_label_y,
            f"{label.split(' ')[0]} saturates ≈ {sat:g} RPS",
            transform=ax.get_xaxis_transform(),
            color=color,
            fontsize=9,
            rotation=90,
            va="bottom",
            ha="right",
        )


def plot_goodput(
    summaries: list[RunSummary],
    slo: SLO,
    output_path: Path,
    synthesized: bool,
    git_sha: str,
) -> None:
    """Render the single goodput-vs-rate graph and save as PNG (300 DPI) + PDF.

    ``output_path`` is the PNG path; a sibling PDF with a matching basename
    is also written.
    """
    if not summaries:
        raise ValueError("no summaries to plot")

    by_engine: dict[str, list[RunSummary]] = {}
    for s in summaries:
        by_engine.setdefault(s.engine, []).append(s)

    fig, ax = plt.subplots(figsize=(8, 5))

    all_rates = sorted({s.target_rate for s in summaries})
    max_rate = max(all_rates)

    # y = x reference.
    ax.plot(
        [0, max_rate],
        [0, max_rate],
        color=DIAGONAL_COLOR,
        linestyle="--",
        alpha=0.5,
        label="Perfect goodput",
        zorder=1,
    )

    # vLLM line.
    _draw_line(
        ax,
        by_engine.get("vllm", []),
        color=VLLM_COLOR,
        linestyle="--",
        marker="o",
        label="vLLM (measured)",
        sat_label_y=0.05,
    )

    # Friendli line (measured or modeled).
    friendli_engine = "friendli_modeled" if synthesized else "friendli"
    friendli_label = (
        "Friendli (modeled from public benchmarks)"
        if synthesized
        else "Friendli (measured)"
    )
    _draw_line(
        ax,
        by_engine.get(friendli_engine, []),
        color=FRIENDLI_COLOR,
        linestyle="-",
        marker="s",
        label=friendli_label,
        sat_label_y=0.18,
    )

    ax.set_xlabel("Request rate (RPS)")
    ax.set_ylabel("Goodput (RPS)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")

    ax.set_title("Goodput under Open-Loop Poisson Load", pad=22)
    subtitle = (
        f"SLO: per-request TTFT ≤ {slo.ttft_ms:g}ms AND "
        f"TPOT ≤ {slo.tpot_ms:g}ms"
    )
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555",
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    short_sha = (git_sha or "")[:7] or "unknown"
    if synthesized:
        footer = (
            "vLLM: measured. Friendli: modeled from FriendliAI public "
            "benchmarks (see README §Data Sources). · "
            f"git: {short_sha} · {timestamp}"
        )
    else:
        footer = f"Both engines measured. · git: {short_sha} · {timestamp}"
    fig.text(0.01, 0.005, footer, fontsize=7, color="#888", ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.03, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
