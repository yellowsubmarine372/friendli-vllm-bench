#!/usr/bin/env python3
"""Generate prompts/prompts.jsonl deterministically.

Produces 200 prompts with varied input lengths (short ~50, medium ~200,
long ~500 tokens). The token counts are approximate using a 4-chars-per-token
heuristic; the prompt_tokens field on RequestFuncInput is metadata only — the
streaming output token count used in measurement comes from the runtime
tokenizer applied to the accumulated response, not from this estimate.

Synthetic generation with a fixed seed is preferred over ShareGPT-style
external datasets so the benchmark is reproducible without an external
download. This follows the rationale validated by vLLM's benchmark_serving.py.

Usage:
    uv run python scripts/build_prompts.py --output prompts/prompts.jsonl --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TOPICS = [
    "photosynthesis",
    "the water cycle",
    "plate tectonics",
    "natural selection",
    "the human immune system",
    "general relativity",
    "quantum entanglement",
    "the central limit theorem",
    "public-key cryptography",
    "the second law of thermodynamics",
    "the supply curve in economics",
    "the prisoner's dilemma",
    "the anchoring bias",
    "the Krebs cycle",
    "the doppler effect",
    "electromagnetic induction",
    "the standard model of particle physics",
    "the carbon cycle",
    "convolutional neural networks",
    "hash tables",
    "the CAP theorem",
    "compound interest",
    "the bystander effect",
    "the placebo effect",
]

AUDIENCES = [
    "a curious child",
    "a high-school student",
    "a college freshman in a non-science major",
    "a graduate researcher in an unrelated field",
    "a software engineer with no biology background",
    "a non-technical executive",
    "a museum visitor reading a panel",
    "a podcast listener with thirty seconds of attention",
]

STYLES = [
    "with concrete examples drawn from everyday life",
    "using analogies to familiar phenomena",
    "starting from first principles, step by step",
    "in a conversational tone",
    "with explicit attention to common misconceptions",
    "by walking through a single worked example end to end",
]

SHORT_TEMPLATES = [
    (
        "Briefly explain {topic} in two or three sentences. Focus on what the "
        "concept is, why it matters, and what someone encountering the term for "
        "the first time should remember above all else."
    ),
    (
        "Summarize {topic} in roughly one short paragraph. Highlight the single "
        "most important idea, and mention one common misunderstanding without "
        "going into detail about it."
    ),
    (
        "Give a short, accurate definition of {topic} suitable for a glossary "
        "entry. Include only the essential information; avoid historical "
        "background, extended examples, and tangential details."
    ),
]

MEDIUM_TEMPLATE = (
    "Explain the concept of {topic} to {audience} {style}. "
    "Begin with a one-paragraph definition that captures the essential idea "
    "without relying on technical jargon. Then describe the historical "
    "development of the concept, identifying the key thinkers, experiments, "
    "or events that shaped our current understanding. Address two common "
    "misconceptions that newcomers often have, and explain how the correct "
    "understanding differs from each. Finally, provide a present-day example "
    "or application that demonstrates why the concept remains relevant today. "
    "Aim for clarity over completeness, and keep the response approximately "
    "two hundred to three hundred words."
)

LONG_TEMPLATE = (
    "Write a comprehensive overview of {topic} for {audience}. Structure "
    "your response with the following sections, using clear headings or "
    "labels for each one. Section 1: Definition and scope. Provide a precise "
    "definition of the concept and explain its boundaries — what it does "
    "and does not include, and where it overlaps with adjacent ideas a "
    "reader might encounter. Section 2: Historical context. Describe the "
    "key milestones in how the concept came to be understood, naming the "
    "people, experiments, or events that shaped it, and noting why earlier "
    "framings were superseded by the modern synthesis. Section 3: "
    "Underlying mechanism. Explain in detail how the concept works at a "
    "mechanistic level. Use {style} to make the explanation accessible. "
    "Where helpful, walk through a concrete example end to end, including "
    "the inputs, the intermediate steps, and the outputs that follow from "
    "each stage. Section 4: Common misconceptions. Identify three "
    "frequently held but inaccurate beliefs about the topic, and for each "
    "one, explain precisely where the misconception goes wrong and what the "
    "correct understanding is. Section 5: Practical applications. Describe "
    "two or three settings in which understanding this concept materially "
    "changes how a practitioner would think or act, contrasting a decision "
    "made with the concept against one made without it. Section 6: "
    "Executive summary. Close with a one-paragraph synthesis aimed at a "
    "reader who skipped the body, capturing the points you would most want "
    "them to retain. Aim for a thorough but readable response, "
    "approximately five hundred words. Use plain prose throughout, avoid "
    "bullet lists for the main argument, and prefer clear topic sentences "
    "at the start of each section."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_corpus(seed: int, n_short: int, n_medium: int, n_long: int) -> list[dict]:
    rng = random.Random(seed)

    short_pool: list[str] = []
    for _ in range(n_short):
        topic = rng.choice(TOPICS)
        template = rng.choice(SHORT_TEMPLATES)
        short_pool.append(template.format(topic=topic))

    medium_pool: list[str] = []
    for _ in range(n_medium):
        medium_pool.append(MEDIUM_TEMPLATE.format(
            topic=rng.choice(TOPICS),
            audience=rng.choice(AUDIENCES),
            style=rng.choice(STYLES),
        ))

    long_pool: list[str] = []
    for _ in range(n_long):
        long_pool.append(LONG_TEMPLATE.format(
            topic=rng.choice(TOPICS),
            audience=rng.choice(AUDIENCES),
            style=rng.choice(STYLES),
        ))

    pool_iters = [iter(short_pool), iter(medium_pool), iter(long_pool)]
    remaining = [n_short, n_medium, n_long]
    interleaved: list[str] = []
    while any(remaining):
        for i in range(3):
            if remaining[i] > 0:
                interleaved.append(next(pool_iters[i]))
                remaining[i] -= 1

    return [
        {"id": i, "text": text, "approx_tokens": estimate_tokens(text)}
        for i, text in enumerate(interleaved)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompts.jsonl deterministically.")
    parser.add_argument("--output", type=Path, default=Path("prompts/prompts.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-short", type=int, default=67)
    parser.add_argument("--n-medium", type=int, default=67)
    parser.add_argument("--n-long", type=int, default=66)
    args = parser.parse_args()

    corpus = build_corpus(args.seed, args.n_short, args.n_medium, args.n_long)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    short = [c for c in corpus if c["approx_tokens"] < 100]
    medium = [c for c in corpus if 100 <= c["approx_tokens"] < 350]
    long_ = [c for c in corpus if c["approx_tokens"] >= 350]
    print(
        f"Wrote {len(corpus)} prompts to {args.output} "
        f"(short={len(short)}, medium={len(medium)}, long={len(long_)})"
    )


if __name__ == "__main__":
    main()
