#!/usr/bin/env python3
"""
PromptFight demo — runs entirely in mock mode when no API keys are set.
Run:  python demo.py
"""

import os
import sys
import json

# Force mock model when no real API keys are present
has_openai    = bool(os.getenv("OPENAI_API_KEY", "").strip())
has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

print("=" * 60)
print("  PromptFight — A/B Prompt Testing Demo")
print("=" * 60)

if not has_openai and not has_anthropic:
    print("  [mock mode] No API keys found — using deterministic mock backend.")
    print("  Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env for real results.")
    demo_models = ["mock"]
else:
    available = []
    if has_openai:
        available.append(os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        print(f"  OpenAI key found  → model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    if has_anthropic:
        available.append(os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"))
        print(f"  Anthropic key found → model: {os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')}")
    demo_models = available

print()

# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

# Import after env inspection
from promptfight import fight, fmt_table, fmt_json

SAMPLE_TEXT = (
    "Artificial intelligence is transforming industries worldwide. "
    "From healthcare diagnostics to autonomous vehicles, AI systems are "
    "increasingly making decisions that affect human lives. This raises "
    "important questions about accountability, transparency, and the future "
    "of work."
)

demos = [
    {
        "title": "Demo 1 — Summarisation style: detailed vs. TL;DR",
        "prompt_a": "Please provide a detailed summary of the following text: {input}",
        "prompt_b": "TL;DR in one sentence: {input}",
        "input": SAMPLE_TEXT,
        "runs": 3,
    },
    {
        "title": "Demo 2 — Tone: formal vs. casual",
        "prompt_a": "Explain the following in formal academic language: {input}",
        "prompt_b": "Explain this like I'm 10 years old: {input}",
        "input": "Why is the sky blue?",
        "runs": 3,
    },
    {
        "title": "Demo 3 — Instruction specificity",
        "prompt_a": "Answer: {input}",
        "prompt_b": "Think step by step, then answer concisely: {input}",
        "input": "What are three benefits of using version control?",
        "runs": 3,
    },
]

for demo in demos:
    print(f"\n{'─'*60}")
    print(f"  {demo['title']}")
    print(f"  Prompt A: {demo['prompt_a'][:60]}...")
    print(f"  Prompt B: {demo['prompt_b'][:60]}...")
    print(f"  Models:   {demo_models}  |  Runs: {demo['runs']}")
    print()

    results = fight(
        prompt_a=demo["prompt_a"],
        prompt_b=demo["prompt_b"],
        user_input=demo["input"],
        models=demo_models,
        runs=demo["runs"],
    )
    print(fmt_table(results))

# ---------------------------------------------------------------------------
# JSON output demo
# ---------------------------------------------------------------------------
print("\n" + "─"*60)
print("  JSON output (pipe-friendly):")
print()
results = fight(
    prompt_a="Summarize: {input}",
    prompt_b="Key points only: {input}",
    user_input=SAMPLE_TEXT,
    models=["mock"],
    runs=2,
)
print(fmt_json(results))

# ---------------------------------------------------------------------------
# CLI invocation demo
# ---------------------------------------------------------------------------
print("\n" + "─"*60)
print("  CLI equivalents (no API key needed):\n")
print('  python promptfight.py \\')
print('    --prompt-a "Summarize: {input}" \\')
print('    --prompt-b "TL;DR: {input}" \\')
print('    --input "Your text here" \\')
print('    --model mock --runs 3\n')
print('  echo "Your text" | python promptfight.py \\')
print('    --prompt-a "Summarize: {input}" \\')
print('    --prompt-b "TL;DR: {input}" \\')
print('    --format json\n')

print("=" * 60)
print("  Demo complete. See README.md for full usage.")
print("  Built autonomously using NEO — https://heyneo.so")
print("=" * 60)
