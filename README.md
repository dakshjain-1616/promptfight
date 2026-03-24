# PromptFight – A/B test any two LLM prompts and know which one wins

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()
[![Zero Runtime Deps](https://img.shields.io/badge/runtime%20deps-stdlib%20only-orange.svg)]()

**Run your two prompts head-to-head across any model and get win rates, latency, cost, and token counts — using nothing but the Python standard library at runtime.**

---

## What it does

You have two prompts. You want to know which one produces better responses, costs less, and responds faster. PromptFight automates that experiment:

- Sends both prompts to one or more models for N runs each
- Scores every response pair with a heuristic judge (or an LLM judge you choose)
- Reports win rates, average latency, average cost per run, average token counts, and cost savings
- Works offline with the built-in `mock` model — no API key required to try it

Calls OpenAI and Anthropic APIs directly via `urllib`. No SDK dependencies at runtime.

---

## Install

```bash
git clone https://github.com/dakshjain-1616/promptfight
cd promptfight
pip install -r requirements.txt
```

Set your API keys (only needed for real model runs):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Or place them in a `.env` file in the project root — PromptFight loads it automatically via `python-dotenv`.

---

## CLI quickstart

### Mock mode — no API key needed

The fastest way to try PromptFight. The `mock` model returns instant, zero-cost responses so you can verify the workflow before spending any money.

```bash
python -m promptfight \
  --prompt-a "Summarize: {input}" \
  --prompt-b "TL;DR: {input}" \
  --input "Artificial intelligence is transforming software development." \
  --model mock \
  --runs 5
```

Example output:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            PromptFight Results                               │
├───────────┬──────┬────────┬────────┬───────┬──────────┬──────────────────────┤
│ Model     │ Runs │ A Wins │ B Wins │ Ties  │ Winner   │ Win Rate             │
├───────────┼──────┼────────┼────────┼───────┼──────────┼──────────────────────┤
│ mock      │  5   │   3    │   2    │   0   │ A        │ 60.0%                │
├───────────┼──────┼────────┼────────┼───────┼──────────┼──────────────────────┤
│           │ A Latency  │ B Latency  │ A Cost      │ B Cost      │           │
│           │    0ms     │    0ms     │ $0.00000    │ $0.00000    │           │
│           │ A Tokens   │ B Tokens   │ Cost Savings│             │           │
│           │   12.0     │   10.0     │    0.0%     │             │           │
└───────────┴────────────┴────────────┴─────────────┴─────────────┴───────────┘
```

### Real model — gpt-4o-mini

```bash
python -m promptfight \
  --prompt-a "Summarize the following in one sentence: {input}" \
  --prompt-b "TL;DR (one sentence): {input}" \
  --input "Artificial intelligence is transforming software development by automating repetitive tasks, generating code, and helping developers ship faster with fewer bugs." \
  --model gpt-4o-mini \
  --runs 10
```

### Pipe input from stdin

```bash
echo "Long article text here..." | python -m promptfight \
  --prompt-a "Summarize: {input}" \
  --prompt-b "TL;DR: {input}"
```

### Test multiple models in one run

```bash
python -m promptfight \
  --prompt-a "Summarize: {input}" \
  --prompt-b "TL;DR: {input}" \
  --input "Article text here..." \
  --model gpt-4o,gpt-4o-mini \
  --runs 5 \
  --format json
```

### Output as CSV

```bash
python -m promptfight \
  --prompt-a "Summarize: {input}" \
  --prompt-b "TL;DR: {input}" \
  --input "Article text here..." \
  --model gpt-4o-mini \
  --runs 5 \
  --format csv
```

### All CLI flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--prompt-a` | Yes | — | First prompt. Use `{input}` or `{text}` as the placeholder. |
| `--prompt-b` | Yes | — | Second prompt. Same placeholder rules. |
| `--input` | No | stdin | Text substituted into `{input}` / `{text}`. Omit to read from stdin. |
| `--model` | No | `mock` | Comma-separated model names to test against. |
| `--runs` | No | `3` | Number of trials per prompt per model. |
| `--format` | No | `table` | Output format: `table`, `json`, or `csv`. |

---

## Python API

```python
from promptfight import fight, FightResult

results = fight(
    prompt_a="Summarize: {input}",
    prompt_b="TL;DR: {input}",
    user_input="Artificial intelligence is transforming software development.",
    models=["mock"],     # or ["gpt-4o-mini"], or ["gpt-4o", "gpt-4o-mini"]
    runs=10
)

for r in results:
    print(f"Model:              {r.model}")
    print(f"Winner:             Prompt {r.winner} ({r.win_rate_pct:.0f}% win rate)")
    print(f"A wins / B wins / Ties:  {r.a_wins} / {r.b_wins} / {r.ties}")
    print(f"A latency:          {r.a_avg_latency_ms:.0f}ms")
    print(f"B latency:          {r.b_avg_latency_ms:.0f}ms")
    print(f"Latency diff:       {r.latency_diff_ms:+.0f}ms  (B minus A)")
    print(f"A cost:             ${r.a_avg_cost_usd:.5f}")
    print(f"B cost:             ${r.b_avg_cost_usd:.5f}")
    print(f"Cost savings:       {r.cost_savings_pct:.1f}%  (choosing the cheaper prompt)")
    print(f"A avg tokens:       {r.a_avg_tokens:.1f}")
    print(f"B avg tokens:       {r.b_avg_tokens:.1f}")
```

### FightResult fields

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model name used for this result row. |
| `runs` | `int` | Number of trials run per prompt. |
| `a_wins` | `int` | Number of runs Prompt A won. |
| `b_wins` | `int` | Number of runs Prompt B won. |
| `ties` | `int` | Number of runs that ended in a tie. |
| `a_avg_latency_ms` | `float` | Average response time for Prompt A in milliseconds. |
| `b_avg_latency_ms` | `float` | Average response time for Prompt B in milliseconds. |
| `a_avg_cost_usd` | `float` | Average cost per run for Prompt A in USD. |
| `b_avg_cost_usd` | `float` | Average cost per run for Prompt B in USD. |
| `a_avg_tokens` | `float` | Average output token count for Prompt A. |
| `b_avg_tokens` | `float` | Average output token count for Prompt B. |
| `winner` | `str` | `"A"`, `"B"`, or `"tie"`. |
| `win_rate_pct` | `float` | Percentage of runs won by the winner (0–100). |
| `latency_diff_ms` | `float` | B average latency minus A average latency. Negative means B is faster. |
| `cost_savings_pct` | `float` | Percentage cost saved by choosing the cheaper prompt over the more expensive one. |

---

## Supported models

| Model ID | Provider | API Key Required |
|----------|----------|-----------------|
| `mock` | Built-in | No — default for local testing |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `gpt-3.5-turbo` | OpenAI | `OPENAI_API_KEY` |
| `claude-3-5-sonnet-20241022` | Anthropic | `ANTHROPIC_API_KEY` |
| `claude-3-haiku-20240307` | Anthropic | `ANTHROPIC_API_KEY` |
| Any OpenAI-compatible model | Custom endpoint | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |

For local models or other providers, set `OPENAI_BASE_URL` to the base URL and pass the model name as `--model`.

---

## Configuration

All settings can be provided as environment variables or in a `.env` file in the project root.

### API keys and endpoint

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key. |
| `ANTHROPIC_API_KEY` | Anthropic API key. |
| `OPENAI_BASE_URL` | Override the OpenAI API base URL. Default: `https://api.openai.com/v1`. |

### Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPTFIGHT_MODELS` | `mock` | Default model(s) when `--model` is not specified. |
| `PROMPTFIGHT_RUNS` | `3` | Default run count when `--runs` is not specified. |
| `PROMPTFIGHT_FORMAT` | `table` | Default output format: `table`, `json`, or `csv`. |
| `PROMPTFIGHT_MAX_TOKENS` | `512` | Maximum tokens per model response. |
| `PROMPTFIGHT_JUDGE_MODEL` | *(empty)* | Model name for LLM judge. Empty means heuristic judge. |

### Cost overrides

Override the per-token cost used for cost calculations if pricing changes:

| Variable | Model |
|----------|-------|
| `COST_GPT4O` | `gpt-4o` |
| `COST_GPT4O_MINI` | `gpt-4o-mini` |
| `COST_GPT35` | `gpt-3.5-turbo` |
| `COST_CLAUDE_SONNET` | `claude-3-5-sonnet-20241022` |
| `COST_CLAUDE_HAIKU` | `claude-3-haiku-20240307` |

Values are in USD per output token (e.g., `COST_GPT4O_MINI=0.00000060`).

---

## Judging

### Heuristic judge (default)

When `PROMPTFIGHT_JUDGE_MODEL` is not set, PromptFight uses a heuristic scorer. Each response is scored on:

- **Length** — longer responses score higher, capturing detail and completeness
- **Code blocks** — presence of fenced code blocks (`` ``` ``)
- **Structure markers** — presence of `##` headings and numbered lists

The response with the higher composite score wins the run. Equal scores result in a tie.

The heuristic judge is free, instant, and requires no API key — ideal for high-volume tests or when responses differ mainly in structure and thoroughness.

### LLM judge

Set `PROMPTFIGHT_JUDGE_MODEL` to any supported model to have an LLM decide which response is better for each run pair:

```bash
export PROMPTFIGHT_JUDGE_MODEL=gpt-4o-mini
```

The judge receives both responses side-by-side and returns `A`, `B`, or `tie`. This is more nuanced than the heuristic but adds latency and cost proportional to your run count. Use it when response quality differences are subtle or subjective.

---

## Run tests

```bash
pip install pytest pytest-mock
pytest tests/ -q
# 48 passed
```

Verbose output:

```bash
pytest tests/ -v
```

Run a single test file:

```bash
pytest tests/test_promptfight.py -v
```

Tests cover the fight engine, heuristic and LLM judges, CLI argument parsing, API client mocking for both OpenAI and Anthropic, cost calculations, and all three output formatters.

---

## Project structure

```
promptfight/
├── promptfight/               # Main package
│   └── __init__.py            # fight(), FightResult, CLI entry point, API clients, judges
├── tests/
│   ├── __init__.py
│   └── test_promptfight.py    # 48 tests
├── scripts/
│   └── demo.py                # Runnable demo script
├── conftest.py                # Pytest configuration and shared fixtures
├── pytest.ini                 # Pytest settings
├── requirements.txt           # python-dotenv (runtime); pytest, pytest-mock (dev)
└── README.md
```
