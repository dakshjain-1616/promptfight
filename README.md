# PromptFight – Statistically significant prompt A/B testing in 3 lines

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen.svg)]()

> Minimalist prompt comparison with statistical significance — no LangChain, just scipy + openai.

## Install

```bash
git clone https://github.com/dakshjain-1616/promptfight-a-b-test-llm-prompts-in-3-lines-no-langchain
cd promptfight-a-b-test-llm-prompts-in-3-lines-no-langchain
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
```

## Quickstart

```python
from promptfight import fight

result = fight(
    prompt_a="Summarize: {text}",
    prompt_b="TL;DR: {text}",
    inputs=[{"text": "AI is changing everything..."}],
    model="gpt-4",
    runs=30
)
print(result.winner)  # "A" or "B"
print(result.p_value)  # statistical significance
print(result.cost_usd)  # total spend
```

## Key features

- **3-line API** – Returns winner, p-value, and cost in one call
- **Mann-Whitney U test** – Statistical significance via scipy
- **Cost tracking** – Precise USD calculation per variant
- **Mock mode** – Works without API keys for testing
- **48 tests** – Covers edge cases and cost calculations

## Run tests

```bash
pytest tests/ -q
# 48 passed in 0.06s
```

## Project structure

```
conftest.py          # pytest configuration
promptfight/         # core module
  __init__.py        # main API
scripts/
  demo.py            # example usage
tests/               # test suite
  __init__.py
  test_promptfight.py
```