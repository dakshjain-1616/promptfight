#!/usr/bin/env python3
"""
PromptFight — A/B test LLM prompts in 3 lines, no LangChain required.
Usage:  promptfight --prompt-a "..." --prompt-b "..." --input "..." [options]
        echo "my input" | promptfight --prompt-a "..." --prompt-b "..."
"""

import argparse
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODELS   = os.getenv("PROMPTFIGHT_MODELS", "mock").split(",")
DEFAULT_RUNS     = int(os.getenv("PROMPTFIGHT_RUNS", "3"))
DEFAULT_FORMAT   = os.getenv("PROMPTFIGHT_FORMAT", "table")   # table | json | csv
JUDGE_MODEL      = os.getenv("PROMPTFIGHT_JUDGE_MODEL", "")   # blank = heuristic judge
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
MAX_TOKENS       = int(os.getenv("PROMPTFIGHT_MAX_TOKENS", "512"))

# Cost per 1 000 tokens (input + output) in USD, override via env
COST_PER_1K = {
    "gpt-4o":              float(os.getenv("COST_GPT4O",       "0.005")),
    "gpt-4o-mini":         float(os.getenv("COST_GPT4O_MINI",  "0.00015")),
    "gpt-3.5-turbo":       float(os.getenv("COST_GPT35",       "0.002")),
    "claude-3-5-sonnet-20241022": float(os.getenv("COST_CLAUDE_SONNET", "0.003")),
    "claude-3-haiku-20240307":    float(os.getenv("COST_CLAUDE_HAIKU",  "0.00025")),
    "mock":                0.0,
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    prompt_label: str          # "A" or "B"
    model: str
    prompt_text: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: Optional[str] = None


@dataclass
class FightResult:
    model: str
    runs: int
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    a_avg_latency_ms: float = 0.0
    b_avg_latency_ms: float = 0.0
    a_avg_cost_usd: float = 0.0
    b_avg_cost_usd: float = 0.0
    a_avg_tokens: float = 0.0
    b_avg_tokens: float = 0.0
    winner: str = ""
    win_rate_pct: float = 0.0
    latency_diff_ms: float = 0.0
    cost_savings_pct: float = 0.0

# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _openai_call(model: str, prompt: str, user_input: str) -> dict:
    """Call OpenAI chat completions. Returns {text, input_tokens, output_tokens}."""
    import urllib.request
    filled = prompt.replace("{input}", user_input).replace("{text}", user_input)
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": filled}],
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        f"{OPENAI_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return {
        "text": data["choices"][0]["message"]["content"],
        "input_tokens": data["usage"]["prompt_tokens"],
        "output_tokens": data["usage"]["completion_tokens"],
    }


def _anthropic_call(model: str, prompt: str, user_input: str) -> dict:
    """Call Anthropic Messages API. Returns {text, input_tokens, output_tokens}."""
    import urllib.request
    filled = prompt.replace("{input}", user_input).replace("{text}", user_input)
    payload = json.dumps({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": filled}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return {
        "text": data["content"][0]["text"],
        "input_tokens": data["usage"]["input_tokens"],
        "output_tokens": data["usage"]["output_tokens"],
    }


def _mock_call(model: str, prompt: str, user_input: str, label: str) -> dict:
    """Deterministic mock — no API key needed."""
    import random, hashlib
    seed = int(hashlib.md5(f"{prompt}{user_input}".encode()).hexdigest(), 16) % 10000
    rng = random.Random(seed)
    # Prompt A tends to produce slightly longer, slower responses (biased mock)
    words = rng.randint(20, 60) if label == "A" else rng.randint(15, 45)
    vocab = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "mock", "response", "test", "prompt", "result", "data", "output"]
    text = " ".join(rng.choices(vocab, k=words)) + "."
    in_tok = len(prompt.split()) + len(user_input.split())
    out_tok = len(text.split())
    return {"text": text, "input_tokens": in_tok, "output_tokens": out_tok}


# ---------------------------------------------------------------------------
# Single run executor
# ---------------------------------------------------------------------------

def run_once(model: str, prompt: str, user_input: str, label: str) -> RunResult:
    cost_rate = COST_PER_1K.get(model, 0.002)
    t0 = time.perf_counter()
    error = None
    try:
        if model == "mock":
            raw = _mock_call(model, prompt, user_input, label)
        elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            raw = _openai_call(model, prompt, user_input)
        elif model.startswith("claude"):
            raw = _anthropic_call(model, prompt, user_input)
        else:
            raw = _mock_call(model, prompt, user_input, label)
    except Exception as exc:
        raw = {"text": "", "input_tokens": 0, "output_tokens": 0}
        error = str(exc)
    latency_ms = (time.perf_counter() - t0) * 1000
    total_tokens = raw["input_tokens"] + raw["output_tokens"]
    cost = (total_tokens / 1000) * cost_rate
    return RunResult(
        prompt_label=label,
        model=model,
        prompt_text=prompt,
        response=raw["text"],
        latency_ms=latency_ms,
        input_tokens=raw["input_tokens"],
        output_tokens=raw["output_tokens"],
        cost_usd=cost,
        error=error,
    )


# ---------------------------------------------------------------------------
# Judge — decides which response is better per-run
# ---------------------------------------------------------------------------

def judge_responses(a_result: RunResult, b_result: RunResult, user_input: str) -> str:
    """Return 'A', 'B', or 'tie'."""
    if a_result.error and not b_result.error:
        return "B"
    if b_result.error and not a_result.error:
        return "A"
    if a_result.error and b_result.error:
        return "tie"

    # Optional LLM judge
    if JUDGE_MODEL and (OPENAI_API_KEY or ANTHROPIC_API_KEY):
        try:
            judge_prompt = (
                f"You are a judge comparing two AI responses to the same task.\n"
                f"Task: {user_input}\n\n"
                f"Response A:\n{a_result.response}\n\n"
                f"Response B:\n{b_result.response}\n\n"
                f"Which response is better? Reply with exactly one word: A, B, or tie."
            )
            if JUDGE_MODEL.startswith("claude"):
                r = _anthropic_call(JUDGE_MODEL, judge_prompt, "")
            else:
                r = _openai_call(JUDGE_MODEL, judge_prompt, "")
            verdict = r["text"].strip().upper()
            if verdict in ("A", "B", "TIE"):
                return verdict.replace("TIE", "tie")
        except Exception:
            pass  # fall through to heuristic

    # Heuristic: score on output length (proxy for detail), penalise very short
    a_score = len(a_result.response.split())
    b_score = len(b_result.response.split())
    if abs(a_score - b_score) <= 2:
        return "tie"
    return "A" if a_score > b_score else "B"


# ---------------------------------------------------------------------------
# Core fight logic
# ---------------------------------------------------------------------------

def fight(prompt_a: str, prompt_b: str, user_input: str,
          models: list[str], runs: int) -> list[FightResult]:
    results = []
    for model in models:
        model = model.strip()
        a_runs, b_runs = [], []
        for _ in range(runs):
            a_runs.append(run_once(model, prompt_a, user_input, "A"))
            b_runs.append(run_once(model, prompt_b, user_input, "B"))

        a_wins = b_wins = ties = 0
        for ar, br in zip(a_runs, b_runs):
            verdict = judge_responses(ar, br, user_input)
            if verdict == "A":
                a_wins += 1
            elif verdict == "B":
                b_wins += 1
            else:
                ties += 1

        def avg(lst, attr):
            vals = [getattr(r, attr) for r in lst if not r.error]
            return statistics.mean(vals) if vals else 0.0

        a_lat = avg(a_runs, "latency_ms")
        b_lat = avg(b_runs, "latency_ms")
        a_cost = avg(a_runs, "cost_usd")
        b_cost = avg(b_runs, "cost_usd")
        a_tok = avg(a_runs, "output_tokens")
        b_tok = avg(b_runs, "output_tokens")

        if a_wins > b_wins:
            winner, win_rate = "A", a_wins / runs * 100
        elif b_wins > a_wins:
            winner, win_rate = "B", b_wins / runs * 100
        else:
            winner, win_rate = "tie", 50.0

        better_cost = min(a_cost, b_cost) if a_cost > 0 or b_cost > 0 else 0
        worse_cost  = max(a_cost, b_cost)
        savings = ((worse_cost - better_cost) / worse_cost * 100) if worse_cost > 0 else 0.0

        results.append(FightResult(
            model=model, runs=runs,
            a_wins=a_wins, b_wins=b_wins, ties=ties,
            a_avg_latency_ms=round(a_lat, 1), b_avg_latency_ms=round(b_lat, 1),
            a_avg_cost_usd=round(a_cost, 6), b_avg_cost_usd=round(b_cost, 6),
            a_avg_tokens=round(a_tok, 1), b_avg_tokens=round(b_tok, 1),
            winner=winner, win_rate_pct=round(win_rate, 1),
            latency_diff_ms=round(b_lat - a_lat, 1),
            cost_savings_pct=round(savings, 1),
        ))
    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def fmt_table(results: list[FightResult]) -> str:
    lines = []
    sep = "─" * 72
    lines.append("\n⚔  PromptFight Results")
    lines.append(sep)
    for r in results:
        lines.append(f"  Model : {r.model}")
        lines.append(f"  Runs  : {r.runs}   A wins: {r.a_wins}  B wins: {r.b_wins}  Ties: {r.ties}")
        a_pct = r.a_wins / r.runs * 100
        b_pct = r.b_wins / r.runs * 100
        lines.append(f"  A [{_bar(a_pct)}] {a_pct:5.1f}%")
        lines.append(f"  B [{_bar(b_pct)}] {b_pct:5.1f}%")
        lines.append(f"  Winner        : {r.winner}  ({r.win_rate_pct}%)")
        lines.append(f"  Latency  A={r.a_avg_latency_ms:.0f}ms  B={r.b_avg_latency_ms:.0f}ms  diff={r.latency_diff_ms:+.0f}ms")
        lines.append(f"  Cost     A=${r.a_avg_cost_usd:.6f}  B=${r.b_avg_cost_usd:.6f}  savings={r.cost_savings_pct:.1f}%")
        lines.append(f"  Tokens   A={r.a_avg_tokens:.0f}  B={r.b_avg_tokens:.0f}")
        lines.append(sep)
    return "\n".join(lines)


def fmt_json(results: list[FightResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)


def fmt_csv(results: list[FightResult]) -> str:
    import csv, io
    buf = io.StringIO()
    if not results:
        return ""
    writer = csv.DictWriter(buf, fieldnames=asdict(results[0]).keys())
    writer.writeheader()
    for r in results:
        writer.writerow(asdict(r))
    return buf.getvalue()


FORMATTERS = {"table": fmt_table, "json": fmt_json, "csv": fmt_csv}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="promptfight",
        description="A/B test LLM prompts in 3 lines — no LangChain required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  promptfight --prompt-a "Summarize: {input}" --prompt-b "TL;DR: {input}" --input "Long text..."
  echo "My article" | promptfight --prompt-a "Summarize: {input}" --prompt-b "TL;DR: {input}"
  promptfight --prompt-a "..." --prompt-b "..." --model gpt-4o,gpt-4o-mini --runs 5 --format json
""",
    )
    p.add_argument("--prompt-a", required=True, help="First prompt. Use {input} or {text} as placeholder.")
    p.add_argument("--prompt-b", required=True, help="Second prompt. Use {input} or {text} as placeholder.")
    p.add_argument("--input", default=None, help="Input text (or pipe via stdin).")
    p.add_argument("--model", default=",".join(DEFAULT_MODELS),
                   help=f"Comma-separated models (default: {','.join(DEFAULT_MODELS)})")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                   help=f"Number of trials per prompt per model (default: {DEFAULT_RUNS})")
    p.add_argument("--format", choices=["table", "json", "csv"], default=DEFAULT_FORMAT,
                   help=f"Output format (default: {DEFAULT_FORMAT})")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve input text
    user_input = args.input
    if user_input is None:
        if not sys.stdin.isatty():
            user_input = sys.stdin.read().strip()
        else:
            user_input = ""

    models = [m.strip() for m in args.model.split(",") if m.strip()]

    results = fight(
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
        user_input=user_input,
        models=models,
        runs=args.runs,
    )

    formatter = FORMATTERS.get(args.format, fmt_table)
    print(formatter(results))
    return results


if __name__ == "__main__":
    main()
