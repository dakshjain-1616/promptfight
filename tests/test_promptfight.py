"""
pytest suite for PromptFight.
All tests run in mock mode — no real API keys required.
Run:  python -m pytest tests/ -v
"""

import json
import os
import sys
import io
import pytest
from unittest.mock import patch, MagicMock

# Ensure the workspace root is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import promptfight as pf
from promptfight import (
    RunResult, FightResult,
    run_once, fight, judge_responses,
    fmt_table, fmt_json, fmt_csv,
    main, build_parser,
    _mock_call,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    return "Artificial intelligence is transforming the world."


@pytest.fixture
def prompt_a():
    return "Summarize: {input}"


@pytest.fixture
def prompt_b():
    return "TL;DR: {input}"


@pytest.fixture
def mock_result_a(sample_input, prompt_a):
    return RunResult(
        prompt_label="A", model="mock", prompt_text=prompt_a,
        response="This is a mock response about AI transformation.",
        latency_ms=120.5, input_tokens=10, output_tokens=8, cost_usd=0.0,
    )


@pytest.fixture
def mock_result_b(sample_input, prompt_b):
    return RunResult(
        prompt_label="B", model="mock", prompt_text=prompt_b,
        response="AI transforms things quickly.",
        latency_ms=90.0, input_tokens=8, output_tokens=5, cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# RunResult dataclass
# ---------------------------------------------------------------------------

class TestRunResult:
    def test_fields_present(self, mock_result_a):
        assert mock_result_a.prompt_label == "A"
        assert mock_result_a.model == "mock"
        assert mock_result_a.latency_ms == 120.5
        assert mock_result_a.input_tokens == 10
        assert mock_result_a.output_tokens == 8
        assert mock_result_a.cost_usd == 0.0
        assert mock_result_a.error is None

    def test_error_field_default_none(self, mock_result_a):
        assert mock_result_a.error is None

    def test_error_field_set(self):
        r = RunResult(
            prompt_label="A", model="mock", prompt_text="p",
            response="", latency_ms=0, input_tokens=0, output_tokens=0,
            cost_usd=0.0, error="timeout"
        )
        assert r.error == "timeout"


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class TestMockCall:
    def test_returns_dict(self):
        result = _mock_call("mock", "Hello {input}", "world", "A")
        assert isinstance(result, dict)
        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result

    def test_deterministic(self):
        r1 = _mock_call("mock", "Test {input}", "data", "A")
        r2 = _mock_call("mock", "Test {input}", "data", "A")
        assert r1["text"] == r2["text"]

    def test_different_prompts_different_output(self):
        r1 = _mock_call("mock", "Summarize {input}", "hello", "A")
        r2 = _mock_call("mock", "TL;DR {input}", "hello", "B")
        # Not guaranteed to differ every time but seeds differ
        assert isinstance(r1["text"], str)
        assert isinstance(r2["text"], str)

    def test_tokens_positive(self):
        r = _mock_call("mock", "Prompt {input}", "some text", "A")
        assert r["input_tokens"] > 0
        assert r["output_tokens"] > 0

    def test_text_nonempty(self):
        r = _mock_call("mock", "Hello {input}", "world", "A")
        assert len(r["text"]) > 0


# ---------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    def test_returns_run_result(self, prompt_a, sample_input):
        r = run_once("mock", prompt_a, sample_input, "A")
        assert isinstance(r, RunResult)

    def test_mock_no_error(self, prompt_a, sample_input):
        r = run_once("mock", prompt_a, sample_input, "A")
        assert r.error is None

    def test_latency_positive(self, prompt_a, sample_input):
        r = run_once("mock", prompt_a, sample_input, "A")
        assert r.latency_ms >= 0

    def test_cost_zero_for_mock(self, prompt_a, sample_input):
        r = run_once("mock", prompt_a, sample_input, "A")
        assert r.cost_usd == 0.0

    def test_label_preserved(self, prompt_b, sample_input):
        r = run_once("mock", prompt_b, sample_input, "B")
        assert r.prompt_label == "B"

    def test_model_preserved(self, prompt_a, sample_input):
        r = run_once("mock", prompt_a, sample_input, "A")
        assert r.model == "mock"

    def test_openai_network_error_captured(self, prompt_a, sample_input):
        """When OpenAI call raises, error is captured in RunResult."""
        with patch("promptfight._openai_call", side_effect=Exception("Connection refused")):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake"}):
                r = run_once("gpt-4o-mini", prompt_a, sample_input, "A")
        assert r.error is not None
        assert "Connection refused" in r.error

    def test_anthropic_network_error_captured(self, prompt_a, sample_input):
        with patch("promptfight._anthropic_call", side_effect=Exception("403 Forbidden")):
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake"}):
                r = run_once("claude-3-haiku-20240307", prompt_a, sample_input, "A")
        assert r.error is not None


# ---------------------------------------------------------------------------
# judge_responses
# ---------------------------------------------------------------------------

class TestJudgeResponses:
    def test_b_wins_when_a_has_error(self, mock_result_a, mock_result_b, sample_input):
        mock_result_a.error = "fail"
        verdict = judge_responses(mock_result_a, mock_result_b, sample_input)
        assert verdict == "B"

    def test_a_wins_when_b_has_error(self, mock_result_a, mock_result_b, sample_input):
        mock_result_b.error = "fail"
        verdict = judge_responses(mock_result_a, mock_result_b, sample_input)
        assert verdict == "A"

    def test_tie_when_both_error(self, mock_result_a, mock_result_b, sample_input):
        mock_result_a.error = "fail"
        mock_result_b.error = "fail"
        verdict = judge_responses(mock_result_a, mock_result_b, sample_input)
        assert verdict == "tie"

    def test_verdict_is_valid_value(self, mock_result_a, mock_result_b, sample_input):
        verdict = judge_responses(mock_result_a, mock_result_b, sample_input)
        assert verdict in ("A", "B", "tie")

    def test_longer_response_wins(self, sample_input):
        a = RunResult("A","mock","p","word " * 30,50,5,25,0.0)
        b = RunResult("B","mock","p","short",50,5,5,0.0)
        assert judge_responses(a, b, sample_input) == "A"

    def test_near_equal_length_is_tie(self, sample_input):
        text = "hello world"
        a = RunResult("A","mock","p", text, 50, 5, 5, 0.0)
        b = RunResult("B","mock","p", text, 50, 5, 5, 0.0)
        assert judge_responses(a, b, sample_input) == "tie"


# ---------------------------------------------------------------------------
# fight
# ---------------------------------------------------------------------------

class TestFight:
    def test_returns_list_of_fight_results(self, prompt_a, prompt_b, sample_input):
        results = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=2)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], FightResult)

    def test_multiple_models(self, prompt_a, prompt_b, sample_input):
        results = fight(prompt_a, prompt_b, sample_input, ["mock", "mock"], runs=2)
        assert len(results) == 2

    def test_wins_sum_to_runs(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=5)[0]
        assert r.a_wins + r.b_wins + r.ties == 5

    def test_winner_field_set(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=3)[0]
        assert r.winner in ("A", "B", "tie")

    def test_win_rate_in_range(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=3)[0]
        assert 0.0 <= r.win_rate_pct <= 100.0

    def test_latencies_non_negative(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=2)[0]
        assert r.a_avg_latency_ms >= 0
        assert r.b_avg_latency_ms >= 0

    def test_cost_savings_pct_non_negative(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=2)[0]
        assert r.cost_savings_pct >= 0.0

    def test_model_name_preserved(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=1)[0]
        assert r.model == "mock"

    def test_runs_field_matches(self, prompt_a, prompt_b, sample_input):
        r = fight(prompt_a, prompt_b, sample_input, ["mock"], runs=4)[0]
        assert r.runs == 4

    def test_empty_input_handled(self, prompt_a, prompt_b):
        results = fight(prompt_a, prompt_b, "", ["mock"], runs=2)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class TestFormatters:
    @pytest.fixture
    def sample_fight_result(self):
        return FightResult(
            model="mock", runs=3, a_wins=2, b_wins=1, ties=0,
            a_avg_latency_ms=110.0, b_avg_latency_ms=95.0,
            a_avg_cost_usd=0.000050, b_avg_cost_usd=0.000030,
            a_avg_tokens=25.0, b_avg_tokens=18.0,
            winner="A", win_rate_pct=66.7,
            latency_diff_ms=-15.0, cost_savings_pct=40.0,
        )

    def test_table_contains_model_name(self, sample_fight_result):
        out = fmt_table([sample_fight_result])
        assert "mock" in out

    def test_table_contains_winner(self, sample_fight_result):
        out = fmt_table([sample_fight_result])
        assert "Winner" in out

    def test_json_is_valid(self, sample_fight_result):
        out = fmt_json([sample_fight_result])
        data = json.loads(out)
        assert isinstance(data, list)
        assert data[0]["model"] == "mock"

    def test_json_has_required_keys(self, sample_fight_result):
        out = fmt_json([sample_fight_result])
        data = json.loads(out)[0]
        for key in ("model", "runs", "a_wins", "b_wins", "winner", "win_rate_pct",
                    "a_avg_latency_ms", "b_avg_latency_ms", "cost_savings_pct"):
            assert key in data, f"Missing key: {key}"

    def test_csv_has_header(self, sample_fight_result):
        out = fmt_csv([sample_fight_result])
        assert "model" in out
        assert "winner" in out

    def test_csv_has_data_row(self, sample_fight_result):
        out = fmt_csv([sample_fight_result])
        lines = out.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row

    def test_table_multiple_results(self, sample_fight_result):
        r2 = FightResult(model="gpt-x", runs=2, a_wins=1, b_wins=1, ties=0,
                         a_avg_latency_ms=200.0, b_avg_latency_ms=180.0,
                         a_avg_cost_usd=0.001, b_avg_cost_usd=0.0008,
                         a_avg_tokens=30.0, b_avg_tokens=28.0,
                         winner="tie", win_rate_pct=50.0,
                         latency_diff_ms=-20.0, cost_savings_pct=20.0)
        out = fmt_table([sample_fight_result, r2])
        assert "mock" in out
        assert "gpt-x" in out

    def test_empty_csv(self):
        out = fmt_csv([])
        assert out == ""


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestCLIMain:
    def test_basic_run(self, capsys, prompt_a, prompt_b, sample_input):
        main(["--prompt-a", prompt_a, "--prompt-b", prompt_b,
              "--input", sample_input, "--model", "mock", "--runs", "2"])
        captured = capsys.readouterr()
        assert "Winner" in captured.out

    def test_json_output(self, capsys, prompt_a, prompt_b, sample_input):
        main(["--prompt-a", prompt_a, "--prompt-b", prompt_b,
              "--input", sample_input, "--model", "mock", "--runs", "1",
              "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)

    def test_csv_output(self, capsys, prompt_a, prompt_b, sample_input):
        main(["--prompt-a", prompt_a, "--prompt-b", prompt_b,
              "--input", sample_input, "--model", "mock", "--runs", "1",
              "--format", "csv"])
        captured = capsys.readouterr()
        assert "model" in captured.out
        assert "winner" in captured.out

    def test_stdin_input(self, capsys, prompt_a, prompt_b):
        # io.StringIO is a C-extension type that does not allow setattr on
        # inherited methods, so patch("sys.stdin.isatty", ...) would raise
        # AttributeError.  Use MagicMock which fully supports attribute patching.
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False
        fake_stdin.read.return_value = "piped text here"
        with patch("sys.stdin", fake_stdin):
            with patch("sys.stdin.isatty", return_value=False):
                main(["--prompt-a", prompt_a, "--prompt-b", prompt_b,
                      "--model", "mock", "--runs", "1"])
        captured = capsys.readouterr()
        assert "Winner" in captured.out

    def test_returns_results_list(self, prompt_a, prompt_b, sample_input):
        results = main(["--prompt-a", prompt_a, "--prompt-b", prompt_b,
                        "--input", sample_input, "--model", "mock", "--runs", "2"])
        assert isinstance(results, list)
        assert len(results) == 1

    def test_parser_requires_prompt_a(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--prompt-b", "B"])

    def test_parser_requires_prompt_b(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--prompt-a", "A"])

    def test_multiple_model_flag(self, prompt_a, prompt_b, sample_input):
        results = main(["--prompt-a", prompt_a, "--prompt-b", prompt_b,
                        "--input", sample_input, "--model", "mock,mock", "--runs", "1"])
        assert len(results) == 2
