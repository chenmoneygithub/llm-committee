"""Offline tests: these do not authenticate or call any model."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "extension_screen", Path(__file__).parents[1] / "scripts" / "screen_question_extension.py"
)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def response(text, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": text}}]}


def test_valid_visible_verdict():
    text = '{"question_class":"debatable_opinion","reason":"External value judgment."}'
    assert screen.parse_verdict(response(text))["question_class"] == "debatable_opinion"
    blocks = [{"type": "reasoning", "text": "not a verdict"}, {"type": "text", "text": text}]
    assert screen.parse_verdict(response(blocks))["reason"] == "External value judgment."


@pytest.mark.parametrize("text,finish", [
    ('{"question_class":"debatable_opinion","reason":"x"}', "length"),
    ('{"question_class":"agree","reason":"x"}', "stop"),
    ('{"question_class":"unclear","reason":""}', "stop"),
    ('{"question_class":"personal_report"}', "stop"),
])
def test_no_default_label_on_invalid_output(text, finish):
    with pytest.raises(ValueError):
        screen.parse_verdict(response(text, finish))


def test_reasoning_is_included_in_cost():
    judge = {"input_usd_per_million": 5, "output_usd_per_million": 30}
    raw = {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 220}}
    assert screen.usage_cost(raw, judge) == pytest.approx(1.1 * (100 * 5 + 120 * 30) / 1e6)


def test_invalid_usage_fails_closed():
    with pytest.raises(ValueError):
        screen.usage_cost({"usage": {}}, {})
