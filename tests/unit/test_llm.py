from typing import Any, cast

import anthropic
import pytest
from pydantic import BaseModel

from ai_act_copilot.llm.anthropic_client import AnthropicLLM
from ai_act_copilot.llm.base import LLMError
from ai_act_copilot.llm.pricing import Usage, cost_usd
from tests.doubles import FakeAnthropic
from tests.doubles import anthropic_response as _response


class Schema(BaseModel):
    answer: str


def _llm(fake: FakeAnthropic, **kwargs: Any) -> AnthropicLLM:
    return AnthropicLLM(client=cast(anthropic.Anthropic, fake), **kwargs)


def test_cost_uses_reported_usage_including_cache_rates() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost_usd("claude-opus-5", usage) == pytest.approx(30.0)
    # A cached read costs a tenth of an input token.
    assert cost_usd("claude-opus-5", Usage(cache_read_tokens=1_000_000)) == pytest.approx(0.5)
    assert cost_usd("claude-opus-5", Usage(cache_write_tokens=1_000_000)) == pytest.approx(6.25)


def test_unknown_model_costs_nothing_rather_than_guessing() -> None:
    assert cost_usd("some-local-model", Usage(input_tokens=1000)) == 0.0


def test_usage_adds_up_across_calls() -> None:
    total = Usage(input_tokens=10, output_tokens=5) + Usage(input_tokens=3, cache_read_tokens=7)

    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens) == (13, 5, 7)
    assert total.total_tokens == 25


def test_system_prompt_is_marked_cacheable_and_effort_is_sent() -> None:
    fake = FakeAnthropic()

    _llm(fake, effort="medium").complete(
        system="rules", messages=[{"role": "user", "content": "q"}]
    )

    _, kwargs = fake.calls[0]
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["output_config"] == {"effort": "medium"}
    assert "thinking" not in kwargs  # Opus 5 reasons by default


def test_fallbacks_are_enabled_by_default_and_can_be_switched_off() -> None:
    with_fallbacks = FakeAnthropic()
    _llm(with_fallbacks).complete(system="s", messages=[{"role": "user", "content": "q"}])

    without = FakeAnthropic()
    _llm(without, enable_fallbacks=False).complete(
        system="s", messages=[{"role": "user", "content": "q"}]
    )

    surface, kwargs = with_fallbacks.calls[0]
    assert surface == "beta.create"
    assert kwargs["fallbacks"] == "default"
    assert without.calls[0][0] == "create"


def test_structured_output_uses_the_parse_surface() -> None:
    fake = FakeAnthropic(_response(parsed_output=Schema(answer="yes")))

    result = _llm(fake).complete(
        system="s", messages=[{"role": "user", "content": "q"}], output_format=Schema
    )

    assert fake.calls[0][0] == "parse"
    assert fake.calls[0][1]["output_format"] is Schema
    assert isinstance(result.parsed, Schema)


def test_result_carries_usage_cost_and_text() -> None:
    result = _llm(FakeAnthropic()).complete(system="s", messages=[{"role": "user", "content": "q"}])

    assert result.text.startswith("Prohibited practices")
    assert result.usage.input_tokens == 1000
    assert result.usage.cache_read_tokens == 500
    assert result.cost_usd > 0
    assert not result.refused


def test_refusal_is_surfaced_not_swallowed() -> None:
    fake = FakeAnthropic(_response(stop_reason="refusal", content=[]))

    result = _llm(fake).complete(system="s", messages=[{"role": "user", "content": "q"}])

    assert result.refused
    assert result.text == ""


def test_truncated_output_is_flagged() -> None:
    fake = FakeAnthropic(_response(stop_reason="max_tokens"))

    assert _llm(fake).complete(system="s", messages=[{"role": "user", "content": "q"}]).truncated


def test_api_errors_become_actionable_llm_errors() -> None:
    error = anthropic.APIConnectionError(request=cast(Any, None))
    fake = FakeAnthropic(error=error)

    with pytest.raises(LLMError, match="could not reach the Anthropic API"):
        _llm(fake).complete(system="s", messages=[{"role": "user", "content": "q"}])
