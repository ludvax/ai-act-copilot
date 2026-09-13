"""Claude access through the official Anthropic SDK.

One LLM layer for the whole system: the RAG path, the LangGraph agent and the judges all
go through here, so prompt caching, refusal handling, cost accounting and tracing are
implemented once.

Notes on the request shape:

- Claude Opus 5 thinks by default, so no ``thinking`` parameter is sent; depth is steered
  with ``output_config.effort``.
- The system prompt is marked cacheable. It is the stable prefix of every call, and cache
  reads are billed at a tenth of the input rate - ``usage.cache_read_tokens`` proves it.
- Server-side fallbacks are enabled by default: if a safety classifier declines a request,
  the API retries it on a fallback model inside the same call instead of returning nothing.
"""

import logging
from collections.abc import Sequence
from typing import Any

import anthropic
from pydantic import BaseModel

from ai_act_copilot.llm.base import DEFAULT_GENERATION_NAME, LLMError, LLMResult, Message
from ai_act_copilot.llm.pricing import Usage, cost_usd
from ai_act_copilot.observability.tracing import observe, record_generation

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Synchronous Claude client returning provider-neutral results."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        enable_fallbacks: bool = True,
        timeout: float = 120.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.enable_fallbacks = enable_fallbacks
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout)

    # Named generically here and renamed per call site below: the decorator runs once, the
    # name has to say what this particular call was for.
    @observe(
        name=DEFAULT_GENERATION_NAME,
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        max_tokens: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_format: type[BaseModel] | None = None,
        name: str = DEFAULT_GENERATION_NAME,
    ) -> LLMResult:
        """One completion. Raises :class:`LLMError` for API failures."""
        budget = max_tokens or self.max_tokens
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": budget,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": list(messages),
            "output_config": {"effort": self.effort},
        }
        if tools:
            request["tools"] = list(tools)

        # Described before the call, not after. A request that fails never reaches the
        # second update, and a failed call is the one you most want to find and read: it
        # has to carry its name and the conversation that produced it, not the placeholder.
        record_generation(
            name=name,
            model=self.model,
            input=[{"role": "system", "content": system}, *messages],
            model_parameters={"effort": self.effort, "max_tokens": budget},
            metadata={
                "structured_output": output_format.__name__ if output_format else None,
                "tools_offered": [str(tool.get("name", "")) for tool in tools or ()],
                "fallbacks": self.enable_fallbacks and output_format is None,
            },
        )
        try:
            response = self._send(request, output_format)
        except anthropic.APIStatusError as error:
            raise LLMError(
                f"Claude request failed ({error.status_code}): {error.message}"
            ) from error
        except anthropic.APIConnectionError as error:
            raise LLMError(f"could not reach the Anthropic API: {error}") from error

        result = _to_result(response, self.model)
        record_generation(
            name=name,
            # The model that actually served the call: a server-side fallback changes it.
            model=result.model,
            output=_generation_output(result),
            usage_details={
                "input": result.usage.input_tokens,
                "output": result.usage.output_tokens,
                "cache_read_input_tokens": result.usage.cache_read_tokens,
                "cache_creation_input_tokens": result.usage.cache_write_tokens,
            },
            cost_usd=result.cost_usd,
            metadata={"stop_reason": result.stop_reason},
        )
        if result.refused:
            logger.warning("Claude declined the request (stop_reason=refusal)")
        return result

    def _send(self, request: dict[str, Any], output_format: type[BaseModel] | None) -> Any:
        if output_format is not None:
            # Structured output guarantees a schema-valid answer; it is the GA surface, so
            # server-side fallbacks do not apply to it.
            return self._client.messages.parse(output_format=output_format, **request)
        if self.enable_fallbacks:
            return self._client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **request
            )
        return self._client.messages.create(**request)


def _generation_output(result: LLMResult) -> Any:
    """What the model produced, in the most readable form available.

    A tool-calling turn is worth showing as its content blocks - which tool, which
    arguments - because that decision is usually what a trace is being read for.
    """
    if result.parsed is not None:
        return result.parsed.model_dump()
    if any(getattr(block, "type", None) == "tool_use" for block in result.content):
        return [
            block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            for block in result.content
        ]
    return result.text


def _to_result(response: Any, requested_model: str) -> LLMResult:
    usage = _usage(response)
    model = getattr(response, "model", requested_model)
    content = list(getattr(response, "content", []))
    text = "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    ).strip()
    return LLMResult(
        text=text,
        model=model,
        stop_reason=getattr(response, "stop_reason", None),
        usage=usage,
        cost_usd=cost_usd(model, usage),
        content=content,
        parsed=getattr(response, "parsed_output", None),
    )


def _usage(response: Any) -> Usage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage()
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )
