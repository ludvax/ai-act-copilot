"""What every LLM call returns, independent of the provider.

The agent in M4 needs the raw content blocks (tool calls and thinking blocks must be
replayed verbatim), while the RAG path only wants text or a parsed object. Both come back
in one result, together with what the call cost.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from ai_act_copilot.llm.pricing import Usage

# A message as the Messages API expects it.
type Message = dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMResult:
    """One completion, with the accounting that goes with it."""

    text: str
    model: str
    stop_reason: str | None
    usage: Usage
    cost_usd: float
    content: list[Any] = field(default_factory=list)
    parsed: BaseModel | None = None

    @property
    def refused(self) -> bool:
        """The model declined to answer; content must not be treated as an answer."""
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class LLMError(RuntimeError):
    """A call failed in a way the caller is expected to surface, not retry blindly."""


class LLMClient(Protocol):
    """The surface the RAG pipeline and the agent both use."""

    model: str

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        max_tokens: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        output_format: type[BaseModel] | None = None,
    ) -> LLMResult: ...
