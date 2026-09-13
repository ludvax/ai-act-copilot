"""The agent's state.

Messages are Anthropic message dicts, appended and never rewritten: thinking blocks and
tool calls must be replayed to the API exactly as they came back. Every reducer is
therefore additive, which also makes a run replayable from a checkpoint.
"""

import operator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from ai_act_copilot.llm.pricing import Usage
from ai_act_copilot.models import Language


class Route(StrEnum):
    """Where the router sends a question."""

    LOOKUP = "lookup"  # one provision answers it: the plain RAG path is enough
    COMPLEX = "complex"  # needs several steps: the tool loop
    OUT_OF_SCOPE = "out_of_scope"  # the corpus cannot answer: abstain without spending tokens


class AgentState(TypedDict, total=False):
    """What flows through the graph."""

    question: str
    language: Language
    thread_id: str
    route: Route
    messages: Annotated[list[dict[str, Any]], operator.add]
    provisions: Annotated[list[str], operator.add]  # provisions actually seen by the tools
    tool_calls: Annotated[list[str], operator.add]  # fingerprints, for loop detection
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    corrections: int
    needs_correction: bool
    answer: str
    citations: list[str]
    abstained: bool
    stop_reason: str | None
    halted_by: str | None  # which guardrail ended the run, if any


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """The result of a run, with everything needed to audit it."""

    question: str
    text: str
    language: Language
    route: Route
    citations: tuple[str, ...] = ()
    abstained: bool = False
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    provisions_seen: tuple[str, ...] = ()
    halted_by: str | None = None
    thread_id: str | None = None
    # The trace this run produced, so a caller can link to it or score it afterwards.
    trace_id: str | None = None
