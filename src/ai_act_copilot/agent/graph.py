r"""The agent graph.

An explicit StateGraph rather than a prebuilt ReAct agent: the routing, the citation check
and the guardrails are the interesting parts of this system, and they belong in edges that
can be read, drawn and tested. The nodes call our own Claude client, so caching, cost
accounting and tracing stay identical to the non-agent path.

    START -> route -> rag_answer -----------------+
                   -> agent <-> tools            |
                          \-> verify -(bad cite)-/ -> finalize -> END
                   -> abstain --------------------/
"""

import logging
import sqlite3
import uuid
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ai_act_copilot.agent.nodes import (
    AgentDeps,
    abstain_node,
    agent_node,
    finalize_node,
    rag_node,
    route_node,
    tools_node,
    verify_node,
)
from ai_act_copilot.agent.state import AgentAnswer, AgentState, Route
from ai_act_copilot.llm.pricing import Usage
from ai_act_copilot.models import Language
from ai_act_copilot.observability.tracing import (
    current_trace_id,
    observe,
    record_span,
    trace_context,
)

logger = logging.getLogger(__name__)


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Wire the graph. Every branch is a named function, so the diagram matches the code."""
    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)

    graph.add_node("route", partial(route_node, deps=deps))
    graph.add_node("rag_answer", partial(rag_node, deps=deps))
    graph.add_node("agent", partial(agent_node, deps=deps))
    graph.add_node("tools", partial(tools_node, deps=deps))
    graph.add_node("verify", partial(verify_node, deps=deps))
    graph.add_node("abstain", partial(abstain_node, deps=deps))
    graph.add_node("finalize", partial(finalize_node, deps=deps))

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        after_route,
        {"rag_answer": "rag_answer", "agent": "agent", "abstain": "abstain"},
    )
    graph.add_edge("rag_answer", "verify")
    graph.add_edge("abstain", "finalize")
    graph.add_conditional_edges(
        "agent", after_agent, {"tools": "tools", "verify": "verify", "finalize": "finalize"}
    )
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "verify": "verify"})
    graph.add_conditional_edges("verify", after_verify, {"agent": "agent", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def after_route(state: AgentState) -> str:
    route = state.get("route", Route.COMPLEX)
    if route is Route.OUT_OF_SCOPE:
        return "abstain"
    return "rag_answer" if route is Route.LOOKUP else "agent"


def after_agent(state: AgentState) -> str:
    if state.get("halted_by") or state.get("stop_reason") == "refusal":
        return "finalize"
    return "tools" if state.get("stop_reason") == "tool_use" else "verify"


def after_tools(state: AgentState) -> str:
    # An answer appears only when the model called submit_answer.
    return "verify" if state.get("answer") else "agent"


def after_verify(state: AgentState) -> str:
    return "agent" if state.get("needs_correction") else "finalize"


def open_checkpointer(path: Path) -> SqliteSaver:
    """A SQLite checkpointer, so a thread_id can carry a conversation across calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


@observe(name="agent-run", as_type="agent", capture_input=False, capture_output=False)
def run_agent(
    question: str,
    deps: AgentDeps,
    *,
    thread_id: str | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    language: Language | None = None,
    graph: Any | None = None,
) -> AgentAnswer:
    """Answer one question, optionally continuing a previous conversation."""
    compiled = graph if graph is not None else build_graph(deps, checkpointer)
    thread = thread_id or str(uuid.uuid4())

    initial: dict[str, Any] = {
        "question": question,
        "messages": [{"role": "user", "content": question}],
    }
    if language is not None:
        initial["language"] = language

    # One turn is one trace; the thread ties the turns of a conversation into a session,
    # which is the only way a follow-up reads as a follow-up rather than a lone question.
    with trace_context(name="agent-run", session_id=thread):
        final: dict[str, Any] = compiled.invoke(
            initial, config={"configurable": {"thread_id": thread}}
        )

    answer = _to_answer(question, thread, final)
    record_span(
        input=question,
        output={
            "answer": answer.text,
            "citations": list(answer.citations),
            "abstained": answer.abstained,
        },
        metadata={
            "route": str(answer.route),
            "steps": answer.steps,
            "language": answer.language.value,
            "provisions_seen": list(answer.provisions_seen),
            "halted_by": answer.halted_by,
            "cost_usd": round(answer.cost_usd, 6),
            "resumed": thread_id is not None,
        },
    )
    return replace(answer, trace_id=current_trace_id())


def _to_answer(question: str, thread_id: str, final: dict[str, Any]) -> AgentAnswer:
    return AgentAnswer(
        question=question,
        text=final.get("answer", ""),
        language=final.get("language", Language.EN),
        route=final.get("route", Route.COMPLEX),
        citations=tuple(final.get("citations", [])),
        abstained=bool(final.get("abstained", False)),
        steps=int(final.get("steps", 0)),
        usage=Usage(
            input_tokens=int(final.get("input_tokens", 0)),
            output_tokens=int(final.get("output_tokens", 0)),
        ),
        cost_usd=float(final.get("cost_usd", 0.0)),
        provisions_seen=tuple(dict.fromkeys(final.get("provisions", []))),
        halted_by=final.get("halted_by"),
        thread_id=thread_id,
    )


def mermaid(deps: AgentDeps) -> str:
    """The graph as a mermaid diagram, kept in docs/agent_graph.mmd."""
    drawn: str = build_graph(deps).get_graph().draw_mermaid()
    return drawn
