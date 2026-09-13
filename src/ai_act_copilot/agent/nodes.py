"""The graph's nodes.

Each node takes the state and returns only the fields it changes; dependencies arrive
through :class:`AgentDeps` rather than globals, so the graph can be built against fakes in
tests. Every node is traced, which is what makes a run legible afterwards: one span per
step, with the model calls nested inside.
"""

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_act_copilot.agent.guardrails import Budget, check_budget, fingerprint, is_repeat
from ai_act_copilot.agent.state import AgentState, Route
from ai_act_copilot.agent.tools import Tool, ToolContext, default_tools, run_tool
from ai_act_copilot.generation.answer import answer_question
from ai_act_copilot.generation.citations import check_citations
from ai_act_copilot.generation.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from ai_act_copilot.llm.base import LLMClient
from ai_act_copilot.models import Language
from ai_act_copilot.observability.tracing import (
    in_current_trace,
    observe,
    observed,
    record_span,
)
from ai_act_copilot.retrieval.base import Retriever
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.text_analysis import detect_language

logger = logging.getLogger(__name__)

ROUTER_PROMPT = "router_v1"
AGENT_PROMPT = "agent_system_v1"
MAX_CORRECTIONS = 1

_NOT_COVERED = {
    Language.EN: (
        "That question is outside this corpus, which covers the AI Act, the GDPR and EU "
        "guidance on them."
    ),
    Language.FR: (
        "Cette question sort du corpus, qui couvre le reglement IA, le RGPD et les lignes "
        "directrices europeennes associees."
    ),
}


class RouteDecision(BaseModel):
    """What the router returns."""

    route: Route = Field(description="lookup, complex or out_of_scope.")
    reason: str = Field(default="", description="One short sentence.")


@dataclass(slots=True)
class AgentDeps:
    """Everything the nodes need, injected once when the graph is built."""

    llm: LLMClient
    retriever: Retriever
    store: CorpusStore
    budget: Budget = field(default_factory=Budget)
    tools: tuple[Tool, ...] = field(default_factory=default_tools)
    prompts_dir: Path = DEFAULT_PROMPTS_DIR

    def tool_by_name(self, name: str) -> Tool | None:
        return next((tool for tool in self.tools if tool.name == name), None)


@observe(name="route-question", capture_input=False, capture_output=False)
def route_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Send cheap questions down the cheap path, and refuse off-topic ones early."""
    question = state["question"]
    language = state.get("language") or detect_language(question)
    prompt = load_prompt(ROUTER_PROMPT, deps.prompts_dir)

    result = deps.llm.complete(
        system=prompt.text,
        messages=[{"role": "user", "content": question}],
        max_tokens=256,
        output_format=RouteDecision,
        name="route-question",
    )
    decision = result.parsed if isinstance(result.parsed, RouteDecision) else None
    route = decision.route if decision else Route.COMPLEX
    logger.info("routed as %s", route)

    record_span(
        input=question,
        output={"route": route.value, "reason": decision.reason if decision else ""},
        metadata={"language": language.value, "prompt_version": prompt.version},
    )
    return {
        "language": language,
        "route": route,
        **_spend(state, result.usage.input_tokens, result.usage.output_tokens, result.cost_usd),
    }


def rag_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """One retrieval, one grounded answer - the M3 path, reused unchanged.

    Deliberately not observed: it delegates to ``answer_question``, which is already a
    span with the same input and output. A wrapper around a single child adds a level
    to the tree and nothing to read in it.
    """
    answer = answer_question(
        state["question"],
        retriever=deps.retriever,
        llm=deps.llm,
        language=state.get("language"),
        prompts_dir=deps.prompts_dir,
    )
    return {
        "answer": answer.text,
        "citations": list(answer.citations),
        "abstained": answer.abstained,
        "provisions": [provision for hit in answer.passages for provision in hit.provision_ids],
        **_spend(state, answer.usage.input_tokens, answer.usage.output_tokens, answer.cost_usd),
    }


@observe(name="agent-step", capture_input=False, capture_output=False)
def agent_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """One turn of the tool loop."""
    step = state.get("steps", 0)
    verdict = check_budget(state, deps.budget)
    if verdict.halted:
        logger.warning("guardrail: %s", verdict.reason)
        # A run that stopped early is the one worth finding later, so say so on the span.
        record_span(
            output={"halted_by": verdict.reason},
            metadata={"step": step, "guardrail": "halted"},
        )
        return {"halted_by": verdict.reason, "stop_reason": "halted"}

    prompt = load_prompt(AGENT_PROMPT, deps.prompts_dir)
    result = deps.llm.complete(
        system=prompt.text,
        messages=state.get("messages", []),
        tools=[tool.definition() for tool in deps.tools],
        name="agent-turn",
    )
    spent = _spend(state, result.usage.input_tokens, result.usage.output_tokens, result.cost_usd)
    if result.refused:
        return {
            "answer": _NOT_COVERED[state.get("language", Language.EN)],
            "abstained": True,
            "stop_reason": "refusal",
            **spent,
        }

    blocks = _as_dicts(result.content)
    record_span(
        output={
            "stop_reason": result.stop_reason,
            "tool_calls": [
                block.get("name") for block in blocks if block.get("type") == "tool_use"
            ],
        },
        metadata={"step": step + 1, "cost_so_far_usd": round(spent["cost_usd"], 6)},
    )
    return {
        # Content is replayed verbatim: thinking blocks and tool calls must survive intact.
        "messages": [{"role": "assistant", "content": blocks}],
        "steps": step + 1,
        "stop_reason": result.stop_reason,
        **spent,
    }


def _observed_tool(tool: Tool, arguments: dict[str, Any], context: ToolContext) -> tuple[str, bool]:
    """Run one tool inside its own observation.

    Named after the tool the model chose, so a trace shows which action was taken rather
    than that "a tool" ran - and so any retrieval it performs nests underneath it.
    """
    with observed(name=tool.name, as_type="tool", input=arguments) as span:
        text, failed = run_tool(tool, arguments, context)
        span.update(output=text, metadata={"failed": failed, "characters": len(text)})
        return text, failed


@observe(name="run-tools", capture_input=False, capture_output=False)
def tools_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Run every tool the model asked for, returning all results in one message.

    Parallel calls are executed concurrently and their results must come back together:
    splitting them across messages teaches the model to stop asking for parallel calls.
    """
    calls = _tool_calls(state)
    context = ToolContext(
        retriever=deps.retriever, store=deps.store, language=state.get("language", Language.EN)
    )
    previous = state.get("tool_calls", [])
    markers = [
        fingerprint(str(call.get("name", "")), dict(call.get("input", {}))) for call in calls
    ]

    submitted: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending: list[tuple[dict[str, Any], Any]] = []
        for call, marker in zip(calls, markers, strict=True):
            name = str(call.get("name", ""))
            tool = deps.tool_by_name(name)
            if name == "submit_answer":
                submitted = dict(call.get("input", {}))
            if tool is None:
                results.append(_tool_result(call, f"Unknown tool: {name}", is_error=True))
                continue
            if name != "submit_answer" and is_repeat(marker, previous):
                results.append(
                    _tool_result(
                        call,
                        "You already made this exact call. Use the earlier result, or try a "
                        "different query.",
                        is_error=True,
                    )
                )
                continue
            # Wrapped on this thread: the snapshot has to be taken where the parent
            # observation is active, not inside the worker that will run the tool.
            pending.append(
                (
                    call,
                    pool.submit(
                        in_current_trace(_observed_tool),
                        tool,
                        dict(call.get("input", {})),
                        context,
                    ),
                )
            )

        for call, future in pending:
            try:
                text, failed = future.result(timeout=deps.budget.tool_timeout_seconds)
            except TimeoutError:
                seconds = deps.budget.tool_timeout_seconds
                text, failed = f"{call.get('name')} timed out after {seconds:.0f}s", True
            results.append(_tool_result(call, text, is_error=failed))

    record_span(
        input=[{"tool": call.get("name"), "input": call.get("input")} for call in calls],
        output=[
            {"tool": call.get("name"), "is_error": result["is_error"]}
            for call, result in zip(calls, results, strict=False)
        ],
        metadata={
            "calls": len(calls),
            "errors": sum(1 for result in results if result["is_error"]),
            "provisions_seen": list(dict.fromkeys(context.provisions)),
        },
    )
    update: dict[str, Any] = {
        "messages": [{"role": "user", "content": results}],
        "provisions": list(dict.fromkeys(context.provisions)),
        "tool_calls": markers,
    }
    if submitted:
        update |= {
            "answer": str(submitted.get("answer", "")),
            "citations": [str(citation) for citation in submitted.get("citations", [])],
            "abstained": bool(submitted.get("abstained", False)),
        }
    return update


# An evaluator: it judges whether the model's output is supported by what the run read,
# and that verdict is what decides between finalising and asking for a correction.
@observe(name="verify-citations", as_type="evaluator", capture_input=False, capture_output=False)
def verify_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Reject citations the run never retrieved, giving the agent one chance to fix them."""
    allowed = set(state.get("provisions", []))
    cited = list(state.get("citations", []))
    check = check_citations(cited, allowed)
    record_span(
        input={"citations": cited, "retrieved": sorted(allowed)},
        output={"valid": list(check.valid), "unsupported": list(check.invalid)},
        metadata={"corrections_so_far": state.get("corrections", 0)},
    )
    if not check.invalid:
        return {"citations": list(check.valid), "needs_correction": False}

    corrections = state.get("corrections", 0)
    if corrections >= MAX_CORRECTIONS:
        logger.warning("dropping unverifiable citations: %s", check.invalid)
        return {"citations": list(check.valid), "needs_correction": False}

    logger.info("asking the agent to correct citations: %s", check.invalid)
    return {
        "corrections": corrections + 1,
        "citations": list(check.valid),
        "needs_correction": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    "These citations are not in anything you retrieved: "
                    f"{', '.join(check.invalid)}. Retrieve them with get_provision, or submit "
                    "an answer citing only what you have actually read."
                ),
            }
        ],
    }


@observe(name="abstain", capture_input=False, capture_output=False)
def abstain_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Out-of-scope questions cost nothing to refuse."""
    refusal = _NOT_COVERED[state.get("language", Language.EN)]
    record_span(
        input=state["question"],
        output=refusal,
        metadata={"reason": "out of scope for the corpus"},
    )
    return {
        "answer": refusal,
        "abstained": True,
        "citations": [],
    }


def finalize_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Make sure a run always ends with something honest to show."""
    if state.get("answer"):
        return {}
    if spoken := _last_assistant_text(state):
        # The model answered in plain text instead of calling submit_answer.
        return {"answer": spoken}
    halted = state.get("halted_by")
    language = state.get("language", Language.EN)
    text = f"No answer within the run budget: {halted}." if halted else _NOT_COVERED[language]
    return {"answer": text, "abstained": True}


def _last_assistant_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            return content
        return " ".join(
            str(block.get("text", "")) for block in content if block.get("type") == "text"
        ).strip()
    return ""


def _spend(state: AgentState, input_tokens: int, output_tokens: int, cost: float) -> dict[str, Any]:
    return {
        "input_tokens": state.get("input_tokens", 0) + input_tokens,
        "output_tokens": state.get("output_tokens", 0) + output_tokens,
        "cost_usd": state.get("cost_usd", 0.0) + cost,
    }


def _as_dicts(content: Sequence[Any]) -> list[dict[str, Any]]:
    """Plain dicts, so the state can be checkpointed and replayed to the API."""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict):
            blocks.append(block)
        elif hasattr(block, "model_dump"):
            blocks.append(block.model_dump(exclude_none=True))
    return blocks


def _tool_calls(state: AgentState) -> list[dict[str, Any]]:
    messages = state.get("messages", [])
    if not messages:
        return []
    content = messages[-1].get("content", [])
    if isinstance(content, str):
        return []
    return [block for block in content if block.get("type") == "tool_use"]


def _tool_result(call: dict[str, Any], text: str, *, is_error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": call.get("id", ""),
        "content": text,
        "is_error": is_error,
    }
