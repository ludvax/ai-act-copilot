# 6. An explicit LangGraph StateGraph, with nodes that call our own Claude client

- Status: accepted
- Date: 2026-09-12

## Context

Some questions need more than one retrieval: classifying a system and then deriving the
obligations that follow, or comparing what two regulations require. That is an agent. The
question is how much of it to write, and how much to inherit from a framework.

Two decisions follow: which LangGraph surface to use, and how the agent talks to Claude.

## Decision

**An explicit `StateGraph`, not `create_react_agent`.** The prebuilt agent hides exactly
the parts that make this system trustworthy — the routing decision, the citation check, the
budget guardrails. As edges and nodes they can be read, drawn
([docs/agent_graph.mmd](agent_graph.mmd)) and tested one path at a time.

**Nodes call the project's own Anthropic client, not `ChatAnthropic`.** One LLM layer for
the RAG path, the agent and the judges means prompt caching, refusal handling, cost
accounting and tracing behave identically everywhere, and the agent's messages are
Anthropic message dicts — which matters because thinking blocks and tool calls must be
replayed verbatim. LangGraph supplies the state machine, persistence and streaming; it does
not need to supply the model client too.

**A router in front of the loop.** A question answerable by one provision takes the single
retrieval path; only genuinely multi-step questions pay for the tool loop; out-of-scope
questions are refused after one cheap call.

**Four tools, with strict schemas generated from Pydantic models**: `search_regulations`,
`get_provision`, `get_definition`, and `submit_answer` as the single terminal action — so
the final answer and its citations arrive as structured arguments rather than prose to
parse.

**Guardrails live in the edges**, not in prompt wording: a step limit, a cost limit, a token
limit, a per-tool timeout, and rejection of a tool call identical to one already made. A
halted run says which limit stopped it.

**The citation check can send the agent back once.** Citations are verified against the
provisions the run actually retrieved; if some are not, the agent gets one chance to fix
them, after which unverifiable citations are dropped.

**State is append-only and checkpointed to SQLite**, so a `thread_id` continues a
conversation and a run can be replayed from its checkpoint.

## Alternatives considered

- **`create_react_agent`.** Fewer lines, but the routing, verification and guardrails would
  have to live in prompt text or wrappers, and the interesting behaviour would be untestable.
- **A hand-written loop with no framework** (the original plan). Viable, and M3's path is
  exactly that; LangGraph earns its place through persistence, conditional edges and the
  drawable graph, which a hand-rolled loop would end up reimplementing.
- **`langchain-anthropic` chat models with `ToolNode`.** Idiomatic LangChain, but it adds a
  second LLM abstraction with its own message format, caching and cost behaviour.

## Consequences

- Every run reports its route, step count, tokens, cost and whether a guardrail fired.
- The agent is tested end to end with a scripted model: routing, the tool loop, citation
  self-correction, budget halting, repeat-call rejection and thread continuation — all
  offline and free.
- Adding a tool is adding a Pydantic model, a handler and one registry entry.
