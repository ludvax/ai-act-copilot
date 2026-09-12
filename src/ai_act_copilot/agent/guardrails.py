"""Limits that stop an agent run from going wrong quietly.

An agent loop fails in three ways: it never stops, it costs more than the answer is worth,
or it repeats the same call forever. Each has a cheap, explicit check here rather than an
implicit hope that the model behaves.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ai_act_copilot.agent.state import AgentState


@dataclass(frozen=True, slots=True)
class Budget:
    """What a single run is allowed to spend."""

    max_steps: int = 8
    max_cost_usd: float = 0.50
    max_tokens: int = 120_000
    tool_timeout_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether the run must stop, and what to tell the user."""

    halted: bool
    reason: str | None = None

    @staticmethod
    def ok() -> "Verdict":
        return Verdict(halted=False)


def check_budget(state: AgentState, budget: Budget) -> Verdict:
    """Stop before the next model call if the run has spent its allowance."""
    steps = state.get("steps", 0)
    if steps >= budget.max_steps:
        return Verdict(True, f"stopped after {steps} steps (limit {budget.max_steps})")

    cost = state.get("cost_usd", 0.0)
    if cost >= budget.max_cost_usd:
        return Verdict(True, f"stopped at ${cost:.2f} (limit ${budget.max_cost_usd:.2f})")

    tokens = state.get("input_tokens", 0) + state.get("output_tokens", 0)
    if tokens >= budget.max_tokens:
        return Verdict(True, f"stopped at {tokens} tokens (limit {budget.max_tokens})")

    return Verdict.ok()


def fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Identify a tool call by name and arguments, so repeats are detectable."""
    payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{name}:{digest}"


def is_repeat(call: str, previous: list[str]) -> bool:
    """True when this exact call has already been made in this run."""
    return call in previous
