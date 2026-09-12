"""The tools the agent can call.

Each tool is a Pydantic input model plus a handler. The JSON Schema handed to Claude is
generated from the model and tightened for strict mode (every field required, no extra
properties), so tool arguments arrive already valid instead of being defensively parsed.

Tool results are data, never instructions: they are corpus text, and the system prompt
tells the model to treat them as such.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ai_act_copilot.models import Language
from ai_act_copilot.retrieval.base import RetrievedChunk, Retriever
from ai_act_copilot.store.sqlite import CorpusStore

logger = logging.getLogger(__name__)

# Where definitions live in each text.
DEFINITION_PROVISIONS = ("ai_act:art:3", "gdpr:art:4")


@dataclass(slots=True)
class ToolContext:
    """What handlers need, plus what the run has actually seen.

    Tracking retrieved provisions is what lets the citation check reject an answer that
    cites something the agent never looked at.
    """

    retriever: Retriever
    store: CorpusStore
    language: Language
    passages: list[RetrievedChunk] = field(default_factory=list)
    provisions: list[str] = field(default_factory=list)

    def remember_hits(self, hits: Sequence[RetrievedChunk]) -> None:
        self.passages.extend(hits)
        self.provisions.extend(provision for hit in hits for provision in hit.provision_ids)

    def remember_provision(self, provision_id: str) -> None:
        self.provisions.append(provision_id)


class SearchRegulations(BaseModel):
    query: str = Field(description="What to look for, in the language of the question.")
    sources: list[str] | None = Field(
        default=None, description='Restrict to these texts, e.g. ["ai_act"] or ["gdpr"].'
    )
    limit: int | None = Field(default=None, description="How many passages to return (max 8).")


class GetProvision(BaseModel):
    provision_id: str = Field(
        description='Exact id, e.g. "ai_act:art:6", "gdpr:art:35", "ai_act:anx:III".'
    )


class GetDefinition(BaseModel):
    term: str = Field(
        description='A defined term, e.g. "AI system", "donnees a caractere personnel".'
    )


class SubmitAnswer(BaseModel):
    answer: str = Field(description="The final answer, in the language of the question.")
    citations: list[str] = Field(
        default_factory=list, description="provision_id values supporting the answer."
    )
    abstained: bool = Field(
        default=False, description="True when the corpus does not answer the question."
    )


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    handler: Callable[[Any, ToolContext], str]

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_schema(self.schema),
            "strict": True,
        }


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema tightened for strict tool use: all fields required, nothing extra."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    properties: dict[str, Any] = schema.get("properties", {})
    for prop in properties.values():
        prop.pop("default", None)
        prop.pop("title", None)
    schema["required"] = sorted(properties)
    schema["additionalProperties"] = False
    return schema


def _search(payload: SearchRegulations, context: ToolContext) -> str:
    hits = context.retriever.search(
        payload.query,
        language=context.language,
        limit=min(payload.limit or 5, 8),
        sources=payload.sources or None,
    )
    context.remember_hits(hits)
    if not hits:
        return "No passage found. Try different wording, or a broader query."
    return "\n\n".join(
        f"[{', '.join(hit.provision_ids)}] {hit.chunk.header}\n{hit.chunk.text}" for hit in hits
    )


def _get_provision(payload: GetProvision, context: ToolContext) -> str:
    provision = context.store.provision(payload.provision_id, context.language)
    if provision is None:
        return (
            f"No provision {payload.provision_id!r}. Ids look like "
            '"ai_act:art:6", "gdpr:art:35" or "ai_act:anx:III".'
        )
    context.remember_provision(provision.provision_id)
    return f"[{provision.provision_id}] {' > '.join(provision.breadcrumb)}\n{provision.text}"


def _get_definition(payload: GetDefinition, context: ToolContext) -> str:
    needle = payload.term.casefold()
    matches: list[str] = []
    for provision_id in DEFINITION_PROVISIONS:
        provision = context.store.provision(provision_id, context.language)
        if provision is None:
            continue
        for paragraph in provision.paragraphs:
            if needle in paragraph.text.casefold():
                context.remember_provision(provision_id)
                label = f"({paragraph.label}) " if paragraph.label else ""
                matches.append(f"[{provision_id}] {label}{paragraph.text}")
    if not matches:
        return (
            f"No definition of {payload.term!r} in Article 3 of the AI Act or Article 4 of the "
            "GDPR. It may be defined elsewhere: use search_regulations."
        )
    return "\n\n".join(matches[:6])


def _submit(payload: SubmitAnswer, context: ToolContext) -> str:
    # The graph reads the arguments directly; the model only needs an acknowledgement.
    return "Answer recorded."


def default_tools() -> tuple[Tool, ...]:
    """The agent's toolbox."""
    return (
        Tool(
            name="search_regulations",
            description=(
                "Search the corpus (AI Act, GDPR, Commission and CNIL guidance) for passages "
                "relevant to a query. Use it whenever you need wording you have not retrieved yet."
            ),
            schema=SearchRegulations,
            handler=_search,
        ),
        Tool(
            name="get_provision",
            description=(
                "Fetch one provision in full by its id. Use it when the question names an "
                "article, annex or recital, or to read a provision a passage refers to."
            ),
            schema=GetProvision,
            handler=_get_provision,
        ),
        Tool(
            name="get_definition",
            description=(
                "Look up a term in the definitions of the AI Act (Article 3) or the GDPR "
                "(Article 4)."
            ),
            schema=GetDefinition,
            handler=_get_definition,
        ),
        Tool(
            name="submit_answer",
            description=(
                "Give the final answer with its citations. Call this exactly once, when the "
                "retrieved passages support an answer - or to abstain when they do not."
            ),
            schema=SubmitAnswer,
            handler=_submit,
        ),
    )


def run_tool(tool: Tool, raw_input: dict[str, Any], context: ToolContext) -> tuple[str, bool]:
    """Execute a tool call. Returns the result text and whether it is an error."""
    try:
        payload = tool.schema.model_validate(raw_input)
    except ValidationError as error:
        logger.warning("invalid arguments for %s: %s", tool.name, error)
        return f"Invalid arguments for {tool.name}: {error}", True
    try:
        return tool.handler(payload, context), False
    except Exception as error:  # a tool failure must not kill the run
        logger.exception("tool %s failed", tool.name)
        return f"{tool.name} failed: {error}", True
