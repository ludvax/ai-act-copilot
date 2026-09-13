"""The RAG path: retrieve, ground, answer, verify.

Structured output is used so the answer, its citations and the abstention flag come back
as data rather than prose to be parsed. Everything the answer rests on is kept on the
result: the passages, the prompt version, the token usage and the cost.
"""

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from pydantic import BaseModel, Field

from ai_act_copilot.generation.citations import (
    CitationCheck,
    allowed_provisions,
    build_context,
    check_citations,
)
from ai_act_copilot.generation.prompts import DEFAULT_PROMPTS_DIR, load_prompt
from ai_act_copilot.llm.base import LLMClient
from ai_act_copilot.llm.pricing import Usage
from ai_act_copilot.models import Language
from ai_act_copilot.observability.tracing import current_trace_id, observe, record_span
from ai_act_copilot.retrieval.base import RetrievedChunk, Retriever
from ai_act_copilot.store.text_analysis import detect_language

logger = logging.getLogger(__name__)

ANSWER_PROMPT = "answer_v1"

_NO_CONTEXT = {
    Language.EN: "The corpus does not contain anything on this question.",
    Language.FR: "Le corpus ne contient rien sur cette question.",
}


class AnswerSchema(BaseModel):
    """What the model must return."""

    answer: str = Field(description="The answer, in the language of the question.")
    citations: list[str] = Field(
        default_factory=list, description="provision_id values copied from the passages."
    )
    abstained: bool = Field(
        default=False, description="True when the passages do not answer the question."
    )


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer with everything needed to audit it."""

    question: str
    text: str
    language: Language
    citations: tuple[str, ...]
    abstained: bool
    passages: tuple[RetrievedChunk, ...]
    citation_check: CitationCheck
    prompt_version: str
    model: str
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    # Lets a caller link this answer to its trace, and an evaluation attach scores to it.
    trace_id: str | None = None

    @property
    def provisions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.citations))


# A chain: it links a retrieval step to a model call, which is exactly what the type means.
@observe(name="answer-question", as_type="chain", capture_input=False, capture_output=False)
def answer_question(
    question: str,
    *,
    retriever: Retriever,
    llm: LLMClient,
    language: Language | None = None,
    limit: int | None = None,
    prompts_dir: Path | None = None,
) -> GroundedAnswer:
    """Retrieve passages and answer strictly from them."""
    target_language = language or detect_language(question)
    passages = retriever.search(question, language=target_language, limit=limit)
    prompt = load_prompt(ANSWER_PROMPT, prompts_dir or DEFAULT_PROMPTS_DIR)

    if not passages:
        return _record(
            GroundedAnswer(
                question=question,
                text=_NO_CONTEXT[target_language],
                language=target_language,
                citations=(),
                abstained=True,
                passages=(),
                citation_check=CitationCheck((), ()),
                prompt_version=prompt.version,
                model=llm.model,
            )
        )

    result = llm.complete(
        system=prompt.text,
        messages=[
            {
                "role": "user",
                "content": (f"Question: {question}\n\nPassages:\n\n{build_context(passages)}"),
            }
        ],
        output_format=AnswerSchema,
        name="generate-answer",
    )

    if result.refused:
        logger.warning("the model declined to answer")
        return _record(
            GroundedAnswer(
                question=question,
                text=_NO_CONTEXT[target_language],
                language=target_language,
                citations=(),
                abstained=True,
                passages=tuple(passages),
                citation_check=CitationCheck((), ()),
                prompt_version=prompt.version,
                model=result.model,
                usage=result.usage,
                cost_usd=result.cost_usd,
            )
        )

    parsed = result.parsed if isinstance(result.parsed, AnswerSchema) else None
    text = parsed.answer if parsed else result.text
    check = check_citations(parsed.citations if parsed else (), allowed_provisions(passages))
    if check.invalid:
        logger.warning("dropped citations not present in the context: %s", check.invalid)

    return _record(
        GroundedAnswer(
            question=question,
            text=text,
            language=target_language,
            citations=check.valid,
            abstained=bool(parsed and parsed.abstained) or not text.strip(),
            passages=tuple(passages),
            citation_check=check,
            prompt_version=prompt.version,
            model=result.model,
            usage=result.usage,
            cost_usd=result.cost_usd,
        )
    )


def _record(answer: GroundedAnswer) -> GroundedAnswer:
    """Describe the step on its span, whichever way it ended.

    Every exit goes through here - including the two that never call the model - because a
    step that abstained silently is precisely the one someone will come looking for.
    """
    record_span(
        input=answer.question,
        output={
            "answer": answer.text,
            "citations": list(answer.citations),
            "abstained": answer.abstained,
        },
        metadata={
            "language": answer.language.value,
            "prompt_version": answer.prompt_version,
            "passages": len(answer.passages),
            "grounded_on": [
                provision for hit in answer.passages for provision in hit.provision_ids
            ],
            "dropped_citations": list(answer.citation_check.invalid),
        },
    )
    return replace(answer, trace_id=current_trace_id())
