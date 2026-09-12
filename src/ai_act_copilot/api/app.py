"""HTTP API: the same two paths the CLI exposes.

Resources that are expensive to build - the corpus store, the vector store, the indexes -
are opened once at startup and shared, so a request only pays for retrieval and the model.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_act_copilot import __version__
from ai_act_copilot.agent.graph import open_checkpointer, run_agent
from ai_act_copilot.agent.guardrails import Budget
from ai_act_copilot.agent.nodes import AgentDeps
from ai_act_copilot.config import Settings, get_settings
from ai_act_copilot.embeddings.indexer import make_embedder
from ai_act_copilot.generation.answer import answer_question
from ai_act_copilot.llm.anthropic_client import AnthropicLLM
from ai_act_copilot.llm.base import LLMError
from ai_act_copilot.models import Language
from ai_act_copilot.observability.tracing import init_tracing
from ai_act_copilot.retrieval.hybrid import HybridRetriever
from ai_act_copilot.store.sqlite import CorpusStore
from ai_act_copilot.store.vectors import VectorStore

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    language: Language | None = None
    k: int | None = Field(default=None, ge=1, le=20)


class AgentRequest(AskRequest):
    thread_id: str | None = Field(
        default=None, description="Continue an earlier conversation on this thread."
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[str] = []
    abstained: bool = False
    model: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class AgentResponse(AskResponse):
    route: str
    steps: int = 0
    thread_id: str
    provisions_seen: list[str] = []
    halted_by: str | None = None


@dataclass(slots=True)
class Resources:
    settings: Settings
    store: CorpusStore
    vectors: VectorStore
    retriever: HybridRetriever
    llm: AnthropicLLM
    checkpointer: Any

    def agent_deps(self) -> AgentDeps:
        return AgentDeps(
            llm=self.llm,
            retriever=self.retriever,
            store=self.store,
            budget=Budget(
                max_steps=self.settings.agent_max_steps,
                max_cost_usd=self.settings.agent_max_cost_usd,
            ),
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Injecting settings keeps it testable."""
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_tracing(configured)
        if configured.anthropic_api_key is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        store = CorpusStore(configured.database_path)
        vectors = VectorStore(configured.database_path)
        app.state.resources = Resources(
            settings=configured,
            store=store,
            vectors=vectors,
            retriever=HybridRetriever(
                store, vectors, make_embedder(configured), settings=configured
            ),
            llm=AnthropicLLM(
                configured.anthropic_api_key.get_secret_value(),
                model=configured.llm_model,
                effort=configured.llm_effort,
                max_tokens=configured.llm_max_tokens,
            ),
            checkpointer=open_checkpointer(configured.data_dir / "index" / "threads.db"),
        )
        try:
            yield
        finally:
            store.close()
            vectors.close()

    app = FastAPI(
        title="AI Act Copilot",
        version=__version__,
        summary="Answers about EU AI regulation, grounded in the official texts.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        resources: Resources = app.state.resources
        try:
            answer = answer_question(
                request.question,
                retriever=resources.retriever,
                llm=resources.llm,
                language=request.language,
                limit=request.k,
            )
        except LLMError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return AskResponse(
            answer=answer.text,
            citations=list(answer.citations),
            abstained=answer.abstained,
            model=answer.model,
            cost_usd=answer.cost_usd,
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
        )

    @app.post("/v1/agent", response_model=AgentResponse)
    def agent(request: AgentRequest) -> AgentResponse:
        resources: Resources = app.state.resources
        try:
            answer = run_agent(
                request.question,
                resources.agent_deps(),
                thread_id=request.thread_id,
                checkpointer=resources.checkpointer,
                language=request.language,
            )
        except LLMError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return AgentResponse(
            answer=answer.text,
            citations=list(answer.citations),
            abstained=answer.abstained,
            model=resources.llm.model,
            cost_usd=answer.cost_usd,
            input_tokens=answer.usage.input_tokens,
            output_tokens=answer.usage.output_tokens,
            route=str(answer.route),
            steps=answer.steps,
            thread_id=answer.thread_id or "",
            provisions_seen=list(answer.provisions_seen),
            halted_by=answer.halted_by,
        )

    return app
