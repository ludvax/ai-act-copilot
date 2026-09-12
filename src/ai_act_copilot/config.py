"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_act_copilot.models import ChunkStrategy

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Runtime configuration.

    Project variables use the ``AIACT_`` prefix. Third-party credentials keep their
    conventional names (``ANTHROPIC_API_KEY``, ``LANGFUSE_*``) so that SDK defaults and
    CI secrets work unchanged.
    """

    model_config = SettingsConfigDict(
        env_prefix="AIACT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_by_name=True,
    )

    data_dir: Path = Path("data")
    log_level: LogLevel = "INFO"

    chunk_strategy: ChunkStrategy = ChunkStrategy.STRUCTURAL
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunk_min_tokens: int = 48

    embedding_model: str = "bge-m3"
    ollama_base_url: str = "http://localhost:11434"
    retrieval_top_k: int = 8
    retrieval_candidates: int = 40
    rrf_k: int = 60

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_max_tokens: int = 8000

    agent_max_steps: int = 8
    agent_max_cost_usd: float = 0.50
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    tracing_enabled: bool = True
    langfuse_public_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "LANGFUSE_HOST"),
    )

    @property
    def sources_file(self) -> Path:
        return self.data_dir / "sources.yaml"

    @property
    def raw_dir(self) -> Path:
        """Downloaded source files; reproducible from sources.yaml, never versioned."""
        return self.data_dir / "raw"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "index" / "corpus.db"

    @property
    def langfuse_configured(self) -> bool:
        """True when both Langfuse keys are present."""
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once."""
    return Settings()
