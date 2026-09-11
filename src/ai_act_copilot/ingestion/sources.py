"""The corpus manifest: which texts are ingested, from where, under which licence."""

from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ai_act_copilot.models import Language

CELLAR_CELEX_URL = "http://publications.europa.eu/resource/celex/{celex}"

# Cellar negotiates languages with ISO 639-3 codes.
_CELLAR_LANGUAGE = {Language.EN: "eng", Language.FR: "fra"}


class SourceFormat(StrEnum):
    EURLEX_XHTML = "eurlex-xhtml"
    PDF = "pdf"


class Source(BaseModel):
    """One text of the corpus, available in one or more languages."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    short_title: str
    format: SourceFormat
    languages: tuple[Language, ...]
    license: str
    attribution: str
    celex: str | None = None
    urls: dict[Language, str] = {}

    @model_validator(mode="after")
    def _check_every_language_is_reachable(self) -> Self:
        if self.celex:
            return self
        missing = [language for language in self.languages if language not in self.urls]
        if missing:
            raise ValueError(
                f"source {self.id!r} has no celex and no url for: " + ", ".join(sorted(missing))
            )
        return self

    def document_id(self, language: Language) -> str:
        return f"{self.id}:{language}"

    def url_for(self, language: Language) -> str:
        if self.celex:
            return CELLAR_CELEX_URL.format(celex=self.celex)
        return self.urls[language]

    def request_headers(self, language: Language) -> dict[str, str]:
        """Cellar returns XHTML only when asked for it; plain HTTP otherwise."""
        if not self.celex:
            return {}
        return {
            "Accept": "application/xhtml+xml",
            "Accept-Language": _CELLAR_LANGUAGE[language],
        }

    def raw_filename(self, language: Language) -> str:
        suffix = "xhtml" if self.format is SourceFormat.EURLEX_XHTML else "pdf"
        return f"{self.id}.{language}.{suffix}"


class SourceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    sources: tuple[Source, ...]

    def get(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        known = ", ".join(source.id for source in self.sources)
        raise KeyError(f"unknown source {source_id!r}; known sources: {known}")


def load_manifest(path: Path) -> SourceManifest:
    """Read and validate data/sources.yaml."""
    return SourceManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
