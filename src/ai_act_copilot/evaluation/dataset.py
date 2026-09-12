"""Golden datasets.

Cases are labelled with the *provisions* a correct answer must rest on, never with chunk
ids: re-chunking or switching strategy leaves the dataset valid, which is what makes two
retrieval configurations comparable.
"""

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ai_act_copilot.models import Language


class Category(StrEnum):
    LOOKUP = "lookup"  # a plain question answered by one provision
    CITATION = "citation"  # the question names the provision explicitly
    DEFINITION = "definition"
    CONCEPT = "concept"  # needs the right article without naming it
    TEMPORAL = "temporal"  # dates of application
    CROSS_TEXT = "cross-text"  # spans two regulations
    OUT_OF_SCOPE = "out-of-scope"  # the corpus cannot answer; abstention is correct


class RetrievalCase(BaseModel):
    """One question and the provisions retrieval must surface."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    language: Language
    expected_provisions: tuple[str, ...]
    category: Category = Category.CONCEPT


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    """Read a JSONL dataset."""
    return tuple(RetrievalCase.model_validate(record) for record in _records(path))


def _records(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
