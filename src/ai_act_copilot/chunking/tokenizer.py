"""Token counting.

Chunk sizes are measured in the *embedding model's* tokens, not characters or words:
bge-m3 is what decides whether a passage fits, so it is what does the counting. Tests and
CI use the offline word counter instead, which keeps them deterministic and network-free.
"""

import re
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# Pinned so chunk boundaries are reproducible across machines and over time.
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens and locates them, so chunkers can cut on token boundaries."""

    def count(self, text: str) -> int: ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of each token, in order."""
        ...


class HuggingFaceTokenCounter:
    """The real bge-m3 tokenizer, downloaded once and cached by huggingface_hub."""

    def __init__(self, model: str = BGE_M3_MODEL, revision: str = BGE_M3_REVISION) -> None:
        self.model = model
        self.revision = revision

    @cached_property
    def _tokenizer(self) -> "Tokenizer":
        from tokenizers import Tokenizer

        return Tokenizer.from_pretrained(self.model, revision=self.revision)

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return list(encoding.offsets)


class WordTokenCounter:
    """Offline approximation: words and punctuation marks count as one token each."""

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in _WORD.finditer(text)]
