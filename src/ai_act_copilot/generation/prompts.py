"""Prompts live in files, not in string literals.

Two reasons: they are reviewed as text in pull requests, and every trace records which
version produced an answer. The version combines the filename (answer_v1) with a short
hash of the content, so an edited prompt is visibly a different prompt in Langfuse even if
nobody bumped the number.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PROMPTS_DIR = Path("prompts")


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    text: str
    digest: str

    @property
    def version(self) -> str:
        """Identifier recorded on every trace, e.g. ``answer_v1@3f9a2c11``."""
        return f"{self.name}@{self.digest}"


@lru_cache(maxsize=16)
def load_prompt(name: str, directory: Path = DEFAULT_PROMPTS_DIR) -> Prompt:
    """Read a prompt by name, e.g. ``answer_v1``."""
    path = directory / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt {name!r} not found at {path}")
    text = path.read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return Prompt(name=name, text=text, digest=digest)
