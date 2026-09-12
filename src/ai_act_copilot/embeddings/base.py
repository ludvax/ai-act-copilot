"""The embedding interface.

Vectors are L2-normalised at the source, so cosine similarity is a dot product and the
index never has to normalise again.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


class Embedder(Protocol):
    """Anything that turns text into normalised vectors."""

    model: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def normalise(vector: npt.ArrayLike) -> Vector:
    """Scale to unit length; a zero vector is returned unchanged."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return (array / norm).astype(np.float32)
