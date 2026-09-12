import numpy as np
import pytest

from ai_act_copilot.embeddings.base import normalise
from ai_act_copilot.store.vector_index import VectorIndex


def _index() -> VectorIndex:
    matrix = np.vstack(
        [
            normalise([1.0, 0.0, 0.0]),
            normalise([0.9, 0.1, 0.0]),
            normalise([0.0, 1.0, 0.0]),
        ]
    )
    return VectorIndex(["a", "b", "c"], matrix)


def test_returns_closest_vectors_first() -> None:
    hits = _index().search(normalise([1.0, 0.0, 0.0]), limit=2)

    assert [chunk_id for chunk_id, _ in hits] == ["a", "b"]
    assert hits[0][1] > hits[1][1]


def test_respects_the_limit() -> None:
    assert len(_index().search(normalise([1.0, 1.0, 0.0]), limit=1)) == 1


def test_allowed_filter_restricts_candidates() -> None:
    hits = _index().search(normalise([1.0, 0.0, 0.0]), limit=2, allowed={"c"})

    assert [chunk_id for chunk_id, _ in hits] == ["c"]


def test_empty_index_returns_nothing() -> None:
    empty = VectorIndex([], np.zeros((0, 0), dtype=np.float32))

    assert empty.search(normalise([1.0, 0.0]), limit=5) == []
    assert len(empty) == 0


def test_mismatched_ids_and_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="2 ids for 1 vectors"):
        VectorIndex(["a", "b"], np.vstack([normalise([1.0, 0.0])]))


def test_similarity_is_a_dot_product_on_normalised_vectors() -> None:
    hits = _index().search(normalise([1.0, 0.0, 0.0]), limit=1)

    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
