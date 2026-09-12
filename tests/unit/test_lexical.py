from ai_act_copilot.models import Language
from ai_act_copilot.store.bm25 import BM25Index
from ai_act_copilot.store.text_analysis import analyse, detect_language


def test_numbers_survive_analysis_because_they_are_the_citation() -> None:
    tokens = analyse("Article 6(2) and Annex III", Language.EN)

    assert "6" in tokens
    assert "2" in tokens
    assert "iii" in tokens


def test_french_elisions_and_stopwords_are_removed() -> None:
    tokens = analyse("l'article et les systemes d'IA dans le marche", Language.FR)

    assert "et" not in tokens
    assert "les" not in tokens
    assert "l" not in tokens  # the elision fragment must not survive tokenisation
    assert "ia" in tokens


def test_french_stemming_matches_singular_and_plural() -> None:
    singular = analyse("traitement de donnee", Language.FR)
    plural = analyse("traitements des donnees", Language.FR)

    assert singular == plural


def test_english_stemming_matches_inflections() -> None:
    assert analyse("providers deploying systems", Language.EN) == analyse(
        "provider deploys system", Language.EN
    )


def test_detects_query_language() -> None:
    assert (
        detect_language("Quelles sont les pratiques interdites par le reglement ?") is Language.FR
    )
    assert detect_language("Which AI practices are prohibited?") is Language.EN


def test_detect_language_defaults_to_english_when_undecidable() -> None:
    assert detect_language("Article 6") is Language.EN


def _index() -> BM25Index:
    return BM25Index.build(
        [
            ("a", analyse("Prohibited AI practices include social scoring", Language.EN)),
            ("b", analyse("High-risk AI systems must keep technical documentation", Language.EN)),
            ("c", analyse("Social scoring by public authorities is prohibited", Language.EN)),
        ]
    )


def test_bm25_ranks_documents_sharing_rare_terms_first() -> None:
    hits = _index().search(analyse("social scoring", Language.EN), limit=3)

    assert {chunk_id for chunk_id, _ in hits[:2]} == {"a", "c"}
    assert all(score > 0 for _, score in hits)


def test_bm25_ignores_unknown_terms_and_empty_queries() -> None:
    index = _index()

    assert index.search(analyse("blockchain", Language.EN), limit=3) == []
    assert index.search([], limit=3) == []
    assert len(index) == 3


def test_bm25_scores_are_deterministic_for_ties() -> None:
    index = _index()

    assert index.search(analyse("AI", Language.EN), limit=3) == index.search(
        analyse("AI", Language.EN), limit=3
    )


def test_empty_index_returns_nothing() -> None:
    assert BM25Index.build([]).search(["anything"], limit=5) == []
