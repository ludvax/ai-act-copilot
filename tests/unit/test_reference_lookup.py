from ai_act_copilot.models import ProvisionKind
from ai_act_copilot.retrieval.reference_lookup import find_references


def test_finds_a_plain_article() -> None:
    (reference,) = find_references("What does Article 6 require?")

    assert reference.kind is ProvisionKind.ARTICLE
    assert reference.number == "6"
    assert reference.paragraph is None


def test_finds_article_with_paragraph_in_both_notations() -> None:
    bracket = find_references("Article 6(2) AI Act")[0]
    french = find_references("l'article 6, paragraphe 2 du reglement")[0]

    assert (bracket.number, bracket.paragraph) == ("6", "2")
    assert (french.number, french.paragraph) == ("6", "2")


def test_article_premier_is_article_one() -> None:
    (reference,) = find_references("Que dit l'article premier ?")

    assert reference.number == "1"


def test_finds_annexes_and_recitals() -> None:
    annex = find_references("Annex III lists high-risk systems")[0]
    recital = find_references("voir le considerant 27")[0]

    assert (annex.kind, annex.number) == (ProvisionKind.ANNEX, "III")
    assert (recital.kind, recital.number) == (ProvisionKind.RECITAL, "27")


def test_detects_which_act_is_meant() -> None:
    ai_act = find_references("Article 5 of the AI Act")[0]
    gdpr = find_references("article 6 du RGPD")[0]
    unknown = find_references("article 9")[0]

    assert ai_act.provision_id() == "ai_act:art:5"
    assert gdpr.provision_id() == "gdpr:art:6"
    assert unknown.source_id is None
    assert unknown.provision_id() is None
    assert unknown.provision_id("ai_act") == "ai_act:art:9"


def test_questions_without_citations_yield_nothing() -> None:
    assert find_references("Which practices are prohibited for emotion recognition?") == []


def test_finds_several_references_in_one_question() -> None:
    references = find_references("Compare Article 6 and Annex III of the AI Act")

    assert [reference.number for reference in references] == ["6", "III"]
