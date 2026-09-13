"""Redaction has to remove identifiers without damaging the corpus.

Both directions matter equally. A pattern that misses an email leaks personal data to a
hosted backend; a pattern that eats "Regulation (EU) 2024/1689" destroys the citations
that make an answer auditable, and nobody notices until they read a trace.
"""

import pytest

from ai_act_copilot.observability.redaction import redact

REDACTED = [
    ("contact jean.dupont@acme.fr about it", "[EMAIL]"),
    ("virement sur FR76 3000 6000 0112 3456 7890 189", "[IBAN]"),
    ("carte 4111 1111 1111 1111", "[CARD]"),
    ("numero de securite sociale 1 84 12 75 116 001 42", "[NIR]"),
    ("appelle le 06 12 34 56 78", "[PHONE]"),
    ("call +33 6 12 34 56 78", "[PHONE]"),
]

# Everything a regulatory corpus is made of, which a greedy pattern would happily destroy.
PRESERVED = [
    "Article 6(2) of Regulation (EU) 2024/1689 applies from 2 August 2026",
    "Annex III, point 4(a); recital 52; Article 5(1)(a)",
    "Directive 2011/833/EU, OJ L 330, 14.12.2011, p. 39",
    "ai_act:art:6, gdpr:art:35, ai_act:anx:III",
    "Regulation (EU) 2016/679 (General Data Protection Regulation)",
    "Les systemes vises a l'annexe III, point 5 b), sont a haut risque.",
]


@pytest.mark.parametrize(("text", "placeholder"), REDACTED)
def test_identifiers_are_replaced_by_a_label(text: str, placeholder: str) -> None:
    redacted = redact(text)

    assert placeholder in redacted
    # The surrounding sentence has to survive, or the trace becomes unreadable.
    assert redacted.split()[0] == text.split()[0]


@pytest.mark.parametrize("text", PRESERVED)
def test_legal_references_are_left_alone(text: str) -> None:
    assert redact(text) == text


def test_several_identifiers_in_one_string() -> None:
    redacted = redact("write to a@b.fr or call 06 12 34 56 78")

    assert redacted == "write to [EMAIL] or call [PHONE]"


def test_redaction_is_idempotent() -> None:
    once = redact("mail: someone@example.org")

    assert redact(once) == once
