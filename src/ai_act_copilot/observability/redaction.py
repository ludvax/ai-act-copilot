"""Redaction of personal data before traces leave the machine.

An assistant on the AI Act and the GDPR gets asked about real situations, and those
questions carry personal data: "can I keep jean.dupont@acme.fr in the training set?".
Sending that verbatim to a hosted tracing backend is exactly the processing the corpus
warns about, so it is removed at the export boundary rather than at each call site -
one place to audit, and impossible to forget when adding an observation.

The patterns are deliberately narrow. Legal text is full of numbered references
("Article 6(2)", "Regulation (EU) 2024/1689", "2 August 2026") and a greedy digit
pattern would shred the corpus passages that make an answer auditable; every pattern
here requires a shape that regulatory prose does not produce. The tests assert both
directions: identifiers disappear, corpus text survives.
"""

import re
from collections.abc import Sequence

from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanPatch,
)

__all__ = ["mask_otel_spans", "redact"]

# (replacement, pattern), applied in order: the most specific shapes first, so a French
# mobile number is not half-eaten by the generic international pattern.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*[\w]")),
    # An IBAN is 15-34 characters, so the last group is not always a full four.
    ("[IBAN]", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    ("[CARD]", re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b")),
    # French social security number: 15 digits in a fixed shape (sex, year, month, ...).
    ("[NIR]", re.compile(r"\b[12] ?\d{2} ?(?:0[1-9]|1[0-2]) ?\d{2} ?\d{3} ?\d{3} ?\d{2}\b")),
    ("[PHONE]", re.compile(r"\b(?:\+33|0)[ .-]?[1-9](?:[ .-]?\d{2}){4}\b")),
    ("[PHONE]", re.compile(r"\+\d{1,3}[ .-]?(?:\d[ .-]?){7,12}\d")),
)


def redact(text: str) -> str:
    """Replace personal identifiers with a placeholder naming what was removed."""
    for replacement, pattern in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_all(value: object) -> list[str] | None:
    """Redact a homogeneous sequence of strings, or None when there is nothing to change."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    items = [item for item in value if isinstance(item, str)]
    if len(items) != len(value):
        return None  # not a string sequence; OpenTelemetry forbids mixing types anyway
    redacted = [redact(item) for item in items]
    return redacted if redacted != items else None


def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    """Langfuse export hook: redact every string attribute in one export batch.

    Chosen over the legacy ``mask`` hook because it runs at export, on the final
    OpenTelemetry attributes - so it also covers anything a future instrumentation
    library adds, not only the values this codebase sets by hand.

    Patches are sparse: spans with nothing to redact are left untouched.
    """
    patches: dict[object, OtelSpanPatch] = {}
    for identifier, span in params.spans.items():
        changed: dict[str, object] = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                if (masked := redact(value)) != value:
                    changed[key] = masked
            elif (sequence := _redact_all(value)) is not None:
                changed[key] = sequence
        if changed:
            patches[identifier] = OtelSpanPatch(set_attributes=changed)  # type: ignore[arg-type]

    if not patches:
        return None  # nothing to change in this batch
    return MaskOtelSpansResult(span_patches=patches)  # type: ignore[arg-type]
