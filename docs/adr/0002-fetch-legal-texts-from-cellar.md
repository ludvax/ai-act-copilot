# 2. Fetch legal texts from Cellar, not from the EUR-Lex website

- Status: accepted
- Date: 2026-09-11

## Context

The corpus needs the AI Act and the GDPR in English and French, in a form stable enough to
parse into articles and paragraphs. Two routes exist: scraping the EUR-Lex web pages, or
asking the EU Publications Office's Cellar repository for a specific expression of a
document by its CELEX identifier.

Scraping a public website for a corpus is brittle by nature (markup churn, bot protection,
rate limits) and it is the kind of thing a client's legal team asks about.

## Decision

Fetch by CELEX identifier from Cellar (`publications.europa.eu/resource/celex/{id}`) using
HTTP content negotiation: `Accept: application/xhtml+xml` with `Accept-Language: eng|fra`.

Measured: both acts are served as structured XHTML (AI Act 1.3 MB, GDPR 0.8 MB) through a
single redirect, with no bot challenge. The same markup is used for both, although they
were published eight years apart, so one parser covers the whole legal corpus.

Guidance documents (Commission, CNIL) have no CELEX identifier and are fetched from their
publisher URL as PDF.

Downloads are content-addressed: each file's SHA-256 is recorded in `data/raw/downloads.json`,
so re-running ingestion costs nothing and a changed source is visible.

## Alternatives considered

- **Scraping EUR-Lex HTML pages.** Rejected: unstable markup and bot protection, for no gain.
- **Formex XML (the official structured format).** Richer and fully typed, but a heavier
  parser for no measured benefit — the XHTML already carries the structure we need
  (`art_6`, `006.001`, chapter and section titles). Worth revisiting if the XHTML
  degrades.

## Consequences

- Raw files are never versioned: `data/sources.yaml` plus checksums make ingestion reproducible.
- Licences are recorded per source and surfaced in the README: EU texts are reusable with
  acknowledgement of the source (Decision 2011/833/EU), CNIL content under the Licence
  Ouverte, both requiring attribution.
- If Cellar changes its negotiation behaviour, ingestion fails loudly at download time
  rather than silently producing an empty corpus.
