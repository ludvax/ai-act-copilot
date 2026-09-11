# 3. Chunk along the structure of the text, and label evaluation data by provision

- Status: accepted
- Date: 2026-09-11

## Context

Regulatory answers must cite exact provisions ("Article 6(2)", "Annex III"). Chunking
decides what the retriever can find and what the answer can cite, and it is the parameter
most often tuned blindly in RAG systems.

A second problem is evaluation: if the golden dataset points at chunk ids, every change to
the chunking invalidates the dataset, which makes strategies impossible to compare.

## Decision

1. **Two strategies, both implemented**: `fixed` (sliding token windows over the raw text,
   the usual tutorial approach) as the baseline, and `structural` (default) which follows
   the boundaries the legislator already provides.
2. **A chunk never spans two provisions.** Citations stay exact, and a retrieved passage
   maps to one article.
3. **Every structural chunk carries a breadcrumb header** ("AI Act > Chapter III >
   Article 6 — Classification rules"), embedded with the text, so an isolated paragraph
   still says what it belongs to.
4. **Sizes are measured in bge-m3 tokens** — the embedding model decides what fits, so it
   does the counting. The tokenizer revision is pinned for reproducible boundaries.
5. **`max_tokens` is a target, not a limit.** A short trailing fragment is folded back into
   the previous chunk even if that slightly exceeds the target: bge-m3 accepts 8192 tokens,
   and a three-token chunk carries no retrievable signal.
6. **Golden datasets reference provisions, never chunks** (`ai_act:art:6`). Re-chunking
   never invalidates the evaluation set, which is what makes strategies comparable in M5.

## Alternatives considered

- **Fixed-size only.** Simpler, and it is kept as the baseline — but it cuts mid-sentence
  and mid-article, so citations cannot be trusted.
- **One chunk per paragraph.** Maximum precision, but paragraphs read out of context
  ("It shall apply from 2 August 2026") embed poorly.
- **Semantic/embedding-based chunking.** Expensive and unnecessary when the document
  already carries authoritative boundaries.

## Consequences

- The real corpus produces 1 896 chunks over 7 documents (median 277 tokens, p95 484),
  small enough that brute-force vector search remains viable (see the M2 ADR).
- The comparison "fixed vs structural" is a measured result in M5, not an opinion.
- Guidance PDFs have no articles: they are split on numbered headings, and section numbers
  are de-duplicated because real documents reuse them. PDF extraction needs defensive
  filtering, found by inspecting the real output: table-of-contents entries were being
  parsed as sections (and stealing the real heading numbers), and footnote markers were
  read as headings. Dotted-leader lines are dropped, headings must carry a dot or be
  multi-level, and sections with no real content are discarded.
