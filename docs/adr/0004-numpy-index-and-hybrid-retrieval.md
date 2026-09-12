# 4. Brute-force vector search in SQLite, and a measured mix of retrieval signals

- Status: accepted
- Date: 2026-09-12

## Context

Retrieval decides the ceiling on answer quality: the generator can only be as good as the
passages it is given. Two decisions matter here — where vectors live, and how results from
different search methods are combined.

The corpus is 1 896 chunks. At 1 024 dimensions in float32 that is roughly 8 MB of vectors.

## Decision

**No vector database.** Vectors are stored as BLOBs in the same SQLite file as the corpus
and loaded into a numpy matrix; a query is one matrix product over 1 896 rows, which is
about a millisecond. A vector database would add a service to run, back up and version for
a dataset that fits in L3 cache. `VectorIndex` exposes the interface a vector store would,
so swapping it later is one class — and the ADR to revisit is this one, when the corpus
grows by an order of magnitude or needs filtered ANN search.

**Embeddings are cached by text hash, not chunk id.** Re-chunking or switching strategy
re-embeds only genuinely new passages, which makes chunking experiments cheap to run.

**Three retrieval signals, fused by rank — but only two are on by default.** Each signal
fails differently, so the combination was measured rather than assumed (15-question golden
set, k=5):

| Configuration | hit@5 | MRR | nDCG |
| --- | --- | --- | --- |
| BM25 only | 0.47 | 0.36 | 0.39 |
| Dense only | 0.93 | 0.88 | 0.89 |
| Dense + BM25 | 0.87 | 0.66 | 0.71 |
| **Dense + reference lookup (default)** | **0.93** | **0.93** | **0.93** |
| All three | 0.93 | 0.73 | 0.78 |

Adding BM25 to dense retrieval *lowers* ranking quality on this corpus: legal text shares
so much vocabulary that lexical matches are rarely discriminating, and they displace better
dense hits. Resolving explicit citations does the opposite — it puts the cited article at
rank 1. So the default is **references + dense**; BM25 stays implemented, tested and one
argument away, because a corpus with rare identifiers (product codes, case numbers) would
likely reverse this result.

**Reciprocal rank fusion rather than score fusion.** A BM25 score and a cosine similarity
are not on the same scale, and normalising them is guesswork that silently favours one
retriever. RRF uses only positions: `weight/(k + rank)`, with k = 60 damping the very top
ranks. Weights are explicit — an earlier version defaulted them when callers passed none,
which made an A/B comparison silently compare a configuration with itself.

**Results are de-duplicated per provision**, so one article cannot occupy several slots in
both languages — the context window goes to distinct provisions instead.

## Alternatives considered

- **Qdrant / pgvector / Chroma.** Justified past a few hundred thousand vectors, or when
  ANN with metadata filters is needed. Here it is operational cost for no measurable gain.
- **Dense-only retrieval.** Closest competitor (MRR 0.88): it only loses on questions
  naming a provision, which is why the reference lookup exists.
- **Keeping BM25 in the default because hybrid search is conventional.** Rejected on the
  measurement above. The 15-question sample is small, so M5 re-runs this comparison on
  the full golden set before the result is treated as settled.
- **A reranker (cross-encoder).** Deliberately deferred: it adds a heavyweight dependency
  and latency per query. It earns its place only if the evaluation shows retrieval, not
  generation, is the bottleneck.

## Consequences

- The whole index rebuilds from `aiact ingest && aiact index` with no external service.
- Which signal found a passage is recorded on every result, so a bad answer can be traced
  to the retrieval step that produced it.
- `aiact eval-retrieval` measures each configuration on the golden dataset, so this ADR is
  backed by numbers rather than intuition.
