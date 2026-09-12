# 7. An in-house evaluation harness, with calibrated judges

- Status: accepted
- Date: 2026-09-13

## Context

Every earlier decision in this repository claims to be measured. That claim is only worth
something if the measurement itself is trustworthy — which means knowing what the numbers
cover, how they were produced, and how far to trust the instrument that produced them.

## Decision

**Deterministic checks before LLM judges.** Whether the answer cited provisions the run
actually retrieved, whether it abstained when the dataset says it should have, and whether
the router chose the labelled path are all facts. They are computed in code: free,
instant, and not subject to a model's mood. Judges are reserved for what genuinely requires
reading — grounding and substantive correctness.

**Two judges, scoring 0-2, on separate prompts.** Faithfulness sees the passages but not the
reference answer, so it grades grounding rather than agreement. Correctness sees the
reference but not the passages, so it grades substance rather than phrasing.

**Judges are calibrated against human labels.** `aiact eval-calibrate` reports exact
agreement, agreement within one point, and Cohen's kappa — chance-corrected, which matters
because most answers score 2 and a judge that always says "2" would otherwise look
excellent. Below kappa 0.4 the judge is reported as unusable rather than quietly trusted.

**Abstention is a measured outcome, not an edge case.** The golden set contains
out-of-scope questions and false premises where the correct behaviour is to decline. A
system that answers everything scores badly on this corpus, as it should.

**Labels reference provisions, so datasets survive re-chunking** (see ADR 0003), and the
same file drives retrieval metrics, routing accuracy, abstention accuracy and answer
grading.

**Reports state their own caveats.** Each report records the dataset, how much of it a
human has reviewed, the judge model and the run cost. The current golden set is 40 cases
drafted from the texts and **not yet human-reviewed**, so its numbers are provisional and
the reports say so.

## Alternatives considered

- **Ragas.** A good library, and its faithfulness and context-precision metrics overlap
  with these judges. Rejected as the primary harness for two reasons: the interesting
  metrics here are domain-specific (citation validity against retrieved provisions,
  abstention correctness, routing accuracy), and a portfolio whose central claim is
  "measured decisions" should not outsource the measurement it cannot explain. Worth adding
  later as a cross-check on faithfulness, precisely because an independent implementation
  disagreeing would be informative.
- **Judging with the same model that answered.** Cheaper in setup, but a model grading its
  own output is the least reliable arrangement available. The judge model is configurable
  and separate.
- **Human evaluation only.** The most trustworthy and the least repeatable. Humans grade
  the calibration sample; the judge scales that judgement to every run.

## Consequences

- A configuration change can be evaluated end to end with one command, and the report says
  what it cost.
- The harness runs offline in tests with a scripted model, so the evaluation code itself is
  covered by CI without spending anything.
- Judge scores are only as good as the calibration sample: until ~20 human labels exist,
  the reports carry the numbers and the warning together.
