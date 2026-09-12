# 5. Grounded generation: structured answers, verified citations, explicit abstention

- Status: accepted
- Date: 2026-09-12

## Context

On a regulatory corpus, a confident wrong answer is the failure mode that matters. Users
cannot tell a correct citation from an invented one without opening the Official Journal,
which is precisely the work the system is supposed to save.

## Decision

**One LLM layer.** The RAG path, the agent (M4) and the judges (M5) all call the same
client, so prompt caching, refusal handling, cost accounting and tracing exist once. Claude
is called through the official Anthropic SDK, never through a compatibility shim.

**Structured output, not prose parsing.** The model returns `{answer, citations,
abstained}`. Citations arrive as data, abstention is a boolean rather than a phrase to
pattern-match, and the schema is enforced by the API.

**Citations are verified in code.** Every returned `provision_id` is checked against the
passages actually retrieved. Unsupported citations are dropped, counted, and reported —
citation precision becomes a metric rather than an impression. This is the check that turns
"it cites Article 99" from an undetected error into a visible one.

**Abstention is a first-class outcome.** No passages retrieved, a model refusal, or an
empty answer all produce an abstention rather than an improvised answer. The evaluation set
deliberately contains out-of-scope questions to measure this.

**Prompt caching on the system prompt.** It is the stable prefix of every call; cache reads
bill at a tenth of the input rate, and `usage.cache_read_tokens` is recorded on every trace
so the saving is observed rather than assumed.

**Effort instead of thinking budgets.** Claude Opus 5 reasons by default; depth is steered
with `output_config.effort` (configurable per route), which M5 measures against cost.

**Server-side refusal fallbacks are enabled by default** on the non-structured path: a
safety decline is retried on a fallback model inside the same call. This is a beta feature
and can be switched off with one flag.

**Prompts are files, not literals.** `prompts/answer_v1.md` is reviewed as text, and each
trace records `answer_v1@<hash>` so an edited prompt is a different prompt in the traces.

## Alternatives considered

- **The native Citations API** (document blocks with character-level spans). More precise
  attribution, but it ties the citation format to one provider and applies awkwardly to
  agent tool results. Revisit if span-level highlighting becomes a requirement.
- **Free-text answers parsed with a regex.** Fragile, and it makes abstention ambiguous.
- **Letting the model cite from memory.** It knows these regulations well enough to sound
  right, which is exactly the problem: unverifiable answers.

## Consequences

- An answer can always be audited: passages, citations, prompt version, model, tokens, cost.
- Hallucinated citations are visible in the logs and measurable in the evaluation.
- The system says "the corpus does not cover this" more often than a chatty assistant would,
  which is the intended trade-off.
