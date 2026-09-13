# 8. Tracing that can be audited, and tested without a Langfuse account

- Status: accepted
- Date: 2026-09-13

## Context

Tracing was wired in at M0 and left largely alone: every step carried an `@observe`
decorator, so the spans had names and a shape, and generations carried model, tokens and
cost. That was enough to believe it worked.

Auditing it against Langfuse's own guidance ([what does a good trace look
like?](https://langfuse.com/docs/observability/best-practices)) showed it did not. Every
decorator was declared `capture_input=False, capture_output=False` — a defensible reflex,
since the default captures a function's whole argument list including clients and
credentials, but nothing was set in their place. The result was a tree of correctly named,
completely empty observations: a trace could tell you that an answer took 6 seconds and
cost $0.04, and nothing at all about what was asked, what was retrieved, or what came back.

Three smaller gaps came with it. Every observation was a plain `span` apart from model
calls, so retrieval, tools and judges were indistinguishable to any filter. Questions were
sent verbatim to a hosted backend, in an application whose entire subject is what may be
done with personal data. And a run of forty evaluation cases arrived as one trace.

## Decision

**Every observation states its input and output explicitly.** The decorators still capture
no arguments — a signature is a poor description of a step — but each one now calls
`record_span` or `record_generation` with what it received and what it produced. The
question, the passages with the signal that found each one, the tool arguments and their
results, the conversation the model actually saw, the citations that were rejected.

**Observations are typed.** `retriever` for retrieval, `tool` for each tool call, `agent`
for the run, `chain` for the RAG pipeline, `evaluator` for the judges and the citation
check, `embedding` for bge-m3. Types are what the agent graph is drawn from and what
evaluators and dashboards filter on.

**Generations are named after the job, never the model.** The old name was `claude`, which
would have broken every saved filter the day the model changed. Calls now pass
`name="generate-answer"`, `"route-question"`, `"agent-turn"`, `"judge-faithfulness"`. The
name is set *before* the request rather than after, so a call that fails still arrives
named and carrying its prompt — a failed call is the one worth finding.

**One trace is one question.** A follow-up on the same `thread_id` is its own trace, tied
to the first by a Langfuse session. An evaluation case is its own trace too, so a graded
case looks exactly like a production one and can carry scores; the run that produced it is
recovered from a tag and a run id in metadata, not from a session, which on the agent path
already means "this conversation".

**Evaluation results are sent back as scores.** Faithfulness, correctness, citation
precision and recall, abstention and routing correctness are attached to the trace of the
case that produced them. Latency and cost are visible on a trace by construction; quality
is the one dimension that has to be put there deliberately.

**Personal data is redacted at the export boundary.** A `mask_otel_spans` hook rewrites
emails, phone numbers, IBANs, card numbers and French social-security numbers into
placeholders on the way out, so it applies to everything exported rather than to the call
sites someone remembered. Patterns are deliberately narrow: legal text is full of numbered
references, and a greedy digit pattern would destroy the citations that make an answer
auditable. Tests assert both directions.

## Consequences

Traces are readable, and the parts of the system that are worth distrusting — which
passages grounded an answer, which tool call went wrong, which citation was dropped — are
the parts a trace now shows.

**The instrumentation is tested offline.** The Langfuse SDK is OpenTelemetry underneath, so
a real client pointed at an in-memory span exporter emits exactly what a live project would
receive. `tests/unit/test_trace_shape.py` asserts the tree, the types, the input and output,
model and cost, sessions, and that a question containing an email address never reaches the
exporter — with no account, no keys and no network, on every commit. Instrumentation is
the one thing that can be entirely broken while every other test passes; this closes that
gap. The tests assert structure rather than chronology, because the Windows timer is coarse
enough that sibling steps share a start time.

**A second Langfuse client in one process disables tracing.** The SDK refuses to trace when
several clients are registered, to avoid leaking spans across projects, so the test fixture
tears the session-wide disabled client down and puts it back.

**Redaction can be wrong in both directions.** A pattern that misses an identifier leaks it;
one that is too eager mangles the corpus. The patterns here are conservative, which means
they will miss unusual formats — a name in free text is not redacted at all. Masking can be
turned off with `AIACT_MASK_TRACES=false` for a project where that is acceptable.

**Costs are reported, not estimated.** They come from `usage` on each response, so the
Langfuse cost view is only as right as the price table in `llm/pricing.py`.
