# 1. Record architecture decisions

- Status: accepted
- Date: 2026-09-11

## Context

This repository is built the way a Forward Deployed Engineer would build a system for a
client: the reasoning behind each technical choice matters as much as the code. Choices
must be reviewable by someone who was not in the room, and revisitable when evaluation
data contradicts them.

## Decision

Architecturally significant decisions are recorded as lightweight ADRs (Michael Nygard
format) in `docs/adr/`, numbered sequentially. Each ADR states the context, the decision,
the alternatives considered and the consequences. When an evaluation result motivates or
overturns a decision, the ADR links to the corresponding report in `evals/results/`.

## Consequences

- Trade-offs are explicit and, where possible, backed by measurements.
- ADRs are never deleted: a reversed decision is marked `Superseded by NNNN`.
- Writing an ADR is part of the definition of done for any milestone that introduces a
  new component or dependency.
