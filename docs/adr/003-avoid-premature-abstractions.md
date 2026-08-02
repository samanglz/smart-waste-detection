# ADR-003: Avoid Premature Abstractions

## Status
Accepted

## Context

Framework projects often introduce patterns such as factories, registries, and
plugin systems too early.

Although these patterns can improve extensibility, unnecessary abstractions
increase complexity and make the code harder to maintain.

## Decision

Follow a pragmatic approach:

- Apply abstraction only when there is a real need.
- Avoid introducing Factory, Registry, or Plugin systems until multiple
  implementations require dynamic selection or extension.
- Prefer simple polymorphism and clear interfaces.

The framework should balance:

- YAGNI (You Aren't Gonna Need It)
- Open/Closed Principle
- Maintainable architecture

## Consequences

### Positive

- Lower initial complexity.
- Faster development and easier debugging.
- Architecture evolves based on real requirements.

### Negative

- Some refactoring may be required when the system grows significantly.