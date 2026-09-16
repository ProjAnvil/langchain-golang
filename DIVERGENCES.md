# Divergences from Python LangChain/LangGraph

Deliberate design decisions where this port does not mirror Python. Each entry states why. Anything not listed here aims for parity; file an issue if you find a gap.

## Never planned

- **LangGraph Platform / Server / Studio / CLI** — commercial product lines, out of scope for an OSS port.
- **Local transformers models** — the Go ecosystem has no comparable runtime; use API providers (or ollama for local serving).
- **Command-Send / remaining_steps** — superseded upstream patterns whose Go equivalents (explicit routing, run control) already exist.

## Deliberate design differences

- **Cache short-circuit middleware** — Go's `cache` middleware short-circuits identical requests instead of Python's instrumentation-only behavior; Go idiom favors explicit memoization points.
- **Blank-import provider registration + shim re-exports** — partner packages self-register via `init()` and top-level `langchain/` shims re-export the stable surface, mirroring Go stdlib plugin patterns rather than Python's explicit imports.

## Deferred (upstream-triggered)

- **deepagents** — upstream is pre-1.0 (0.7.x) with an unstable API; re-evaluate when it reaches 1.0.
- **AWS Bedrock provider, RemoteGraph client** — demand-triggered; both are on the backlog.

## Notable per-adapter behaviors

- **anthropic `tool_choice=none`** — the Messages API has no `none` type; binding fails loudly instead of silently misrouting (bind no tools instead).
- **TracePolicy processor failures** — upstream records the untransformed payload when a trace processor errors; the Go port drops the payload (fail-closed) since the feature's motivation is PII/compliance.
- **Error-handler abort semantics** — a failed superstep's successful handler outcomes are not committed (Go commits writes at superstep end vs Python's per-task `put_writes`), so a resume after a sibling-task failure re-runs an already-succeeded handler; effects must therefore be idempotent.

## Cleared as non-gaps (audited 2026-09-16)

- `tool_choice=any` (openai→required, anthropic→any) and openai multimodal inputs: verified implemented with tests.
- `BindToolsOptions.ParallelToolCalls` core plumbing (since the bind_tools options batch); provider serialization landed in v0.8.0.
