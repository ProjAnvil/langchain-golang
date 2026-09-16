# Divergences from Python LangChain/LangGraph

Deliberate design decisions where this port does not mirror Python. Each entry states why. Anything not listed here aims for parity; file an issue if you find a gap.

## Never planned

- **LangGraph Platform / Server / Studio / CLI** — commercial product lines, out of scope for an OSS port.
- **Local transformers models** — the Go ecosystem has no comparable runtime; use API providers (or ollama for local serving).
- **Command-Send / remaining_steps** — superseded upstream patterns whose Go equivalents (explicit routing, run control) already exist.

## Deliberate design differences

- **Cache short-circuit middleware** — Go's `cache` middleware short-circuits identical requests instead of Python's instrumentation-only behavior; Go idiom favors explicit memoization points.
- **Blank-import provider registration + shim re-exports** — partner packages self-register via `init()` and top-level `langchain/` shims re-export the stable surface, mirroring Go stdlib plugin patterns rather than Python's explicit imports.
- **SQL toolkit read-only guardrails** — langchain-community's SQLDatabaseToolkit executes whatever the model sends; the Go sqltoolkit rejects anything but a single SELECT/WITH statement (comments stripped before analysis, write keywords rejected in the statement body) on both `sql_db_query` and `sql_db_query_checker`, since an LLM with write access to a database is a footgun the port refuses to ship. The checker also validates via the database planner (EXPLAIN) instead of an extra LLM call, `sql_db_schema` accepts empty input to describe all tables, and sample-row values are tab-joined (upstream comma-joins row values but tab-joins headers).

## Deferred (upstream-triggered)

- **deepagents** — upstream is pre-1.0 (0.7.x) with an unstable API; re-evaluate when it reaches 1.0.
- **AWS Bedrock provider, RemoteGraph client** — demand-triggered; both are on the backlog.

## Notable per-adapter behaviors

- **anthropic `tool_choice=none`** — the Messages API has no `none` type; binding fails loudly instead of silently misrouting (bind no tools instead).
- **gemini adapter scope** — targets the Gemini API backend (generativelanguage.googleapis.com, API-key auth) via the official genai SDK only; the Vertex AI backend (Python's `use_vertexai` path) is not covered, and `GOOGLE_GENAI_USE_VERTEXAI` is deliberately ignored (backend pinned) so the adapter cannot be silently rerouted.
- **gemini streaming usage metadata** — the API reports cumulative token counts on every SSE response; Python subtracts previous usage to yield per-chunk deltas, the Go adapter instead surfaces usage only on the terminal chunk (plus a terminal usage-only chunk when counts arrive after the finish chunk), keeping aggregating consumers correct with a single attachment point.
- **gemini `ParallelToolCalls`** — the Gemini API has no parallel-tool-calls payload field; the bind option is accepted and dropped (allowedFunctionNames is already fully determined by the tool-choice mode).
- **TracePolicy processor failures** — upstream records the untransformed payload when a trace processor errors; the Go port drops the payload (fail-closed) since the feature's motivation is PII/compliance.
- **Error-handler abort semantics** — a failed superstep's successful handler outcomes are not committed (Go commits writes at superstep end vs Python's per-task `put_writes`), so a resume after a sibling-task failure re-runs an already-succeeded handler; effects must therefore be idempotent.

## Cleared as non-gaps (audited 2026-09-16)

- `tool_choice=any` (openai→required, anthropic→any) and openai multimodal inputs: verified implemented with tests.
- `BindToolsOptions.ParallelToolCalls` core plumbing (since the bind_tools options batch); provider serialization landed in v0.8.0.
