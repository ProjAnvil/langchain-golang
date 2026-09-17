# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

## [0.9.2] - 2026-09-17

### Added
- Test coverage gate: unit tests for all parity catch-up packages brought to ≥95% statement coverage each (merged 98.3%) — pgvector 99.8%, redisvector 99.2%, MCP 98.6%, Gemini 100%, SQL toolkit 97.7%, Cohere/Jina rerank 98.1%, vectorstores 99.4%.
- `make coverage` / `make coverage-html` targets with a 95% red line over the tracked package list; a CI coverage job reports the per-package table to the job summary and fails below the threshold.
- CONTRIBUTING: documented the coverage gate, tracked packages, and the exemption process.

### Testing
- Error-injection coverage across the new partners (driver/rows errors, HTTP 5xx and mid-stream failures, malformed responses, elicitation failure branches, SQL guard lexer token paths) — all offline via fakes and mocks; env-gated e2e suites unchanged.

**Full Changelog**: https://github.com/ProjAnvil/langchain-golang/compare/v0.9.1...v0.9.2

## [0.9.1] - 2026-09-17

Full parity catch-up release: the complete RAG stack, MCP tool integration, a native Gemini provider, document loaders, the SQL toolkit, runnable examples, and a bilingual docs site. (Milestones M1–M3 of the parity catch-up program, released together.)

### Added — RAG stack
- vectorstores: declarative filter capability interface — `SearchOptions` + optional `OptionSearcher` (zero changes to the base interface), an 11-operator `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$between/$exists/$like` DSL matching Python's `SearchArgs.filter`, retriever kwargs pass-through, and in-memory/chroma adapters (af1b2af)
- partners/pgvector: pgx/v5 vector store — per-collection tables with auto DDL + HNSW/IVFFlat indexes, cosine/L2/IP distances, jsonb filter SQL (fully parameterized), pgxmock unit tests + env-gated conformance (c6fb101)
- partners/redisvector: RediSearch store — FT.CREATE vector schema, KNN search, declared-metadata JSON filtering, RESP2/RESP3 reply parsing, env-gated conformance against Redis Stack (933bd11)
- retrievers: `DocumentCompressor` abstraction + `ContextualCompressionRetriever`; `MultiQueryRetriever` (model-generated query variants), `ParentDocumentRetriever` (child retrieval → parent reassembly via a BaseStore docstore), `EnsembleRetriever` (RRF fusion, Python rank semantics) (9af9be2, 1906b50)
- partners/cohere & partners/jina: hosted rerank adapters (hand-written REST, zero SDK deps) (f118582)

### Added — models & tools
- partners/mcp: MCP adapter following langchain 1.4.0's `langchain.mcp` blueprint (mcp-go) — MCPConfig + ClientGroup connection models with namespaced tools, discovery cache modes, the four-quadrant result mapping (content blocks / structured artifact / error ToolMessage / transport raise), **elicitation bridged to LangGraph interrupts** (transport-independent, accept/decline/cancel), and destructiveHint-based HITL gating (f7d3e77)
- langgraph: `graph.InterruptSupported` context probe for partner adapters (34b42bd)
- partners/gemini: native Gemini chat model on the official genai SDK — tool calling + tool-choice mapping, structured output (responseJsonSchema), SSE streaming with terminal usage, multimodal inputs, provider self-registration; passes all five standardtests chat-model conformance layers (7b6533b)
- documentloaders: HTML (goquery), Web (http + HTML extraction, Python-aligned headers/metadata), and PDF (ledongthuc/pdf) loaders with golden-file tests (8a6b385)

### Added — toolkit, examples, docs
- langchain/toolkits/sqltoolkit: the four SQL tools (query / list_tables / schema / checker) over database/sql with **read-only guardrails** (single SELECT/WITH only, comment-stripped lexical analysis, write keywords rejected; guardrails verified against real bypass attempts) and sqlite/postgres introspection (917c7a3)
- examples/: 12 runnable examples — quickstart, HITL, streaming, subgraph resume, fault tolerance (retry + error handler), TracePolicy scrubbing, full RAG chain, MCP tools, Gemini agent, SQL agent, middleware suite, advanced retrievers (25ce7c3)
- docs site: mkdocs-material bilingual site with six new bilingual guides (RAG stack, MCP, Gemini, SQL toolkit, fault tolerance, loaders) and a GitHub Pages workflow (5381a38)
- CI: dependency pin alignment check keeping root and nested module pgx/go-redis versions in lockstep (8758bec)

### Fixed
- RRF rank base corrected to Python's 1-based semantics; Gemini stream double-Close panic and ctx registration leak; MCP adapter connection cleanup on partial startup failure; bounded waits on elicitation cancel; `$between` mixed-type validation; MMR double-embedding in pgvector/redisvector (audit fixes e70c9b7, 2d599a7, 12b9eae)

**Full Changelog**: https://github.com/ProjAnvil/langchain-golang/compare/v0.8.1...v0.9.1

## [0.8.1] - 2026-09-17

### Added
- langgraph: node-level error handlers (`NodePolicies.ErrorHandler`, mirroring langgraph 1.2.0 `error_handler=`): handlers receive the state snapshot and a typed `NodeError` after retries are exhausted and may return a plain update or a `Command`; the task error is persisted as a checkpoint ERROR write before the handler runs, so a crash mid-handler resumes into a handler re-run instead of the node.
- langgraph: per-node `TracePolicy` (`NodePolicies.Trace`, mirroring langgraph 1.2.11) — emit-side payload transforms applied before any tracer observes chain events; `OmitPayload` helper included. Processor panics fail closed (deliberate divergence, see DIVERGENCES.md).
- agents: middleware `TracePolicyProvider` hook (mirroring langchain 1.3.15) for per-middleware payload scrubbing.

## [0.8.0] - 2026-09-16

### Added
- CI (GitHub Actions): root module tests on a go 1.26/1.27 matrix, nested checkpoint module suites (sqlite, redis via miniredis, postgres via embedded binaries), golangci-lint v2.13.2 gate.
- openai: `parallel_tool_calls` serialization on both the Responses and Chat Completions paths via `BindToolsOptions.ParallelToolCalls`.
- anthropic: `ParallelToolCalls` now synthesizes `tool_choice.disable_parallel_tool_use`, defaulting tool_choice to `{"type":"auto"}` when unset.
- Community files: CONTRIBUTING, DIVERGENCES, SECURITY, and this CHANGELOG.
- Package-level godoc comments for all core/langchain/modelprofiles packages.

### Changed
- Error strings lowercased to Go convention across agents/embeddings/middleware (ST1005); lint findings cleared (86 → 0).

## [0.7.1] - 2026-09-16

### Changed
- Modernized to Go 1.26 idioms throughout (3a24fdf).

### Chore
- Pin langchain-golang v0.7.1 in nested checkpoint modules (29cacb1).

## [0.7.0] - 2026-09-12

### Breaking
- Graph-level default retry: `DefaultRetryOn` now retries all errors unless a custom `RetryOn` is set (Python parity).
- Subgraph checkpoint namespaces changed to NS-routed values; interrupted subgraphs pause the parent and resume via the new namespace scheme.
- Anthropic streaming now yields a terminal usage-only chunk on `message_delta` (aligns usage accounting with invoke).

### Added
- langgraph: interrupted-subgraph pause/resume across processes; durable pause writes (async/exit); per-run durability override and per-task subgraph checkpoints; graph-level default retry policy; semantic search for InMemoryStore (`index=` parity).
- runnables/callbacks: StreamEvents (Runnable-level `astream_events`, v2 projection); chain lifecycle events across all combinators; Bind/Pick/Each/BatchAsCompleted and fallback error filtering.
- agents: interrupt-based human-in-the-loop with cross-process approval; middleware tool/state auto-collection; tool-returned Commands and agent-hook jumps; `write_todos` state; per-call response_format/tool_choice/model_settings overrides; apply update-only Commands from `wrap_model_call` middleware.
- tracers: LangSmith run-tree tracing with a batched background client.
- partners/openaicompat: OpenAI-compatible provider registry (groq, mistralai, deepseek, xai, openrouter, fireworks, perplexity).
- partners/openai: multimodal inputs, Responses reasoning effort, streaming usage, sampling params (top_p/stop/seed/penalties/logit_bias/n/logprobs), honest capability declarations.
- partners/anthropic: usage cache token details, stop_sequences, explicit errors for non-base64 data URIs.
- language/agents: bind_tools options with tool_choice (ToolStrategy forces "any").
- standardtests: layered chat-model conformance suites (tool calling/choice, structured output, multimodal, streaming) wired to openai/anthropic/ollama.
- prompts: Partial and FewShotChatMessagePromptTemplate.
- core/language: real GPT-2 BPE (r50k_base) fallback token counting; messages usage-metadata scaling.

### Fixed
- openai: Chat Completions path serializes structured-output response_format; ToolStrategy narrowing override filters bindings and request tools.
- Durable pause writes, run-id tree, minted message ids, tracer payload ownership (final-review fixes).

## [0.6.5] - 2026-09-03
create_agent parity: return_direct, structured-output retry, routing, 9999 recursion default, dynamic model; nested module pin refresh.

## [0.6.4] - 2026-09-01
Graph visualization parity: `get_graph`, Mermaid options, xray, router probing, box-drawing ASCII.

## [0.6.3] - 2026-08-27
Streaming robustness, bounded batch concurrency, constructor error returns (rectification batch).

## [0.6.2] - 2026-08-26
tiktoken-go/tokenizer v0.8.1; Go floor raised to 1.26.

## [0.6.1] - 2026-08-26
tiktoken-based token counting (tokenizer v0.7.0, go 1.23 floor); anthropic count_tokens API-backed message counting.

## [0.6.0] - 2026-08-25
Initial public parity line: agents, graphs, checkpoint savers, partners (openai/anthropic/ollama/chroma), textsplitters, standardtests.

## [0.5.x] - 2026-08
Early development line preceding the parity baseline.

[Unreleased]: https://github.com/ProjAnvil/langchain-golang/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/ProjAnvil/langchain-golang/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.8.1...v0.9.1
[0.8.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.6.5...v0.7.0
