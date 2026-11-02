# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

## [0.12.5] - 2026-11-03

Nested Stream emitter regression — v0.12.0's #9458 strip also dropped a
nested Stream's OWN emitter, so a graph node that streamed a child graph
with its own ThreadID received zero chunks (the run ended cleanly with no
events). Found by woca's goal mode (outer goal graph node streaming the
inner agent graph per cycle). Fix: `CompiledGraph.run` takes Stream's
emitter as an `own` parameter and installs it after the #9458 strip, so
the strip only ever drops carriers inherited from an enclosing run.

### Fixed
- langgraph: a nested own-thread `Stream` delivers its own chunks again
  (v0.11.1 semantics); a nested same-thread `Stream` keeps shadowing the
  caller's emitter; the #9458 drop for nested own-thread Invoke/AddSubgraph
  behavior is unchanged. Pinned by TestNestedStreamOwnThreadStillYields and
  TestNestedStreamSameThreadShadowsCallerEmitter.



tool_end payload fix — closes the v0.12.3 known issue where default-converted tool errors emitted a nil ToolResult.

### Fixed
- agents: `tool_end` StreamEvents are emitted from the tools node's ordered post-conversion outcomes instead of from inside the wrap chain (which only observes raw errors since the v0.12.3 layering move). Default-converted tool errors carry their converted content and `status=error` again; a declined error (`ToolErrorMiddleware` propagation path, run failing) emits no `tool_end`, matching Python's no-`on_tool_end`-for-failing-tools shape; and concurrent siblings' `tool_end`s follow the call order rather than the completion order (batched at the tools node). In a mixed batch where one sibling's error is declined, the completed siblings' `tool_end`s are suppressed with it. TDD: three reproduction tests plus two regression pins (lifecycle ordering, middleware-converted path).

## [0.12.3] - 2026-11-02

ToolErrorMiddleware port — Python's per-exception tool-error disclosure middleware (#38781) as a first-class agent middleware, plus the ToolNode error-path layering it needs to observe raw tool errors. Plan: `docs/superpowers/plans/2026-11-02-tool-error-middleware.md`.

### Added
- agents: `middleware.ToolErrorMiddleware` — `OnError` receives each tool-execution error with its `ToolCallRequest`; returning content (string or content blocks) converts it into an error `ToolMessage` (status=error, tool name, call id), declining propagates the original error and fails the run (`tools.UnhandledToolError` sentinel bypasses the default conversion). Tool filter (`WithToolErrorToolInstances`), constructor validation, control-flow signals never reach the handler, and the ToolRetryMiddleware composition ordering (retry inner / error-format outer) is documented and pinned. Upstream's async-only split (`aon_error`) has no Go analog — single execution path.
- tools: `UnhandledToolError` — the decline marker `ToolErrorMiddleware` wraps propagated errors in; `ToolNode` strips it (all direct layers) at the conversion point and re-raises the original unconverted.

### Changed
- tools: `ToolNode` applies `handleToolErrors` at the runOne level, OUTSIDE the wrap chain, mirroring langgraph's `_run_one` — wrap middleware now observe raw tool errors instead of pre-converted messages. With no wrap chain the observable behavior is unchanged; one intended flip for existing `ToolRetryMiddleware` users inside create_agent's always-installed chain: retries now actually fire on tool failures (previously the in-chain conversion hid them, so the retry path was dead) and "continue"-mode failures carry the retry-formatted text. Wrapper-thrown errors (e.g. an HITL edit naming an unregistered tool) are now default-converted where they previously failed the run — matching Python's `handle_tool_errors=True` mode.

### Known issues
- With create_agent's always-installed wrap chain, `tool_end` stream events for default-converted tool errors carry a nil ToolResult (previously the converted text). Python emits no `on_tool_end` at all for failing tools (only `on_tool_error`), so neither shape has an upstream analog; see DIVERGENCES.md.

## [0.12.2] - 2026-11-02

Divergence-backlog round — four portable divergences paid down against the wave-4 references (upstream refresh check: no releases past langchain-core 1.7.3 / langchain 1.4.7 / langgraph 1.2.19 / langgraph-redis 0.5.2). Plan: `docs/superpowers/plans/2026-11-02-v0.12.2-backlog-divergences.md`.

### Fixed
- anthropic: `claude-fable-5*`/`claude-sonnet-5-5*` reject `top_k` and non-default (`!= 1`) `top_p`/`temperature` during request validation with Python's verbatim messages (#40106, #40882), and fable-5 payloads drop `temperature`/`top_k`/`top_p` at build — ordered after the adaptive-thinking constraint injection so the pops win. The Go model exposes no call-time sampling channel, so constructor values are validated (Python's per-request-kwargs half is unreachable in Go).
- anthropic: a leading system message's tool-change blocks forward unconditionally (Python `preserve_tool_changes=True`, #40758) instead of being gated on the profile's `mid_conversation_tools`; the in-place mid-conversation gate remains a documented divergence (Python forwards unconditionally there too; the Go gate is a client-side guard for the sonnet-5 server rejection the upstream profile data flags).
- langgraph: pinned (time-travel) resumes from an exit-mode anchor checkpoint no longer double-apply an already-hydrated task's write — stale synthetic-id anchored writes are excluded from the resume hydration seed and their delta channels sealed, mirroring Python's forced-snapshot protection for addressed resumes where reapply is disabled (#8548). The wave-4 invariant holds: the resumed task still re-executes, and a failed final save still does not make it look done.
- openai: chat-completions stream assembly keeps the first-seen tool-call id/name per index (sticky first-wins, Python `_merge_chunk_into_store`, #36834); args concatenation and per-slice forwarding are unchanged.

## [0.12.1] - 2026-11-02

Residuals wave — the review-owed debts of parity wave 4: openai streaming tool-call assembly, anthropic system-block narrowing / thinking validations / betas, and the langgraph pause-repair and interrupt_after persistence gaps, plus a mechanical gofmt gate. Spec/plan: `docs/superpowers/{specs,plans}/2026-11-02-v0.12.1-residuals*`.

### Fixed
- openai: chat-completions stream assembly keeps sparse-index tool calls — the final chunk iterated a dense 0..n-1 range over a streamed-index map, silently dropping any call whose index left gaps below the highest streamed index (Python `_merge_chunk_into_store` iterates the present indices in sorted order); dense index sets are unchanged.
- anthropic: unrecognized non-text/non-tool-change system blocks are dropped with the Python-shaped warning on the hoist path too (previously only in-place turns narrowed; the hoist kept the first block and failed); manual thinking config is rejected on the thinking-pinned model families (`claude-fable-5*`/`claude-sonnet-5-5*` reject any manual thinking, `claude-opus-5*` rejects enabled) with Python's verbatim remedies, effort-independently.
- langgraph: the #9235 injected-failure pause-repair family is uniform across durability modes — repair payloads keep snapshot-suppressed channels blob-free (a suppressed pause checkpoint cannot be re-materialized by its own repair), the exit-mode pause checkpoint gets the same failure repair its sibling write paths perform, and interrupt_after pause checkpoints persist the triggering superstep's below-cadence delta writes instead of deferring them until the run resumes.

### Added
- anthropic: the remaining #41180/#41536 request surfaces — thinking `display: updates` and `user_profile_id` auto-append their betas (task_budget → thinking-display → tool-change → user-profiles order), `WithUserProfileID` sets the constructor field, and `WithOutputConfig(map[string]any)` emits the constructor `output_config` (effort/format/task_budget, dict-then-`WithEffort` merge, `task_budget` auto-append).
- CI: gofmt gate — the `gofmt` formatter in golangci-lint (root module) plus a `make fmt-check` step; gofmt is module-blind so one `gofmt -l .` walk covers the nested checkpoint modules too, which `golangci-lint run ./...` cannot reach.

### Testing
- langgraph: dedicated unit test pinning the nil-ChannelValues decode guard (a tuple with no channel values must decode to a non-nil assignable map).

## [0.12.0] - 2026-11-01

Parity wave 4 — the late-October 2026 upstream releases (langchain-core 1.7.3 · langchain 1.4.7 · langgraph 1.2.19 · checkpoint 4.3.1 · prebuilt 1.1.1): the second DeltaChannel-correctness wave, the core 1.7.0 tool-call stream protocol, and partner catch-up. Spec/plan: `docs/superpowers/{specs,plans}/2026-11-01-parity-wave4*`; gap analysis: `reports/14-latest-release-port-gap.md`.

### Added
- langgraph: delta-channel persistence correctness (#8544/#9114) — every write carries its dispatch-position task path and sqlite/postgres/redis replay pending writes in live dispatch order; resumed sends dispatch above restored tasks, resumed siblings' delta writes apply exactly once, and resumed exit runs replay in live order.
- langgraph: exit-durability runs against a superseded checkpoint snapshot onto their own fork instead of contaminating the newer branch (#9269), covering the new-turn and pinned-at-latest cases.
- langgraph: every message written to a messages delta channel is stamped with a stable ID before apply/persist (#9275), so RemoveMessage targeting and ID dedupe survive persisted writes.
- langgraph: cooperative draining (#9281) — `Options.Control`/`Runtime.Control` RunControl requests stop the run at the next superstep boundary (breakpoints still win) with the requesting superstep committed, returning `*types.GraphDrainedError`; subgraphs and nested invokes inherit the caller's control (see DIVERGENCES.md).
- langgraph: messages streaming emits node-returned values (#9463); nested invokes with their own thread/checkpoint drop inherited stream carriers (#9458); graph export lists a ToolNode's tool names as `__tools` metadata via `AddNodeWithBound`/`prebuilt.AddToolNode` (#9449).
- langgraph: a delta channel's value survives its field changing to a plain reducer (#9302); a failed delta write repairs its checkpoint instead of stranding it (#9235); store namespace labels (incl. NUL bytes) are validated on every entry point (#9383); redis `GetTuple` returns pending writes, locked by test (langgraph-redis #220).
- core/partners: tool-call args stream as `tool-call-delta` protocol slices (#41257) — `content-block-start` skeleton carrying id/name, per-slice args deltas without repetition — emitted incrementally by openai (Responses and Chat Completions) and anthropic and assembled by `ChatModelStream.Dispatch` alongside the legacy shape; gemini/ollama stay whole-block (see DIVERGENCES.md).
- core: notification metadata helpers (#41540) — `lc_source`/`lc_data` add/get on messages, deep-merged across `MergeRuns`; LangSmith tracer accepts `address` (#41006); llm-cache global accessors.
- anthropic: `WithEffort`/`reasoning_effort` — `output_config` emission, profile-gated adaptive-thinking default, and the opus-5 thinking/effort conflict validation (#41536); mid-conversation system messages and tool changes gated on the model's profile with tool-change beta auto-append and beta-endpoint routing (#41180).
- modelprofiles: `mid_conversation_*` profile keys plus refreshed materialized data (#41180).

### Fixed
- langgraph: `trace_policy` scopes to tracers only (#9486) — usage aggregation, the messages bridge, and StreamEvents observe real payloads again (`callbacks.RawPayloadHandler` opt-out; language-forced scoping difference in DIVERGENCES.md).
- langgraph: a new turn's exit input write no longer lands on a paused thread's collision slot (#9114); pause checkpoints no longer strand sibling writes beside anchored snapshots (#9235 review).
- openai: Responses stream items land on the content block they started on after `output_index` drift (#41121); both chunk bridges keep wire indices collision-free.
- core: Mermaid export escapes node labels and drops nested containers from metadata labels (#41512); notification `lc_data` deep-merges instead of overwriting (#41540).

### Changed
- langgraph: **delta snapshot cadence defaults to 100 updates** (was 1000, #9420) — tighter replay-latency bound, more frequent full snapshots; pass an explicit `snapshotFrequency` to `NewDeltaChannel`/`NewMessagesDeltaChannel` to keep the old spacing.
- core/partners: **tool-call streaming uses the `tool-call-delta` protocol** (#41257) — consumers of content-block events see a start skeleton plus id/name-free args slices instead of repeated self-describing chunks; the assembled final chunk is unchanged.
- anthropic: requests requiring beta flags (in-place tool-change system blocks) automatically append the tool-change betas and post to `/messages?beta=true` (#41180).
- Publish checkpoint/sqlite and checkpoint/postgres v0.3.9 with the root module pinned to v0.12.0.

## [0.11.1] - 2026-10-24

### Fixed
- langgraph: reserve exit-durability delta anchor IDs before loop checkpoint IDs so an anchor created across a millisecond boundary cannot shadow the final checkpoint as the latest state or lose the next turn's delta input.
- runnables: make batch lifecycle tests verify independent per-run start/end pairs under controlled concurrency instead of assuming globally serial callback events.
- checkpoint/sqlite and checkpoint/postgres: preserve operation errors together with unexpected close/rollback failures instead of discarding cleanup errors; already-finished transactions remain a normal cleanup outcome.

### Changed
- Modernize Go 1.26 collection, iteration, pointer, and worker-pool idioms across modules while preserving public APIs, JSON formats, copy semantics, and nil/empty container behavior.
- Publish checkpoint/sqlite and checkpoint/postgres v0.3.8 with the root module pinned to v0.11.1.

### Testing
- Add deterministic checkpoint clock-boundary and concurrent batch lifecycle regressions, cleanup-error fault injection, and stable-ranking coverage.
- Validate the root and nested checkpoint modules with build, vet, race, and lint checks, plus offline integration type checks and the existing per-package coverage gate.

## [0.11.0] - 2026-10-13

Parity wave 3 — upstream releases of 2026-10-08 (langchain-core 1.6.9 · langchain 1.4.4 · openai 1.7.0) plus the TypeSafe/Jev partner and provider catch-up. Spec/plan: `docs/superpowers/{specs,plans}/2026-10-12-parity-wave3*`; gap analysis: `reports/13-latest-release-port-gap.md`.

### Added
- partners/typesafe: TypeSafe classifier for the Jev decision model (#40542/#40570/#40659) — Noul/Choice/Score questions with probability/confidence answers, recursive state serialization (messages anywhere inside JSON), `/v1/systemone` client with Bearer auth, and a typed `APIError` layer embedding `lcerrors.ProviderError` (safe `Error()` that never leaks body/headers; `errors.Is` sentinels and `errors.As` both work); retry-after-ms/-after/HTTP-date parsing.
- partners/typesafe/middleware: experimental `AutoModeMiddleware` (Noul risk gate, fail-closed, threshold 0.5, verbatim blocked notices) and `ModelRouterMiddleware` (Choice-based per-run model routing, JSON-native `model_route` state).
- partners/openai/decisions: OpenAI Decisions API client (#41125) — Predicate/Choice/Score questions, OpenAI-shape message mapping, string/Human-sequence/JSON input assembly, `POST /decisions` with lcerrors mapping and answer normalization.
- partners/openai/middleware: `AutoMode` (Decisions risk gate with with-images/text-only variants, 400/413 downgrade, verbatim notices) and `ModelRouter` (Choice routing with per-call model override via `WithModel`, omit-payload trace policy).
- partners/exa: `exa_search_results_json` / `exa_find_similar_results_json` tools and `SearchRetriever`, REST-direct (exa 1.1.0 parameter surface).
- partners/nomic: text embeddings via the Atlas API (nomic 1.0.1) — method-driven task_type defaults, explicit four-value validation, optional Matryoshka dimensionality.

### Fixed
- core/security: SSRF — IPv4-compatible (`::/96`, int>1) addresses now unwrap onto the IPv4 check path (`::127.0.0.1`, `::10.0.0.5`, `::169.254.169.254` previously bypassed loopback/private/metadata rules); NAT64 discovery prefix `64:ff9b:1::/48` joins the blocked ranges (langchain core #41151).
- summarization: the summary step retries once on `ErrModelContextOverflow` with the middle of the history cut out (half the measured tokens, capped at 40% of `max_input_tokens`) (#41159).

## [0.10.0] - 2026-10-06

Core-semantics parity with the October 2026 upstream releases (langchain-core 1.6.6 · langchain 1.4.3 · langgraph 1.2.13). Audited by an independent review pass before release; audit-driven hardening folded in (durable-serde safety for new state, Python-exact notice text, wired internal-call filtering).

### Added
- core/lcerrors: standard model error kinds (#39538) — `ModelErrorKind` classification (auth/permission/invalid-request/not-found/rate-limit/server/connection/timeout/context-overflow with 400-body sniffing), fine-grained `ErrModel*` sentinels (subsumed by `ErrProvider` for compatibility), `ProviderError.IsModelRetryable()` following Python's `is_retryable` defaults, and transport connection-refused/reset classification.
- agents: `DefaultRetryOn` predicate (Python `default_retry_on`, #39538) is now the default `RetryOn` for ModelRetryMiddleware and ToolRetryMiddleware — classified model errors retry only when retryable; unclassified errors retry.
- gemini: genai `APIError`s map onto the standard model error kinds (`errors.Is`-able sentinels, original error preserved in the chain).
- langgraph: `Interrupt.ResponseSchema` + `graph.InterruptWithSchema` (#8886) — surfaced interrupts carry the JSON Schema of the expected resume value, round-tripping through the durable checkpoint serde (resume values themselves are not validated against the schema; see DIVERGENCES.md).
- agents/middleware (HITL): edit decisions keep the model's original call on the AIMessage, record the reviewer's action in the `hitl_edited_tool_calls` state key (JSON-native, durable-saver serializable; rewritten every review round so stale edits cannot leak), substitute it at execution time via `WrapToolCall` (tool re-resolution included; an edited name binding no tool fails loudly), and prepend the edit notice naming the executed action (Python #40463 text; `WithEditNotice("")` disables). After-tools return-direct routing resolves executed tool names from client-tool messages, falling back to the model's recorded name.
- callbacks/langgraph: `callbacks.WithInternalCall`/`IsInternalCall` context marking filters middleware-internal model calls out of messages-mode projections and StreamEvents chat-model events (#39252; mechanism divergence, see DIVERGENCES.md); wired into the tool-selection and tool-emulator middleware model calls.

### Fixed
- agents: invalid tool calls are answered with error ToolMessages so the model receives corrective feedback (#40530); custom HITL rejection reasons are framed as user rejections (#39773).
- langgraph: delta channels are hydrated from the ancestor history on the run/resume/new-turn/UpdateState paths, so resumed runs continue accumulated values and update checkpoints include ancestor state (#8548, #9170, #9165, #9142); never-written delta channels short-circuit the history walk via ChannelVersions (#9141); delta counters advance over registered-but-never-materialized protos and exit-durability flush no longer double-advances.

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

[Unreleased]: https://github.com/ProjAnvil/langchain-golang/compare/v0.12.1...HEAD
[0.12.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/ProjAnvil/langchain-golang/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.8.1...v0.9.1
[0.8.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.6.5...v0.7.0
