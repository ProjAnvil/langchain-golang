# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

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

[Unreleased]: https://github.com/ProjAnvil/langchain-golang/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/ProjAnvil/langchain-golang/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ProjAnvil/langchain-golang/compare/v0.6.5...v0.7.0
