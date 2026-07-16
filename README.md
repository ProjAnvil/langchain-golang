# langchain-golang

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A community **Go port** of [LangChain](https://github.com/langchain-ai/langchain) — the Python AI application framework. Build LLM agents and LLM applications in Go using LangChain's abstractions: chat models, tools, prompts, output parsers, messages, vector stores, retrievers, and the `create_agent` factory.

> **Not affiliated with or endorsed by LangChain, Inc.** Preview quality (`v0.3.1`); the public API may still change before `v1.0.0`.

## What this is

A Go port of:

- **`langchain_core`** → `core/` — base abstractions and interfaces.
- **`langchain`** (the actively-maintained `langchain_v1` package) → `langchain/` — concrete implementations, the agent factory, middleware, tools.
- **`langchain_text_splitters`** → `textsplitters/`.
- **`langchain-tests`** → `standardtests/` — shared conformance suites.
- **`model-profiles`** → `modelprofiles/` + the `cmd/langchain-profiles` CLI.

It is **not** a port of `langchain_classic` (the legacy package), and **not** a full port of `langgraph` — only the minimal graph runtime that `create_agent` depends on is internalized as a private package (see [Not supported](#not-supported--out-of-scope)).

All tests green: `go test ./...` — 920+ tests across 51 packages.

---

## ✅ Supported

### Core (`core/`)

`messages` (unified struct, content blocks, tool calls, trimming, serialization) · `runnables` (composition, batch, stream, fallback, branch, router, JSON/ASCII/Mermaid graph export) · `language` (`ChatModel` / `LLM` interfaces, fake models, **`ChatModel.Stream`**, rate-limiter hooks) · `tools` (base tool, render helpers, retriever-tool adapter) · `prompts` (string + structured + templated, local JSON loading) · `outputparser` (all parser variants, format instructions, partial parsing) · `callbacks` (manager fan-out, stdout/streaming/file handlers, usage aggregation) · `streamevents` (v3 content-block protocol, `ChatModelStream` projection) · `documents` · `documentloaders` · `indexing` (incl. SQL record manager) · `embeddings` · `vectorstores` (in-memory, filtering, MMR, retriever adapters) · `retrievers` · `exampleselectors` · `tracers` (context/root listener, memory, filtering, replay, event streaming, stdout) · `load` · `stores` · `caches` · `ratelimiters` · `retry` · `_api` (deprecation) · `_security` (SSRF protection, transport validation) · `utils` · `chathistory` · `httpclient` · `modelconfig` · `outputs` · `structuredoutput` · `schema`.

### LangChain v1 (`langchain/`)

- `chatmodels` / `embeddings` — provider registries, parsing, init-spec boundaries.
- `tools.ToolNode` — concurrent dispatch, unknown-tool errors, configurable error handling, `ToolCallWrapper`.
- `messages`, `ratelimiters`.

#### `agents.CreateAgent` — the Go equivalent of Python's `create_agent`

A model ↔ tools agent loop built on an internal graph runtime, with:

- **Middleware chain** — `WrapModelCall` / `WrapToolCall` (outermost-first), `BeforeModel` / `AfterModel` / `BeforeAgent` / `AfterAgent` hooks, `jump_to` short-circuit convention, `context.Context` on every hook (for interrupt).
- **15 middleware modules**: human-in-the-loop, model-call-limit, model-fallback, model/tool-retry, tool-call-limit, context-editing, file-search (ripgrep fast path), pii/redaction, provider-tool-search, shell, summarization, todo, tool-emulator, tool-selection.
- **`system_prompt`** — plain string **and** templated `PromptTemplate` (with per-call variables).
- **`state_schema`** — custom graph-state fields via `StateField` + reducers (`WithAgentStateFields`).
- **`context_schema`** — read-only runtime context over Go `context.Context` (`WithContextValues` / `ContextValue`).
- **`response_format`** — `ToolStrategy`, `ProviderStrategy` (provider-native via `language.StructuredCaller` when the model implements it), and `AutoStrategy` (auto-selects between the two from the model's capabilities).
- **`store`** — cross-thread KV store, injected into each tool call (`WithAgentStore`).
- **`cache`** — model-response cache wired into the model-call path (`WithAgentCache`).
- **`interrupt_before` / `interrupt_after`** — pause at named graph nodes (`WithAgentInterruptBefore` / `WithAgentInterruptAfter`).
- **`model`** — pass a constructed `language.ChatModel`, **or** a bare `provider:model` string resolved via `chatmodels.Resolve` (`WithAgentModel("openai:gpt-4o")`).
- **`tools`** — explicit tools, **or** Go callables reflected into tools via `core/tools.FromFunc` (the `@tool` equivalent).
- **`checkpointer`** — in-memory saver; **interrupt / resume** round trips.
- `recursion_limit`, `name`, `debug`.
- **Streaming** — `Agent.StreamEvents`: real per-token streaming (model deltas + tool/node lifecycle events) over `runnables.Stream[StreamEvent]`.
- **Subagents (agent-as-tool)** — one agent delegates to a named inner agent via a hand-rolled tool whose body calls the inner agent's `InvokeWithState` (mirrors Python's `@tool` + `agent.invoke()`); the nested run is distinguishable by name via `NameFromContext`. A non-streaming nested invoke no longer leaks its events into a streaming parent's stream. See the [Subagents guide](docs/usage/agents.md).

### Text splitters, standard tests, model profiles

- `textsplitters/` — full port (character, HTML, Markdown, code, recursive, header; sentence-transformers / NLTK / spaCy / KoNLPy adapter interfaces).
- `standardtests/` — chat-model / embeddings / retriever / vector-store / runnable conformance suites.
- `modelprofiles/` — profile registry, Markdown summary, the `langchain-profiles refresh` CLI (merges models.dev data + TOML overrides → `profiles.json`).

### Partner packages

`partners/openai` · `partners/anthropic` · `partners/ollama` (chat models & embeddings) · `partners/chroma` (vector store). `partners/openai` is a full integration — its `ChatModel` (Responses API: Invoke/Stream/tool-calling) implements `language.StructuredCaller` and self-registers into `chatmodels.Resolve`, so `WithAgentModel("openai:gpt-4o")` works out of the box. The others are usable integrations and validation aids; adapter slots for more partners.

---

## ❌ Not supported / out of scope

### Deliberately not ported

- **`langchain_classic`** — legacy chains, agents, memory, tools, retrievers, vectorstores, storage. The classic `AgentExecutor` is gone; use `agents.CreateAgent`.
- **A full `langgraph` port**. Only the minimal subset `create_agent` depends on lives here, internalized at `langchain/internal/agentruntime/` (package `agentruntime`, **not exported**). Intentionally absent: subgraphs, streaming modes beyond `events`, time-travel / state history, caching/retry policies, the functional `@entrypoint`/`@task` API, persistent Postgres/SQLite checkpoint backends, and the langgraph CLI/SDK.
- **Subagent transformer (`transformers` / `run.subagents`)** — not exposed. `transformers` is a langgraph stream-mode construct, and this port holds the `agentruntime` boundary (no stream modes). The motivating feature — **PII streaming-delta redaction** — IS delivered, via a bounded middleware delta layer (`WrapModelStreamHook` + `PIIStreamTransformer`'s lookback buffer); batch redaction also works.
- **Functional `@entrypoint`/`@task` API**, **time-travel**, **subgraphs** — see above.

### Limited partner coverage

- Only `openai`, `anthropic`, `ollama`, `chroma`. **No Google/Gemini, AWS, Azure, Pinecone, etc.** — community contributions welcome.
- `langchain/chatmodels` parses a model name to a `ChatModelSpec` **and** resolves it to a constructed partner `ChatModel` via the Go provider registry (`Resolve` + `RegisterProvider`); `WithAgentModel("openai:gpt-4o")` works end-to-end. (anthropic/ollama/chroma are not yet registered as real Go factories — pass a constructed `language.ChatModel` for those.)
- `langchain/tools.ToolNode` does **not** support `Command`/`Send` returned from tools, or reflection-based `InjectedState` / `InjectedStore` / `ToolRuntime` argument injection.

### Other gaps

- `core/prompts` does not load YAML, Jinja templates, or `lc://` Hub prompts (string + local JSON only).
- `core/runnables` PNG graph rendering is unsupported (JSON/ASCII/Mermaid are).
- Python-style dynamic provider import / instance construction is unsupported — construct concrete models in Go.
- **File tools** (`Read`/`Write`/`Edit`/`Bash`) and **sandboxing** are out of scope — those are provided by [`claude-agent-sdk-golang`](https://github.com/ProjAnvil/claude-agent-sdk-golang), not by LangChain.

The support / gap tables above are the canonical compatibility reference. Open an issue if you need detail on a specific gap.

---

## Installation

```bash
go get github.com/projanvil/langchain-golang@v0.3.1
```

Requires Go 1.23+.

## Quick start

A minimal runnable example using the in-tree fake model (swap in a partner `ChatModel` for production):

```go
package main

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents"
)

func main() {
	model := language.NewFakeChatModel(
		language.WithResponses(messages.AI("It's sunny in Shanghai.")),
	)

	agent, err := agents.CreateAgent(model, nil,
		agents.WithAgentSystemPrompt("You are a helpful assistant."),
		agents.WithAgentName("my-agent"),
	)
	if err != nil {
		panic(err)
	}

	// Non-streaming:
	reply, _ := agent.Invoke(context.Background(), []messages.Message{
		messages.User("What's the weather?"),
	})
	fmt.Println(reply[len(reply)-1].Content)

	// Streaming:
	stream, _ := agent.StreamEvents(context.Background(), []messages.Message{
		messages.User("Tell me a story."),
	})
	for {
		ev, ok, _ := stream.Next(context.Background())
		if !ok {
			break
		}
		if ev.Type == agents.StreamModelDelta && ev.Text != "" {
			fmt.Print(ev.Text)
		}
	}
}
```

For a real model, either construct a `language.ChatModel` from a partner package (e.g. `partners/openai`, `partners/anthropic`, `partners/ollama`) and pass it positionally, **or** resolve one from a bare name string: `agents.CreateAgent(nil, nil, agents.WithAgentModel("openai:gpt-4o"))` (configure via `OPENAI_API_KEY` / `OPENAI_BASE_URL` env vars).

## Documentation

Usage guides live under [`docs/`](docs/) — example-driven, with every snippet offline-friendly (using the in-tree fake model unless noted):

- [Getting started](docs/usage/getting-started.md) — install, configure a provider, run your first agent.
- [Composing runnables (LCEL)](docs/usage/composition.md) — `Pipe` / `Pipe3-6` / `Parallel` / `Branch` / `Fallbacks` / `Retry`, the Go equivalent of Python's `prompt | model | parser`.
- [Agents — `CreateAgent`](docs/usage/agents.md) — system prompts, tools, the 15-module middleware chain, structured output, interrupts.
- [Streaming](docs/usage/streaming.md) — `Agent.StreamEvents`: per-token model deltas + tool/node lifecycle events.

For the full API reference, see the package docs at [pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang). Compile-checked examples also live in each package's `example_test.go`.

## Repository layout

```
langchain-golang/
├── core/                  # langchain_core port
├── langchain/             # langchain (v1) port
│   ├── agents/            # CreateAgent + 15 middleware
│   ├── chatmodels/ embeddings/ messages/ tools/ ratelimiters/
│   └── internal/agentruntime/   # internal graph runtime (not exported)
├── textsplitters/         # langchain_text_splitters port
├── standardtests/         # langchain-tests conformance port
├── modelprofiles/         # model-profiles port
├── partners/              # openai, anthropic, ollama, chroma
└── cmd/langchain-profiles # profiles refresh CLI
```

## Acknowledgments

This project is a Go port of [LangChain](https://github.com/langchain-ai/langchain) (MIT License, Copyright © LangChain, Inc.) and [LangGraph](https://github.com/langchain-ai/langgraph). All credit for the original design and abstractions belongs to the LangChain team.

## License

[MIT](LICENSE) — Copyright © 2026 ProjAnvil.
