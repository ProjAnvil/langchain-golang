# langchain-golang

[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1382%20passing-brightgreen)]()
[![Packages](https://img.shields.io/badge/packages-62-blue)]()

**Languages:** English | [简体中文](README.zh-CN.md)

A community **Go port** of [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph) — build production-grade LLM agents and applications in pure Go.

> **Not affiliated with or endorsed by LangChain, Inc.** Preview quality; the public API may change before `v1.0.0`.

---

## Why langchain-golang?

| | langchain-golang | Python LangChain |
|---|---|---|
| **Language** | Go (statically typed, compiled) | Python |
| **Concurrency** | Goroutines (native) | asyncio / threading |
| **Deployment** | Single binary, no runtime deps | Python interpreter + venv |
| **Type safety** | Compile-time checked | Runtime (Pydantic) |
| **Checkpointer backends** | In-memory, SQLite, PostgreSQL | Same + Redis, MongoDB |

## Features

### 🤖 Agent Framework — `agents.CreateAgent`

The Go equivalent of Python's `create_agent`, built on the `langgraph/` runtime:

- **Model ↔ Tools loop** with streaming, interrupts, and checkpointing
- **17 middleware modules**: human-in-the-loop, context-editing (summarization), model/tool retry, fallback, call limits, PII redaction, shell execution, file search, TODO tracking, tool selection/emulation
- **Structured output**: tool strategy, provider-native strategy, or auto-detect
- **Cross-thread memory** via `store.Store` (semantic key-value with namespace hierarchy)
- **Model-response caching** with per-node `CachePolicy`
- **Interrupt / resume** round trips with versioned checkpoint history and time travel

### 🔄 Graph Runtime — `langgraph/`

A Pregel-style state graph executor mirroring Python's LangGraph 1.2.x:

- **StateGraph builder** with typed channels: `LastValue`, `Topic`, `BinaryOperator`, `Ephemeral`, `Barrier`, **`DeltaChannel`** (sentinel-only checkpoint storage with counter-based snapshot cadence), **`Overwrite`** reducer
- **Checkpointing**: in-memory, SQLite (`modernc.org/sqlite`, pure Go), PostgreSQL (`pgx/v5`) — all sharing a conformance suite
- **Durability modes**: `sync` (default), `async` (background goroutine writer), `exit` (deferred flush) via `checkpointSink`
- **Stream API**: `values` / `updates` / `debug` / `messages` / `custom` / `delta` modes
- **Functional API**: `@entrypoint` / `@task` equivalent (`langgraph/fn`)
- **Subgraphs**, join edges (`AddJoinEdge`), per-node retry/cache policies
- **TimeoutPolicy**: `run_timeout` / `idle_timeout` with heartbeat refresh

### 🧱 Core Abstractions — `core/`

30 packages porting `langchain_core`:

- **Messages**: unified `Message` struct, typed content blocks (text/image/audio/video/file/reasoning/citation/tool-call), tool calls, trimming, serialization
- **Runnable composition (LCEL)**: `Pipe` / `Parallel` / `Branch` / `Fallbacks` / `Retry` with JSON/ASCII/Mermaid graph export
- **Chat models**: `ChatModel` / `LLM` interfaces, streaming (`ChatModel.Stream`), rate limiting
- **Tools**: base tool, `FromFunc` reflection-based tool creation (the `@tool` equivalent), retriever-tool adapter
- **Prompts**: string, structured, and templated (`PromptTemplate`)
- **Output parsers**: all variants with format instructions
- **Vector stores**: in-memory with filtering, MMR, retriever adapters
- **Callbacks, tracers, document loaders, indexers, embeddings, retrievers, example selectors**

### 🔌 Partner Integrations

| Partner | Chat Model | Embeddings | Vector Store | Self-registered |
|---------|:---:|:---:|:---:|:---:|
| **OpenAI** | ✅ | ✅ | — | ✅ (`init()`) |
| **Anthropic** | ✅ | — | — | ✅ (`init()`) |
| **Ollama** | ✅ | ✅ | — | ✅ (`init()`) |
| **Chroma** | — | — | ✅ | — |

All three chat-model providers self-register via `init()`, so `WithAgentModel("openai:gpt-4o")` resolves end-to-end. Configure via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`).

---

## Installation

```bash
go get github.com/projanvil/langchain-golang
```

Requires **Go 1.23+**.

For checkpoint backends:

```bash
# SQLite (pure Go, no CGO)
go get github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite

# PostgreSQL
go get github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres

# Redis
go get github.com/projanvil/langchain-golang/langgraph/checkpoint/redis
```

---

## Quick Start

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
	// Use a fake model for this example; swap with a real provider for production:
	//   agents.WithAgentModel("openai:gpt-4o")
	model := language.NewFakeChatModel(
		language.WithResponses(messages.AI("It's sunny in Shanghai.")),
	)

	agent, err := agents.CreateAgent(model, nil,
		agents.WithAgentSystemPrompt("You are a helpful assistant."),
	)
	if err != nil {
		panic(err)
	}

	// Invoke:
	reply, _ := agent.Invoke(context.Background(), []messages.Message{
		messages.User("What's the weather?"),
	})
	fmt.Println(reply[len(reply)-1].Content)

	// Stream:
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

### With Tools

```go
getWeather := coretools.FromFunc("get_weather",
	"Get the weather for a city",
	func(ctx context.Context, city string) (string, error) {
		return "Sunny, 25°C", nil
	})

agent, _ := agents.CreateAgent(model, []language.Tool{getWeather},
	agents.WithAgentSystemPrompt("Use tools to answer questions."))
```

### With Checkpointing + Interrupts

```go
saver := checkpoint.NewMemorySaver()
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentCheckpointer(saver),
	agents.WithAgentInterruptBefore("tools"))

// First invoke pauses before tools:
result, _ := agent.Invoke(ctx, msgs, agents.Options{ThreadID: "t1"})
// result.Interrupts is non-empty

// Resume:
result, _ = agent.Invoke(ctx, nil, agents.Options{ThreadID: "t1"})

// Inspect state at any time:
state, _ := agent.Graph.GetState(ctx, graph.Options{ThreadID: "t1"})
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   agents.CreateAgent                   │
│         (model ↔ tools loop + middleware chain)        │
├──────────────────────────────────────────────────────┤
│                    langgraph/                          │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────┐  │
│  │ StateGraph│  │ checkpoint │  │   channels         │  │
│  │  builder  │→ │  (memory / │  │ LastValue/Topic/  │  │
│  │  + Pregel │  │  sqlite /  │  │ BinOp/Delta/      │  │
│  │  executor │  │  postgres) │  │ Overwrite/Barrier │  │
│  └──────────┘  └────────────┘  └───────────────────┘  │
├──────────────────────────────────────────────────────┤
│                     core/                             │
│  messages · runnables · language · tools · prompts    │
│  outputparser · callbacks · vectorstores · tracers    │
├──────────────────────────────────────────────────────┤
│              partners/ (openai, anthropic, ...)        │
└──────────────────────────────────────────────────────┘
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/usage/getting-started.md) | Install, configure a provider, run your first agent |
| [Runnable Composition (LCEL)](docs/usage/composition.md) | `Pipe` / `Parallel` / `Branch` / `Fallbacks` — the Go LCEL |
| [Agents — CreateAgent](docs/usage/agents.md) | System prompts, tools, middleware, structured output, interrupts |
| [Streaming](docs/usage/streaming.md) | Per-token model deltas + tool/node lifecycle events |
| [Graph Runtime](docs/usage/langgraph.md) | Stream modes, DeltaChannel, checkpoint serde, SQLite/Postgres savers |

API reference: [pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang)

---

## Repository Layout

```
langchain-golang/
├── core/                      # langchain_core port (30 packages)
├── langgraph/                 # langgraph port
│   ├── channels/              # LastValue, Topic, BinOp, Delta, Overwrite, Barrier, Ephemeral
│   ├── checkpoint/            # Saver interface + MemorySaver
│   │   ├── sqlite/            # nested module: pure-Go SQLite saver
│   │   ├── postgres/          # nested module: PostgreSQL saver (pgx/v5)
│   │   ├── savertest/         # shared conformance suite
│   │   └── serde/             # JSON serializer + type registry
│   ├── graph/                 # StateGraph builder + Pregel executor + checkpointSink
│   ├── runtime/               # Runtime[ContextT] (context + store + heartbeat + stream)
│   ├── store/                 # cross-thread BaseStore (semantic KV + InMemoryStore)
│   ├── fn/                    # functional API (@entrypoint / @task)
│   ├── prebuilt/              # ToolNode graph adapter
│   └── types/                 # Send, Command, Interrupt, Durability, Overwrite
├── langchain/                 # langchain (v1) port
│   ├── agents/                # CreateAgent + 17 middleware modules
│   │   └── middleware/        # context-editing, summarization, retry, PII, shell, ...
│   ├── chatmodels/            # provider registry (Resolve / RegisterProvider)
│   ├── tools/                 # ToolNode (concurrent dispatch)
│   └── messages/              # langchain-level message helpers
├── partners/                  # openai, anthropic, ollama, chroma
├── textsplitters/             # langchain_text_splitters port
├── standardtests/             # conformance suites
├── modelprofiles/             # model-profiles registry + CLI
├── cmd/langchain-profiles     # profiles refresh CLI
├── docs/                      # bilingual usage guides (EN + zh-CN)
└── integration/               # integration tests
```

---

## Python Alignment

This is a **faithful port** — every design decision defaults to "what Python does." The codebase tracks:

- **langchain-core** 1.4.9
- **langchain** v1 1.3.13
- **langgraph** 1.2.10
- **langgraph-checkpoint** 4.2.0

### Key design decisions

| Area | Go approach | Python equivalent |
|------|------------|-------------------|
| Node function signature | `func(rt runtime.Runtime, state map[string]any) (any, error)` | `def node(state, runtime):` |
| Context passing | `runtime.Runtime` (satisfies `context.Context`) | `RunnableConfig` configurable |
| Streaming | `iter.Seq2[StreamChunk, error]` (explicit types) | async generator |
| Checkpoint serde | `Serializer` interface + JSON type registry | `JsonPlusSerializer` |
| Durability modes | `checkpointSink` (single worker goroutine) | `PregelLoop` futures |
| Content blocks | Sealed interface + concrete types | Pydantic union |

### Not ported (by design)

- `langchain_classic` (legacy chains/agents/memory) — replaced by `CreateAgent`
- langgraph CLI / SDK / Server
- `defer=True` nodes (`NamedBarrierValueAfterFinish`)
- Dynamic provider import (`init_chat_model` dynamic chain)
- YAML / Jinja / Hub prompts (string + local JSON only)

---

## Testing

```bash
# Full suite with race detector
go test -race ./...

# Checkpoint backends
make test-sqlite       # pure-Go SQLite saver
make test-postgres     # embedded PostgreSQL saver
```

1382 tests across 62 packages, all passing with `-race`.

---

## Contributing

Contributions welcome — especially new partner integrations (Google Gemini, AWS Bedrock, Pinecone, etc.).

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Ensure `go build ./... && go vet ./... && go test -race ./...` passes
4. Match the existing code style and Python-parity conventions
5. Submit a pull request

### Conventions

- **Python is authoritative**: when in doubt, check what the Python source does
- **Trust `go build/vet/test`**, not editor diagnostics (gopls may show false positives)
- Every package should have compile-checked examples in `example_test.go`
- Bilingual docs: add both `guide.md` and `guide.zh-CN.md`

---

## Acknowledgments

This project is a Go port of [LangChain](https://github.com/langchain-ai/langchain) (MIT License, Copyright © LangChain, Inc.) and [LangGraph](https://github.com/langchain-ai/langgraph). All credit for the original design and abstractions belongs to the LangChain team.

## License

[MIT](LICENSE) — Copyright © 2026 ProjAnvil.
