# langchain-golang documentation

**Languages:** English | [简体中文](README.zh-CN.md)

This directory holds the usage documentation for **langchain-golang**, a
community Go port of [LangChain](https://github.com/langchain-ai/langchain).

For the full API reference, see the package docs at
[pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang). For
supported features, scope, and known gaps, see the top-level
[README](../README.md).

## Usage guides

| Guide | What it covers |
|------|----------------|
| [Getting started](usage/getting-started.md) | Install, configure a provider, run your first agent |
| [Composing runnables (LCEL)](usage/composition.md) | `Pipe` / `Pipe3-6` / `Parallel` / `Branch` / `Fallbacks` / `Retry` — the Go equivalent of Python's `prompt \| model \| parser` |
| [Agents — `CreateAgent`](usage/agents.md) | System prompts, tools, middleware, structured output, interrupts, state/context schema |
| [Graph runtime (langgraph/)](usage/langgraph.md) | StateGraph, checkpoints, Stream modes, savers, join edges, functional API |
| [Streaming](usage/streaming.md) | `Agent.StreamEvents`: per-token model deltas + tool/node lifecycle events |
| [RAG stack](mkdocs/rag-stack.md) | Filter DSL, pgvector, Redis vector store, rerankers, advanced retrievers |
| [MCP tools](mkdocs/mcp.md) | MCP servers as agent tools, elicitation, HITL gating |
| [Google Gemini](mkdocs/gemini.md) | Native Gemini chat model via the official genai SDK |
| [SQL toolkit](mkdocs/sql-toolkit.md) | Read-only SQL agent toolkit over `database/sql` |
| [Fault tolerance & trace privacy](mkdocs/fault-tolerance.md) | Retry policies, node error handlers, TracePolicy scrubbing |
| [Document loaders](mkdocs/loaders.md) | HTML, web, and PDF loaders |

## Documentation site

The same guides (plus bilingual home pages) are built into an mkdocs-material
site from [`mkdocs/`](mkdocs/) — see [`mkdocs.yml`](../mkdocs.yml) at the
repository root. Preview locally with
`pip3 install --user -r requirements-docs.txt && python3 -m mkdocs serve`.

## Bilingual convention

Documentation in this repository is bilingual (English + Simplified Chinese):

1. The English file is the primary document; the Chinese translation lives as a
   sibling file named `<name>.zh-CN.md` in the same directory (site content
   under `mkdocs/` keeps its zh-CN versions in the `mkdocs/zh-CN/`
   subdirectory).
2. Both files carry a language-switch line (`English | 简体中文`) at the top,
   one blank line below the title, linking to each other.
3. New or changed documentation must be written in both languages in the same
   change — do not land an English-only update and translate it later.

## Examples in the repo

Runnable, compile-checked examples live alongside the code as
`example_test.go` files (`go test ./...` verifies their `// Output:` blocks):

- [agents example](../langchain/agents/example_test.go) — minimal `CreateAgent`, `provider:model` string resolution
- [chatmodels example](../langchain/chatmodels/example_test.go)
- [core/tools example](../core/tools/example_test.go)
- [core/language example](../core/language/example_test.go)

## Conventions

- All examples are offline by default — they use `language.FakeChatModel` so you
  can run them without an API key. Swap in a partner `ChatModel`
  (`partners/openai`, `partners/anthropic`, `partners/ollama`) for real usage.
- Go 1.26+ is required (v0.6.1 and earlier: Go 1.23+).
