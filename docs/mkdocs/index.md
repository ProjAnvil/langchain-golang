# langchain-golang

**Languages:** English | [简体中文](zh-CN/index.md)

A community **Go port** of [LangChain](https://github.com/langchain-ai/langchain)
and [LangGraph](https://github.com/langchain-ai/langgraph) — build
production-grade LLM agents and applications in pure Go.

> **Not affiliated with or endorsed by LangChain, Inc.** Preview quality; the
> public API may change before `v1.0.0`.

## Guides

| Guide | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | Install, configure a provider, run your first agent |
| [Runnable composition (LCEL)](composition.md) | `Pipe` / `Parallel` / `Branch` / `Fallbacks` — the Go LCEL |
| [Agents — `CreateAgent`](agents.md) | System prompts, tools, middleware, structured output, interrupts |
| [Streaming](streaming.md) | Per-token model deltas + tool/node lifecycle events |
| [Graph runtime](langgraph.md) | StateGraph, checkpoints, stream modes, savers, functional API |
| [RAG stack](rag-stack.md) | Filter DSL, pgvector, Redis vector store, rerankers, advanced retrievers |
| [MCP tools](mcp.md) | Model Context Protocol servers as agent tools, elicitation, HITL gating |
| [Google Gemini](gemini.md) | Native Gemini chat model via the official genai SDK |
| [SQL toolkit](sql-toolkit.md) | Read-only SQL agent toolkit over `database/sql` |
| [Fault tolerance & trace privacy](fault-tolerance.md) | Retry policies, node error handlers, TracePolicy scrubbing |
| [Document loaders](loaders.md) | HTML, web, and PDF loaders |

## Examples

Runnable, self-contained examples live in the repository under
[`examples/`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples):
quickstart, streaming, human-in-the-loop interrupts, subgraph resume, fault
tolerance, TracePolicy redaction, the full RAG chain (loader → split → embed →
pgvector → retrieve → rerank → agent), MCP tools, Gemini, SQL agent, advanced
retrievers, and a middleware suite. Each runs offline with a scripted or fake
model and switches to a real provider via environment variables.

## API reference

Package documentation is on
[pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang).
