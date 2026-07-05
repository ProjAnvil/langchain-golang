# langchain-golang documentation

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
| [Streaming](usage/streaming.md) | `Agent.StreamEvents`: per-token model deltas + tool/node lifecycle events |

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
- Go 1.23+ is required.
