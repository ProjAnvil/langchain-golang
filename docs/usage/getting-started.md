# Getting started

This guide takes you from `go get` to a running agent in a few minutes. It
mirrors the README's Quick start section but explains each piece.

## Install

```bash
go get github.com/projanvil/langchain-golang@latest
```

Requires Go 1.23+.

## Your first agent

An agent in LangChain is a **model ↔ tools loop**: the model decides what to
say or which tool to call, the tool runs, the result goes back to the model,
and the loop repeats until the model produces a final answer with no tool
calls. `agents.CreateAgent` builds that loop.

This example uses the in-tree `language.FakeChatModel` so it runs offline with
no API key:

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
		agents.WithAgentName("weather-agent"),
	)
	if err != nil {
		panic(err)
	}

	reply, err := agent.Invoke(context.Background(), []messages.Message{
		messages.User("What's the weather?"),
	})
	if err != nil {
		panic(err)
	}
	// reply is the full message history; the last entry is the assistant answer.
	fmt.Println(reply[len(reply)-1].Content)
	// Output: It's sunny in Shanghai.
}
```

### What just happened

- `CreateAgent(model, tools, opts...)` returns an `*Agent` whose `.Graph` is a
  compiled model↔tools loop. `model` is any `language.ChatModel`; `tools` is a
  slice of `core/tools.Tool` (here `nil`, so the model just answers).
- `agent.Invoke(ctx, messages)` runs the loop to completion and returns the
  final message history.
- `WithAgentSystemPrompt` and `WithAgentName` are functional options — there are
  ~15 of them covering middleware, structured output, persistence, recursion
  limits, and more (see the [agents guide](agents.md)).

## Using a real model

For production, swap the `FakeChatModel` for a partner `ChatModel`. You have
two ways to supply it:

### 1. Construct it positionally

```go
import (
	"context"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/partners/openai"
)

func realAgent() {
	model := openai.NewChatModel(/* model="gpt-4o", settings... */)

	agent, _ := agents.CreateAgent(model, nil,
		agents.WithAgentSystemPrompt("You are a helpful assistant."),
	)
	// ... agent.Invoke(...)
}
```

`partners/openai` self-registers into the `chatmodels` provider registry when
imported (its `init()` runs the registration), so once imported the bare-string
form below also works for `"openai:..."`.

### 2. Resolve from a `"provider:model"` string

```go
import (
	_ "github.com/projanvil/langchain-golang/partners/openai" // register the "openai" factory
	"github.com/projanvil/langchain-golang/langchain/agents"
)

func stringAgent() {
	// model is nil positionally; WithAgentModel resolves "openai:gpt-4o".
	agent, _ := agents.CreateAgent(nil, nil,
		agents.WithAgentModel("openai:gpt-4o"),
		agents.WithAgentSystemPrompt("You are a helpful assistant."),
	)
	_ = agent
}
```

### Environment variables

Each partner reads its credentials from the environment:

| Partner | Env vars |
|---------|----------|
| `partners/openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| `partners/anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (or `ANTHROPIC_API_URL`) |
| `partners/ollama` | `OLLAMA_HOST` (default `http://localhost:11434`) |

> `anthropic` and `ollama` are integrated but not yet registered as
> auto-resolving factories — for those, construct the `ChatModel` positionally
> (form 1). `openai` resolves end-to-end via `WithAgentModel("openai:...")`.

## Where to go next

- [Composing runnables (LCEL)](composition.md) — chain prompts, models, and
  parsers with `Pipe` / `Pipe3-6`, the Go equivalent of Python's `|`.
- [Agents — `CreateAgent`](agents.md) — tools, middleware, structured output,
  interrupts, and streaming.
- [Streaming](streaming.md) — per-token model deltas via `StreamEvents`.
