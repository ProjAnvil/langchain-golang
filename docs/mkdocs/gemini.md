# Google Gemini

**Languages:** English | [简体中文](zh-CN/gemini.zh-CN.md)

`partners/gemini` is a native Gemini chat model — not an OpenAI-compat
shim — built on the official
[`google.golang.org/genai`](https://pkg.go.dev/google.golang.org/genai) SDK.
It implements the full `language.ChatModel` surface: invoke, batch, stream,
tool calling, tool choice, and provider-native structured output. Runnable
example:
[`examples/gemini-agent`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/gemini-agent).

## Installation

```bash
go get github.com/projanvil/langchain-golang
```

## Configuration

| Env variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | API key from [AI Studio](https://aistudio.google.com/apikey) |
| `GOOGLE_API_KEY` | Honored as a fallback (SDK precedence: wins when both are set) |

## Construct and use directly

```go
import (
    "os"

    "github.com/projanvil/langchain-golang/core/messages"
    "github.com/projanvil/langchain-golang/core/modelconfig"
    "github.com/projanvil/langchain-golang/partners/gemini"
)

model := gemini.NewChatModel(
    modelconfig.WithAPIKey(os.Getenv("GEMINI_API_KEY")),
    modelconfig.WithModel("gemini-2.5-flash"),
)

reply, err := model.Invoke(ctx, []messages.Message{messages.Human("Hello!")})
```

## Registry resolution

The package self-registers the `"gemini"` provider name via `init()`, so
blank-import it and the `"provider:model"` string form works end to end:

```go
import _ "github.com/projanvil/langchain-golang/partners/gemini"

model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
    Provider: "gemini",
    Model:    "gemini-2.5-flash",
})
```

## Agent with tools

```go
wordcount, _ := coretools.NewSimple("count_words",
    "counts the words in the input text",
    func(_ context.Context, text string) (coretools.Result, error) {
        return coretools.Result{Content: fmt.Sprintf("%d words", len(strings.Fields(text)))}, nil
    })

agent, _ := agents.CreateAgent(model, []coretools.Tool{wordcount},
    agents.WithAgentSystemPrompt("You are a concise assistant."))
out, _ := agent.Invoke(ctx, []messages.Message{
    messages.Human("How many words are in 'the quick brown fox'?"),
})
```

Streaming (`model.Stream` returns token deltas), `BindToolsWithOptions`
(tool choice), and `WithStructuredOutput` (provider-native response schema)
all follow the same shapes as the openai/anthropic partners.

## Switching points

- **Model**: `modelconfig.WithModel("gemini-2.5-pro")` — flash for
  cost/latency, pro for harder reasoning.
- **Other providers**: swap `gemini.NewChatModel(...)` for
  `openai.NewChatModel(...)` / `anthropic.NewChatModel(...)`; everything
  downstream takes the `language.ChatModel` interface.
- **Endpoint**: `modelconfig.WithBaseURL(...)` (plus `WithHeader` /
  `WithHTTPClient`) routes to a proxy or Vertex-style gateway when needed.
