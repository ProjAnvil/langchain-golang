# Google Gemini

**Languages:** [English](../gemini.md) | 简体中文

`partners/gemini` 是原生 Gemini 聊天模型——不是 OpenAI 兼容转接——构建在
官方 [`google.golang.org/genai`](https://pkg.go.dev/google.golang.org/genai)
SDK 之上。它实现完整的 `language.ChatModel` 接口面：invoke、batch、流式、
工具调用、工具选择，以及 provider 原生结构化输出。可运行示例：
[`examples/gemini-agent`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/gemini-agent)。

## 安装

```bash
go get github.com/projanvil/langchain-golang
```

## 配置

| 环境变量 | 用途 |
|---|---|
| `GEMINI_API_KEY` | 来自 [AI Studio](https://aistudio.google.com/apikey) 的 API key |
| `GOOGLE_API_KEY` | 作为回退被识别（SDK 优先级：两者都设时它生效） |

## 构造并直接使用

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

reply, err := model.Invoke(ctx, []messages.Message{messages.Human("你好！")})
```

## 注册表解析

该包经 `init()` 自注册 `"gemini"` provider 名，因此 blank import 后
`"provider:model"` 字符串形式即可端到端工作：

```go
import _ "github.com/projanvil/langchain-golang/partners/gemini"

model, err := chatmodels.Resolve(chatmodels.ChatModelSpec{
    Provider: "gemini",
    Model:    "gemini-2.5-flash",
})
```

## 带工具的智能体

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

流式（`model.Stream` 返回逐 token 增量）、`BindToolsWithOptions`（工具
选择）与 `WithStructuredOutput`（provider 原生 response schema）都与
openai/anthropic partner 同构。

## 切换点

- **模型**：`modelconfig.WithModel("gemini-2.5-pro")`——flash 便宜快，
  pro 适合更难的推理。
- **其他 provider**：把 `gemini.NewChatModel(...)` 换成
  `openai.NewChatModel(...)` / `anthropic.NewChatModel(...)`；下游全部面向
  `language.ChatModel` 接口。
- **端点**：`modelconfig.WithBaseURL(...)`（配合 `WithHeader` /
  `WithHTTPClient`）在需要时路由到代理或 Vertex 风格网关。
