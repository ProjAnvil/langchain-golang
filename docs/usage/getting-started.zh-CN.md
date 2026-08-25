# 快速上手

**Languages:** [English](getting-started.md) | 简体中文

本指南带你在几分钟内从 `go get` 走到一个可运行的 agent。内容对应 README
的 Quick start 一节，但逐一讲解每个部分。

## 安装

```bash
go get github.com/projanvil/langchain-golang@latest
```

要求 Go 1.26+（v0.6.1 及更早版本：Go 1.23+）。

## 你的第一个 agent

LangChain 中的 agent 是一个**模型 ↔ 工具循环**：模型决定要说什么或调用
哪个工具，工具运行，结果回到模型，循环往复，直到模型产出不再含工具调用
的最终答案。`agents.CreateAgent` 构建的就是这个循环。

这个例子使用仓库内置的 `language.FakeChatModel`，无需 API key 即可离线
运行：

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

### 刚才发生了什么

- `CreateAgent(model, tools, opts...)` 返回一个 `*Agent`，其 `.Graph` 是
  编译好的模型↔工具循环。`model` 是任意 `language.ChatModel`；`tools` 是
  `core/tools.Tool` 切片（这里为 `nil`，模型直接作答）。
- `agent.Invoke(ctx, messages)` 把循环跑到底，返回最终的消息历史。
- `WithAgentSystemPrompt` 与 `WithAgentName` 是函数式选项 —— 共有约 15
  个，覆盖 middleware、结构化输出、持久化、递归上限等等（见
  [agents 指南](agents.md)）。

## 使用真实模型

生产环境中把 `FakeChatModel` 换成 partner 的 `ChatModel`。有两种提供
方式：

### 1. 位置参数直接构造

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

`partners/openai` 被 import 时会自注册进 `chatmodels` provider 注册表
（其 `init()` 执行注册），因此一旦 import，下面的裸字符串形式对
`"openai:..."` 同样可用。

### 2. 从 `"provider:model"` 字符串解析

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

### 环境变量

各 partner 从环境变量读取凭据：

| Partner | 环境变量 |
|---------|----------|
| `partners/openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| `partners/anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (or `ANTHROPIC_API_URL`) |
| `partners/ollama` | `OLLAMA_HOST` (default `http://localhost:11434`) |

> 三个 partner 都会在 import 时自注册自动解析工厂，因此一旦 import 相应
> 包，`WithAgentModel("anthropic:...")` 与 `WithAgentModel("ollama:...")`
> 与 `"openai:..."` 一样端到端可用。

## 接下来看什么

- [组合 runnable（LCEL）](composition.md) —— 用 `Pipe` / `Pipe3-6` 串联
  prompt、模型与 parser，即 Python `|` 的 Go 等价物。
- [Agents —— `CreateAgent`](agents.md) —— 工具、middleware、结构化输出、
  interrupt 与 streaming。
- [图运行时（langgraph/）](langgraph.md) —— checkpoint、stream 模式、
  saver、join 边、函数式 API。
- [Streaming](streaming.md) —— 经 `StreamEvents` 的逐 token 模型增量。
