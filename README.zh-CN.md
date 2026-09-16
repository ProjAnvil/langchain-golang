# langchain-golang

[![CI](https://github.com/ProjAnvil/langchain-golang/actions/workflows/ci.yml/badge.svg)](https://github.com/ProjAnvil/langchain-golang/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.26+-00ADD8?logo=go)](https://go.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-3111%20passing-brightgreen)]()
[![Packages](https://img.shields.io/badge/packages-65-blue)]()

**语言:** [English](README.md) | 简体中文

[LangChain](https://github.com/langchain-ai/langchain) 和 [LangGraph](https://github.com/langchain-ai/langgraph) 的社区 **Go 语言端口** —— 用纯 Go 构建生产级 LLM 智能体和应用。

> **未与 LangChain, Inc. 有关联或受其认可。** 预览质量；公开 API 在 `v1.0.0` 前可能变化。

## 最新动态

**v0.8.1** —— 容错与追踪隐私 parity：

- **节点级错误处理器**：`NodePolicies.ErrorHandler`（langgraph 1.2.0 `error_handler=`）——重试耗尽后恢复函数可读状态、收到类型化 `NodeError`，返回 update 或 `Command`；ERROR write 先行落盘，崩溃后 resume 重跑 handler 而非节点。
- **TracePolicy**：节点级与 middleware 级追踪载荷脱敏（langgraph 1.2.11 / langchain 1.3.15），在任何 tracer 观察到事件之前变换；processor panic 时 fail-closed。

**v0.8.0** —— 工程化加固版本：

- **CI**：GitHub Actions 覆盖主模块与全部 checkpoint 嵌套模块（go 1.26/1.27 矩阵）+ golangci-lint 门禁，存量 86 条 lint 告警清零。
- **`parallel_tool_calls`**：openai（Responses + Chat Completions 双路径）与 anthropic（`tool_choice.disable_parallel_tool_use`）现支持 `BindToolsOptions.ParallelToolCalls`。
- **社区文件**：[CHANGELOG](CHANGELOG.md)、[CONTRIBUTING](CONTRIBUTING.md)、[DIVERGENCES](DIVERGENCES.md)、[SECURITY](SECURITY.md)。

完整历史见 [CHANGELOG](CHANGELOG.md)。

---

## 为什么选择 langchain-golang？

| | langchain-golang | Python LangChain |
|---|---|---|
| **语言** | Go（静态类型，编译型） | Python |
| **并发** | Goroutine（原生） | asyncio / 线程 |
| **部署** | 单一二进制，无运行时依赖 | Python 解释器 + venv |
| **类型安全** | 编译期检查 | 运行时（Pydantic） |
| **Checkpoint 后端** | 内存、SQLite、PostgreSQL、Redis | 同上 + MongoDB |

## 功能特性

### 🤖 智能体框架 — `agents.CreateAgent`

Python `create_agent` 的 Go 等价物，构建在 `langgraph/` 运行时之上：

- **模型 ↔ 工具循环**，支持流式输出、中断和检查点
- **17 个中间件模块**：人机交互、上下文编辑（摘要）、模型/工具重试、回退、调用限制、PII 脱敏、Shell 执行、文件搜索、TODO 追踪、工具选择/模拟
- **结构化输出**：工具策略、Provider 原生策略、或自动检测
- **跨线程记忆** 通过 `store.Store`（带命名空间层级的语义键值存储）
- **模型响应缓存**，支持每节点 `CachePolicy`
- **中断 / 恢复** 往返，带版本化检查点历史和时间旅行

### 🔄 图运行时 — `langgraph/`

镜像 Python LangGraph 1.2.x 的 Pregel 风格状态图执行器：

- **StateGraph 构建器**，带类型化通道：`LastValue`、`Topic`、`BinaryOperator`、`Ephemeral`、`Barrier`、**`DeltaChannel`**（仅哨兵检查点存储 + 基于计数器的快照节拍）、**`Overwrite`** reducer
- **检查点**：内存、SQLite（`modernc.org/sqlite`，纯 Go）、PostgreSQL（`pgx/v5`）、Redis（`go-redis/v9`）—— 共享一致性测试套件
- **持久化模式**：`sync`（默认）、`async`（后台 goroutine 写入器）、`exit`（延迟刷新），通过 `checkpointSink`
- **流式 API**：`values` / `updates` / `debug` / `messages` / `custom` / `delta` 模式
- **函数式 API**：`@entrypoint` / `@task` 的等价物（`langgraph/fn`）
- **子图**、连接边（`AddJoinEdge`）、每节点重试/缓存策略
- **超时策略**：`run_timeout` / `idle_timeout`，带心跳刷新

### 🧱 核心抽象 — `core/`

31 个包，移植 `langchain_core`：

- **消息**：统一 `Message` 结构体、类型化内容块（文本/图片/音频/视频/文件/推理/引用/工具调用）、工具调用、裁剪、序列化
- **Runnable 组合（LCEL）**：`Pipe` / `Parallel` / `Branch` / `Fallbacks` / `Retry`，支持 JSON/ASCII/Mermaid 图导出
- **聊天模型**：`ChatModel` / `LLM` 接口、流式（`ChatModel.Stream`）、速率限制
- **工具**：基础工具、`FromFunc` 反射式工具创建（`@tool` 等价物）、检索工具适配器
- **提示词**：字符串、结构化、模板化（`PromptTemplate`）
- **输出解析器**：全变体，带格式指令
- **向量存储**：内存存储，带过滤、MMR、检索器适配器
- **回调、追踪器、文档加载器、索引器、嵌入、检索器、示例选择器**

### 🔌 Partner 集成

| Partner | 聊天模型 | 嵌入 | 向量存储 | 自注册 |
|---------|:---:|:---:|:---:|:---:|
| **OpenAI** | ✅ | ✅ | — | ✅ (`init()`) |
| **Anthropic** | ✅ | — | — | ✅ (`init()`) |
| **Ollama** | ✅ | ✅ | — | ✅ (`init()`) |
| **Groq / Mistral / DeepSeek / xAI / OpenRouter / Fireworks / Perplexity** | ✅ | — | — | ✅（`init()`，经 `partners/openaicompat`） |
| **Chroma** | — | — | ✅ | — |

全部十个聊天模型 provider 名字都经 `init()` 自注册（`partners/openai`、`partners/anthropic`、`partners/ollama`，加上 `partners/openaicompat` 里的七个 OpenAI 兼容名字），因此 blank import 后 `WithAgentModel("openai:gpt-4o")` 与 `WithAgentModel("groq:openai/gpt-oss-20b")` 均可端到端解析。通过环境变量配置（`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`OLLAMA_HOST`、`GROQ_API_KEY`、`DEEPSEEK_API_KEY` 等）。

---

## 安装

```bash
go get github.com/projanvil/langchain-golang
```

需要 **Go 1.26+**（v0.6.1 及更早版本：Go 1.23+）。

检查点后端：

```bash
# SQLite（纯 Go，无需 CGO）
go get github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite

# PostgreSQL
go get github.com/projanvil/langchain-golang/langgraph/checkpoint/postgres

# Redis
go get github.com/projanvil/langchain-golang/langgraph/checkpoint/redis
```

---

## 快速开始

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
	// 此示例使用 fake model；生产环境替换为真实 provider：
	//   agents.WithAgentModel("openai:gpt-4o")
	model := language.NewFakeChatModel(
		language.WithResponses(messages.AI("上海今天晴天。")),
	)

	agent, err := agents.CreateAgent(model, nil,
		agents.WithAgentSystemPrompt("你是一个有用的助手。"),
	)
	if err != nil {
		panic(err)
	}

	// 调用：
	reply, _ := agent.Invoke(context.Background(), []messages.Message{
		messages.User("天气怎么样？"),
	})
	fmt.Println(reply[len(reply)-1].Content)

	// 流式输出：
	stream, _ := agent.StreamEvents(context.Background(), []messages.Message{
		messages.User("讲个故事。"),
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

### 使用工具

```go
getWeather := coretools.FromFunc("get_weather",
	"获取城市天气",
	func(ctx context.Context, city string) (string, error) {
		return "晴天，25°C", nil
	})

agent, _ := agents.CreateAgent(model, []language.Tool{getWeather},
	agents.WithAgentSystemPrompt("使用工具回答问题。"))
```

### 使用检查点 + 中断

```go
saver := checkpoint.NewMemorySaver()
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentCheckpointer(saver),
	agents.WithAgentInterruptBefore("tools"))

// 首次调用在工具前暂停：
result, _ := agent.Invoke(ctx, msgs, agents.Options{ThreadID: "t1"})
// result.Interrupts 非空

// 恢复：
result, _ = agent.Invoke(ctx, nil, agents.Options{ThreadID: "t1"})

// 随时检查状态：
state, _ := agent.Graph.GetState(ctx, graph.Options{ThreadID: "t1"})
```

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│                   agents.CreateAgent                   │
│           （模型 ↔ 工具循环 + 中间件链）                │
├──────────────────────────────────────────────────────┤
│                    langgraph/                          │
│  ┌──────────┐  ┌────────────┐  ┌───────────────────┐  │
│  │ StateGraph│  │ checkpoint │  │   channels         │  │
│  │  构建器   │→ │ (内存/sql/ │  │ LastValue/Topic/  │  │
│  │ + Pregel  │  │  pg/redis) │  │ BinOp/Delta/      │  │
│  │  执行器   │  │            │  │ Overwrite/Barrier │  │
│  └──────────┘  └────────────┘  └───────────────────┘  │
├──────────────────────────────────────────────────────┤
│                     core/                             │
│  messages · runnables · language · tools · prompts    │
│  outputparser · callbacks · vectorstores · tracers    │
├──────────────────────────────────────────────────────┤
│           partners/ (openai, anthropic, ...)           │
└──────────────────────────────────────────────────────┘
```

---

## 文档

| 指南 | 描述 |
|------|------|
| [入门指南](docs/usage/getting-started.zh-CN.md) | 安装、配置 provider、运行你的第一个智能体 |
| [Runnable 组合（LCEL）](docs/usage/composition.zh-CN.md) | `Pipe` / `Parallel` / `Branch` / `Fallbacks` —— Go 版 LCEL |
| [智能体 — CreateAgent](docs/usage/agents.zh-CN.md) | 系统提示词、工具、中间件、结构化输出、中断 |
| [流式输出](docs/usage/streaming.zh-CN.md) | 逐 token 模型增量 + 工具/节点生命周期事件 |
| [图运行时](docs/usage/langgraph.zh-CN.md) | 流模式、DeltaChannel、checkpoint serde、SQLite/Postgres saver |

API 参考：[pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang)

---

## 仓库结构

```
langchain-golang/
├── core/                      # langchain_core 移植（31 个包）
├── langgraph/                 # langgraph 移植
│   ├── channels/              # LastValue, Topic, BinOp, Delta, Overwrite, Barrier, Ephemeral
│   ├── checkpoint/            # Saver 接口 + MemorySaver
│   │   ├── sqlite/            # 嵌套模块：纯 Go SQLite saver
│   │   ├── postgres/          # 嵌套模块：PostgreSQL saver (pgx/v5)
│   │   ├── redis/             # 嵌套模块：Redis saver (go-redis/v9)
│   │   ├── savertest/         # 共享一致性测试套件
│   │   └── serde/             # JSON 序列化器 + 类型注册表
│   ├── graph/                 # StateGraph 构建器 + Pregel 执行器 + checkpointSink
│   ├── runtime/               # Runtime[ContextT]（context + store + heartbeat + stream）
│   ├── store/                 # 跨线程 BaseStore（语义 KV + InMemoryStore）
│   ├── fn/                    # 函数式 API（@entrypoint / @task）
│   ├── prebuilt/              # ToolNode 图适配器
│   └── types/                 # Send, Command, Interrupt, Durability, Overwrite
├── langchain/                 # langchain (v1) 移植
│   ├── agents/                # CreateAgent + 17 个中间件模块
│   │   └── middleware/        # 上下文编辑、摘要、重试、PII、Shell...
│   ├── chatmodels/            # provider 注册表（Resolve / RegisterProvider）
│   ├── tools/                 # ToolNode（并发分发）
│   └── messages/              # langchain 层消息辅助函数
├── partners/                  # openai, anthropic, ollama, openaicompat（7 家 provider）, chroma
├── textsplitters/             # langchain_text_splitters 移植
├── standardtests/             # 一致性测试套件
├── modelprofiles/             # 模型配置注册表 + CLI
├── cmd/langchain-profiles     # profiles 刷新 CLI
├── docs/                      # 双语使用指南（EN + zh-CN）
└── integration/               # 集成测试
```

---

## Python 对齐

这是一个**忠实移植** —— 每个设计决策都以"Python 怎么做"为默认。代码库追踪：

- **langchain-core** 1.4.9
- **langchain** v1 1.3.13
- **langgraph** 1.2.10
- **langgraph-checkpoint** 4.2.0

### 关键设计决策

| 领域 | Go 方案 | Python 等价物 |
|------|--------|-------------|
| 节点函数签名 | `func(rt runtime.Runtime, state map[string]any) (any, error)` | `def node(state, runtime):` |
| Context 传递 | `runtime.Runtime`（满足 `context.Context`） | `RunnableConfig` configurable |
| 流式输出 | `iter.Seq2[StreamChunk, error]`（显式类型） | async generator |
| Checkpoint serde | `Serializer` 接口 + JSON 类型注册表 | `JsonPlusSerializer` |
| 持久化模式 | `checkpointSink`（单一 worker goroutine） | `PregelLoop` futures |
| 内容块 | 密封接口 + 具体类型 | Pydantic union |

### 未移植（设计决策）

- `langchain_classic`（遗留链/智能体/记忆）—— 被 `CreateAgent` 替代
- langgraph CLI / SDK / Server
- `defer=True` 节点（`NamedBarrierValueAfterFinish`）
- 运行时动态导入 provider —— `chatmodels.InitChatModel` / `ParseModelString` 可解析 `"provider:model"` 字符串，`Resolve` 从 `RegisterProvider` 注册表构建模型，但 partner 包必须在编译期链接（blank import）；Go 无法像 Python `init_chat_model` 那样按字符串运行时导入包
- YAML / Jinja 提示词（仅字符串 + 本地 JSON；`lc://` Hub 加载在上游同样已移除）

---

## 测试

```bash
# 全量测试 + 竞争检测器
go test -race ./...

# Checkpoint 后端（各为独立嵌套模块）
make test-sqlite       # 纯 Go SQLite saver
make test-redis        # Redis saver（miniredis，离线）
make test-postgres     # 嵌入式 PostgreSQL saver
```

65 个包共 3111 个测试，全部通过 `-race` 检测，由 [CI](.github/workflows/ci.yml)（go 1.26/1.27 矩阵 + golangci-lint）保障。

---

## 贡献

欢迎贡献 —— 特别是新的 Partner 集成（Google Gemini、AWS Bedrock、Pinecone 等）。

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feat/your-feature`）
3. 确保 `go build ./... && go vet ./... && go test -race ./...` 通过
4. 遵循现有代码风格和 Python 对齐约定
5. 提交 Pull Request

### 约定

- **Python 是权威**：不确定时查看 Python 源码的做法
- **信任 `go build/vet/test`**，不信编辑器诊断（gopls 可能显示误报）
- 每个包应有编译检查的示例在 `example_test.go` 中
- 双语文档：同时添加 `guide.md` 和 `guide.zh-CN.md`

---

## 致谢

本项目是 [LangChain](https://github.com/langchain-ai/langchain)（MIT 许可证，Copyright © LangChain, Inc.）和 [LangGraph](https://github.com/langchain-ai/langgraph) 的 Go 移植。原始设计和抽象的全部功劳归属于 LangChain 团队。

## 许可证

[MIT](LICENSE) — Copyright © 2026 ProjAnvil.
