# langchain-golang

**Languages:** [English](README.md) | 简体中文

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[LangChain](https://github.com/langchain-ai/langchain)——Python AI 应用框架——的社区 **Go 移植版**。用 LangChain 的抽象在 Go 中构建 LLM 智能体与 LLM 应用：聊天模型、工具、提示词、输出解析器、消息、向量存储、检索器，以及 `create_agent` 工厂。

> **与 LangChain, Inc. 无隶属关系，也未获其背书。** 预览质量（`v0.3.1`）；公开 API 在 `v1.0.0` 之前仍可能变动。

## 这是什么

以下是这些项目的 Go 移植：

- **`langchain_core`** → `core/` —— 基础抽象与接口。
- **`langchain`**（积极维护的 `langchain_v1` 包）→ `langchain/` —— 具体实现、智能体工厂、中间件、工具。
- **`langchain_text_splitters`** → `textsplitters/`。
- **`langchain-tests`** → `standardtests/` —— 共享契约测试套件。
- **`model-profiles`** → `modelprofiles/` + `cmd/langchain-profiles` CLI。

它**不是** `langchain_classic`（遗留包）的移植，也**不是** `langgraph` 的完整移植——已移植的图运行时（channel 对象、带历史的版本化 checkpoint、时间旅行、子图、流模式、checkpoint serde、join 边（`AddJoinEdge`）、Postgres checkpoint saver，以及函数式 API（`fn` 包））位于公开的顶层 `langgraph/` 包中（见[不支持](#-不支持--超出范围)与[图运行时指南](docs/usage/langgraph.md)）。

全部测试通过：`go test ./...` —— 52 个包，1250+ 个测试。

---

## ✅ 支持

### Core（`core/`）

`messages`（统一结构体、内容块、工具调用、裁剪、序列化）· `runnables`（组合、批量、流式、fallback、分支、路由、JSON/ASCII/Mermaid 图导出）· `language`（`ChatModel` / `LLM` 接口、假模型、**`ChatModel.Stream`**、限速器钩子）· `tools`（基础工具、渲染辅助、检索器-工具适配器）· `prompts`（字符串 + 结构化 + 模板化，本地 JSON 加载）· `outputparser`（全部解析器变体、格式指令、部分解析）· `callbacks`（管理器扇出、stdout/流式/文件处理器、用量聚合）· `streamevents`（v3 内容块协议、`ChatModelStream` 投影）· `documents` · `documentloaders` · `indexing`（含 SQL record manager）· `embeddings` · `vectorstores`（内存、过滤、MMR、检索器适配器）· `retrievers` · `exampleselectors` · `tracers`（context/root 监听器、内存、过滤、重放、事件流、stdout）· `load` · `stores` · `caches` · `ratelimiters` · `retry` · `_api`（废弃标记）· `_security`（SSRF 防护、传输校验）· `utils` · `chathistory` · `httpclient` · `modelconfig` · `outputs` · `structuredoutput` · `schema`。

### LangChain v1（`langchain/`）

- `chatmodels` / `embeddings` —— 提供者注册表、解析、init-spec 边界。
- `tools.ToolNode` —— 并发分发、未知工具错误、可配置的错误处理、`ToolCallWrapper`。
- `messages`、`ratelimiters`。

#### `agents.CreateAgent` —— Python `create_agent` 的 Go 等价物

构建在公开的 `langgraph/` 图运行时之上的模型 ↔ 工具智能体循环，具备：

- **中间件链** —— `WrapModelCall` / `WrapToolCall`（最外层优先），`BeforeModel` / `AfterModel` / `BeforeAgent` / `AfterAgent` 钩子，`jump_to` 短路约定，每个钩子都带 `context.Context`（用于中断）。
- **15 个中间件模块**：human-in-the-loop、model-call-limit、model-fallback、model/tool-retry、tool-call-limit、context-editing、file-search（ripgrep 快路径）、pii/redaction、provider-tool-search、shell、summarization、todo、tool-emulator、tool-selection。
- **`system_prompt`** —— 纯字符串**与**模板化 `PromptTemplate`（带每次调用的变量）。
- **`state_schema`** —— 通过 `StateField` + reducer 自定义图状态字段（`WithAgentStateFields`）。
- **`context_schema`** —— 基于 Go `context.Context` 的只读运行时上下文（`WithContextValues` / `ContextValue`）。
- **`response_format`** —— `ToolStrategy`、`ProviderStrategy`（当模型实现 `language.StructuredCaller` 时走提供者原生路径），以及 `AutoStrategy`（根据模型能力在两者之间自动选择）。
- **`store`** —— 跨线程 KV 存储，注入每次工具调用（`WithAgentStore`）。
- **`cache`** —— 接入模型调用路径的模型响应缓存（`WithAgentCache`）。
- **`interrupt_before` / `interrupt_after`** —— 在指定图节点处暂停（`WithAgentInterruptBefore` / `WithAgentInterruptAfter`）。
- **`model`** —— 传入构造好的 `language.ChatModel`，**或**一个裸 `provider:model` 字符串，经 `chatmodels.Resolve` 解析（`WithAgentModel("openai:gpt-4o")`）。
- **`tools`** —— 显式工具，**或**经 `core/tools.FromFunc` 反射为工具的 Go 可调用体（`@tool` 的等价物）。
- **`checkpointer`** —— 带版本化 checkpoint + 历史的内存 saver；**中断 / 恢复**往返，外加 `GetState` / `GetStateHistory` / `UpdateState`，以及在 `Agent.Graph` 上通过 `Options.CheckpointID` 做时间旅行。
- `recursion_limit`、`name`、`debug`。
- **流式** —— `Agent.StreamEvents`：基于 `runnables.Stream[StreamEvent]` 的真正逐 token 流式（模型增量 + 工具/节点生命周期事件）。
- **子智能体（agent-as-tool）** —— 一个智能体通过手工打造的工具委托给具名的内部智能体，工具体调用内部智能体的 `InvokeWithState`（对齐 Python 的 `@tool` + `agent.invoke()`）；嵌套运行可经 `NameFromContext` 按名称区分。非流式的嵌套调用不再把自己的事件泄漏进流式父运行的流中。见[子智能体指南](docs/usage/agents.md)。

### 文本切分、标准测试、模型档案

- `textsplitters/` —— 完整移植（character、HTML、Markdown、code、recursive、header；sentence-transformers / NLTK / spaCy / KoNLPy 适配器接口）。
- `standardtests/` —— 聊天模型 / embeddings / 检索器 / 向量存储 / runnable 契约测试套件。
- `modelprofiles/` —— 档案注册表、Markdown 摘要、`langchain-profiles refresh` CLI（合并 models.dev 数据 + TOML 覆盖 → `profiles.json`）。

### Partner 包

`partners/openai` · `partners/anthropic` · `partners/ollama`（聊天模型与 embeddings）· `partners/chroma`（向量存储）。`partners/openai` 是完整集成——其 `ChatModel`（Responses API：Invoke/Stream/工具调用）实现了 `language.StructuredCaller` 并自行注册进 `chatmodels.Resolve`，因此 `WithAgentModel("openai:gpt-4o")` 开箱即用。其余为可用的集成与验证辅助；为更多 partner 预留了适配器插槽。

---

## ❌ 不支持 / 超出范围

### 有意不移植

- **`langchain_classic`** —— 遗留的 chains、agents、memory、tools、retrievers、vectorstores、storage。经典的 `AgentExecutor` 已不复存在；请使用 `agents.CreateAgent`。
- **完整的 `langgraph` 移植**。已移植的部分位于公开的顶层 `langgraph/` 包（`langgraph/{types,channels,checkpoint,graph,prebuilt,fn}`），覆盖 `create_agent` 运行时以及：channel 对象（经 `AddChannel` 的 `NewLastValue` / `NewTopic` / `NewBinaryOperator` / `NewEphemeral`）、带历史的版本化 checkpoint（`GetState` / `GetStateHistory` / `UpdateState`）、时间旅行（`Options.CheckpointID`）、子图（`AddSubgraph`，面向父图的 `Command{Graph: types.ParentGraph}`）、具有 Python 对等流模式的 `CompiledGraph.Stream` API（基于 `iter.Seq2[StreamChunk, error]` 的 `values` / `updates` / `debug` / `messages` / `custom`）、checkpoint serde（`checkpoint.Serializer` + `checkpoint/serde` 的带封闭类型注册表的 JSON 序列化器）、逐节点 retry/cache 策略（经 `AddNodeWithPolicies` 的 `RetryPolicy` / `CachePolicy`，配合 `WithCache` + `checkpoint.NewInMemoryCache`）、多父 barrier join 边（`AddJoinEdge`，Python `add_edge((a, b), c)` waiting edge 的等价物）、函数式 API（`langgraph/fn`：`NewEntrypoint` / `NewEntrypointFinal` / `NewTask`，`@entrypoint`/`@task` 的等价物）、Saver 接口扩展（`ListOptions.Filter` 元数据过滤、`PutWrites` 任务路径，以及共享的 `langgraph/checkpoint/savertest` 契约测试套件），以及 `prebuilt.ToolNode`（`langchain/tools.ToolNode` 的图节点适配器）；`langchain/internal/agentruntime/` 仅作为废弃的别名垫片保留。有意缺席：图级默认重试策略、langgraph CLI/SDK、`defer=True` 节点（`NamedBarrierValueAfterFinish`）、Shallow saver 变体、delta channel history 快路径，以及函数式 API 的 `store` 参数（Python `@entrypoint(checkpointer=..., store=...)` 的跨线程 BaseStore 未移植）。SQLite checkpoint 后端**确实存在**，为嵌套 module `langgraph/checkpoint/sqlite`（自带 `go.mod`；本移植的第一个第三方依赖 `modernc.org/sqlite`；经 `make test-sqlite` 测试）。持久化的 Postgres checkpoint 后端同样**确实存在**，为嵌套 module `langgraph/checkpoint/postgres`（自带 `go.mod`；本移植的第二个第三方依赖 `pgx/v5`；显式调用 `Setup` 执行 migrations；经 `make test-postgres` 测试，它会拉起一个内嵌 Postgres 并运行共享的 `savertest` 契约套件）。见[图运行时指南](docs/usage/langgraph.md)。
- **子智能体 transformer（`transformers` / `run.subagents`）** —— 未暴露。`transformers` 是基于 Python 可变形流输出的 langgraph 流模式构件，Go 的 `Stream` API 有意以显式类型的 `StreamChunk` 取而代之。其动机性功能——**PII 流式增量脱敏**——**已交付**，通过有界中间件增量层（`WrapModelStreamHook` + `PIIStreamTransformer` 的回看缓冲）实现；批量脱敏同样可用。

### Partner 覆盖有限

- 仅有 `openai`、`anthropic`、`ollama`、`chroma`。**没有 Google/Gemini、AWS、Azure、Pinecone 等**——欢迎社区贡献。
- `langchain/chatmodels` 把模型名解析为 `ChatModelSpec`，**并**经 Go 提供者注册表（`Resolve` + `RegisterProvider`）解析为构造好的 partner `ChatModel`；`WithAgentModel("openai:gpt-4o")` 端到端可用。（anthropic/ollama/chroma 尚未注册为真正的 Go 工厂——这些请传入构造好的 `language.ChatModel`。）
- `langchain/tools.ToolNode` **不**支持工具返回 `Send`，也不支持基于反射的 `InjectedState` / `InjectedStore` / `ToolRuntime` 参数注入。工具**可以**通过在 `Result.Artifact` 中放置 `*types.Command` 来发出图控制流信号（由 `ToolNode.InvokeToolCallsFull` 浮现）；`langgraph/prebuilt.ToolNode` 把该约定应用到图状态——见[图运行时指南](docs/usage/langgraph.md)。

### 其他缺口

- `core/prompts` 不加载 YAML、Jinja 模板或 `lc://` Hub 提示词（仅字符串 + 本地 JSON）。
- `core/runnables` 不支持 PNG 图渲染（JSON/ASCII/Mermaid 支持）。
- 不支持 Python 风格的动态提供者导入 / 实例构造——请在 Go 中构造具体模型。
- **文件工具**（`Read`/`Write`/`Edit`/`Bash`）与**沙箱**超出范围——它们由 [`claude-agent-sdk-golang`](https://github.com/ProjAnvil/claude-agent-sdk-golang) 提供，不属于 LangChain。

上面的支持 / 缺口清单是权威的兼容性参考。如需了解某个具体缺口的细节，请开 issue。

---

## 安装

```bash
go get github.com/projanvil/langchain-golang@v0.3.1
```

需要 Go 1.23+。

## 快速开始

一个使用仓库内假模型的最小可运行示例（生产环境请换用 partner `ChatModel`）：

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

要使用真实模型，可以从 partner 包（如 `partners/openai`、`partners/anthropic`、`partners/ollama`）构造一个 `language.ChatModel` 并按位置参数传入，**或**从一个裸名称字符串解析：`agents.CreateAgent(nil, nil, agents.WithAgentModel("openai:gpt-4o"))`（通过 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 环境变量配置）。

## 文档

使用指南位于 [`docs/`](docs/) —— 示例驱动，每个片段都可离线运行（除非注明，均使用仓库内假模型）：

- [快速上手](docs/usage/getting-started.md) —— 安装、配置提供者、运行你的第一个智能体。
- [组合 runnable（LCEL）](docs/usage/composition.md) —— `Pipe` / `Pipe3-6` / `Parallel` / `Branch` / `Fallbacks` / `Retry`，Python `prompt | model | parser` 的 Go 等价物。
- [智能体 —— `CreateAgent`](docs/usage/agents.md) —— 系统提示词、工具、15 模块中间件链、结构化输出、中断。
- [流式](docs/usage/streaming.md) —— `Agent.StreamEvents`：逐 token 模型增量 + 工具/节点生命周期事件。
- [图运行时（`langgraph/`）](docs/usage/langgraph.md) —— `CompiledGraph.Stream` 流模式、checkpoint serde、SQLite 与 Postgres checkpoint saver、逐节点 retry/cache 策略、join 边（`AddJoinEdge`）、函数式 API（`langgraph/fn`）、`prebuilt.ToolNode`。

完整 API 参考见 [pkg.go.dev](https://pkg.go.dev/github.com/projanvil/langchain-golang) 的包文档。经编译检查的示例也存放在各包的 `example_test.go` 中。

## 仓库布局

```
langchain-golang/
├── core/                  # langchain_core port
├── langgraph/             # langgraph port: StateGraph builder, Pregel executor, channel objects, versioned checkpoints + history, time travel, subgraphs, Stream API (values/updates/debug/messages/custom), checkpoint serde, per-node retry/cache policies, join edges (AddJoinEdge), functional API (fn), Postgres checkpoint saver, prebuilt.ToolNode
│   ├── checkpoint/sqlite/ # nested Go module: SQLite checkpoint saver (modernc.org/sqlite, pure Go); test it with `make test-sqlite`
│   ├── checkpoint/postgres/ # nested Go module: Postgres checkpoint saver (pgx/v5); test it with make test-postgres
│   ├── checkpoint/savertest/ # shared checkpoint.Saver conformance suite (mirrors the standardtests/ philosophy)
│   └── fn/                # functional API: NewEntrypoint / NewTask (Python @entrypoint/@task)
├── langchain/             # langchain (v1) port
│   ├── agents/            # CreateAgent + 15 middleware
│   ├── chatmodels/ embeddings/ messages/ tools/ ratelimiters/
│   └── internal/agentruntime/   # deprecated alias shim over langgraph/
├── textsplitters/         # langchain_text_splitters port
├── standardtests/         # langchain-tests conformance port
├── modelprofiles/         # model-profiles port
├── partners/              # openai, anthropic, ollama, chroma
└── cmd/langchain-profiles # profiles refresh CLI
```

## 致谢

本项目是 [LangChain](https://github.com/langchain-ai/langchain)（MIT License，Copyright © LangChain, Inc.）与 [LangGraph](https://github.com/langchain-ai/langgraph) 的 Go 移植。原始设计与抽象的全部荣誉属于 LangChain 团队。

## 许可证

[MIT](LICENSE) —— Copyright © 2026 ProjAnvil。
