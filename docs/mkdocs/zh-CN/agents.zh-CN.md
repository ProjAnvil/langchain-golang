# Agents —— `CreateAgent`

**Languages:** [English](../agents.md) | 简体中文

`agents.CreateAgent` 是 Python `langchain.agents.create_agent` 的 Go 等价物。
它在公开的 `langgraph/` 图运行时之上构建模型↔工具循环，并在每次模型调用
与每次工具调用外围挂上可组合的 middleware 钩子。

## 签名

```go
func CreateAgent(
	model language.ChatModel,
	toolList []coretools.Tool,
	opts ...AgentOption,
) (*Agent, error)
```

- 当 `WithAgentModel` 提供 `"provider:model"` 字符串时，`model` 可为 `nil`。
- 纯对话 agent 的 `toolList` 可为 `nil`。
- 返回的 `*Agent` 暴露 `Invoke`、`InvokeWithState` 与 `StreamEvents`。

对范围内参数集，Go 移植与 Python `create_agent` 达到了完全的参数对齐
（17 个中的 16 个）。唯一未移植的参数是 `transformers`（逐 callable 的输出
变换，例如流式 PII 脱敏）；等价的流式脱敏改由 `WrapModelStreamHook`
middleware 提供。

## System prompt

纯字符串：

```go
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentSystemPrompt("You are a helpful assistant."),
)
```

模板 —— 每次模型调用时经 `core/prompts` 渲染，变量由构建期默认值与每次
`Invoke` 的覆盖值合并：

```go
tmpl, _ := prompts.NewPromptTemplate("You are a {{.role}}.")
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentSystemPromptTemplate(tmpl, map[string]any{"role": "analyst"}),
)
```

## 工具

用 `core/tools` 定义工具。最简单的是为类型化函数使用 `NewSimple`：

```go
echo, _ := coretools.NewSimple("echo", "echo its input",
	func(ctx context.Context, input string) (coretools.Result, error) {
		return coretools.Result{Content: "echo:" + input}, nil
	})
```

`FromFunc` 把任意 Go 函数反射成工具 —— 即 `@tool` 的等价物。函数的参数
struct 定义 JSON schema：

```go
search, _ := coretools.FromFunc("search", "search the web",
	func(args struct {
		Query string `json:"query"`
	}) (string, error) {
		return runSearch(args.Query), nil
	})
```

把它们传给 `CreateAgent`；由模型决定何时调用：

```go
agent, _ := agents.CreateAgent(model, []coretools.Tool{echo, search})
```

工具也可以写图状态（或跳转）：在工具的 `Result.Artifact` 中放置
`*types.Command`（它实现了 `messages.ToolOutput`）。内置 tools 节点会消费
这些命令 —— 各命令的 update 合并进节点的状态更新（`messages` 拼接，其他键
按后写覆盖），`Goto` 目标拼接进路由；`Command.Resume` 或非空的
`Command.Graph` 会使运行失败。`TodoListMiddleware` 的 `write_todos` 工具
正是借此把 todos 落进图状态。

## Middleware

Middleware 包装模型调用与每次工具调用。用 `WithAgentMiddleware` 组合：

```go
import "github.com/projanvil/langchain-golang/langchain/agents/middleware"

agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentMiddleware(
		middleware.NewModelFallbackMiddleware(model, fallbackModel), // switch model on error
		middleware.NewModelRetryMiddleware(),                        // retry transient failures
	),
)
```

仓库内置十五个 middleware 模块：

| Middleware | 用途 |
|------------|---------|
| `NewModelFallbackMiddleware` | 出错时回退到备选模型 |
| `NewModelRetryMiddleware` | 带退避地重试模型调用 |
| `NewSummarizationMiddleware` | 压缩过长的对话历史 |
| `NewModelCallLimitMiddleware` | 限制每次运行 / 每个线程的模型调用数 |
| `NewToolCallLimitMiddleware` | 限制对特定工具的调用数 |
| `NewToolRetryMiddleware` | 重试失败的工具调用 |
| `NewInterruptHumanInTheLoopMiddleware` / `NewHumanInTheLoopMiddleware` | 暂停等待人工批准（基于 interrupt / 同步 `Decide` 回调） |
| `NewPIIMiddleware` / `NewPIIStreamTransformer` | PII 脱敏（批量与流式） |
| `NewContextEditingMiddleware` | 改写模型调用的上下文 |
| `NewFilesystemFileSearchMiddleware` | 基于 ripgrep 的文件搜索工具 |
| `NewProviderToolSearchMiddleware` | provider 侧工具搜索 |
| `NewShellToolMiddleware` | 持久 shell 会话工具 |
| `NewTodoListMiddleware` | 任务跟踪工具 |
| `NewLLMToolEmulator` | 用 LLM 模拟工具调用 |
| `NewLLMToolSelectorMiddleware` | 由 LLM 选取工具子集 |

钩子执行顺序（从最外层开始）：

```
BeforeAgent → BeforeModel → WrapModelCall → WrapToolCall → AfterModel → AfterAgent
```

每个钩子都收到一个 `context.Context`，因此任何一个都可以调用
`graphpkg.Interrupt` 暂停运行、等待外部输入（见下文 *Human-in-the-loop*）。
钩子还可以通过把 `update["jump_to"]` 设为 `"model"`、`"tools"` 或 `"end"`
来短路路由：对 `before_agent` 而言，该跳转决定运行是否进入 model↔tools
循环（`"model"`/`"tools"`），还是直接经 `after_agent` 走向结束（`"end"`）；
对返回 update 的 `after_agent` 钩子，`"model"` 会重新进入循环，而
`"end"`（或不跳转）完成本次运行。

Middleware 贡献的工具与状态字段会被自动收集：实现了
`middleware.ToolProvider`（`ProvidedTools`）或
`middleware.StateSchemaContributor`（`StateSchema`）的 middleware 会把其
工具与状态键注册进 agent —— 无需手工 append 工具列表，也无需在
`WithAgentStateFields` 里重复声明。两个同名 middleware（经
`MiddlewareNamer` 显式给出 `Name()`，或同为某个 Go 类型）会在构建期被
拒绝，对齐 Python 的重名 middleware 检查。

`wrap_model_call` 钩子可以返回 `middleware.ExtendedModelResponse`（实现
`WrapModelCallResultHook`）：其中内嵌 `Command` 的 `Update` 会被叠加应用到
模型节点自身的状态更新之上（冲突时 middleware 的键胜出），而携带
`Goto` / `Resume` / `Graph` 的 `Command` 会使运行失败 —— 路由控制仍归上面的
`jump_to` 约定。

## 结构化输出

通过 `WithAgentResponseFormat` 把最终响应约束到某个 schema。有三种策略：

```go
import "github.com/projanvil/langchain-golang/core/schema"

sentimentSchema := schema.Object(map[string]schema.Schema{
	"sentiment": schema.Schema{"type": "string", "enum": []any{"pos", "neg", "neu"}},
	"score":     schema.Integer("sentiment score 0-100"),
}, "sentiment", "score")

toolStrategy := agents.NewToolStrategy(sentimentSchema)
//   → schema bound as a callable tool; a matching tool call ends the run.

providerStrategy := agents.NewProviderStrategy(sentimentSchema)
//   → ask the provider for native structured output (best-effort).

autoStrategy := agents.NewAutoStrategy(sentimentSchema)
//   → resolved at build time from the model's capabilities (ToolStrategy when
//     the model supports tool calling, else ProviderStrategy).

agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentResponseFormat(autoStrategy),
)

state, _ := agent.InvokeWithState(ctx, msgs)
result := state["structured_response"] // parsed per the schema
```

> `ProviderStrategy` 的 provider 原生 model-kwargs 绑定是尽力而为的：模型
> 被单独配置（或提示）输出匹配的 JSON，然后最终文本响应按 schema 解析。

Middleware 可以在 `wrap_model_call` 内部经 `request.Override(...)` 调整
绑定。`middleware.WithResponseFormat` —— 包括 ToolStrategy↔ProviderStrategy
切换（ToolStrategy 覆盖只能收窄到 agent 原始 response format 中声明的结构化
工具）—— `middleware.WithToolChoice` 与 `middleware.WithModelSettings` 按每次
模型调用生效（覆盖型 middleware 每执行一次就生效一次），流式也包括在内；
`AutoStrategy` 会在每次调用时针对当前模型重新解析，因此 `DynamicModel` 换模
后会重新检查。

## Human-in-the-loop（interrupt 模式）

`middleware.NewInterruptHumanInTheLoopMiddleware` 会在模型产出已登记审查的
工具调用时暂停运行 —— 对齐 Python 基于 interrupt 的
`HumanInTheLoopMiddleware`。该暂停是真正的 langgraph interrupt：运行停下，
待处理的调用以 `Result.Interrupts` 浮现，随后的 resume 把人的应答带回。

两个前置条件决定暂停是否可恢复：checkpointer（`WithAgentCheckpointer`）
与每次运行的 `graphpkg.Options.ThreadID`。缺了它们 interrupt 仍会触发，但
没有可恢复的线程。

```go
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(
		map[string]middleware.InterruptConfig{
			"transfer_funds": {
				AllowedDecisions: []middleware.DecisionType{
					middleware.DecisionApprove, middleware.DecisionEdit,
					middleware.DecisionReject, middleware.DecisionRespond,
				},
				// 可选：When、Description / DescriptionFunc、ArgsSchema
				// 进一步筛选哪些调用被审查、审查者看到什么。
			},
		},
	)),
	agents.WithAgentCheckpointer(checkpoint.NewMemorySaver()),
)

// 首次运行：模型请求被审查的工具后暂停。暂停在这里不是错误 ——
// 返回已提交的状态加上 interrupts。
values, interrupts, err := agent.InvokeWithStateOptions(ctx, msgs,
	graphpkg.Options{ThreadID: "thread-1"})

request, _ := middleware.HITLRequestFromInterrupt(interrupts[0])
// request.ActionRequests —— 待处理的调用（name / args / description）
// request.ReviewConfigs —— 各工具的审查策略（允许的决策、schema）

// 应答暂停并在同一线程上继续：
values, _, err = agent.Resume(ctx, graphpkg.Options{
	ThreadID: "thread-1",
	Resume: middleware.HITLResponse{Decisions: []middleware.Decision{{
		Type: middleware.DecisionApprove,
	}}},
})
```

审查运行在专属的 `"hitl"` 图节点中，布线在模型节点与 tools 节点之间
（Python 把每个 `after_model` middleware 作为独立节点，同理）：暂停发生时
模型的 AI 消息**已提交**，因此 resume 只重跑审查节点 —— 被暂停的那次调用
绝不会重新调用模型，决策作用于已提交、稳定的 AIMessage。修订后的消息按
消息 ID 原位替换已提交的那条（无 ID 的 AI 消息会为此铸出一个确定性 ID），
因此 reject/respond 不会留下重复副本。一个 agent 至多支持一个 interrupt
模式的 HITL middleware（`"hitl"` 节点是共享的）；它可以与 Decide 模式的
HITL（见下）组合，后者内联运行。

`HITLResponse` 按顺序为每个被审查的工具调用携带一个 `Decision`。要求决策
数与被审查调用数严格相等，且每个 `Type` 必须在该工具的
`AllowedDecisions` 内 —— 否则 resume 报错。四个分支：

- **approve** —— 原样执行该调用。
- **edit** —— 经 `Decision.EditedAction`（`*middleware.ToolCall`）替换
  name/args；调用 ID 保留，对话缝合不破坏。
- **reject** —— 不执行；`Decision.Message` 成为应答该调用的错误
  `ToolMessage`，模型能看到并作出反应。
- **respond** —— 不执行；`Decision.Message` 代表人直接作为工具的成功应答。

不带值地 resume（`Options.Resume` 留 nil）会重跑审查节点并以相同 ID 再次
抛出同一 interrupt —— 即再暂停，适合轮询式 UI。resume 值也可以是其 JSON
wire 形态（`map[string]any{"decisions": [...]}`，键为 `"type"` /
`"edited_action"` / `"message"`）—— 即 checkpoint 经 JSON saver 往返或非
Go 生产者留下的形态；`middleware.DecodeHITLResponse` 两种都接受。

当有多个 interrupt 待处理时，`Options.Resume` 是按 interrupt NS（优先）或
ID（其次）寻址的 `map[string]any`。NS 也是嵌套保持可寻址的关键：

```go
// supervisor 把 worker agent 作为子图嵌入。worker 自身不带
// checkpointer：它共享父运行的那个。
worker, _ := agents.CreateAgent(workerModel, workerTools,
	agents.WithAgentMiddleware(middleware.NewInterruptHumanInTheLoopMiddleware(...)))

parent := graphpkg.NewStateGraph()
parent.AddSubgraph("worker", worker.Graph)
parent.AddEdge(types.START, "worker")
parent.AddEdge("worker", types.END)
supervisor, _ := parent.Compile(graphpkg.WithCheckpointer(checkpoint.NewMemorySaver()))

res, _ := supervisor.InvokeWithOptions(ctx,
	map[string]any{"messages": msgs},
	graphpkg.Options{ThreadID: "sup-1"})
// worker 的审查会暂停父运行；res.Interrupts[0].NS 带嵌套前缀
// "worker:<task>/hitl:<task>"。

supervisor.InvokeWithOptions(ctx, nil, graphpkg.Options{
	ThreadID: "sup-1",
	Resume: map[string]any{res.Interrupts[0].NS: middleware.HITLResponse{
		Decisions: []middleware.Decision{{Type: middleware.DecisionApprove}},
	}},
})
```

`Options.Graph` 把一次 resume 圈定到某个命名空间：设为子图任务的命名空间
（interrupt NS 去掉末尾 `/hitl:<task>` 段），标量 `Resume` 就只喂给该子运行；
`Graph` 未命中任何待处理 interrupt 的 NS 时会给出描述性错误。

### HITL 的限制与边界

- **结构化输出不被审查。** `ToolStrategy` 下，结构化输出工具调用在模型
  节点内就结束运行 —— 先于审查顺序 —— 因此不会为 HITL 暂停（Python
  顺序相同）。
- **流式把暂停表现为错误。** `StreamEvents`（以及普通的
  `Invoke`/`InvokeWithState`）把被中断的运行视为终止性错误。这类线程用
  `InvokeWithStateOptions`/`Resume`（非流式）恢复；感知暂停的流式属于
  后续工作。
- **`jump_to: "tools"` 绕过审查。** 钩子直接跳到 tools 节点时是按名直达
  的；插入审查的重映射只作用于模型节点的常规路由（对齐 Python 的
  `jump_to`）。
- **工具内嵌套 agent 无法暂停父运行。** 工具函数体里调用另一个 agent 的
  `InvokeWithState` 是一次独立的嵌套运行，其 interrupt 不会上浮。需要可
  暂停的 supervisor/worker 组合时，用 `StateGraph.AddSubgraph` 嵌入
  worker 并让父运行带 checkpointer，如上例。

### 同步 Decide 模式（兼容）

`middleware.NewHumanInTheLoopMiddleware(interruptOn, decide)` 是较早的 Go
形态：它不暂停，而是在模型节点内（`AfterModel`）内联调用 `decide` 回调，
传入同一个 `HITLRequest`，并以完全一致的四分支逻辑应用返回的
`[]Decision`。它不需要 checkpointer，但人必须在调用内部应答 —— 没有可
恢复的暂停。两种模式在每个 middleware 实例上互斥（`Decide` 为 nil 即选定
interrupt 模式），并可作为独立 middleware 共存于一个 agent；内联的
`AfterModel` 钩子恒先于专属 hitl 节点运行。

## Interrupt 边界与 checkpoint 历史

与 HITL 正交地，用 `WithAgentInterruptBefore` / `WithAgentInterruptAfter`
在具名节点处暂停运行，然后用相同的 `ThreadID` 经 `Agent.Resume`（或
`Agent.Graph.InvokeWithOptions`）恢复。这需要一个 checkpointer。边界
interrupt 以 nil 的 `Options.Resume` 恢复；它们与 hitl 节点可组合 ——
`WithAgentInterruptBefore(agents.ToolsNodeName)` 的边界在 HITL 决策被批准
之后、tools 分发之前触发（`agents.HITLNodeName` 同样可如此寻址）。

```go
agent, _ := agents.CreateAgent(model, tools,
	agents.WithAgentCheckpointer(checkpointer),
	agents.WithAgentInterruptBefore(agents.ToolsNodeName), // pause before tools run
)

// First run pauses; resume the same thread:
result, _ := agent.Graph.InvokeWithOptions(ctx,
	map[string]any{"messages": msgs},
	graphpkg.Options{ThreadID: "thread-1"}, // nil Resume resumes a boundary interrupt
)
```

> **API 说明：** checkpointer 类型（`checkpoint.Saver`、
> `checkpoint.NewMemorySaver`）位于公开包
> `github.com/projanvil/langchain-golang/langgraph/checkpoint`。要接入自
> 定义 saver，实现带版本的 `Saver` 接口 —— `GetTuple` / `List`
> （`ListOptions.Filter` 提供 metadata 过滤）/ `Put` / `PutWrites`（带
> `taskPath` 参数；`checkpoint.Write` 携带 `TaskPath`）/ `DeleteThread`，
> 以 `checkpoint.Config`（线程 ID + checkpoint namespace + checkpoint ID）
> 为键；往返形态见
> `langchain/agents/create_agent_test.go`（`TestCreateAgent_InterruptBeforeNode`）。
> 该接口先是取代了 M1 的 `Get` / `Put` / `Delete` 接口，后又被 M5 扩展 ——
> 两次都是 spec 允许的 1.0 前 break；before/after 签名与迁移指引见
> [图运行时指南](../langgraph.md)的 *Breaking 变更*一节。

### Checkpoint 历史与时间旅行

装了 checkpointer 之后，每个超步都会写入一个不可变的、可按 ID 寻址的
checkpoint，因此一个线程会累积出完整历史。编译好的图（`Agent.Graph`）
直接暴露这些能力：

- `GetState(ctx, checkpoint.Config{ThreadID: ...})` —— 最新（或指定）
  checkpoint 的 `StateSnapshot`：channel 值、下一批节点、config、待处理
  的 interrupt。
- `GetStateHistory(ctx, cfg, opts)` —— 该线程的快照，新的在前。
- `UpdateState(ctx, cfg, values, asNode)` —— 应用一批归属于某节点的写入
  并保存为新 checkpoint（human-in-the-loop 的状态编辑）。
- 时间旅行：向 `InvokeWithOptions` 传 `graphpkg.Options{ThreadID: ...,
  CheckpointID: ...}` 钉住某个历史 checkpoint —— 运行从它分叉，而不是从
  线程的最新状态继续。

相关的 `graph.Interrupt(ctx, value)` 原语 —— 在节点*内部*暂停并在恢复时
喂回一个值 —— 经公开包
`github.com/projanvil/langchain-golang/langgraph/graph` 提供给
middleware/节点作者使用。

> **持久化 checkpoint：** `checkpoint.NewMemorySaver` 是内存实现。需要持久
> 后端时，嵌套模块 `langgraph/checkpoint/sqlite` 提供 SQLite saver ——
> `sqlite.New(path, serde.NewJSONSerializer())` —— 可用于任何接受
> `checkpoint.Saver` 的地方（包括 `WithAgentCheckpointer`）。嵌套模块
> `langgraph/checkpoint/postgres` 提供 Postgres saver ——
> `postgres.NewFromConnString(ctx, dsn, serde.NewJSONSerializer())` 外加一
> 次显式的 `Setup(ctx)` 调用 —— 用法相同。详见
> [图运行时指南](../langgraph.md)。

## State 与 context schema

- **`WithAgentStateFields`** —— 注册自带 reducer 的自定义图状态字段（对应
  Python 的 `state_schema`）。名称与默认键（`messages` / `jump_to` /
  `structured_response`）冲突的字段会覆盖该键的 reducer。
- **`WithAgentContextSchema`** + **`WithContextValues`** / **`ContextValue`** ——
  声明并读取随 Go `context.Context` 传递的每次运行、只读的上下文（对应
  Python 的 `context_schema`）。

完整的 `WithAgent*` 选项集（递归上限、名称、debug、store、cache 等）见
`agents` 包的 godoc。

## Streaming

需要实时输出时使用 `agent.StreamEvents` —— 见
[streaming 指南](../streaming.md)。更底层的图层面，编译好的图
（`Agent.Graph`）还暴露带 Python 对齐 stream 模式（`values` / `updates` /
`debug` / `messages` / `custom`）的 `Stream` —— 见
[图运行时指南](../langgraph.md)。

## 子 agent（agent 即工具）

"子 agent" 是从另一个 agent 的工具内部调用的具名 agent。没有专门的 API：
照 Python `langchain.agents` 的写法，写一个工具，其函数体调用内层 agent 的
`InvokeWithState`，并返回最后一条 AI 消息的文本。

```go
// A named inner agent — the name is what makes it distinguishable.
weather, err := agents.CreateAgent(model, nil, agents.WithAgentName("weather_agent"))

// Hand-rolled subagent tool (the Go equivalent of Python's
//   @tool
//   def call_weather(city): return weather.invoke(...)["messages"][-1].text).
callWeather, err := coretools.NewFunc(
    "call_weather", "Call the weather agent.",
    schema.Object(map[string]schema.Schema{"city": schema.String("city")}, "city"),
    func(ctx context.Context, input map[string]any) (coretools.Result, error) {
        city, _ := input["city"].(string)
        state, err := weather.InvokeWithState(ctx, []messages.Message{messages.Human("weather in " + city)})
        if err != nil {
            return coretools.Result{}, err
        }
        msgs, _ := state["messages"].([]messages.Message)
        for i := len(msgs) - 1; i >= 0; i-- {
            if msgs[i].Role == messages.RoleAI {
                return coretools.Result{Content: messages.Text(msgs[i])}, nil
            }
        }
        return coretools.Result{}, fmt.Errorf("weather agent produced no output")
    },
)

// Supervisor delegates via the tool.
supervisor, err := agents.CreateAgent(model, []coretools.Tool{callWeather}, agents.WithAgentName("supervisor"))
```

在嵌套运行内部，`agents.NameFromContext(ctx)` 返回的是内层 agent 的名字
（`"weather_agent"`）而非 supervisor 的，因为 `InvokeWithState` 会重新绑定
run-name 上下文标签。用 `WithAgentName` 构建内层 agent，使其对 middleware、
日志与 tracing 可区分。该名字还会被盖到 agent 模型产出的每条 AI 消息上，
因此从多个子 agent 汇集的消息历史仍可归因。

内层 agent 的错误会穿过工具、以错误 `ToolMessage` 的形式浮现（经由
`ToolNode` 默认的 `HandleToolErrors`），因此 supervisor 的运行仍会完成，
模型可以作出反应。嵌套可递归：每次 `InvokeWithState` 都是一次独立的图运行，
有自己的递归上限。

**Streaming 限制。** 当 supervisor 经 `StreamEvents` 运行时，嵌套的 agent 以
非 streaming 方式运行 —— 只有它的最终结果作为工具结果浮现；嵌套运行不会
向父流发出 `model_delta` 事件。把子 agent 的实时事件在独立句柄
（`run.subagents`）下有界浮现的能力未提供；它属于被延后的
stream-transformer 工作（v1-final-parity spec 中的 Design Decision 4）。

## 有意不做的部分

遵循有界移植立场（只移植 `langgraph` 的一个子集）：

- **`transformers` / `run.subagents`** —— 不暴露；流式 PII 脱敏改由
  `WrapModelStreamHook` middleware 的增量层提供。
- **工具返回 `Send`** —— 不支持（没有按工具调用的 Send 分发；并行工具调用
  改为在单一节点内并发执行）。
