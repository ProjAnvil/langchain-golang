# Streaming

**Languages:** [English](../streaming.md) | 简体中文

`Agent.StreamEvents` 返回一个拉取式的 `StreamEvent` 流，让你实时观察运行
过程：逐 token 的模型增量、工具分发生命周期与节点边界。它是 Python
`astream_events` 的 agent 级对应物；面向任意 `core/runnables` Runnable 的
Runnable 级 v2 事件流是下文的
[`runnables.StreamEvents`](#runnable-级事件runnablestreamevents)，LangSmith
tracing 也挂接在同一层 callback 上。

## 事件类型

| 常量 | 何时发出 | 填充的关键字段 |
|----------|--------------|----------------------|
| `StreamNodeStart` / `StreamNodeEnd` | 每个节点（`before_agent`、`model`、`tools`、`after_agent`）前后 | `Node` |
| `StreamModelDelta` | 每个模型 chunk | `Node`、`Delta`、`Text` |
| `StreamModelEnd` | 每次模型调用一次，携带组装好的 AI 消息 | `Node`、`Message` |
| `StreamToolStart` | 每次工具分发前 | `Node`、`ToolName`、`ToolArgs` |
| `StreamToolEnd` | 每次工具分发后 | `Node`、`ToolName`、`ToolResult` |
| `StreamEnd` | 终止事件，最后恰好发出一次 | `State`、`Message`（或 `Err`） |

所有常量都在 `agents` 包中，例如 `agents.StreamModelDelta`。

## 最小示例：打印文本增量与工具调用

```go
stream, err := agent.StreamEvents(ctx, []messages.Message{
	messages.User("Summarize the latest commits."),
})
if err != nil {
	panic(err)
}
for {
	ev, ok, err := stream.Next(ctx)
	if err != nil {
		panic(err)
	}
	if !ok {
		break
	}
	switch ev.Type {
	case agents.StreamModelDelta:
		fmt.Print(ev.Text)
	case agents.StreamToolStart:
		fmt.Printf("\n[tool %s args=%v]\n", ev.ToolName, ev.ToolArgs)
	case agents.StreamToolEnd:
		fmt.Printf("\n[tool %s done result=%v]\n", ev.ToolName, ev.ToolResult)
	case agents.StreamEnd:
		if ev.Err != nil {
			log.Printf("run ended: %v", ev.Err)
		}
	}
}
```

`ev.Text` 是一个便捷字符串，承载 `StreamModelDelta` 的文本增量（非文本
增量 —— 如 reasoning 或 tool-call 增量 —— 时为空）。如果你需要原始的
content-block 协议事件（例如 reasoning 增量），读 `ev.Delta`。

## 顺序保证

- `node_start` / `node_end` 对每次节点调用总是成对出现，即使在出错或
  interrupt 路径上也是如此。
- 在一个 `model` 节点内：零或多个 `model_delta` 事件，然后恰好一个携带
  完整组装 AI 消息的 `model_end`。
- 在一个 `tools` 节点内：每个被分发的工具对应一对 `tool_start` /
  `tool_end`。
- 流关闭前，恰好一个终止 `StreamEnd` 作为最后一个事件。

当图扇出（同一超步内多个任务活跃）时，它们的事件在流上交错 —— 通过
`Node` 字段区分。

## Runnable 级事件（`runnables.StreamEvents`）

`core/runnables.StreamEvents` 是 Python `astream_events(v2)` 的 Go 对应物，
作用于**任意** `Runnable` —— 用 `Pipe` / `Parallel` / `Retry` 组合的链、
`NewFunc` lambda、chat 模型 —— 而不限于 agent。它把一次
`Runnable.Stream` 调用包进一个收集 callback 的 manager，并把扁平的
callback 流投影到 v2 的 `runnables.StreamEvent` 形状：

```go
import (
	"context"
	"strings"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
)

double := runnables.NewFunc(
	func(_ context.Context, in string, _ ...runnables.Option) (string, error) {
		return in + in, nil
	}, schema.String(""), schema.String(""))
upper := runnables.NewFunc(
	func(_ context.Context, in string, _ ...runnables.Option) (string, error) {
		return strings.ToUpper(in), nil
	}, schema.String(""), schema.String(""))
chain := runnables.Pipe(double, upper)

for ev, err := range runnables.StreamEvents(ctx, chain, "go",
	runnables.StreamEventOptions{}) {
	if err != nil {
		return err // run failure is yielded as the final pair
	}
	fmt.Println(ev.Event, ev.Name, ev.RunID, ev.ParentIDs)
}
```

- **事件形状** —— `Event`（`on_chain_start` / `on_chain_stream` /
  `on_chain_end` / `on_chain_error`、`on_chat_model_*`、`on_llm_*`、
  `on_tool_*`、`on_retriever_*`）、`RunID`、`Name`、`Tags`、`Metadata`、
  根优先的 `ParentIDs`，以及 `Data` 负载（start 携带 `Input`、stream 携带
  `Chunk`、end 携带 `Output`、error 携带 `Error`）。根运行的 `ParentIDs`
  为 nil；`Pipe` 内部的 step 以链的 run ID 作为其唯一父级。
- **过滤** —— `StreamEventOptions` 对齐 Python 的 `include_*` /
  `exclude_*` 过滤器：`IncludeNames` / `ExcludeNames`、
  `IncludeTypes` / `ExcludeTypes`（run 类型 `chain` / `chat_model` / `llm`
  / `tool` / `retriever`）与 `IncludeTags` / `ExcludeTags`。每个已设置的
  include 列表都必须放行该事件（include-OR）；每个 exclude 列表都不得
  命中。过滤只发生在投影输出端，因此被过滤掉的运行不会破坏幸存后代的
  `parent_ids`。
- **Chat-model 聚合** —— 每个模型运行一个聚合器，从流式 chunk 组装
  `on_chat_model_end` 的输出，并在 provider 同时发出两种形式时优先采用 v3
  content-block 协议而非旧版消息 chunk：只有 content-block 增量会以
  `on_chat_model_stream` 浮出（chunk 负载是协议事件本身 —— 与 v2 的
  `AIMessageChunk` 不同，是已文档化的分歧）；message-start/finish 与块
  边界折叠进前后的 start/end 事件。
- **生命周期** —— 迭代器单次使用；运行失败在所有事件之后作为最后一对
  `(零值, err)` 吐出。提前跳出 `range` 会取消运行并等待生产者 goroutine
  退出，因此没有泄漏。并发的子运行自由交错 —— 按 `RunID` 而非全局嵌套
  顺序配对事件。

`Agent.StreamEvents`（上述七种领域事件）与 `runnables.StreamEvents`
（Runnable 树的 v2 事件）是不同层的两个投影 —— 一次运行选一个面；二者
不互通。

## LangSmith tracing

`core/tracers` 内置 LangSmith tracer（`tracers.NewLangChainTracer`）：它
从同一层 callback 流重建运行树，并在后台 goroutine 上发往 LangSmith 批量
API —— 每批最多 64 个操作、500ms 刷新节拍、重试一次后交给 `OnError`。
tracing 永远不会拖垮业务调用：handler 错误被吞掉，POST 失败只上报、不
传播。

用环境变量启用（可选变量遵循 Python 客户端 `LANGSMITH_` 先于
`LANGCHAIN_` 的优先级）：

```bash
export LANGSMITH_TRACING_V2=true  # 或 LANGCHAIN_TRACING_V2 / LANGSMITH_TRACING / LANGCHAIN_TRACING
export LANGSMITH_API_KEY=ls__...  # 或 LANGCHAIN_API_KEY —— 必需
export LANGSMITH_PROJECT=my-app   # 可选；LANGSMITH_ENDPOINT 覆盖 API base
```

tracing **默认关闭** —— 除非 tracing 标志与 API key 同时设置，否则什么也
不会发送。显式构造 tracer（环境未启用时返回 nil，且所有方法 nil-safe，
因此可以无条件使用）并自行管理其生命周期：

```go
tracer := tracers.NewLangChainTracer()
defer tracer.Close() // flushes the remainder; idempotent
```

环境启用 tracing 时，`runnables.StreamEvents` 会自动附加该 tracer，因此
流式运行的追踪无需向 options 接线。自己挂 tracer 时请保持环境变量未设置
—— 重复的 tracer 会在服务端造成冲突。`NewLangChainTracerWithOptions`
用显式 `LangSmithOptions`（endpoint、project、批大小、刷新节拍、
`OnError`）构造 tracer。

## Streaming 与非 streaming

`agent.Invoke` 把循环跑到底并返回最终消息历史。`agent.StreamEvents` 运行
同一个循环，但边跑边发事件。两条路径的状态语义完全一致；streaming 是附加
的可观测性，不是另一种执行模型。

## 关于缓存的说明

配置了 `WithAgentCache` 时，缓存只在非 streaming 的 `Invoke` 路径上被
查询。`StreamEvents` 总是绕过缓存，使 `model_delta` / `model_end` 事件在
每次运行都会触发 —— 否则缓存命中会短路模型调用，什么事件也发不出来。

## Provider 说明

- **`partners/openai` 的 usage chunk。** OpenAI chat 模型的 `Stream` 会
  opt in usage 统计（`stream_options.include_usage`）：流的末尾会出现一个
  仅含 usage 的 chunk —— 一条携带 `UsageMetadata` 的空 AI 消息，并在
  provider 上报时包含缓存 token / reasoning token 细节。它经
  `StreamEvents` 不会发出 `model_delta`（文本为空）；需要读取它请直接消费
  `model.Stream`。
- **流式下的结构化输出。** 流式路径应用与 `Invoke` 相同的按次绑定：
  middleware 的 `WithResponseFormat` / `WithToolChoice` /
  `WithModelSettings` 覆盖，以及 `ProviderStrategy` 的 model kwargs（在模型
  实现 `ModelSettingsBinder` 时）同样作用于 `Stream` 调用，而不仅限非流式
  路径；不具备该能力的模型保持对组装后消息的事后 JSON 解析。
