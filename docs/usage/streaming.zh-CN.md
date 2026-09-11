# Streaming

**Languages:** [English](streaming.md) | 简体中文

`Agent.StreamEvents` 返回一个拉取式的 `StreamEvent` 流，让你实时观察运行
过程：逐 token 的模型增量、工具分发生命周期与节点边界。它是 Python
`astream_events` 的 Go 等价物。

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
