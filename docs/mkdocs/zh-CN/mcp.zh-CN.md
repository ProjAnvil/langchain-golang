# MCP 工具：Model Context Protocol 适配器

**Languages:** [English](../mcp.md) | 简体中文

`partners/mcp` 把 Model Context Protocol（MCP）服务器适配成标准的
`core/tools` 工具，对齐 langchain 1.4.0 的 `langchain.mcp` 快照。智能体并
不知道工具在远端：它像调用任何其他工具一样调用它。可运行的完整示例见
[`examples/mcp-tools`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/mcp-tools)。

## 安装

```bash
go get github.com/projanvil/langchain-golang
```

适配器的线上协议基于
[`mark3labs/mcp-go`](https://github.com/mark3labs/mcp-go)。

## 连接一个 fleet（MCPConfig）

`mcp.New` 接收一个 `"mcpServers"` 风格的 map 并连接每个条目。一个 fleet
里可以自由混用传输方式：

```go
import langchainmcp "github.com/projanvil/langchain-golang/partners/mcp"

adapter, err := langchainmcp.New(ctx, langchainmcp.MCPConfig{
    Servers: map[string]langchainmcp.Server{
        "calc":   &langchainmcp.StdioServer{Command: "uvx", Args: []string{"mcp-server-calculator"}},
        "remote": &langchainmcp.HTTPServer{URL: "https://example.com/mcp"},
    },
})
if err != nil { /* ... */ }
defer adapter.Close()

toolList, err := adapter.LoadTools(ctx) // list_tools + schema
```

工具名带 `{server}_{tool}` 命名空间（"calc_add"），因此远端名字冲突的
fleet 依然可区分。已经构建好的 mcp-go 客户端（自带认证、自带 handler）则
用 `mcp.NewClientGroup` 替代 `MCPConfig`。

## 基于 MCP 工具的智能体

```go
agent, err := agents.CreateAgent(model, toolList,
    agents.WithAgentSystemPrompt("Use the available tools to answer."))
out, err := agent.Invoke(ctx, []messages.Message{messages.Human("what is 19 plus 23?")})
```

工具结果遵循上游映射：文本成为模型可见内容；图片/音频/文件内容以内容块
形式保留在 `*ToolArtifact` 上；`isError=true` 映射为 `*ToolError`（渲染成
模型可自我纠正的错误 `ToolMessage`）；传输层故障则以 error 抛出。

## Elicitation（服务器向人要输入）经 interrupt 桥接

当服务器发出 elicitation 请求（2026-07-28 协议形态）时，适配器把它桥接
为 LangGraph interrupt。前置条件：工具调用必须运行在**带 checkpointer**
的图节点内——`CreateAgent` 传了 `agents.WithAgentCheckpointer(saver)` 并
使用 `ThreadID` 后即满足：

```go
result, _ := agent.Invoke(ctx, msgs, agents.Options{ThreadID: "t1"})
// result.Interrupts 携带待处理的 elicitation；把请求展示给
// 人工，再按服务器的请求 key 回答：

resume := mcp.ElicitationResponses{Responses: map[string]mcp.ElicitationAnswer{
    "confirm": {Action: mcp.ElicitationAccept, Content: map[string]any{"amount": 42}},
}}
result, _ = agent.Invoke(ctx, nil, agents.Options{ThreadID: "t1", Resume: resume})
```

动作有 `ElicitationAccept`（content 必须匹配请求的 schema）、
`ElicitationDecline`（工具调用继续）、`ElicitationCancel`（放弃整个调用）。

## 破坏性工具的 HITL 门控

MCP 工具注解可以把工具标记为破坏性（`destructiveHint=true`）。
`mcp.InterruptOnConfigs` 一行就把它们变成人机交互中间件配置：

```go
import agentsmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"

mw := agentsmiddleware.NewInterruptHumanInTheLoopMiddleware(
    mcp.InterruptOnConfigs(toolList))
agent, _ := agents.CreateAgent(model, toolList, agents.WithAgentMiddleware(mw))
```

此后每个破坏性 MCP 工具执行前都会暂停等待人工批准——与其他 HITL
中间件共用同一套 interrupt/resume 往返。

## 切换点

- **传输**：`InProcessServer`（内嵌 mcp-go 服务器，离线测试）→
  `StdioServer` / `HTTPServer`，`New` 之后的代码零改动。
- **Elicitation 开关**：`mcp.NewClientGroup(..., mcp.WithoutElicitation())`
  为不应暂停的客户端关闭该能力。
- **缓存**：`LoadTools(ctx, mcp.WithCacheMode(mcp.CacheModeRefresh))`——
  默认 `CacheModeUse` 会在有缓存时直接使用缓存的工具清单。
