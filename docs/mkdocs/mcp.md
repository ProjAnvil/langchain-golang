# MCP tools: Model Context Protocol adapters

**Languages:** English | [简体中文](zh-CN/mcp.zh-CN.md)

`partners/mcp` adapts Model Context Protocol (MCP) servers into standard
`core/tools` tools, mirroring the langchain 1.4.0 `langchain.mcp` snapshot.
An agent does not know or care that a tool is remote: it calls it like any
other tool. For a runnable walkthrough see
[`examples/mcp-tools`](https://github.com/ProjAnvil/langchain-golang/tree/main/examples/mcp-tools).

## Installation

```bash
go get github.com/projanvil/langchain-golang
```

The adapter builds on [`mark3labs/mcp-go`](https://github.com/mark3labs/mcp-go)
for the wire protocol.

## Connecting a fleet (MCPConfig)

`mcp.New` takes an `"mcpServers"`-style map and connects every entry. Transports
mix freely in one fleet:

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

toolList, err := adapter.LoadTools(ctx) // list_tools + schemas
```

Tool names are namespaced `{server}_{tool}` ("calc_add"), so fleets with
colliding remote names stay distinguishable. Already-built mcp-go clients
(own auth, own handlers) use `mcp.NewClientGroup` instead of `MCPConfig`.

## An agent over MCP tools

```go
agent, err := agents.CreateAgent(model, toolList,
    agents.WithAgentSystemPrompt("Use the available tools to answer."))
out, err := agent.Invoke(ctx, []messages.Message{messages.Human("what is 19 plus 23?")})
```

Tool results follow the upstream mapping: text becomes model-visible content;
image/audio/file content is preserved as content blocks on a `*ToolArtifact`;
`isError=true` maps to a `*ToolError` (rendered as an error `ToolMessage` the
model can self-correct on); transport failures raise as errors.

## Elicitation (server-asked input) via interrupts

When a server sends an elicitation request (protocol 2026-07-28 form), the
adapter bridges it to a LangGraph interrupt. Requirements: the tool call must
run inside a graph node **with a checkpointer** — which `CreateAgent` provides
once you pass `agents.WithAgentCheckpointer(saver)` and a `ThreadID`:

```go
result, _ := agent.Invoke(ctx, msgs, agents.Options{ThreadID: "t1"})
// result.Interrupts carries the pending elicitation; show the request to
// a human, then answer keyed by the server's request key:

resume := mcp.ElicitationResponses{Responses: map[string]mcp.ElicitationAnswer{
    "confirm": {Action: mcp.ElicitationAccept, Content: map[string]any{"amount": 42}},
}}
result, _ = agent.Invoke(ctx, nil, agents.Options{ThreadID: "t1", Resume: resume})
```

Actions are `ElicitationAccept` (content must match the requested schema),
`ElicitationDecline` (the tool call continues), and `ElicitationCancel`
(abandons the whole call).

## HITL gating for destructive tools

MCP tool annotations can mark a tool as destructive
(`destructiveHint=true`). `mcp.InterruptOnConfigs` turns those into
human-in-the-loop middleware config in one line:

```go
import agentsmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"

mw := agentsmiddleware.NewInterruptHumanInTheLoopMiddleware(
    mcp.InterruptOnConfigs(toolList))
agent, _ := agents.CreateAgent(model, toolList, agents.WithAgentMiddleware(mw))
```

Now every destructive MCP tool pauses for human approval before executing —
the same interrupt/resume round trip as any other HITL middleware.

## Switching points

- **Transport**: `InProcessServer` (embedded mcp-go server, offline tests) →
  `StdioServer` / `HTTPServer` without touching anything after `New`.
- **Elicitation on/off**: `mcp.NewClientGroup(..., mcp.WithoutElicitation())`
  opts out for clients that should never pause.
- **Caching**: `LoadTools(ctx, mcp.WithCacheMode(mcp.CacheModeRefresh))` —
  the default `CacheModeUse` serves a cached listing when present.
