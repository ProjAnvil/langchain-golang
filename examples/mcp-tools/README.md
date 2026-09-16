# mcp-tools

MCP tool use: an in-process mcp-go server is attached through `partners/mcp` (`New` + `LoadTools`), its tools are discovered with `{server}_{tool}` namespacing, and an offline scripted agent calls one. Swap `InProcessServer` for `StdioServer`/`HTTPServer` in the same `MCPConfig` to reach real servers.

## Run

```sh
go run ./examples/mcp-tools
```

No environment variables required (in-process MCP server + fake model).
