// Package mcp adapts Model Context Protocol servers into core/tools tools,
// mirroring the langchain 1.4.0 `langchain.mcp` snapshot (API surface locked
// to that docs snapshot; later upstream drift is not tracked here).
//
// Two connection mechanisms exist, matching the upstream snapshot:
//
//   - MCPConfig: an aggregated multi-server fleet ("mcpServers" dict) built
//     by New. Every tool is prefixed with its config key, so duplicate tool
//     names across backends stay distinguishable. Upstream negotiates a
//     single protocol era across the fleet; the mcp-go clients underneath
//     each negotiate independently, so the Go port documents this as
//     per-connection negotiation instead.
//   - ClientGroup: prebuilt, independent mcp-go clients (own auth, own
//     handlers), routing each call back to the client that advertised the
//     tool. Tools are likewise namespaced {server}_{tool}.
//
// Both expose LoadTools (list_tools with a cache mode) and Close.
//
// Tool result mapping follows the upstream four quadrants: text becomes the
// model-visible content, image/audio/file content is preserved as
// standardized content blocks on the artifact, structured output attaches a
// *ToolArtifact (never folded into model-visible text), isError=true maps to
// a *ToolError (so langchain/tools.HandleToolErrors renders the error
// ToolMessage), and transport/session failures raise (a model cannot act on
// a dropped connection).
//
// Elicitation: modern multi-round-trip input requests (protocol 2026-07-28,
// the only form upstream supports) are bridged to LangGraph interrupts. The
// tool call must run inside a graph node with a checkpointer; resuming
// supplies answers keyed by the server's request key with accept/decline/
// cancel actions. Legacy handshake-session elicitation is bridged only for
// in-process connections; on remote transports mcp-go dispatches it outside
// the caller's goroutine, where a panic-based interrupt cannot unwind, so it
// fails with a descriptive error.
//
// Tool-level HITL gating: InterruptOnConfigs maps tools whose MCP
// annotations carry destructiveHint to HumanInTheLoopMiddleware InterruptOn
// entries (langchain/agents/human_in_the_loop.go).
package mcp
