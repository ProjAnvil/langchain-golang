package mcp

import (
	"github.com/mark3labs/mcp-go/mcp"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	agentsmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// InterruptOnOption configures InterruptOnConfigs.
type InterruptOnOption func(*interruptOnConfig)

type interruptOnConfig struct {
	allowedDecisions []agentsmiddleware.DecisionType
}

// WithAllowedDecisions overrides the default approve/reject decision pair
// for the produced InterruptOn entries.
func WithAllowedDecisions(decisions ...agentsmiddleware.DecisionType) InterruptOnOption {
	return func(c *interruptOnConfig) { c.allowedDecisions = decisions }
}

// InterruptOnConfigs maps MCP tools whose server annotations carry
// destructiveHint=true to HumanInTheLoopMiddleware InterruptOn entries
// (langchain/agents/human_in_the_loop.go), mirroring the upstream pairing
// of destructiveHint with InterruptOnConfig. Non-MCP tools and
// non-destructive MCP tools are skipped; the returned map keys are the
// namespaced tool names, ready for NewInterruptHumanInTheLoopMiddleware.
//
// The default allowed decisions are approve and reject (the upstream
// example); the tool's args schema is attached so reviewers can inspect and
// edit the pending arguments.
func InterruptOnConfigs(tools []coretools.Tool, opts ...InterruptOnOption) map[string]agentsmiddleware.InterruptConfig {
	cfg := interruptOnConfig{
		allowedDecisions: []agentsmiddleware.DecisionType{
			agentsmiddleware.DecisionApprove,
			agentsmiddleware.DecisionReject,
		},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	out := make(map[string]agentsmiddleware.InterruptConfig)
	for _, t := range tools {
		mt, ok := t.(*Tool)
		if !ok {
			continue
		}
		if !isDestructive(mt.Annotations()) {
			continue
		}
		out[mt.Name()] = agentsmiddleware.InterruptConfig{
			AllowedDecisions: cfg.allowedDecisions,
			ArgsSchema:       mt.ArgsSchema(),
		}
	}
	return out
}

// isDestructive reads an annotation set the way upstream's is_destructive
// helper does: an explicit destructiveHint=true gates, an explicit
// readOnlyHint=true never does, and an absent destructiveHint is treated as
// not destructive. Note that the MCP wire default for destructiveHint is
// true and mcp-go's NewTool materializes those defaults server-side, so
// tools from mcp-go servers that did not override their annotations gate
// (the spec-safe reading); servers that omit the fields entirely do not.
func isDestructive(a mcp.ToolAnnotation) bool {
	if a.ReadOnlyHint != nil && *a.ReadOnlyHint {
		return false
	}
	return a.DestructiveHint != nil && *a.DestructiveHint
}
