package mcp

// HITL gating tests: mapping MCP tool annotations (destructiveHint) to
// HumanInTheLoopMiddleware InterruptOn configs.

import (
	"slices"
	"testing"

	coretools "github.com/projanvil/langchain-golang/core/tools"
	agentsmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// TestAnnotationsExposed: the destructive annotation round-trips through the
// loaded tool.
func TestAnnotationsExposed(t *testing.T) {
	danger := loadOne(t, "srv", "srv_danger")
	echo := loadOne(t, "srv", "srv_echo")
	if danger.Annotations().DestructiveHint == nil || !*danger.Annotations().DestructiveHint {
		t.Fatalf("danger annotations = %+v, want destructiveHint=true", danger.Annotations())
	}
	if echo.Annotations().DestructiveHint != nil && *echo.Annotations().DestructiveHint {
		t.Fatalf("echo annotations = %+v, want no destructive hint", echo.Annotations())
	}
}

// TestInterruptOnConfigs: only destructive tools produce InterruptOn
// entries, with approve/reject decisions by default and the tool's args
// schema attached.
func TestInterruptOnConfigs(t *testing.T) {
	danger := loadOne(t, "srv", "srv_danger")
	echo := loadOne(t, "srv", "srv_echo")

	interruptOn := InterruptOnConfigs([]coretools.Tool{danger, echo})
	if len(interruptOn) != 1 {
		t.Fatalf("InterruptOnConfigs() = %#v, want exactly the destructive tool", interruptOn)
	}
	cfg, ok := interruptOn["srv_danger"]
	if !ok {
		t.Fatalf("InterruptOnConfigs() missing srv_danger: %#v", interruptOn)
	}
	if !slices.Contains(cfg.AllowedDecisions, agentsmiddleware.DecisionApprove) ||
		!slices.Contains(cfg.AllowedDecisions, agentsmiddleware.DecisionReject) {
		t.Fatalf("AllowedDecisions = %v, want approve and reject", cfg.AllowedDecisions)
	}
	if cfg.ArgsSchema == nil {
		t.Fatal("ArgsSchema not attached")
	}

	// Custom decisions override the default pair.
	custom := InterruptOnConfigs([]coretools.Tool{danger},
		WithAllowedDecisions(agentsmiddleware.DecisionApprove))
	if len(custom["srv_danger"].AllowedDecisions) != 1 {
		t.Fatalf("custom decisions = %v", custom["srv_danger"].AllowedDecisions)
	}
}

// TestInterruptOnConfigsFeedMiddleware: the produced map wires directly into
// the interrupt-mode HumanInTheLoopMiddleware.
func TestInterruptOnConfigsFeedMiddleware(t *testing.T) {
	danger := loadOne(t, "srv", "srv_danger")
	mw := agentsmiddleware.NewInterruptHumanInTheLoopMiddleware(
		InterruptOnConfigs([]coretools.Tool{danger}))
	if !mw.HitlInterruptEnabled() {
		t.Fatal("middleware must be in interrupt mode")
	}
	if _, ok := mw.InterruptOn["srv_danger"]; !ok {
		t.Fatalf("InterruptOn = %#v", mw.InterruptOn)
	}
}
