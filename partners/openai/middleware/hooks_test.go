package middleware

import (
	"testing"

	"github.com/projanvil/langchain-golang/langchain/agents"
	agentsmiddleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// Compile-time checks that the middleware satisfy the hook surfaces
// CreateAgent discovers structurally (WrapToolCallHook, BeforeAgentHook,
// WrapModelCallHook, StateSchemaContributor, TracePolicyProvider). These
// are the Go equivalents of Python's AgentMiddleware method overrides, and
// importing the agents package here (test-only) guarantees the signatures
// stay compatible without adding a runtime dependency.
var (
	_ agents.WrapToolCallHook                 = (*AutoMode)(nil)
	_ agents.BeforeAgentHook                  = (*ModelRouter)(nil)
	_ agents.WrapModelCallHook                = (*ModelRouter)(nil)
	_ agentsmiddleware.StateSchemaContributor = (*ModelRouter)(nil)
	_ agentsmiddleware.TracePolicyProvider    = (*ModelRouter)(nil)
)

func TestHooksCompile(t *testing.T) {
	// No runtime behavior; the var block above is the assertion.
}
