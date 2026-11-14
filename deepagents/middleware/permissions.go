package middleware

import (
	"slices"

	"github.com/projanvil/langchain-golang/deepagents/backends"
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// fsInterruptToolOrder lists the seven filesystem tools in the fixed key
// order of _FS_TOOL_PATH_ARGS in the Python reference,
// deepagents/middleware/_fs_interrupt.py:39-47, so
// InterruptConfigsFromPermissions builds its map deterministically (Go map
// iteration itself is unordered).
var fsInterruptToolOrder = []string{
	"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
}

// fsInterruptDecisions is the full decision set offered on every derived
// entry, mirroring the `allowed` list in _build_interrupt_on_from_permissions
// (UP _fs_interrupt.py:189-194): approve, edit, reject, and respond are all
// human-controlled, so the human stays the authorization gate — an edited
// call re-enters the tool and hits its pre-execution deny check, and respond
// skips execution entirely.
var fsInterruptDecisions = []agentsmw.DecisionType{
	agentsmw.DecisionApprove,
	agentsmw.DecisionEdit,
	agentsmw.DecisionReject,
	agentsmw.DecisionRespond,
}

// InterruptConfigsFromPermissions derives the interrupt_on map for
// NewInterruptHumanInTheLoopMiddleware from interrupt-mode permission rules,
// mirroring _build_interrupt_on_from_permissions in the Python reference,
// deepagents/middleware/_fs_interrupt.py:174-203. It returns one entry for
// each filesystem tool whose operation class is covered by an interrupt-mode
// rule: a read-op interrupt rule yields entries for ls/read_file/glob/grep,
// a write-op rule yields write_file/edit_file/delete (the tool→operation
// mapping is fsToolOperations, mirroring _FS_TOOL_PATH_ARGS' operation
// column). The FIRST interrupt rule matching a tool's op class wins (entries
// are otherwise identical, so first-wins and any-match coincide); allow and
// deny rules never produce entries. CreateDeepAgent feeds the result to
// NewInterruptHumanInTheLoopMiddleware at assembly time (Task 9).
//
// Divergences from the reference, intentional (port plan D11):
//   - the reference scopes each entry with a `when` predicate firing only
//     when the call's path argument matches an interrupt-mode rule (and no
//     earlier deny rule matches); this port applies each entry to EVERY call
//     of the covered tool. The port is strictly more conservative: it may
//     pause calls the reference would let through, never fewer, and a deny
//     rule still wins once a paused call is approved or edited into running,
//     because the tool's own deny gate (denyGate) runs before the backend.
//   - no backend parameter: the reference also matches the symlink-resolved
//     target of the call's path inside its predicate; the when-less entries
//     this port derives need no path matching at all.
//
// Interrupt-mode contract at the tool gate: the HITL middleware reviews the
// model's tool calls in its after-model node and pauses the run BEFORE the
// tools node executes them, so a wired agent's filesystem tools never observe
// an interrupt-mode disposition in practice. FilesystemMiddleware.Check still
// reports PermissionInterrupt for such rules, but denyGate intentionally
// treats interrupt as allow — the interrupt decision path is owned entirely
// by the HITL middleware. When the HITL middleware is not installed, the same
// allow fall-through applies: interrupt rules gate only through HITL, never
// by blocking the tool themselves.
func InterruptConfigsFromPermissions(perms []backends.PermissionRule) map[string]agentsmw.InterruptConfig {
	configs := make(map[string]agentsmw.InterruptConfig)
	for _, tool := range fsInterruptToolOrder {
		operation := fsToolOperations[tool]
		for _, rule := range perms {
			if rule.Mode != backends.PermissionInterrupt {
				continue
			}
			if !slices.Contains(rule.Operations, operation) {
				continue
			}
			configs[tool] = agentsmw.InterruptConfig{AllowedDecisions: fsInterruptDecisions}
			break // first interrupt rule per tool wins
		}
	}
	return configs
}
