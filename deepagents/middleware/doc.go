// Package middleware ports the deepagents middleware layer to Go. It mirrors
// the Python reference under deepagents/middleware/ in the upstream
// deepagents repository (libs/deepagents/deepagents/middleware/).
//
// The centerpiece of this task is FilesystemMiddleware (filesystem.go): it
// binds the filesystem tools to a deepagents/backends Backend, evaluates
// permission rules for filesystem tool calls (first match wins, default
// allow), and — for a *backends.StateBackend — injects the files state into
// each tool call's context via agents.WrapToolCallHook. state.go carries the
// subagent state-key contracts: ExcludedStateKeys and PrivateStateKeys,
// filtered when building subagent state and when applying subagent updates.
//
// The delegation half is SubAgentMiddleware (subagents.go): it provides the
// `task` tool, which dispatches a description to a compiled subagent —
// declared as a SubAgentSpec (subagent.go) or passed pre-built as a
// *CompiledSubAgent, with GeneralPurposeSubAgent as the auto-added default —
// runs it as an isolated graph on a per-call thread, and writes the
// subagent's changed public state back to the parent through the tool
// result's Command artifact.
//
// The package depends on langchain's agent middleware, imported under the
// agentsmw alias because both packages are named middleware:
//
//	import agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
//
// The dependency is one-directional: langchain/agents/middleware predates
// this package and cannot import it.
package middleware
