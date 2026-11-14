// Package deepagents assembles Deep Agents — tool-calling agents with a
// virtual filesystem, ephemeral subagents, context summarization, and
// permission-gated human approval — on top of langchain/agents.CreateAgent.
//
// It is the Go port of the Python deepagents package, pinned to upstream
// v0.7.38 and scoped to its MVP Tier 1-4 subset:
//
//   - deepagents/backends: the Backend interface, the on-disk
//     FilesystemBackend, and the state-backed StateBackend with the files
//     MapMergeReducer;
//   - deepagents/middleware: FilesystemMiddleware (seven file tools,
//     permission evaluation, state-view injection, parallel-mutation guard),
//     permission-derived HITL interrupts, SubAgentSpec compilation, and the
//     SubAgentMiddleware task tool;
//   - this package: the CreateDeepAgent assembler (agent.go) that stacks the
//     middleware and delegates to agents.CreateAgent.
//
// Upstream file:line citations across these packages refer to that pinned
// version. Start with CreateDeepAgent and DeepAgentOptions in agent.go, and
// examples/deepagents/main.go for a runnable end-to-end example.
//
// Behavior authority and audit trail: the implementation plan at
// docs/superpowers/plans/2026-11-14-deepagents-port.md, the per-task briefs
// and reports under .superpowers/sdd/2026-11-14-deepagents-port/, and the
// spec's divergence register mirrored in DIVERGENCES.md (section
// "deepagents", D1-D14 plus the port's parked items).
package deepagents
