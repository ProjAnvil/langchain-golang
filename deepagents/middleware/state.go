package middleware

import (
	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// ExcludedStateKeys are the agent-state keys excluded when passing state to
// subagents and when returning updates from subagents (consumed by the
// subagent middleware). It mirrors _EXCLUDED_STATE_KEYS in the Python
// reference, deepagents/middleware/subagents.py:447-456, verbatim, with that
// module's symbolic names resolved to their defining literals:
// _SKILL_TOOLS_DISCLOSED_KEY = "_skill_tools_disclosed"
// (deepagents/middleware/_skill_tools.py:43), _FORKED_CONTEXT_KEY =
// "_deepagents_forked_context" (subagents.py:74), and _RUBRIC_KEY = "rubric"
// (deepagents/middleware/rubric.py:121).
//
// Upstream rationale (subagents.py:457-476), condensed: the messages key is
// handled explicitly so only the subagent's final message returns; todos and
// structured_response have no defined reducer or subagent meaning;
// skills_metadata, _skill_tools_disclosed, and pinned_skills describe the
// parent agent's own skill state; _deepagents_forked_context marks forked
// subagents; and rubric grades the caller's whole goal, not the child's task.
var ExcludedStateKeys = []string{
	"messages",
	"todos",
	"structured_response",
	"skills_metadata",
	"pinned_skills",
	"_skill_tools_disclosed",
	"_deepagents_forked_context",
	"rubric",
}

// PrivateStateKeys are the agent-private state keys filtered from both
// subagent inputs and subagent outputs, in addition to ExcludedStateKeys.
// It is the Go counterpart of the keys upstream declares with
// PrivateStateAttr (subagents.py:740-749 collects them as
// private_state_keys); this port's one private key today is the HITL
// middleware's edited-tool-calls scratch state, which describes one agent's
// pending tool-call edits and must not leak across the parent/subagent
// boundary.
var PrivateStateKeys = []string{agentsmw.EditedToolCallsStateKey}
