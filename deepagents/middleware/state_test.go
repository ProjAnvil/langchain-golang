package middleware

import (
	"reflect"
	"testing"

	agentsmw "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// TestExcludedStateKeys pins the content of ExcludedStateKeys to the values
// of _EXCLUDED_STATE_KEYS in the Python reference,
// deepagents/middleware/subagents.py:447-456, with that module's symbolic
// names resolved to their defining literals.
func TestExcludedStateKeys(t *testing.T) {
	want := []string{
		"messages",
		"todos",
		"structured_response",
		"skills_metadata",
		"pinned_skills",
		"_skill_tools_disclosed",
		"_deepagents_forked_context",
		"rubric",
	}
	if !reflect.DeepEqual(ExcludedStateKeys, want) {
		t.Fatalf("ExcludedStateKeys = %v, want %v", ExcludedStateKeys, want)
	}
}

// TestExcludedStateKeysAreASet guards against duplicated entries slipping in.
func TestExcludedStateKeysAreASet(t *testing.T) {
	seen := make(map[string]bool, len(ExcludedStateKeys))
	for _, key := range ExcludedStateKeys {
		if seen[key] {
			t.Fatalf("ExcludedStateKeys contains a duplicate: %q", key)
		}
		seen[key] = true
	}
}

// TestPrivateStateKeys pins the agent-private state keys: the HITL
// middleware's edited-tool-calls scratch state, whose key constant lives in
// langchain/agents/middleware (human_in_the_loop.go).
func TestPrivateStateKeys(t *testing.T) {
	want := []string{agentsmw.EditedToolCallsStateKey}
	if !reflect.DeepEqual(PrivateStateKeys, want) {
		t.Fatalf("PrivateStateKeys = %v, want %v", PrivateStateKeys, want)
	}
	// Pin the constant itself so an upstream rename surfaces here.
	if agentsmw.EditedToolCallsStateKey != "hitl_edited_tool_calls" {
		t.Fatalf("EditedToolCallsStateKey = %q, want %q",
			agentsmw.EditedToolCallsStateKey, "hitl_edited_tool_calls")
	}
}

// TestPrivateStateKeysDisjointFromExcluded keys documents the two lists'
// relationship: private keys are filtered in addition to excluded keys when
// subagent state is built, so an overlap would be a modeling bug.
func TestPrivateStateKeysDisjointFromExcluded(t *testing.T) {
	excluded := make(map[string]bool, len(ExcludedStateKeys))
	for _, key := range ExcludedStateKeys {
		excluded[key] = true
	}
	for _, key := range PrivateStateKeys {
		if excluded[key] {
			t.Fatalf("key %q appears in both ExcludedStateKeys and PrivateStateKeys", key)
		}
	}
}
