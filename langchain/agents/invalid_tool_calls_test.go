package agents

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
)

// TestCreateAgentRepairsInvalidToolCalls verifies that invalid tool calls on
// a model response are answered with error ToolMessages so the model receives
// corrective feedback on its next turn instead of the calls silently
// disappearing (Python langchain #40530).
func TestCreateAgentRepairsInvalidToolCalls(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			InvalidToolCalls: []messages.ToolCall{
				{ID: "bad_1", Name: "search", Args: map[string]any{"q": "unclosed"}},
			},
		},
		messages.AI("let me retry properly"),
	}}

	agent, err := CreateAgent(model, []coretools.Tool{newEchoTool(t)})
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	msgs, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// human, AI(invalid), error ToolMessage, AI(final)
	repairedIdx := -1
	for i := range msgs {
		if msgs[i].ToolCallID == "bad_1" {
			repairedIdx = i
		}
	}
	if repairedIdx < 0 {
		t.Fatalf("no ToolMessage answering the invalid call: %+v", msgs)
	}
	repaired := msgs[repairedIdx]
	if repaired.Role != messages.RoleTool {
		t.Fatalf("answering message role = %v, want tool", repaired.Role)
	}
	if repaired.Name != "search" {
		t.Fatalf("answering message Name = %q, want search", repaired.Name)
	}
	if repaired.ResponseMetadata["status"] != "error" {
		t.Fatalf("answering message status = %v, want error", repaired.ResponseMetadata["status"])
	}
	want := "Tool call search with id bad_1 could not be executed - arguments were malformed or truncated."
	if repaired.Content != want {
		t.Fatalf("answering message content = %q, want %q", repaired.Content, want)
	}
}

// TestCreateAgentSkipsUnanswerableAndAnsweredInvalidCalls covers the two
// guard rails: invalid calls without an ID cannot be answered; invalid calls
// already answered by a ToolMessage are not answered twice.
func TestCreateAgentSkipsUnanswerableAndAnsweredInvalidCalls(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{
		{
			Role: messages.RoleAI,
			InvalidToolCalls: []messages.ToolCall{
				{Name: "ghost"}, // no ID: not answerable
				{ID: "bad_2", Name: "search", Args: nil},
			},
		},
		messages.AI("final"),
	}}

	agent, err := CreateAgent(model, nil)
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	msgs, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	counts := map[string]int{}
	for _, m := range msgs {
		if m.Role == messages.RoleTool {
			counts[m.ToolCallID]++
		}
	}
	if counts[""] != 0 {
		t.Fatalf("ID-less invalid call must not produce a ToolMessage: %+v", msgs)
	}
	if counts["bad_2"] != 1 {
		t.Fatalf("bad_2 answered %d times, want exactly 1", counts["bad_2"])
	}
}
