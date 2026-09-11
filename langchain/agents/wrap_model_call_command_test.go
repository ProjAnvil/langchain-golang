package agents

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// commandWrapMiddleware is a WrapModelCallResultHook middleware that returns
// its inner handler's response wrapped in an ExtendedModelResponse carrying a
// fixed Command, mirroring Python middleware whose wrap_model_call returns
// `ExtendedModelResponse(model_response=..., command=Command(update=...))`.
type commandWrapMiddleware struct {
	command *middleware.Command
}

// Name distinguishes instances for CreateAgent's duplicate-middleware check
// (derived from the wrapped command's address; a nil command falls back to
// the type name).
func (m *commandWrapMiddleware) Name() string {
	if m.command == nil {
		return "commandWrapMiddleware"
	}
	return fmt.Sprintf("commandWrapMiddleware(%p)", m.command)
}

func (m *commandWrapMiddleware) WrapModelCallResult(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
	resp, err := handler(ctx, request)
	if err != nil {
		return nil, err
	}
	return middleware.ExtendedModelResponse{ModelResponse: resp, Command: m.command}, nil
}

// TestWrapModelCallCommandUpdateAppliedToState: an update-only Command
// returned from wrap_model_call is applied to the final graph state in
// addition to the node's default messages update (mirroring
// factory._build_commands' `commands.extend`, factory.py:227-232).
func TestWrapModelCallCommandUpdateAppliedToState(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("model output")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(&commandWrapMiddleware{
		command: &middleware.Command{Update: map[string]any{"foo": "bar"}},
	}))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if state["foo"] != "bar" {
		t.Fatalf("expected state foo=bar from middleware Command update, got %#v", state["foo"])
	}
	msgs, ok := state["messages"].([]messages.Message)
	if !ok || len(msgs) != 2 || msgs[1].Content != "model output" {
		t.Fatalf("expected normal messages update alongside the command update, got %#v", state["messages"])
	}
}

// TestWrapModelCallCommandRoutingNotSupported: a wrap_model_call Command
// carrying goto/resume/graph fails the run with an error naming the offending
// field (mirroring factory._build_commands' NotImplementedError,
// factory.py:210-216).
func TestWrapModelCallCommandRoutingNotSupported(t *testing.T) {
	cases := []struct {
		name    string
		command *middleware.Command
		wantSub string
	}{
		{"goto", &middleware.Command{Goto: "tools"}, "goto"},
		{"resume", &middleware.Command{Resume: "resumed"}, "resume"},
		{"graph", &middleware.Command{Graph: "other"}, "graph"},
	}
	for _, tc := range cases {
		model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}
		agent, err := CreateAgent(model, nil, WithAgentMiddleware(&commandWrapMiddleware{command: tc.command}))
		if err != nil {
			t.Fatalf("%s: create agent: %v", tc.name, err)
		}
		_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
		if err == nil || !strings.Contains(err.Error(), tc.wantSub) ||
			!strings.Contains(err.Error(), "wrap_model_call") {
			t.Fatalf("%s: expected unsupported-command error containing %q, got %v", tc.name, tc.wantSub, err)
		}
	}
}

// TestWrapModelCallCommandMessagesUpdateFollowsPythonOrder: when the
// middleware Command updates the "messages" key, Python semantics are the two
// sequential Commands (default first, middleware second) applied by the
// add_messages reducer — so the middleware message lands AFTER the model
// output, and both survive.
func TestWrapModelCallCommandMessagesUpdateFollowsPythonOrder(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("model output")}}
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(&commandWrapMiddleware{
		command: &middleware.Command{Update: map[string]any{
			"messages": []messages.Message{messages.AI("override")},
		}},
	}))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	msgs, ok := state["messages"].([]messages.Message)
	if !ok || len(msgs) != 3 {
		t.Fatalf("expected 3 messages (human, model output, override), got %#v", state["messages"])
	}
	want := []string{"hi", "model output", "override"}
	for i, w := range want {
		if msgs[i].Content != w {
			t.Fatalf("message %d: want %q, got %#v", i, w, msgs[i])
		}
	}
}

// TestWrapModelCallCommandUpdateAppliedStreaming: the streaming path
// (invokeModelStreaming behind an active event sink) flows through the same
// WrapModelCall composition boundary, so an update-only Command applies there
// too — asserted via the terminal StreamEnd event's final state.
func TestWrapModelCallCommandUpdateAppliedStreaming(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(
		messages.AI("stream"),
		messages.AI(" output"),
	))
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(&commandWrapMiddleware{
		command: &middleware.Command{Update: map[string]any{"foo": "bar"}},
	}))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	stream, err := agent.StreamEvents(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream events: %v", err)
	}
	defer stream.Close()

	var end *StreamEvent
	for _, ev := range drainStream(t, stream) {
		if ev.Type == StreamEnd {
			e := ev
			end = &e
		}
	}
	if end == nil {
		t.Fatal("no StreamEnd event")
	}
	if end.Err != nil {
		t.Fatalf("stream end error: %v", end.Err)
	}
	if end.State["foo"] != "bar" {
		t.Fatalf("expected final streaming state foo=bar, got %#v", end.State["foo"])
	}
	msgs, ok := end.State["messages"].([]messages.Message)
	if !ok || len(msgs) != 2 || msgs[1].Content != "stream output" {
		t.Fatalf("expected assembled streamed message in final state, got %#v", end.State["messages"])
	}
}

// TestWrapModelCallCommandMultiLayerOrdering: with two wrap_model_call layers
// each returning a Command, both updates apply; commands accumulate
// inner-first then outer (factory._chain_model_call_handlers,
// factory.py:236-333), so on a conflicting key the outer middleware's write —
// applied later — wins, exactly like Python's sequential Commands.
func TestWrapModelCallCommandMultiLayerOrdering(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}
	inner := &commandWrapMiddleware{command: &middleware.Command{Update: map[string]any{
		"inner": "1", "winner": "inner",
	}}}
	outer := &commandWrapMiddleware{command: &middleware.Command{Update: map[string]any{
		"outer": "2", "winner": "outer",
	}}}
	// First middleware in the list is the outermost wrap layer.
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(outer, inner))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if state["inner"] != "1" || state["outer"] != "2" {
		t.Fatalf("expected both layers' updates applied, got inner=%#v outer=%#v", state["inner"], state["outer"])
	}
	if state["winner"] != "outer" {
		t.Fatalf("expected outer (later-applied) command to win the conflict, got %#v", state["winner"])
	}
}

// TestWrapModelCallCommandAppliedOnStructuredOutputPath: a matched structured
// response ends the run through detectStructuredOutput's terminal Command;
// the middleware Command's update must still be applied on that exit path,
// alongside the structured_response.
func TestWrapModelCallCommandAppliedOnStructuredOutputPath(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI(`{"temperature":72,"condition":"sunny"}`)}}
	agent, err := CreateAgent(model, nil,
		WithAgentResponseFormat(NewProviderStrategy(weatherSchema())),
		WithAgentMiddleware(&commandWrapMiddleware{
			command: &middleware.Command{Update: map[string]any{"foo": "bar"}},
		}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	state, err := agent.InvokeWithState(context.Background(), []messages.Message{messages.Human("weather in Tokyo?")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	structured, ok := state["structured_response"].(map[string]any)
	if !ok || structured["condition"] != "sunny" {
		t.Fatalf("expected structured_response.condition=sunny, got %#v", state["structured_response"])
	}
	if state["foo"] != "bar" {
		t.Fatalf("expected middleware Command update applied on the structured exit path, got %#v", state["foo"])
	}
}
