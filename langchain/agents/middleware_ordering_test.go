package agents

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// fullStackRecorder implements all six hook interfaces (BeforeAgent / BeforeModel
// / WrapModelCall / WrapToolCall / AfterModel / AfterAgent), appending tagged
// events to a shared log. Used to assert the full composition order through
// CreateAgent in a single run. Extends the recordingWrapModelCallMiddleware
// pattern (create_agent_test.go:180-190) across every hook surface.
type fullStackRecorder struct {
	tag string
	log *[]string
}

func (r *fullStackRecorder) BeforeAgent(_ context.Context, _ map[string]any) (map[string]any, error) {
	*r.log = append(*r.log, r.tag+":before_agent")
	return nil, nil
}

func (r *fullStackRecorder) AfterAgent(_ context.Context, _ map[string]any) error {
	*r.log = append(*r.log, r.tag+":after_agent")
	return nil
}

func (r *fullStackRecorder) BeforeModel(_ context.Context, _ map[string]any) (map[string]any, error) {
	*r.log = append(*r.log, r.tag+":before_model")
	return nil, nil
}

func (r *fullStackRecorder) AfterModel(_ context.Context, _ map[string]any) (map[string]any, error) {
	*r.log = append(*r.log, r.tag+":after_model")
	return nil, nil
}

func (r *fullStackRecorder) WrapModelCall(
	ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler,
) (middleware.ModelResponse, error) {
	*r.log = append(*r.log, r.tag+":wrap_model:before")
	resp, err := handler(ctx, request)
	*r.log = append(*r.log, r.tag+":wrap_model:after")
	return resp, err
}

func (r *fullStackRecorder) WrapToolCall(
	ctx context.Context, request middleware.ToolCallRequest, handler middleware.ToolHandler,
) (messages.Message, error) {
	*r.log = append(*r.log, r.tag+":wrap_tool:before")
	resp, err := handler(ctx, request)
	*r.log = append(*r.log, r.tag+":wrap_tool:after")
	return resp, err
}

// TestCreateAgentFullStackMiddlewareOrdering asserts the full composition order
// across BeforeAgent → (BeforeModel → WrapModelCall → [WrapToolCall] →
// AfterModel) → AfterAgent when two middleware are stacked (first-listed is
// outermost, per Python's wrap_model_call docstring types.py:503).
func TestCreateAgentFullStackMiddlewareOrdering(t *testing.T) {
	echo := newEchoTool(t) // package-local helper (create_agent_test.go:89)
	// First model response requests the echo tool BY ITS ACTUAL NAME so the run
	// traverses the tool node (no hardcoded name — robust to newEchoTool's
	// choice); second is the plain final answer.
	model := &sequenceModel{responses: []messages.Message{
		aiWithToolCall("call_1", echo.Name()),
		messages.AI("done"),
	}}
	var log []string

	agent, err := CreateAgent(model, []tools.Tool{echo},
		WithAgentMiddleware(
			&fullStackRecorder{tag: "A", log: &log},
			&fullStackRecorder{tag: "B", log: &log},
		),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("go")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	// Assert the salient ordering properties (NOT exact-equality on the whole
	// log — the precise interleaving across two model-loop iterations is verbose;
	// the regression value is the outer→inner nesting and the before/after
	// bracketing of each surface).
	assertOrder := func(a, b string) {
		ai, bi := indexOrFatal(t, log, a), indexOrFatal(t, log, b)
		if ai >= bi {
			t.Fatalf("expected %q before %q; log=%v", a, b, log)
		}
	}

	// BeforeAgent fires once, before any model-loop hook.
	assertOrder("A:before_agent", "A:before_model")
	assertOrder("B:before_agent", "B:before_model")

	// First-listed is OUTERMOST on WrapModelCall (A wraps B wraps model).
	assertOrder("A:wrap_model:before", "B:wrap_model:before")
	assertOrder("B:wrap_model:after", "A:wrap_model:after")

	// WrapToolCall brackets the tool execution (one tool call in iteration 1),
	// with the same outer→inner nesting.
	assertOrder("A:wrap_tool:before", "B:wrap_tool:before")
	assertOrder("B:wrap_tool:after", "A:wrap_tool:after")

	// AfterAgent fires once per middleware AFTER all model-loop activity: every
	// ":after_agent" event must come after every ":after_model" event. (Does NOT
	// assume which middleware's AfterAgent runs last — that ordering is not part
	// of the documented contract.)
	maxAfterModel := -1
	for i, e := range log {
		if strings.HasSuffix(e, ":after_model") && i > maxAfterModel {
			maxAfterModel = i
		}
	}
	for i, e := range log {
		if strings.HasSuffix(e, ":after_agent") && i <= maxAfterModel {
			t.Fatalf("after_agent %q at %d must follow last after_model at %d; log=%v", e, i, maxAfterModel, log)
		}
	}
	if !hasEvent(log, "A:after_agent") || !hasEvent(log, "B:after_agent") {
		t.Fatalf("both A and B after_agent must fire; log=%v", log)
	}
}

func indexOrFatal(t *testing.T, log []string, want string) int {
	t.Helper()
	for i, e := range log {
		if e == want {
			return i
		}
	}
	t.Fatalf("event %q not in log: %v", want, log)
	return -1
}

func hasEvent(log []string, want string) bool {
	for _, e := range log {
		if e == want {
			return true
		}
	}
	return false
}

// aiWithToolCall builds an AI message that requests one tool call (drives the
// run into the tool node so WrapToolCall fires).
func aiWithToolCall(id, name string) messages.Message {
	m := messages.AI("")
	m.ToolCalls = []messages.ToolCall{{ID: id, Name: name, Args: map[string]any{}}}
	return m
}
