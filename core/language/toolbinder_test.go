package language

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// mustAdderTool builds a minimal invocable tool for bind assertions.
func mustAdderTool(t *testing.T) tools.Tool {
	t.Helper()
	tool, err := tools.NewFunc(
		"adder",
		"adds integers",
		schema.Object(map[string]schema.Schema{}),
		func(_ context.Context, _ map[string]any) (tools.Result, error) {
			return tools.Result{Content: "3"}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}
	return tool
}

// TestFakeChatModelImplementsToolBinder guards the optional capability
// interface: FakeChatModel must satisfy ToolBinder so agent code can thread
// bind_tools options (tool_choice) into test doubles.
func TestFakeChatModelImplementsToolBinder(t *testing.T) {
	var _ ToolBinder = NewFakeChatModel()
}

// TestFakeChatModelBindToolsWithOptionsRecordsToolChoice covers the recording
// contract: BindToolsWithOptions stores the received ToolChoice where tests
// can assert it (BoundToolChoice on the receiver and on the returned copy).
func TestFakeChatModelBindToolsWithOptionsRecordsToolChoice(t *testing.T) {
	model := NewFakeChatModel(WithCapabilities(ChatModelCapabilities{ToolCalling: true}))

	boundAny, err := model.BindToolsWithOptions([]tools.Tool{mustAdderTool(t)}, BindToolsOptions{
		ToolChoice: ToolChoiceAny,
	})
	if err != nil {
		t.Fatalf("bind tools with options: %v", err)
	}
	bound, ok := boundAny.(*FakeChatModel)
	if !ok {
		t.Fatalf("bound model type: %T", boundAny)
	}
	if got := bound.BoundToolChoice(); got != ToolChoiceAny {
		t.Fatalf("bound copy ToolChoice = %q, want %q", got, ToolChoiceAny)
	}
	if got := model.BoundToolChoice(); got != ToolChoiceAny {
		t.Fatalf("receiver ToolChoice = %q, want %q (must record for assertions)", got, ToolChoiceAny)
	}
	if len(bound.BoundTools()) != 1 {
		t.Fatalf("bound tools: got %d want 1", len(bound.BoundTools()))
	}
}

// TestFakeChatModelBindToolsWithOptionsZeroEquivalence pins the godoc
// contract: BindTools(tools) is equivalent to BindToolsWithOptions(tools,
// BindToolsOptions{}) — same bound tools, zero (unconstrained) tool choice.
func TestFakeChatModelBindToolsWithOptionsZeroEquivalence(t *testing.T) {
	model := NewFakeChatModel(WithCapabilities(ChatModelCapabilities{ToolCalling: true}))

	viaBase, err := model.BindTools([]tools.Tool{mustAdderTool(t)})
	if err != nil {
		t.Fatalf("BindTools: %v", err)
	}
	viaOptions, err := model.BindToolsWithOptions([]tools.Tool{mustAdderTool(t)}, BindToolsOptions{})
	if err != nil {
		t.Fatalf("BindToolsWithOptions: %v", err)
	}

	base, ok := viaBase.(*FakeChatModel)
	if !ok {
		t.Fatalf("base bound model type: %T", viaBase)
	}
	opts, ok := viaOptions.(*FakeChatModel)
	if !ok {
		t.Fatalf("options bound model type: %T", viaOptions)
	}
	if len(base.BoundTools()) != len(opts.BoundTools()) {
		t.Fatalf("bound tools mismatch: %d vs %d", len(base.BoundTools()), len(opts.BoundTools()))
	}
	if got := opts.BoundToolChoice(); got != "" {
		t.Fatalf("zero-value options ToolChoice = %q, want empty", got)
	}
	if got := base.BoundToolChoice(); got != "" {
		t.Fatalf("BindTools ToolChoice = %q, want empty", got)
	}
}

// TestFakeChatModelBindToolsWithOptionsUnsupported keeps the capability gate
// aligned with BindTools: no ToolCalling capability + tools => error.
func TestFakeChatModelBindToolsWithOptionsUnsupported(t *testing.T) {
	model := NewFakeChatModel()
	_, err := model.BindToolsWithOptions([]tools.Tool{mustAdderTool(t)}, BindToolsOptions{
		ToolChoice: ToolChoiceAny,
	})
	if err == nil {
		t.Fatal("expected unsupported tool calling error")
	}
}

// TestBindToolsOptionsParallelToolCallsField sanity-checks the struct surface
// (pointer so nil = provider default survives round-trips).
func TestBindToolsOptionsParallelToolCallsField(t *testing.T) {
	opts := BindToolsOptions{}
	if opts.ParallelToolCalls != nil {
		t.Fatal("zero-value ParallelToolCalls must be nil")
	}
	disable := false
	opts.ParallelToolCalls = &disable
	if opts.ParallelToolCalls == nil || *opts.ParallelToolCalls {
		t.Fatal("ParallelToolCalls round-trip failed")
	}
}
