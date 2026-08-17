package agents

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

func TestWithAgentContextSchemaRecordsFields(t *testing.T) {
	fields := []ContextField{
		{Name: "user_id"},
		{Name: "tenant", Type: "string"},
	}
	opts := AgentOptions{}
	WithAgentContextSchema(fields...)(&opts)
	if len(opts.ContextSchema) != 2 || opts.ContextSchema[0].Name != "user_id" || opts.ContextSchema[1].Name != "tenant" {
		t.Fatalf("context schema not recorded: %#v", opts.ContextSchema)
	}

	// The last call wins, replacing any previously declared schema.
	WithAgentContextSchema(ContextField{Name: "request_id"})(&opts)
	if len(opts.ContextSchema) != 1 || opts.ContextSchema[0].Name != "request_id" {
		t.Fatalf("expected last WithAgentContextSchema call to replace the schema: %#v", opts.ContextSchema)
	}
}

func TestWithAgentContextSchemaAcceptedByCreateAgent(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	agent, err := CreateAgent(model, nil,
		WithAgentContextSchema(ContextField{Name: "user_id"}),
	)
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(WithContextValues(context.Background(), map[string]any{"user_id": "u1"}),
		[]messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) == 0 || out[len(out)-1].Content != "done" {
		t.Fatalf("unexpected output: %#v", out)
	}
}

func TestRuntimeValueReadsContextBag(t *testing.T) {
	rt := runtime.Runtime{Context: map[string]any{"user_id": "u1"}}
	v, ok := RuntimeValue(rt, "user_id")
	if !ok || v != "u1" {
		t.Fatalf("expected (u1, true), got (%v, %v)", v, ok)
	}
	if v, ok := RuntimeValue(rt, "missing"); ok || v != nil {
		t.Fatalf("expected (nil, false) for missing key, got (%v, %v)", v, ok)
	}

	// A Runtime without a context bag yields (nil, false) for every key.
	empty := runtime.Runtime{}
	if v, ok := RuntimeValue(empty, "user_id"); ok || v != nil {
		t.Fatalf("expected (nil, false) without a context bag, got (%v, %v)", v, ok)
	}
}
