package graph

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// interruptSchemaGraph builds a single-node graph whose node interrupts with
// a response schema describing the expected resume value (Python langgraph
// 1.2.13 interrupt(response_schema=...), #8886).
func interruptSchemaGraph(t *testing.T, saver checkpoint.Saver, calls *int, schema map[string]any) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		*calls++
		v := InterruptWithSchema(rt, "pick one", schema)
		return map[string]any{"answer": v}, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

func TestInterruptResponseSchemaSurfaced(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"choice": map[string]any{"type": "string"}},
		"required":   []any{"choice"},
	}
	calls := 0
	cg := interruptSchemaGraph(t, checkpoint.NewMemorySaver(), &calls, schema)
	ctx := t.Context()

	result, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(result.Interrupts) != 1 {
		t.Fatalf("expected one interrupt, got %+v", result.Interrupts)
	}
	intr := result.Interrupts[0]
	if intr.Value != "pick one" {
		t.Fatalf("Value = %v, want pick one", intr.Value)
	}
	if intr.ResponseSchema == nil {
		t.Fatal("run-result interrupt missing ResponseSchema")
	}
	if got := intr.ResponseSchema["type"]; got != "object" {
		t.Fatalf("ResponseSchema[\"type\"] = %v, want object", got)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if len(snap.Interrupts) != 1 || snap.Interrupts[0].ResponseSchema == nil {
		t.Fatalf("GetState().Interrupts = %+v, want the schema carried", snap.Interrupts)
	}

	// Resume still delivers the value, and the schema does not interfere.
	resumed, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1", Resume: "B"})
	if err != nil {
		t.Fatalf("resume error = %v", err)
	}
	if resumed.Values["answer"] != "B" {
		t.Fatalf("resume produced %v, want B", resumed.Values["answer"])
	}
}

func TestInterruptWithoutSchemaStaysNil(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "plain")
		return map[string]any{"answer": v}, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	result, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t2"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(result.Interrupts) != 1 || result.Interrupts[0].ResponseSchema != nil {
		t.Fatalf("plain interrupt must keep ResponseSchema nil: %+v", result.Interrupts)
	}
}
