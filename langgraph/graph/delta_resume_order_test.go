package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// resumeOrderReducer concatenates a []string base with each []string update
// in the batch — an order-sensitive BatchReducer, so a doubled or permuted
// write changes the rebuilt value.
func resumeOrderReducer(existing any, updates []any) (any, error) {
	base, _ := existing.([]string)
	out := make([]string, len(base))
	copy(out, base)
	for _, u := range updates {
		add, _ := u.([]string)
		out = append(out, add...)
	}
	return out, nil
}

// resumeOrderGraph builds START -> fork -> (p || q) -> r over a DeltaChannel
// with snapshotFrequency=100 (it never snapshots, so the value lives only as
// pending writes reconstructed by walks). p completes in the paused
// superstep; q interrupts and produces its write only after the resume; r
// runs in a later superstep.
func resumeOrderGraph(t *testing.T, opts ...CompileOption) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(resumeOrderReducer, func() any { return []string{} }, 100))
	g.AddNode("fork", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("p", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"msgs": []string{"p"}}, nil
	})
	g.AddNode("q", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		if Interrupt(rt, "q-asks") != nil {
			return map[string]any{"msgs": []string{"q"}}, nil
		}
		return nil, nil
	})
	g.AddNode("r", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"msgs": []string{"r"}}, nil
	})
	g.AddEdge(types.START, "fork")
	g.AddEdge("fork", "p")
	g.AddEdge("fork", "q")
	g.AddEdge("p", "r")
	g.AddEdge("q", "r")
	g.AddEdge("r", types.END)
	cg, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestResumeAppliesCompletedSiblingDeltaWriteOnce pins the duplication half
// of Python #9114 for the Go resume path: the pause checkpoint holds the
// completed sibling p's delta write as a pending write. The resumed run
// hydrates delta channels from the ancestor history AND replays completed
// siblings' writes — seeding p's write into the hydration would apply it
// twice, and the live result would disagree with the persisted history
// (`[in p p q r]` vs `[in p q r]`).
func TestResumeAppliesCompletedSiblingDeltaWriteOnce(t *testing.T) {
	cg := resumeOrderGraph(t, WithCheckpointer(checkpoint.NewMemorySaver()))
	ctx := t.Context()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"in"}}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("pause Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "q-asks" {
		t.Fatalf("expected one q-asks interrupt, got %+v", first.Interrupts)
	}

	second, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "answer"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	want := []string{"in", "p", "q", "r"}
	if got, _ := second.Values["msgs"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("live msgs = %v, want %v (completed sibling p applied twice)", got, want)
	}

	// The persisted history must agree: GetState replays the stored writes,
	// so a doubled live application or a reordered store both show here.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["msgs"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState msgs = %v, want %v", got, want)
	}
}
