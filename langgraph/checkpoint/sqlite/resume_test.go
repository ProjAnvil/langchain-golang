package sqlite_test

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestResumeChainedInterruptPrefixSurvivesSlotCollapse is the regression test
// for the sqlite reserved-slot collapse: __resume__ maps to the fixed write
// idx -4 (INSERT OR REPLACE, last-write-wins), so persisting one write per
// consumed resume value would keep only the LAST value. The executor instead
// persists ONE write carrying the whole ordered prefix list (Python parity,
// types.py:905-925), which this test pins end to end: a single node chains
// three sequential interrupts across four invocations, and the third pause
// checkpoint must carry the full ["a","b"] prefix so the chain completes in
// order.
func TestResumeChainedInterruptPrefixSurvivesSlotCollapse(t *testing.T) {
	ctx := context.Background()
	saver := newSaver(t, dbPath(t))

	g := graph.NewStateGraph()
	g.AddNode("chain", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		a := graph.Interrupt(rt, "q0")
		b := graph.Interrupt(rt, "q1")
		c := graph.Interrupt(rt, "q2")
		return map[string]any{"data": fmt.Sprintf("%v,%v,%v", a, b, c)}, nil
	})
	g.AddEdge(types.START, "chain")
	g.AddEdge("chain", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	r1, err := cg.InvokeWithOptions(ctx, map[string]any{}, graph.Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("invoke 1: %v", err)
	}
	if len(r1.Interrupts) != 1 || r1.Interrupts[0].Value != "q0" {
		t.Fatalf("invoke 1 Interrupts = %+v, want one interrupt (q0)", r1.Interrupts)
	}
	r2, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t", Resume: "a"})
	if err != nil {
		t.Fatalf("invoke 2: %v", err)
	}
	if len(r2.Interrupts) != 1 || r2.Interrupts[0].Value != "q1" {
		t.Fatalf("invoke 2 Interrupts = %+v, want one interrupt (q1)", r2.Interrupts)
	}
	r3, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t", Resume: "b"})
	if err != nil {
		t.Fatalf("invoke 3: %v", err)
	}
	if len(r3.Interrupts) != 1 || r3.Interrupts[0].Value != "q2" {
		t.Fatalf("invoke 3 Interrupts = %+v, want one interrupt (q2)", r3.Interrupts)
	}

	// The third pause checkpoint carries the two-value consumed prefix as ONE
	// full-list ReservedResume write — the shape that survives idx -4.
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	var resumes []checkpoint.Write
	sawInterrupt := false
	for _, w := range tup.PendingWrites {
		switch w.Channel {
		case checkpoint.ReservedInterrupt:
			sawInterrupt = true
		case checkpoint.ReservedResume:
			resumes = append(resumes, w)
		}
	}
	if !sawInterrupt {
		t.Fatal("pause checkpoint pending writes missing the ReservedInterrupt write")
	}
	if len(resumes) != 1 {
		t.Fatalf("ReservedResume writes = %+v, want exactly ONE full-list write", resumes)
	}
	if want := []any{"a", "b"}; !reflect.DeepEqual(resumes[0].Value, want) {
		t.Fatalf("ReservedResume value = %v (%T), want the full prefix %v (slot collapse?)", resumes[0].Value, resumes[0].Value, want)
	}

	r4, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t", Resume: "c"})
	if err != nil {
		t.Fatalf("invoke 4: %v", err)
	}
	if len(r4.Interrupts) != 0 {
		t.Fatalf("invoke 4 Interrupts = %+v, want none (run must complete)", r4.Interrupts)
	}
	if r4.Values["data"] != "a,b,c" {
		t.Fatalf("data = %v, want %q", r4.Values["data"], "a,b,c")
	}
}
