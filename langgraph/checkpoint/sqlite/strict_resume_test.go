package sqlite_test

import (
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Strict resume mode (#9574) saver coverage: these tests mirror the
// table-driven strict-resume cases of langgraph/graph/strict_resume_test.go
// against the sqlite saver — the reserved-slot write mapping (interrupt at
// idx -3, resume at idx -4, INSERT OR REPLACE) is the shape where a stale or
// malformed resume must still be rejected BEFORE anything is written. The
// graph package's own tests cannot import this module (this module requires
// the root one), so the sqlite leg lives here.

// strictResumeAskGraph builds a one-node graph whose node interrupts once and
// echoes the resume value into state["answer"].
func strictResumeAskGraph(t *testing.T, saver checkpoint.Saver) *graph.CompiledGraph {
	t.Helper()
	g := graph.NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := graph.Interrupt(rt, "q1")
		return map[string]any{"answer": v}, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// threadCheckpointCount counts the checkpoints stored for thread, used to pin
// the strict-resume "fails BEFORE anything is written" guarantee.
func threadCheckpointCount(t *testing.T, saver checkpoint.Saver, thread string) int {
	t.Helper()
	tups, err := saver.List(t.Context(), checkpoint.Config{ThreadID: thread}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	return len(tups)
}

// TestStrictResumeUnknownIDSqlite is the sqlite leg of
// TestStrictResumeUnknownID: a resume map naming no pending interrupt fails
// with *types.UnknownInterruptError (Reason "not_pending") and writes no new
// checkpoint.
func TestStrictResumeUnknownIDSqlite(t *testing.T) {
	ctx := t.Context()
	saver := newSaver(t, dbPath(t))
	cg := strictResumeAskGraph(t, saver)
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, graph.Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != "ask-1" {
		t.Fatalf("Interrupts = %+v, want one interrupt (id ask-1)", res.Interrupts)
	}

	before := threadCheckpointCount(t, saver, "t")
	_, err = cg.InvokeWithOptions(ctx, nil, graph.Options{
		ThreadID:     "t",
		Resume:       map[string]any{"bogus-id": 42},
		StrictResume: true,
	})
	var uiErr *types.UnknownInterruptError
	if !errors.As(err, &uiErr) {
		t.Fatalf("resume error = %v (%T), want *types.UnknownInterruptError", err, err)
	}
	if uiErr.Reason != "not_pending" {
		t.Fatalf("Reason = %q, want %q", uiErr.Reason, "not_pending")
	}
	if !reflect.DeepEqual(uiErr.InterruptIDs, []string{"bogus-id"}) {
		t.Fatalf("InterruptIDs = %v, want [bogus-id]", uiErr.InterruptIDs)
	}
	if !reflect.DeepEqual(uiErr.PendingIDs, []string{"ask-1"}) {
		t.Fatalf("PendingIDs = %v, want [ask-1]", uiErr.PendingIDs)
	}
	if got := threadCheckpointCount(t, saver, "t"); got != before {
		t.Fatalf("checkpoints after failed resume = %d, want %d (nothing written)", got, before)
	}
}

// TestStrictResumeEmptyMapSqlite is the sqlite leg of TestStrictResumeEmptyMap.
func TestStrictResumeEmptyMapSqlite(t *testing.T) {
	ctx := t.Context()
	saver := newSaver(t, dbPath(t))
	cg := strictResumeAskGraph(t, saver)
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, graph.Options{ThreadID: "t"}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	before := threadCheckpointCount(t, saver, "t")
	_, err := cg.InvokeWithOptions(ctx, nil, graph.Options{
		ThreadID:     "t",
		Resume:       map[string]any{},
		StrictResume: true,
	})
	var uiErr *types.UnknownInterruptError
	if !errors.As(err, &uiErr) {
		t.Fatalf("resume error = %v (%T), want *types.UnknownInterruptError", err, err)
	}
	if uiErr.Reason != "empty_map" {
		t.Fatalf("Reason = %q, want %q", uiErr.Reason, "empty_map")
	}
	if !reflect.DeepEqual(uiErr.PendingIDs, []string{"ask-1"}) {
		t.Fatalf("PendingIDs = %v, want [ask-1]", uiErr.PendingIDs)
	}
	if got := threadCheckpointCount(t, saver, "t"); got != before {
		t.Fatalf("checkpoints after failed resume = %d, want %d (nothing written)", got, before)
	}
}

// TestStrictResumeAmbiguousScalarSqlite is the sqlite leg of
// TestStrictResumeAmbiguousScalar: a scalar resume against two pending
// interrupts fails with Reason "ambiguous" under strict mode.
func TestStrictResumeAmbiguousScalarSqlite(t *testing.T) {
	ctx := t.Context()
	saver := newSaver(t, dbPath(t))
	g := graph.NewStateGraph()
	g.AddNode("start", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := graph.Interrupt(rt, "qa")
		return map[string]any{"ra": v}, nil
	})
	g.AddNode("b", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := graph.Interrupt(rt, "qb")
		return map[string]any{"rb": v}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "a")
	g.AddEdge("start", "b")
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, graph.Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 2 {
		t.Fatalf("Interrupts = %+v, want two pending interrupts", res.Interrupts)
	}

	_, err = cg.InvokeWithOptions(ctx, nil, graph.Options{
		ThreadID:     "t",
		Resume:       42,
		StrictResume: true,
	})
	var uiErr *types.UnknownInterruptError
	if !errors.As(err, &uiErr) {
		t.Fatalf("strict resume error = %v (%T), want *types.UnknownInterruptError", err, err)
	}
	if uiErr.Reason != "ambiguous" {
		t.Fatalf("Reason = %q, want %q", uiErr.Reason, "ambiguous")
	}
	if !reflect.DeepEqual(uiErr.PendingIDs, []string{"a-1", "b-1"}) {
		t.Fatalf("PendingIDs = %v, want [a-1 b-1]", uiErr.PendingIDs)
	}
}
