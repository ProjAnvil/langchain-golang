package graph

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Strict resume mode (langgraph 1.2.20, #9574, CONFIG_KEY_STRICT_RESUME):
// opting in via Options.StrictResume makes a resume that names an interrupt
// which is not pending (stale/typo'd id), an empty resume map, or a scalar
// resume meeting several pending interrupts fail with
// *types.UnknownInterruptError BEFORE anything is written. Default
// (non-strict) behavior is unchanged: unknown map keys are silently ignored
// and the ambiguous scalar case keeps its plain error.
//
// Saver coverage: the table-driven cases below run against the in-memory
// saver here; the same cases run against the sqlite saver in the sqlite
// module (langgraph/checkpoint/sqlite/strict_resume_test.go) — the sqlite
// saver lives in a nested module that depends on this one, so it cannot be
// imported from this package's tests.

// strictResumeAskGraph builds a one-node graph whose node interrupts once and
// echoes the resume value into state["answer"].
func strictResumeAskGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "q1")
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

// threadSaverFingerprint captures a saver's stored state for thread as one
// comparable string: the ordered set of stored checkpoint IDs, plus — for the
// latest (head) tuple — the head checkpoint ID, its full checkpoint content
// (including Next), and its serialized pending writes. Comparing fingerprints
// around a rejected resume pins the "fails BEFORE anything is written"
// guarantee: a bare checkpoint-count check would miss a same-ID overwrite or
// a stray PutWrites against an existing checkpoint.
func threadSaverFingerprint(t *testing.T, saver checkpoint.Saver, thread string) string {
	t.Helper()
	ctx := t.Context()
	tups, err := saver.List(ctx, checkpoint.Config{ThreadID: thread}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	var sb strings.Builder
	for _, tup := range tups {
		fmt.Fprintf(&sb, "cp:%s;", tup.Config.CheckpointID)
	}
	head, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: thread})
	if err != nil {
		t.Fatalf("GetTuple() error = %v", err)
	}
	if head != nil {
		fmt.Fprintf(&sb, "head:%s;ckpt:%#v;", head.Config.CheckpointID, head.Checkpoint)
		for _, w := range head.PendingWrites {
			fmt.Fprintf(&sb, "pw:%s/%s/%#v;", w.TaskID, w.Channel, w.Value)
		}
	}
	return sb.String()
}

// TestStrictResumeUnknownID: resuming with a map key that names no pending
// interrupt fails with *types.UnknownInterruptError (Reason "not_pending")
// and writes no new checkpoint.
func TestStrictResumeUnknownID(t *testing.T) {
	savers := []struct {
		name  string
		saver checkpoint.Saver
	}{
		{"memory", checkpoint.NewMemorySaver()},
	}
	for _, tc := range savers {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			cg := strictResumeAskGraph(t, tc.saver)
			res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
			if err != nil {
				t.Fatalf("first Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 1 || res.Interrupts[0].ID != "ask-1" {
				t.Fatalf("Interrupts = %+v, want one interrupt (id ask-1)", res.Interrupts)
			}

			before := threadSaverFingerprint(t, tc.saver, "t")
			_, err = cg.InvokeWithOptions(ctx, nil, Options{
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
			if want := "Unknown or already-consumed interrupt: bogus-id"; uiErr.Error() != want {
				t.Fatalf("Error() = %q, want %q", uiErr.Error(), want)
			}
			if got := threadSaverFingerprint(t, tc.saver, "t"); got != before {
				t.Fatalf("saver fingerprint changed after failed resume\nbefore: %s\nafter:  %s", before, got)
			}
		})
	}
}

// TestStrictResumeAnsweredID: an id that WAS pending in an earlier pause but
// is already consumed by the time of this resume is "not pending" too.
func TestStrictResumeAnsweredID(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("chain", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		a := Interrupt(rt, "q0")
		b := Interrupt(rt, "q1")
		return map[string]any{"data": fmt.Sprintf("%v,%v", a, b)}, nil
	})
	g.AddEdge(types.START, "chain")
	g.AddEdge("chain", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil || len(res.Interrupts) != 1 || res.Interrupts[0].ID != "chain-1" {
		t.Fatalf("first Invoke() = (%+v, %v), want one interrupt (chain-1)", res.Interrupts, err)
	}
	// Answer the first interrupt for real — this resume must succeed.
	res, err = cg.InvokeWithOptions(ctx, nil, Options{
		ThreadID:     "t",
		Resume:       map[string]any{"chain-1": "a"},
		StrictResume: true,
	})
	if err != nil {
		t.Fatalf("second Invoke() error = %v, want pause at q1", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != "chain-2" {
		t.Fatalf("second Invoke() Interrupts = %+v, want one interrupt (chain-2)", res.Interrupts)
	}

	// chain-1 is already consumed: re-answering it is not_pending — and the
	// rejection must leave the saver byte-identical (no same-ID overwrite, no
	// stray PutWrites).
	before := threadSaverFingerprint(t, saver, "t")
	_, err = cg.InvokeWithOptions(ctx, nil, Options{
		ThreadID:     "t",
		Resume:       map[string]any{"chain-1": "again"},
		StrictResume: true,
	})
	var uiErr *types.UnknownInterruptError
	if !errors.As(err, &uiErr) {
		t.Fatalf("resume error = %v (%T), want *types.UnknownInterruptError", err, err)
	}
	if uiErr.Reason != "not_pending" {
		t.Fatalf("Reason = %q, want %q", uiErr.Reason, "not_pending")
	}
	if !reflect.DeepEqual(uiErr.InterruptIDs, []string{"chain-1"}) {
		t.Fatalf("InterruptIDs = %v, want [chain-1]", uiErr.InterruptIDs)
	}
	if !reflect.DeepEqual(uiErr.PendingIDs, []string{"chain-2"}) {
		t.Fatalf("PendingIDs = %v, want [chain-2]", uiErr.PendingIDs)
	}
	if got := threadSaverFingerprint(t, saver, "t"); got != before {
		t.Fatalf("saver fingerprint changed after failed resume\nbefore: %s\nafter:  %s", before, got)
	}
}

// TestStrictResumeEmptyMap: an empty resume map under strict mode fails with
// Reason "empty_map" and writes nothing.
func TestStrictResumeEmptyMap(t *testing.T) {
	savers := []struct {
		name  string
		saver checkpoint.Saver
	}{
		{"memory", checkpoint.NewMemorySaver()},
	}
	for _, tc := range savers {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			cg := strictResumeAskGraph(t, tc.saver)
			if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"}); err != nil {
				t.Fatalf("first Invoke() error = %v", err)
			}
			before := threadSaverFingerprint(t, tc.saver, "t")
			_, err := cg.InvokeWithOptions(ctx, nil, Options{
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
			if want := "The resume map names no interrupt"; uiErr.Error() != want {
				t.Fatalf("Error() = %q, want %q", uiErr.Error(), want)
			}
			if got := threadSaverFingerprint(t, tc.saver, "t"); got != before {
				t.Fatalf("saver fingerprint changed after failed resume\nbefore: %s\nafter:  %s", before, got)
			}
		})
	}
}

// strictResumeDualInterruptGraph builds a graph whose two parallel nodes each
// interrupt in the same superstep, leaving two pending interrupts.
func strictResumeDualInterruptGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("start", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "qa")
		return map[string]any{"ra": v}, nil
	})
	g.AddNode("b", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "qb")
		return map[string]any{"rb": v}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "a")
	g.AddEdge("start", "b")
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestStrictResumeAmbiguousScalar: a scalar resume against two pending
// interrupts fails with Reason "ambiguous" under strict mode; the same input
// without strict mode keeps the existing plain error (now carrying the #9590
// docs URL), not an UnknownInterruptError.
func TestStrictResumeAmbiguousScalar(t *testing.T) {
	savers := []struct {
		name  string
		saver checkpoint.Saver
	}{
		{"memory", checkpoint.NewMemorySaver()},
	}
	for _, tc := range savers {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			cg := strictResumeDualInterruptGraph(t, tc.saver)
			res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
			if err != nil {
				t.Fatalf("first Invoke() error = %v", err)
			}
			if len(res.Interrupts) != 2 {
				t.Fatalf("Interrupts = %+v, want two pending interrupts", res.Interrupts)
			}

			before := threadSaverFingerprint(t, tc.saver, "t")
			_, err = cg.InvokeWithOptions(ctx, nil, Options{
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
			if got := threadSaverFingerprint(t, tc.saver, "t"); got != before {
				t.Fatalf("saver fingerprint changed after failed strict resume\nbefore: %s\nafter:  %s", before, got)
			}
			wantMsg := "When there are multiple pending interrupts, you must specify the interrupt id when resuming. " +
				"Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts."
			if uiErr.Error() != wantMsg {
				t.Fatalf("Error() = %q, want %q", uiErr.Error(), wantMsg)
			}

			// Non-strict: same input keeps the existing plain error (#9590
			// appends the docs URL), NOT an UnknownInterruptError.
			_, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: 42})
			if err == nil {
				t.Fatal("non-strict scalar resume with two pending interrupts: error = nil, want the map-required error")
			}
			if errors.As(err, &uiErr) {
				t.Fatalf("non-strict resume error = %v, must NOT be *types.UnknownInterruptError", err)
			}
			if !strings.Contains(err.Error(), "requires a map[string]any") {
				t.Fatalf("non-strict resume error = %q, want the map-required message", err)
			}
			if !strings.Contains(err.Error(), "Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts.") {
				t.Fatalf("non-strict resume error = %q, want the #9590 docs URL appended", err)
			}
		})
	}
}

// TestNonStrictResumeUnknownIDIgnored: without StrictResume, an unknown map
// key keeps today's behavior — silently ignored, the pending interrupt
// re-fires and the run re-pauses with it.
func TestNonStrictResumeUnknownIDIgnored(t *testing.T) {
	ctx := t.Context()
	cg := strictResumeAskGraph(t, checkpoint.NewMemorySaver())
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(ctx, nil, Options{
		ThreadID: "t",
		Resume:   map[string]any{"bogus-id": 42},
	})
	if err != nil {
		t.Fatalf("resume error = %v, want a silent re-pause", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != "ask-1" {
		t.Fatalf("Interrupts = %+v, want the same interrupt re-fired (ask-1)", res.Interrupts)
	}
}

// TestStaticBreakpointScalarResumeAllowed: a run paused by interrupt_before
// has exactly one pending (boundary) interrupt; a scalar resume under strict
// mode is legal (Python: a single static-breakpoint value is still accepted).
func TestStaticBreakpointScalarResumeAllowed(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	var bRuns int
	g := NewStateGraph()
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		bRuns++
		return map[string]any{"b_ran": true}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("b"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != "interrupt-before-b" {
		t.Fatalf("Interrupts = %+v, want the interrupt-before-b boundary interrupt", res.Interrupts)
	}

	res, err = cg.InvokeWithOptions(ctx, nil, Options{
		ThreadID:     "t",
		Resume:       "go",
		StrictResume: true,
	})
	if err != nil {
		t.Fatalf("strict scalar resume of a static breakpoint error = %v, want success", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("Interrupts = %+v, want none (run must complete)", res.Interrupts)
	}
	if bRuns != 1 || res.Values["b_ran"] != true {
		t.Fatalf("bRuns = %d, b_ran = %v, want b executed once", bRuns, res.Values["b_ran"])
	}
}

// TestStrictResumeNotPropagatedToSubgraph: the AddSubgraph wrapper builds the
// child run's Options field-by-field and never copies StrictResume (Python
// skips the check for nested graphs). A parent strict resume answering BOTH a
// root-level interrupt and a child interrupt with one map must succeed: at
// the child level the root interrupt's key is not pending, so a propagated
// strict flag would wrongly reject it there.
func TestStrictResumeNotPropagatedToSubgraph(t *testing.T) {
	ctx := t.Context()

	child := NewStateGraph()
	child.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "child-q")
		return map[string]any{"child_answer": v}, nil
	})
	child.AddEdge(types.START, "ask")
	child.AddEdge("ask", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	saver := checkpoint.NewMemorySaver()
	top := NewStateGraph()
	top.AddNode("start", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	top.AddNode("rootq", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "root-q")
		return map[string]any{"root_answer": v}, nil
	})
	top.AddSubgraph("sub", childCG)
	top.AddEdge(types.START, "start")
	top.AddEdge("start", "rootq")
	top.AddEdge("start", "sub")
	top.AddEdge("rootq", types.END)
	top.AddEdge("sub", types.END)
	cg, err := top.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("top Compile() error = %v", err)
	}

	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 2 {
		t.Fatalf("Interrupts = %+v, want two (root + propagated child)", res.Interrupts)
	}
	childNS := ""
	for _, intr := range res.Interrupts {
		if intr.Value == "child-q" {
			childNS = intr.NS
		}
	}
	if childNS == "" {
		t.Fatalf("Interrupts = %+v, want the propagated child interrupt", res.Interrupts)
	}

	// Both keys are pending at the ROOT level (the child interrupt by its
	// full nested NS), so the strict root check passes. The map is forwarded
	// verbatim into the child, where "rootq-1" is NOT pending — legal only
	// because strict mode is not propagated.
	res, err = cg.InvokeWithOptions(ctx, nil, Options{
		ThreadID:     "t1",
		Resume:       map[string]any{"rootq-1": "r1", childNS: "r2"},
		StrictResume: true,
	})
	if err != nil {
		t.Fatalf("resume error = %v, want completion (strict must not reach the child run)", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("Interrupts = %+v, want none (run must complete)", res.Interrupts)
	}
	if res.Values["root_answer"] != "r1" {
		t.Fatalf("root_answer = %v, want %q", res.Values["root_answer"], "r1")
	}
	if res.Values["child_answer"] != "r2" {
		t.Fatalf("child_answer = %v, want %q", res.Values["child_answer"], "r2")
	}
}

// TestStrictResumePinnedHeadCheckpointAllowed: a run pinned to a checkpoint
// (Options.CheckpointID) skips the strict check entirely (upstream skips it
// for checkpoint-superseded resumes): a bogus resume key is silently ignored
// and the pending interrupt re-fires, exactly like non-strict mode.
func TestStrictResumePinnedHeadCheckpointAllowed(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := strictResumeAskGraph(t, saver)
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want one", res.Interrupts)
	}
	head, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || head == nil {
		t.Fatalf("GetTuple() = (%v, %v), want the pause checkpoint", head, err)
	}

	res, err = cg.InvokeWithOptions(ctx, nil, Options{
		ThreadID:     "t",
		CheckpointID: head.Config.CheckpointID,
		Resume:       map[string]any{"bogus-id": 42},
		StrictResume: true,
	})
	if err != nil {
		var uiErr *types.UnknownInterruptError
		if errors.As(err, &uiErr) {
			t.Fatalf("pinned resume error = %v, want the strict check SKIPPED for pinned resumes", err)
		}
		t.Fatalf("pinned resume error = %v, want a silent re-pause", err)
	}
	if len(res.Interrupts) != 1 || res.Interrupts[0].ID != "ask-1" {
		t.Fatalf("Interrupts = %+v, want the same interrupt re-fired (ask-1)", res.Interrupts)
	}
}
