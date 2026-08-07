package graph

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestResumeReplaysGotoOnlySibling pins the empty-update-map replay shape: a
// completed sibling whose Command carries ONLY Goto destinations (no state
// update) persists nothing but ReservedTasks pending writes, and on resume its
// sends rejoin the task queue with no state writes replayed — the sibling is
// not re-run and its destinations run exactly once.
func TestResumeReplaysGotoOnlySibling(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	var aRuns, bRuns, cRuns int32
	g.AddNode("start", func(_ context.Context, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&aRuns, 1)
		return &types.Command{Goto: To("c")}, nil // routing only, no update
	})
	g.AddNode("b", func(ctx context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&bRuns, 1)
		Interrupt(ctx, "pause-b")
		return nil, nil
	})
	g.AddNode("c", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&cRuns, 1)
		return map[string]any{"c_ran": true}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "a")
	g.AddEdge("start", "b")
	g.AddEdge("b", types.END)
	g.AddEdge("c", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(context.Background(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "pause-b" {
		t.Fatalf("expected one interrupt (pause-b), got %+v", first.Interrupts)
	}

	// The goto-only sibling's persisted pending writes are ONLY the
	// ReservedTasks send: no plain channel writes (the empty-update-map shape).
	tup, err := saver.GetTuple(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	aTaskID := ""
	for _, pt := range tup.Checkpoint.Next {
		if pt.Node == "a" {
			aTaskID = pt.ID
		}
	}
	if aTaskID == "" {
		t.Fatalf("pause checkpoint Next = %+v, want sibling a planned", tup.Checkpoint.Next)
	}
	sawSend := false
	for _, w := range tup.PendingWrites {
		if w.TaskID != aTaskID {
			continue
		}
		if w.Channel != checkpoint.ReservedTasks {
			t.Fatalf("goto-only sibling persisted a %q channel write, want ReservedTasks writes only", w.Channel)
		}
		sawSend = true
	}
	if !sawSend {
		t.Fatal("goto-only sibling persisted no ReservedTasks send")
	}

	second, err := cg.InvokeWithOptions(context.Background(), nil, Options{ThreadID: "t1", Resume: "go"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if aRuns != 1 {
		t.Fatalf("goto-only sibling a must NOT re-run on resume, ran %d times", aRuns)
	}
	if bRuns != 2 {
		t.Fatalf("interrupted sibling b must re-run exactly once, ran %d times", bRuns)
	}
	if cRuns != 1 {
		t.Fatalf("a's replayed send must dispatch c exactly once, ran %d times", cRuns)
	}
	if second.Values["c_ran"] != true {
		t.Fatalf("c_ran = %v, want true", second.Values["c_ran"])
	}
}

// TestInterruptUpdateStateResumeHITL walks the human-in-the-loop flow: a node
// interrupts for approval, the human records the decision via UpdateState
// attributed to the interrupting node's predecessor (so the update
// checkpoint's re-resolved Next re-plans the interrupted node), and a
// nil-input resume re-runs the interrupted node, which observes the updated
// state and completes instead of re-interrupting. The update checkpoint steps
// past the pause checkpoint (S6).
func TestInterruptUpdateStateResumeHITL(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	var draftRuns, reviewRuns int32
	g.AddNode("draft", func(_ context.Context, _ map[string]any) (any, error) {
		atomic.AddInt32(&draftRuns, 1)
		return map[string]any{"draft": "v1"}, nil
	})
	g.AddNode("review", func(ctx context.Context, state map[string]any) (any, error) {
		atomic.AddInt32(&reviewRuns, 1)
		if state["approved"] == true {
			return map[string]any{"status": "published"}, nil
		}
		Interrupt(ctx, "needs-approval")
		return nil, nil
	})
	g.AddEdge(types.START, "draft")
	g.AddEdge("draft", "review")
	g.AddEdge("review", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"topic": "x"}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "needs-approval" {
		t.Fatalf("expected one interrupt (needs-approval), got %+v", first.Interrupts)
	}
	if draftRuns != 1 || reviewRuns != 1 {
		t.Fatalf("runs at pause: draft=%d review=%d, want 1 each", draftRuns, reviewRuns)
	}
	if _, ok := first.Values["status"]; ok {
		t.Fatalf("status must not be set at the pause, got %+v", first.Values)
	}

	pause, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() of pause checkpoint error = %v", err)
	}
	if len(pause.Next) != 1 || pause.Next[0] != "review" {
		t.Fatalf("pause checkpoint Next = %+v, want [review]", pause.Next)
	}

	// The human records the approval as a write attributed to "draft", whose
	// re-resolved successor is "review" — so the update checkpoint re-plans
	// the interrupted node.
	newCfg, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: "t1"},
		map[string]any{"approved": true}, "draft")
	if err != nil {
		t.Fatalf("UpdateState() error = %v", err)
	}
	update, err := cg.GetState(ctx, newCfg)
	if err != nil {
		t.Fatalf("GetState() of update checkpoint error = %v", err)
	}
	if update.Metadata.Source != "update" || update.Metadata.Step != pause.Metadata.Step+1 {
		t.Fatalf("update checkpoint Metadata = %+v, want source update step %d (pause step + 1, S6)",
			update.Metadata, pause.Metadata.Step+1)
	}
	if len(update.Next) != 1 || update.Next[0] != "review" {
		t.Fatalf("update checkpoint Next = %+v, want [review] (draft's re-resolved successor)", update.Next)
	}

	// Resume with nil input (Python's invoke(None)): the update checkpoint has
	// no pending writes for "review", so the node re-executes from the start,
	// observes the approved state, and completes.
	resumed, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(resumed.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", resumed.Interrupts)
	}
	if resumed.Values["status"] != "published" {
		t.Fatalf("status = %v, want published", resumed.Values["status"])
	}
	if resumed.Values["approved"] != true || resumed.Values["draft"] != "v1" {
		t.Fatalf("resumed values = %+v, want approved=true draft=v1", resumed.Values)
	}
	if draftRuns != 1 {
		t.Fatalf("draft must NOT re-run on resume, ran %d times", draftRuns)
	}
	if reviewRuns != 2 {
		t.Fatalf("review must re-run exactly once after the update, ran %d times", reviewRuns)
	}
}
