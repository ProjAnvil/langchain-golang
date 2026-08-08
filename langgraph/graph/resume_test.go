package graph

import (
	"context"
	"fmt"
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

// multiInterruptGraph builds the StateGraph of
// test_node_before_multiple_interrupt_cycles_graph_api
// (langgraph tests/test_pregel.py:6048-6090): a prepare node (count+10)
// feeding a node that raises two sequential in-node interrupts and joins
// their resume values.
func multiInterruptGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode("prepare", func(_ context.Context, state map[string]any) (any, error) {
		count, _ := state["count"].(int)
		return map[string]any{"count": count + 10}, nil
	})
	g.AddNode("multi_interrupt", func(ctx context.Context, _ map[string]any) (any, error) {
		first := Interrupt(ctx, "First question?")
		second := Interrupt(ctx, "Second question?")
		return map[string]any{"data": fmt.Sprintf("%v,%v", first, second)}, nil
	})
	g.AddEdge(types.START, "prepare")
	g.AddEdge("prepare", "multi_interrupt")
	g.AddEdge("multi_interrupt", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestResumeSequentialInterruptsGraphAPI ports
// test_node_before_multiple_interrupt_cycles_graph_api
// (langgraph tests/test_pregel.py:6048-6090): a node running before an
// interrupt node must not interfere with multiple interrupt/resume cycles.
// Each resume value feeds the NEXT unconsumed interrupt — not the queue
// head — because the pause checkpoint persists the already-consumed resume
// prefix (ReservedResume writes) and resume rebuilds the full ordered queue.
func TestResumeSequentialInterruptsGraphAPI(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := multiInterruptGraph(t, saver)
	ctx := context.Background()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"count": 0, "data": ""}, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "First question?" {
		t.Fatalf("first Invoke() Interrupts = %+v, want one interrupt (First question?)", first.Interrupts)
	}

	second, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "1", Resume: "first_answer"})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 1 || second.Interrupts[0].Value != "Second question?" {
		t.Fatalf("second Invoke() Interrupts = %+v, want one interrupt (Second question?)", second.Interrupts)
	}

	third, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "1", Resume: "second_answer"})
	if err != nil {
		t.Fatalf("third Invoke() error = %v", err)
	}
	if len(third.Interrupts) != 0 {
		t.Fatalf("third Invoke() Interrupts = %+v, want none (run must complete)", third.Interrupts)
	}
	if third.Values["count"] != 10 {
		t.Fatalf("count = %v, want 10", third.Values["count"])
	}
	if third.Values["data"] != "first_answer,second_answer" {
		t.Fatalf("data = %v, want %q", third.Values["data"], "first_answer,second_answer")
	}
}

// TestResumePauseCheckpointPersistsResumePrefix pins the pause checkpoint's
// pending-writes shape after a SECOND sequential interrupt fires: the paused
// task carries one ReservedInterrupt write (the freshly raised interrupt)
// followed by one ReservedResume write per already-consumed resume value, in
// consumption order (interrupt writes first, resume writes after).
func TestResumePauseCheckpointPersistsResumePrefix(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := multiInterruptGraph(t, saver)
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"count": 0, "data": ""}, Options{ThreadID: "1"}); err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	second, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "1", Resume: "first_answer"})
	if err != nil {
		t.Fatalf("second Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 1 || second.Interrupts[0].Value != "Second question?" {
		t.Fatalf("second Invoke() Interrupts = %+v, want one interrupt (Second question?)", second.Interrupts)
	}

	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "1"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	taskID := ""
	for _, pt := range tup.Checkpoint.Next {
		if pt.Node == "multi_interrupt" {
			taskID = pt.ID
		}
	}
	if taskID == "" {
		t.Fatalf("pause checkpoint Next = %+v, want multi_interrupt planned", tup.Checkpoint.Next)
	}
	var writes []checkpoint.Write
	for _, w := range tup.PendingWrites {
		if w.TaskID == taskID {
			writes = append(writes, w)
		}
	}
	if len(writes) != 2 {
		t.Fatalf("paused task pending writes = %+v, want exactly 2 (interrupt + resume)", writes)
	}
	if writes[0].Channel != checkpoint.ReservedInterrupt {
		t.Fatalf("writes[0].Channel = %q, want ReservedInterrupt (interrupt writes come first)", writes[0].Channel)
	}
	intr, ok := writes[0].Value.(types.Interrupt)
	if !ok || intr.Value != "Second question?" {
		t.Fatalf("writes[0].Value = %+v, want types.Interrupt (Second question?)", writes[0].Value)
	}
	if writes[1].Channel != checkpoint.ReservedResume {
		t.Fatalf("writes[1].Channel = %q, want ReservedResume", writes[1].Channel)
	}
	if writes[1].Value != "first_answer" {
		t.Fatalf("writes[1].Value = %v, want %q", writes[1].Value, "first_answer")
	}
}

// TestResumeChainedInterruptPrefixAccumulates drives a single node through
// three sequential interrupts across four invocations. Each pause checkpoint
// must carry the task's FULL consumed resume prefix, so the queue rebuilds
// in order and the chain advances instead of re-feeding the newest value to
// the first interrupt (the pre-fix misalignment loop).
func TestResumeChainedInterruptPrefixAccumulates(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("chain", func(ctx context.Context, _ map[string]any) (any, error) {
		a := Interrupt(ctx, "q0")
		b := Interrupt(ctx, "q1")
		c := Interrupt(ctx, "q2")
		return map[string]any{"data": fmt.Sprintf("%v,%v,%v", a, b, c)}, nil
	})
	g.AddEdge(types.START, "chain")
	g.AddEdge("chain", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := context.Background()

	r1, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("invoke 1 error = %v", err)
	}
	if len(r1.Interrupts) != 1 || r1.Interrupts[0].Value != "q0" {
		t.Fatalf("invoke 1 Interrupts = %+v, want one interrupt (q0)", r1.Interrupts)
	}
	r2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "a"})
	if err != nil {
		t.Fatalf("invoke 2 error = %v", err)
	}
	if len(r2.Interrupts) != 1 || r2.Interrupts[0].Value != "q1" {
		t.Fatalf("invoke 2 Interrupts = %+v, want one interrupt (q1)", r2.Interrupts)
	}
	r3, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "b"})
	if err != nil {
		t.Fatalf("invoke 3 error = %v", err)
	}
	if len(r3.Interrupts) != 1 || r3.Interrupts[0].Value != "q2" {
		t.Fatalf("invoke 3 Interrupts = %+v, want one interrupt (q2)", r3.Interrupts)
	}

	// The third pause checkpoint carries the two-value consumed prefix, one
	// ReservedResume write per value, in consumption order.
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	var resumes []any
	sawInterrupt := false
	for _, w := range tup.PendingWrites {
		switch w.Channel {
		case checkpoint.ReservedInterrupt:
			sawInterrupt = true
		case checkpoint.ReservedResume:
			resumes = append(resumes, w.Value)
		}
	}
	if !sawInterrupt {
		t.Fatal("pause checkpoint pending writes missing the ReservedInterrupt write")
	}
	if len(resumes) != 2 || resumes[0] != "a" || resumes[1] != "b" {
		t.Fatalf("ReservedResume writes = %v, want [a b] in order", resumes)
	}

	r4, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "c"})
	if err != nil {
		t.Fatalf("invoke 4 error = %v", err)
	}
	if len(r4.Interrupts) != 0 {
		t.Fatalf("invoke 4 Interrupts = %+v, want none (run must complete)", r4.Interrupts)
	}
	if r4.Values["data"] != "a,b,c" {
		t.Fatalf("data = %v, want %q", r4.Values["data"], "a,b,c")
	}
}

// TestResumeNilResumeRepauses pins the unchanged nil-resume semantic:
// resuming a paused run with nil resume (Python's invoke(None)) re-fires the
// same pending interrupt instead of answering it with nil.
func TestResumeNilResumeRepauses(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := multiInterruptGraph(t, saver)
	ctx := context.Background()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"count": 0, "data": ""}, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "First question?" {
		t.Fatalf("first Invoke() Interrupts = %+v, want one interrupt (First question?)", first.Interrupts)
	}

	// invoke(None): nil input, no Resume — the pending interrupt re-fires.
	again, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "1"})
	if err != nil {
		t.Fatalf("nil-resume Invoke() error = %v", err)
	}
	if len(again.Interrupts) != 1 || again.Interrupts[0].Value != "First question?" {
		t.Fatalf("nil-resume Invoke() Interrupts = %+v, want the same interrupt re-fired (First question?)", again.Interrupts)
	}
}
