package graph

import (
	"fmt"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
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
	var aRuns, bRuns, cRuns atomic.Int32
	g.AddNode("start", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		aRuns.Add(1)
		return &types.Command{Goto: To("c")}, nil // routing only, no update
	})
	g.AddNode("b", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		bRuns.Add(1)
		Interrupt(ctx, "pause-b")
		return nil, nil
	})
	g.AddNode("c", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		cRuns.Add(1)
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

	first, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "pause-b" {
		t.Fatalf("expected one interrupt (pause-b), got %+v", first.Interrupts)
	}

	// The goto-only sibling's persisted pending writes are ONLY the
	// ReservedTasks send: no plain channel writes (the empty-update-map shape).
	tup, err := saver.GetTuple(t.Context(), checkpoint.Config{ThreadID: "t1"})
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

	second, err := cg.InvokeWithOptions(t.Context(), nil, Options{ThreadID: "t1", Resume: "go"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if aRuns.Load() != 1 {
		t.Fatalf("goto-only sibling a must NOT re-run on resume, ran %d times", aRuns.Load())
	}
	if bRuns.Load() != 2 {
		t.Fatalf("interrupted sibling b must re-run exactly once, ran %d times", bRuns.Load())
	}
	if cRuns.Load() != 1 {
		t.Fatalf("a's replayed send must dispatch c exactly once, ran %d times", cRuns.Load())
	}
	if second.Values["c_ran"] != true {
		t.Fatalf("c_ran = %v, want true", second.Values["c_ran"])
	}
}

// TestResumeMultiDestinationGotoAllSurvive pins the F1 regression: a
// goto-only sibling whose Command routes to TWO destinations persists TWO
// ReservedTasks pending writes in one batch — both must survive the pause
// checkpoint (the sqlite/postgres reserved-slot -2 mapping collapsed them to
// one) — and on resume both destinations dispatch exactly once.
func TestResumeMultiDestinationGotoAllSurvive(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	var aRuns, bRuns, cRuns, dRuns atomic.Int32
	g.AddNode("start", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		aRuns.Add(1)
		return &types.Command{Goto: To("c", "d")}, nil // routing only, no update
	})
	g.AddNode("b", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		bRuns.Add(1)
		Interrupt(ctx, "pause-b")
		return nil, nil
	})
	g.AddNode("c", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		cRuns.Add(1)
		return map[string]any{"c_ran": true}, nil
	})
	g.AddNode("d", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		dRuns.Add(1)
		return map[string]any{"d_ran": true}, nil
	})
	g.AddEdge(types.START, "start")
	g.AddEdge("start", "a")
	g.AddEdge("start", "b")
	g.AddEdge("b", types.END)
	g.AddEdge("c", types.END)
	g.AddEdge("d", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	first, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "pause-b" {
		t.Fatalf("expected one interrupt (pause-b), got %+v", first.Interrupts)
	}

	// The pause checkpoint must carry BOTH of a's ReservedTasks sends.
	tup, err := saver.GetTuple(t.Context(), checkpoint.Config{ThreadID: "t1"})
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
	var sends []types.Send
	for _, w := range tup.PendingWrites {
		if w.TaskID != aTaskID {
			continue
		}
		if w.Channel != checkpoint.ReservedTasks {
			t.Fatalf("goto-only sibling persisted a %q channel write, want ReservedTasks writes only", w.Channel)
		}
		send, ok := w.Value.(types.Send)
		if !ok {
			t.Fatalf("ReservedTasks write value = %T, want types.Send", w.Value)
		}
		sends = append(sends, send)
	}
	if len(sends) != 2 {
		t.Fatalf("multi-destination goto persisted %d ReservedTasks writes, want 2 (both must survive): %+v", len(sends), sends)
	}

	second, err := cg.InvokeWithOptions(t.Context(), nil, Options{ThreadID: "t1", Resume: "go"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if aRuns.Load() != 1 {
		t.Fatalf("goto-only sibling a must NOT re-run on resume, ran %d times", aRuns.Load())
	}
	if bRuns.Load() != 2 {
		t.Fatalf("interrupted sibling b must re-run exactly once, ran %d times", bRuns.Load())
	}
	if cRuns.Load() != 1 || dRuns.Load() != 1 {
		t.Fatalf("a's replayed sends must dispatch c and d exactly once each, ran c=%d d=%d", cRuns.Load(), dRuns.Load())
	}
	if second.Values["c_ran"] != true || second.Values["d_ran"] != true {
		t.Fatalf("resumed values = %+v, want c_ran and d_ran true", second.Values)
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
	var draftRuns, reviewRuns atomic.Int32
	g.AddNode("draft", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		draftRuns.Add(1)
		return map[string]any{"draft": "v1"}, nil
	})
	g.AddNode("review", func(ctx runtime.Runtime, state map[string]any) (any, error) {
		reviewRuns.Add(1)
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
	ctx := t.Context()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"topic": "x"}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("first Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "needs-approval" {
		t.Fatalf("expected one interrupt (needs-approval), got %+v", first.Interrupts)
	}
	if draftRuns.Load() != 1 || reviewRuns.Load() != 1 {
		t.Fatalf("runs at pause: draft=%d review=%d, want 1 each", draftRuns.Load(), reviewRuns.Load())
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
	if draftRuns.Load() != 1 {
		t.Fatalf("draft must NOT re-run on resume, ran %d times", draftRuns.Load())
	}
	if reviewRuns.Load() != 2 {
		t.Fatalf("review must re-run exactly once after the update, ran %d times", reviewRuns.Load())
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
	g.AddNode("prepare", func(_ runtime.Runtime, state map[string]any) (any, error) {
		count, _ := state["count"].(int)
		return map[string]any{"count": count + 10}, nil
	})
	g.AddNode("multi_interrupt", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
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
	ctx := t.Context()

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
// followed by ONE ReservedResume write whose value is the whole ordered
// consumed prefix (interrupt write first, resume write after).
func TestResumePauseCheckpointPersistsResumePrefix(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := multiInterruptGraph(t, saver)
	ctx := t.Context()

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
	if want := []any{"first_answer"}; !reflect.DeepEqual(writes[1].Value, want) {
		t.Fatalf("writes[1].Value = %v (%T), want the full prefix %v", writes[1].Value, writes[1].Value, want)
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
	g.AddNode("chain", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
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
	ctx := t.Context()

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

	// The third pause checkpoint carries the two-value consumed prefix as ONE
	// ReservedResume write whose value is the full ordered list.
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
		t.Fatalf("ReservedResume value = %v (%T), want the full prefix %v", resumes[0].Value, resumes[0].Value, want)
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
	ctx := t.Context()

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

// TestResumeNilResumeRepausesKeepsPrefix pins the nil-resume re-pause path
// when the paused task has a NON-EMPTY consumed resume prefix: resuming with
// nil resume (Python's invoke(None)) must re-fire the same pending interrupt
// AND re-persist the full prefix intact, so a later real resume still
// rebuilds the complete ordered queue.
func TestResumeNilResumeRepausesKeepsPrefix(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("chain", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
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
	ctx := t.Context()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"}); err != nil {
		t.Fatalf("invoke 1 error = %v", err)
	}
	r2, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "a"})
	if err != nil {
		t.Fatalf("invoke 2 error = %v", err)
	}
	if len(r2.Interrupts) != 1 || r2.Interrupts[0].Value != "q1" {
		t.Fatalf("invoke 2 Interrupts = %+v, want one interrupt (q1)", r2.Interrupts)
	}

	// invoke(None) on the second pause: nil input, no Resume — q1 re-fires.
	again, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("nil-resume Invoke() error = %v", err)
	}
	if len(again.Interrupts) != 1 || again.Interrupts[0].Value != "q1" {
		t.Fatalf("nil-resume Invoke() Interrupts = %+v, want the same interrupt re-fired (q1)", again.Interrupts)
	}

	// The re-pause checkpoint must still carry the full consumed prefix.
	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("expected pause checkpoint, got tup=%+v err=%v", tup, err)
	}
	var prefix []any
	for _, w := range tup.PendingWrites {
		if w.Channel == checkpoint.ReservedResume {
			if list, ok := w.Value.([]any); ok {
				prefix = list
			}
		}
	}
	if want := []any{"a"}; !reflect.DeepEqual(prefix, want) {
		t.Fatalf("re-pause ReservedResume prefix = %v, want %v (prefix lost)", prefix, want)
	}

	// The chain still advances correctly from the preserved prefix.
	r3, err := cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: "b"})
	if err != nil {
		t.Fatalf("invoke 3 error = %v", err)
	}
	if len(r3.Interrupts) != 1 || r3.Interrupts[0].Value != "q2" {
		t.Fatalf("invoke 3 Interrupts = %+v, want one interrupt (q2)", r3.Interrupts)
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

func TestPutPauseWritesEmptyIsNoOp(t *testing.T) {
	// Even with a failing saver, persisting zero writes must not call it.
	sink := newCheckpointSink(&putWritesErrSaver{Saver: checkpoint.NewMemorySaver()}, DurabilitySync, t.Context(), nil)
	err := sink.putPauseWrites(t.Context(), checkpoint.Config{ThreadID: "t"}, nil, "task")
	if err != nil {
		t.Fatalf("putPauseWrites(nil) error = %v, want nil", err)
	}
	if interruptWrites(nil) != nil {
		t.Fatalf("interruptWrites(nil) = %v, want nil", interruptWrites(nil))
	}
	if interruptAndResumeWrites(nil, nil) != nil {
		t.Fatalf("interruptAndResumeWrites(nil, nil) = %v, want nil", interruptAndResumeWrites(nil, nil))
	}
}

func TestCompletedTaskWritesBadGotoErrors(t *testing.T) {
	_, err := completedTaskWrites(map[string]any{"x": 1}, &types.Command{Goto: []any{42}})
	if err == nil || !strings.Contains(err.Error(), "unsupported routing destination") {
		t.Fatalf("completedTaskWrites() error = %v, want an unsupported routing destination error", err)
	}
}

func TestPlanResumeSkipsEndDestinations(t *testing.T) {
	tup := &checkpoint.Tuple{
		Checkpoint: checkpoint.Checkpoint{
			V:               1,
			ID:              "cp",
			ChannelValues:   map[string]any{},
			ChannelVersions: map[string]int64{},
			VersionsSeen:    map[string]map[string]int64{},
			Next: []checkpoint.PlannedTask{
				{Node: types.END},
				{Node: "a", ID: "task-a"},
			},
		},
	}
	plan, err := (&CompiledGraph{}).planResume(tup, nil, "")
	if err != nil {
		t.Fatalf("planResume() error = %v", err)
	}
	if len(plan.tasks) != 1 || plan.tasks[0].node != "a" {
		t.Fatalf("planResume().tasks = %+v, want only node %q (END destinations skipped)", plan.tasks, "a")
	}
}

// TestInterruptNSIDFormatRootLevel snapshots the NS/ID format of a root-level
// in-node interrupt: ID stays "<node>-<counter>" (backward compatible), NS is
// the task's own checkpoint namespace "<node>:<plannedTaskID>" where the
// planned task ID is the dispatch-time identity (TaskID anchored on the
// checkpoint position the task was planned against — the same ID the run loop
// publishes via plannedTaskIDKey for subgraph namespacing), and the persisted
// ReservedInterrupt copy carries the same NS.
func TestInterruptNSIDFormatRootLevel(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(rt, "q")
		return nil, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want one", res.Interrupts)
	}
	intr := res.Interrupts[0]
	if intr.ID != "ask-1" {
		t.Fatalf("interrupt ID = %q, want %q", intr.ID, "ask-1")
	}

	// The ask task dispatched as superstep 0 against the input checkpoint, so
	// its dispatch-time planned ID is TaskID(inputCheckpoint, 0, "ask", nil).
	tups, err := saver.List(ctx, checkpoint.Config{ThreadID: "t"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	inputID := ""
	for _, tt := range tups {
		if tt.Metadata.Source == "input" {
			inputID = tt.Config.CheckpointID
		}
	}
	if inputID == "" {
		t.Fatalf("no input checkpoint in history: %+v", tups)
	}
	wantNS := "ask:" + TaskID(inputID, 0, "ask", nil)
	if intr.NS != wantNS {
		t.Fatalf("interrupt NS = %q, want %q (node + dispatch-time planned task ID)", intr.NS, wantNS)
	}
	// Format snapshot: "<node>:<16-hex task ID>".
	if !strings.HasPrefix(intr.NS, "ask:") || len(intr.NS) != len("ask:")+16 {
		t.Fatalf("interrupt NS = %q, want <node>:<16-hex task ID>", intr.NS)
	}

	tup, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || tup == nil {
		t.Fatalf("GetTuple() = (%v, %v), want the pause checkpoint", tup, err)
	}
	if len(tup.Checkpoint.Next) != 1 || tup.Checkpoint.Next[0].Node != "ask" {
		t.Fatalf("pause checkpoint Next = %+v, want the ask task", tup.Checkpoint.Next)
	}
	sawCopy := false
	for _, w := range tup.PendingWrites {
		if w.Channel != checkpoint.ReservedInterrupt {
			continue
		}
		cp, ok := w.Value.(types.Interrupt)
		if !ok {
			t.Fatalf("ReservedInterrupt write value is %T, want types.Interrupt", w.Value)
		}
		if cp.NS != wantNS || cp.ID != "ask-1" {
			t.Fatalf("persisted interrupt copy = %+v, want NS %q ID %q", cp, wantNS, "ask-1")
		}
		sawCopy = true
	}
	if !sawCopy {
		t.Fatalf("pause checkpoint has no ReservedInterrupt pending write: %+v", tup.PendingWrites)
	}
}

// TestBoundaryInterruptNSRootLevel pins the boundary interrupt NS format at
// the root level: node-only namespace (no task ID), ID unchanged.
func TestBoundaryInterruptNSRootLevel(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("b", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("b"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("Interrupts = %+v, want one", res.Interrupts)
	}
	if res.Interrupts[0].NS != "b" {
		t.Fatalf("boundary interrupt NS = %q, want %q", res.Interrupts[0].NS, "b")
	}
	if res.Interrupts[0].ID != "interrupt-before-b" {
		t.Fatalf("boundary interrupt ID = %q, want %q", res.Interrupts[0].ID, "interrupt-before-b")
	}
}

// TestResumeMapByNSAddressesRootInterrupt verifies a map resume keyed by an
// interrupt's NS (not its ID) reaches the pending root-level interrupt.
func TestResumeMapByNSAddressesRootInterrupt(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("ask", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		v := Interrupt(rt, "q")
		s, _ := v.(string)
		return map[string]any{"answer": s}, nil
	})
	g.AddEdge(types.START, "ask")
	g.AddEdge("ask", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := cg.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t"})
	if err != nil || len(res.Interrupts) != 1 {
		t.Fatalf("pause Invoke() = (%+v, %v), want one interrupt", res.Interrupts, err)
	}
	res, err = cg.InvokeWithOptions(ctx, nil, Options{ThreadID: "t", Resume: map[string]any{res.Interrupts[0].NS: "42"}})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("resume re-paused with %+v", res.Interrupts)
	}
	if res.Values["answer"] != "42" {
		t.Fatalf("answer = %v, want 42 (NS-keyed resume did not match)", res.Values["answer"])
	}
}

// TestInterruptOwnedBy covers the namespace-ownership predicate used by
// resumeSkipNode: only boundary interrupts owned by the resuming run's own
// namespace may drive the first-superstep skip.
func TestInterruptOwnedBy(t *testing.T) {
	cases := []struct {
		ownNS, interruptNS string
		want               bool
	}{
		{"", "ask:abc", true},                     // root run, root in-node interrupt
		{"", "b", true},                           // root run, root boundary interrupt
		{"", "sub:t1/ask:t2", false},              // root run, subgraph-internal interrupt
		{"", "sub:t1/b", false},                   // root run, subgraph boundary interrupt
		{"sub:t1", "sub:t1/ask:t2", true},         // sub run, own in-node interrupt
		{"sub:t1", "sub:t1/b", true},              // sub run, own boundary interrupt
		{"sub:t1", "sub:t1/mid:t2/ask:t3", false}, // sub run, grandchild interrupt
		{"sub:t1", "other:9/ask:t1", false},       // different branch
		{"sub:t1", "", true},                      // legacy persisted interrupt
	}
	for _, c := range cases {
		if got := interruptOwnedBy(c.ownNS, c.interruptNS); got != c.want {
			t.Fatalf("interruptOwnedBy(%q, %q) = %v, want %v", c.ownNS, c.interruptNS, got, c.want)
		}
	}
}

// TestResumeSkipNodeOwnership pins that a subgraph's internal
// interrupt-before-<node> does not make the parent run skip its own
// same-named node, while owned and legacy boundary interrupts still do.
func TestResumeSkipNodeOwnership(t *testing.T) {
	rootPending := []types.Interrupt{
		{Value: "v", ID: "interrupt-before-b", NS: "sub:t1/b"}, // child's boundary interrupt
	}
	if got := resumeSkipNode(rootPending, ""); got != "" {
		t.Fatalf("resumeSkipNode(child boundary, ownNS root) = %q, want \"\"", got)
	}
	childPending := []types.Interrupt{
		{Value: "v", ID: "interrupt-before-b", NS: "sub:t1/b"},
	}
	if got := resumeSkipNode(childPending, "sub:t1"); got != "b" {
		t.Fatalf("resumeSkipNode(own boundary, ownNS sub:t1) = %q, want %q", got, "b")
	}
	legacy := []types.Interrupt{
		{Value: "v", ID: "interrupt-before-b", NS: ""}, // pre-NS persisted data
	}
	if got := resumeSkipNode(legacy, ""); got != "b" {
		t.Fatalf("resumeSkipNode(legacy boundary, ownNS root) = %q, want %q", got, "b")
	}
	nonBoundary := []types.Interrupt{
		{Value: "v", ID: "ask-1", NS: "ask:t1"},
	}
	if got := resumeSkipNode(nonBoundary, ""); got != "" {
		t.Fatalf("resumeSkipNode(in-node interrupt) = %q, want \"\"", got)
	}
}
