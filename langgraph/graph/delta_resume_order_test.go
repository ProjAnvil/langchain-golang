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

// TestResumeSendDispatchPathAboveRestoredTasks pins the resumed-superstep
// path spacing: a completed sibling's replayed Send rejoins the resumed
// superstep's task queue AFTER the restored (rerun) tasks, so its dispatch
// path must continue ABOVE the planning checkpoint's Next indices. Stamping
// it at its active-slice index would collide with a restored task's path
// (Next=[p(done, emitted Send s), q(interrupted)]: q keeps "0000000001",
// the send lands at active index 1 -> the same "0000000001"), and the tie
// would fall to the task-id hash, permuting replay vs live dispatch order.
func TestResumeSendDispatchPathAboveRestoredTasks(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(resumeOrderReducer, func() any { return []string{} }, 100))
	g.AddNode("fork", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("p", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return &types.Command{
			Update: map[string]any{"msgs": []string{"p"}},
			Goto:   []any{&types.Send{Node: "s"}},
		}, nil
	})
	g.AddNode("q", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		if Interrupt(rt, "q-asks") != nil {
			return map[string]any{"msgs": []string{"q"}}, nil
		}
		return nil, nil
	})
	g.AddNode("s", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"msgs": []string{"s"}}, nil
	})
	g.AddEdge(types.START, "fork")
	g.AddEdge("fork", "p")
	g.AddEdge("fork", "q")
	g.AddEdge("q", types.END)
	g.AddEdge("s", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
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
	want := []string{"in", "p", "q", "s"}
	if got, _ := second.Values["msgs"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("live msgs = %v, want %v", got, want)
	}

	// q's and s's writes anchor on the final loop checkpoint. Their dispatch
	// paths must be distinct and ordered q (the restored task, Next index 1)
	// before s (the replayed send, above the Next index space) — a collision
	// would leave both at "0000000001" and the tie to the fnv task-id hash.
	head, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || head == nil {
		t.Fatalf("GetTuple() = (%v, %v)", head, err)
	}
	type msgWrite struct {
		path   string
		tokens []string
	}
	var writes []msgWrite
	for _, w := range head.PendingWrites {
		if w.Channel != "msgs" {
			continue
		}
		v, ok := w.Value.([]string)
		if !ok {
			t.Fatalf("msgs write = %#v, want []string", w.Value)
		}
		writes = append(writes, msgWrite{path: w.TaskPath, tokens: v})
	}
	if len(writes) != 2 {
		t.Fatalf("head pending msgs writes = %+v, want exactly 2 (q and s)", writes)
	}
	if writes[0].path != "0000000001" || writes[0].tokens[0] != "q" {
		t.Fatalf("first head write = {path %q tokens %v}, want {path %q tokens [q]} (the restored task keeps its Next index)", writes[0].path, writes[0].tokens, "0000000001")
	}
	if writes[1].path != "0000000002" || writes[1].tokens[0] != "s" {
		t.Fatalf("second head write = {path %q tokens %v}, want {path %q tokens [s]} (the replayed send continues above the Next index space, not the collided %q)", writes[1].path, writes[1].tokens, "0000000002", "0000000001")
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["msgs"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState msgs = %v, want %v (replay order != live order)", got, want)
	}
}
