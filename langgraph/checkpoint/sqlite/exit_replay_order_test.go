package sqlite_test

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// stringTokenReducer concatenates a []string base with each []string update
// in the batch — an order-sensitive BatchReducer, so a replay that permutes
// writes rebuilds a different value than live execution produced.
func stringTokenReducer(existing any, updates []any) (any, error) {
	base, _ := existing.([]string)
	out := make([]string, len(base))
	copy(out, base)
	for _, u := range updates {
		add, _ := u.([]string)
		out = append(out, add...)
	}
	return out, nil
}

// exitReplayGraph builds START -> fork -> (p || q) -> r over a DeltaChannel
// with snapshotFrequency=100 (it never snapshots, so every value lives only
// as per-checkpoint pending writes reconstructed by the ancestor walk). p
// writes "p" and completes; q interrupts and writes "q" only after the
// resume; r runs in a LATER superstep and writes "r".
func exitReplayGraph(t *testing.T, saver checkpoint.Saver) *graph.CompiledGraph {
	t.Helper()
	g := graph.NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(stringTokenReducer, func() any { return []string{} }, 100))
	g.AddNode("fork", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return nil, nil
	})
	g.AddNode("p", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"msgs": []string{"p"}}, nil
	})
	g.AddNode("q", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		if graph.Interrupt(rt, "q-asks") != nil {
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
	cg, err := g.Compile(graph.WithCheckpointer(saver), graph.WithDurability(graph.DurabilityExit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitDurabilityResumeReplayOrder pins #9114: a resumed exit-durability
// run's delta writes must replay in LIVE order. The run starts from the
// pause checkpoint, which already holds the finished sibling p's write under
// its real task id; the resumed q's write and the later superstep's r write
// anchor on that same checkpoint under exit-mode synthetic task ids. A saver
// that orders those writes by task id alone replays them before p's loaded
// write (the sort-before synthetic prefix), rebuilding a permuted value.
//
// Live order: input "in", p, q (after resume), r => [in p q r].
func TestExitDurabilityResumeReplayOrder(t *testing.T) {
	saver, err := sqlite.New(filepath.Join(t.TempDir(), "checkpoints.db"), serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { saver.Close() })
	cg := exitReplayGraph(t, saver)
	ctx := t.Context()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"in"}}, graph.Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("pause Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "q-asks" {
		t.Fatalf("expected one q-asks interrupt, got %+v", first.Interrupts)
	}

	second, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t", Resume: "answer"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(second.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", second.Interrupts)
	}
	if got, _ := second.Values["msgs"].([]string); !reflect.DeepEqual(got, []string{"in", "p", "q", "r"}) {
		t.Fatalf("live msgs = %v, want [in p q r]", got)
	}

	// The pause checkpoint is the anchor every exit write landed on: its own
	// pending writes replay oldest-first through the ancestor walk, so their
	// stored order must equal the live order.
	head, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || head == nil {
		t.Fatalf("GetTuple() = (%v, %v), want the final checkpoint", head, err)
	}
	anchor := head.ParentConfig
	if anchor == nil {
		t.Fatalf("final checkpoint %s has no parent: the exit flush must anchor on the pause checkpoint", head.Config.CheckpointID)
	}
	pause, err := saver.GetTuple(ctx, *anchor)
	if err != nil || pause == nil {
		t.Fatalf("GetTuple(anchor) = (%v, %v)", pause, err)
	}
	var replayed []string
	for _, w := range pause.PendingWrites {
		if w.Channel != "msgs" {
			continue
		}
		v, ok := w.Value.([]string)
		if !ok {
			t.Fatalf("msgs write = %#v, want []string", w.Value)
		}
		replayed = append(replayed, v...)
	}
	if want := []string{"in", "p", "q", "r"}; !reflect.DeepEqual(replayed, want) {
		t.Fatalf("anchor pending-writes replay order = %v, want %v (live order)", replayed, want)
	}

	// GetState reconstructs "msgs" purely from that stored order.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["msgs"].([]string); !reflect.DeepEqual(got, []string{"in", "p", "q", "r"}) {
		t.Fatalf("GetState msgs = %v, want [in p q r] (replay order != live order)", got)
	}
}

// TestExitDurabilityPauseAfterDeltaSuperstepReplayOrder covers the exit
// PAUSE path with accumulated writes from EARLIER supersteps: nodes a and b
// write in supersteps 1 and 2, q interrupts in superstep 3, so the pause
// checkpoint holds a's and b's exit writes ("!"-prefixed sort-before step
// paths — they are strictly older than the pause superstep's direct writes)
// alongside the interrupt records, and the resumed q's write (the anchor
// superstep, real dispatch path) flushes onto that same checkpoint. Sorting
// b's write with the flush-path keys instead (a "~~" sort-after late path)
// would replay it after q's real-path write. Replay must equal live order:
// [in a b q].
func TestExitDurabilityPauseAfterDeltaSuperstepReplayOrder(t *testing.T) {
	saver, err := sqlite.New(filepath.Join(t.TempDir(), "checkpoints.db"), serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { saver.Close() })
	g := graph.NewStateGraph()
	g.AddChannel("msgs", channels.NewDeltaChannel(stringTokenReducer, func() any { return []string{} }, 100))
	delta := func(token string) func(runtime.Runtime, map[string]any) (any, error) {
		return func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{"msgs": []string{token}}, nil
		}
	}
	g.AddNode("a", delta("a"))
	g.AddNode("b", delta("b"))
	g.AddNode("q", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		if graph.Interrupt(rt, "q-asks") != nil {
			return map[string]any{"msgs": []string{"q"}}, nil
		}
		return nil, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "q")
	g.AddEdge("q", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver), graph.WithDurability(graph.DurabilityExit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	ctx := t.Context()

	first, err := cg.InvokeWithOptions(ctx, map[string]any{"msgs": []string{"in"}}, graph.Options{ThreadID: "t"})
	if err != nil {
		t.Fatalf("pause Invoke() error = %v", err)
	}
	if len(first.Interrupts) != 1 || first.Interrupts[0].Value != "q-asks" {
		t.Fatalf("expected one q-asks interrupt, got %+v", first.Interrupts)
	}
	second, err := cg.InvokeWithOptions(ctx, nil, graph.Options{ThreadID: "t", Resume: "answer"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	want := []string{"in", "a", "b", "q"}
	if got, _ := second.Values["msgs"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("live msgs = %v, want %v", got, want)
	}

	// The pause checkpoint carries the run's whole delta history: the input
	// write (pathless), a's and b's earlier-superstep exit writes
	// (sort-before early paths), and the resumed q's anchor-superstep write
	// (real dispatch path). Its stored order must equal live order.
	pause, err := saver.GetTuple(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil || pause == nil {
		t.Fatalf("GetTuple() = (%v, %v)", pause, err)
	}
	if pause.ParentConfig == nil {
		t.Fatalf("head %s has no parent: the exit flush must anchor on the pause checkpoint", pause.Config.CheckpointID)
	}
	anchor, err := saver.GetTuple(ctx, *pause.ParentConfig)
	if err != nil || anchor == nil {
		t.Fatalf("GetTuple(anchor) = (%v, %v)", anchor, err)
	}
	var replayed []string
	for _, w := range anchor.PendingWrites {
		if w.Channel != "msgs" {
			continue
		}
		v, ok := w.Value.([]string)
		if !ok {
			t.Fatalf("msgs write = %#v, want []string", w.Value)
		}
		replayed = append(replayed, v...)
	}
	if !reflect.DeepEqual(replayed, want) {
		t.Fatalf("anchor pending-writes replay order = %v, want %v (live order)", replayed, want)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["msgs"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("GetState msgs = %v, want %v (replay order != live order)", got, want)
	}
}
