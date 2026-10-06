package graph

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// counterTestGraph builds n1 -> n2 with a written delta channel ("items") and
// a never-written one ("phantom").
func counterTestGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("items", channels.NewDeltaChannel(appendInts, func() any { return []int{} }, 100))
	g.AddChannel("phantom", channels.NewDeltaChannel(appendInts, func() any { return []int{} }, 100))
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"items": []int{1}}, nil
	})
	g.AddNode("n2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"plain": "v"}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// latestCounters returns the newest checkpoint's delta counters.
func latestCounters(t *testing.T, saver checkpoint.Saver, thread string) map[string][2]int {
	t.Helper()
	tuples, err := saver.List(t.Context(), checkpoint.Config{ThreadID: thread}, checkpoint.ListOptions{})
	if err != nil || len(tuples) == 0 {
		t.Fatalf("List: %d tuples, err=%v", len(tuples), err)
	}
	return tuples[0].Metadata.CountersSinceDeltaSnapshot
}

// TestAdvanceCountersCoverNeverMaterializedChannels verifies the superstep
// counter advances for a registered-but-never-written delta channel, so the
// 5000-superstep staleness bound can eventually fire for it (mirrors Python,
// which materializes every registered channel at graph start).
func TestAdvanceCountersCoverNeverMaterializedChannels(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := counterTestGraph(t, saver)
	if _, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	counters := latestCounters(t, saver, "t1")
	phantom, ok := counters["phantom"]
	if !ok {
		t.Fatalf("phantom missing from CountersSinceDeltaSnapshot: %+v", counters)
	}
	if phantom[1] == 0 {
		t.Fatalf("phantom superstep counter = %d, want > 0", phantom[1])
	}
}

// TestExitDurabilityDeltaCountersMatchSync verifies the single checkpoint an
// exit-durability run persists carries the same delta counters a sync run of
// the same graph produces — flushExit must not double-advance.
func TestExitDurabilityDeltaCountersMatchSync(t *testing.T) {
	syncSaver := checkpoint.NewMemorySaver()
	cgSync := counterTestGraph(t, syncSaver)
	if _, err := cgSync.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "ts"}); err != nil {
		t.Fatalf("sync Invoke: %v", err)
	}
	syncCounters := latestCounters(t, syncSaver, "ts")

	exitSaver := checkpoint.NewMemorySaver()
	cgExit := counterTestGraph(t, exitSaver)
	if _, err := cgExit.InvokeWithOptions(t.Context(), map[string]any{}, Options{
		ThreadID: "te", Durability: DurabilityExit,
	}); err != nil {
		t.Fatalf("exit Invoke: %v", err)
	}
	exitCounters := latestCounters(t, exitSaver, "te")

	for _, name := range []string{"items", "phantom"} {
		sv, sok := syncCounters[name]
		ev, eok := exitCounters[name]
		if sok != eok {
			t.Fatalf("counter presence mismatch for %q: sync=%v exit=%v", name, syncCounters, exitCounters)
		}
		if sok && sv != ev {
			t.Fatalf("counter mismatch for %q: sync=%v exit=%v (double-advance?)", name, sv, ev)
		}
	}
}
