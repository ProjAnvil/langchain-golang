package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// exitPauseNewTurnGraph builds in -> a -> q -> r under exit durability: `a`
// and `r` write to a delta-channel `log` (cadence never fires), and `q`
// interrupts unless the state's "gate" is set. The carry-in repro shape from
// task-2's review: turn 1 pauses at q; turn 2 (a NEW TURN with fresh input)
// completes through r while anchoring on the pause checkpoint.
func exitPauseNewTurnGraph(t *testing.T, saver checkpoint.Saver) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("log", channels.NewDeltaChannel(stringBatchReducer, func() any { return []string{} }, 100))
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"a"}}, nil
	})
	g.AddNode("q", func(rt runtime.Runtime, state map[string]any) (any, error) {
		if gated, _ := state["gate"].(bool); !gated {
			Interrupt(rt, "q")
		}
		return nil, nil
	})
	g.AddNode("r", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"log": []string{"r"}}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "q")
	g.AddEdge("q", "r")
	g.AddEdge("r", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithDurability(DurabilityExit))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// TestExitNewTurnAfterPauseKeepsInputWrite: a completing new-turn exit run on
// a PAUSED thread must keep its input write. Both turns anchor writes on the
// pause checkpoint (turn 1 via putPauseCheckpointExit, turn 2 via flushExit's
// persisted-parent anchor), and their input writes keyed identically
// (exitDeltaTaskID(0, NullTaskID), pathless) — savers' first-write-wins slots
// silently dropped the new turn's input, leaving the stored state reading
// [in1 a a q r] against the live [in1 a in2 a q r] (carry-in ledgered from
// task 2's review; resolved by the #9235 new-turn pending-writes force
// snapshot, which materializes the channel into the completing run's final
// checkpoint and filters its writes out of the colliding anchor slots).
func TestExitNewTurnAfterPauseKeepsInputWrite(t *testing.T) {
	ctx := t.Context()
	saver := checkpoint.NewMemorySaver()
	cg := exitPauseNewTurnGraph(t, saver)
	cfg := Options{ThreadID: "t"}

	// Turn 1: pauses at q.
	res, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in1"}}, cfg)
	if err != nil {
		t.Fatalf("turn 1 Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 1 {
		t.Fatalf("turn 1 interrupts = %+v, want one pause at q", res.Interrupts)
	}

	// Turn 2: a NEW TURN with fresh input, gated through q to completion.
	live, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in2"}, "gate": true}, cfg)
	if err != nil {
		t.Fatalf("turn 2 Invoke() error = %v", err)
	}
	want := []string{"in1", "a", "in2", "a", "r"}
	if got, _ := live.Values["log"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("turn 2 live log = %v, want %v", got, want)
	}

	// The stored state must read exactly what the live run produced — the
	// new turn's input write may not be dropped by colliding with turn 1's
	// pause-anchored input write.
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() error = %v", err)
	}
	if got, _ := snap.Values["log"].([]string); !reflect.DeepEqual(got, want) {
		t.Fatalf("stored log = %v, want %v (the new turn's input write was dropped)", got, want)
	}

	// And a third turn continues the thread correctly from there.
	res3, err := cg.InvokeWithOptions(ctx, map[string]any{"log": []string{"in3"}, "gate": true}, cfg)
	if err != nil {
		t.Fatalf("turn 3 Invoke() error = %v", err)
	}
	want3 := append(append([]string{}, want...), "in3", "a", "r")
	if got, _ := res3.Values["log"].([]string); !reflect.DeepEqual(got, want3) {
		t.Fatalf("turn 3 live log = %v, want %v", got, want3)
	}
	snap3, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t"})
	if err != nil {
		t.Fatalf("GetState() after turn 3 error = %v", err)
	}
	if got, _ := snap3.Values["log"].([]string); !reflect.DeepEqual(got, want3) {
		t.Fatalf("stored log after turn 3 = %v, want %v", got, want3)
	}
}
