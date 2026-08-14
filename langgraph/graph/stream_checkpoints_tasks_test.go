package graph

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestStreamCheckpointsMode pins the "checkpoints" stream mode: every chunk is
// a StreamCheckpoints chunk whose payload is a StateSnapshot (the GetState
// shape, Python's checkpoints-mode contract) with non-empty Values.
func TestStreamCheckpointsMode(t *testing.T) {
	cg := streamLinearGraph(t, WithCheckpointer(checkpoint.NewMemorySaver()))
	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Options: Options{ThreadID: "t"}, Modes: []StreamMode{StreamCheckpoints}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var snapshots []StateSnapshot
	for _, c := range chunks {
		if c.Mode != StreamCheckpoints {
			t.Fatalf("chunk mode = %q, want checkpoints", c.Mode)
		}
		snap, ok := c.Payload.(StateSnapshot)
		if !ok {
			t.Fatalf("checkpoints payload is %T, want StateSnapshot", c.Payload)
		}
		if len(snap.Values) == 0 {
			t.Fatalf("checkpoint %q has empty Values", snap.Config.CheckpointID)
		}
		snapshots = append(snapshots, snap)
	}
	if len(snapshots) == 0 {
		t.Fatal("no checkpoints chunks emitted")
	}
	// The first checkpoint is the input checkpoint (step -1) carrying the
	// initial state.
	if got := snapshots[0].Values["v"]; got != 0 {
		t.Fatalf("first checkpoint v = %v, want 0", got)
	}
}

// TestStreamTasksMode pins the "tasks" stream mode on a single-node graph: one
// TaskEvent for the dispatch and one TaskResultEvent for the completion, both
// naming the node.
func TestStreamTasksMode(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("n", func(_ runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"v": state["v"].(int) + 1}, nil
	})
	g.AddEdge(types.START, "n")
	g.AddEdge("n", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Modes: []StreamMode{StreamTasks}}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}

	var starts, results int
	for _, c := range chunks {
		if c.Mode != StreamTasks {
			t.Fatalf("chunk mode = %q, want tasks", c.Mode)
		}
		switch p := c.Payload.(type) {
		case TaskEvent:
			if p.Name != "n" {
				t.Fatalf("task start Name = %q, want n", p.Name)
			}
			if p.Input == nil {
				t.Fatalf("task start Input is nil")
			}
			starts++
		case TaskResultEvent:
			if p.Name != "n" {
				t.Fatalf("task result Name = %q, want n", p.Name)
			}
			if !reflect.DeepEqual(p.Result, map[string]any{"v": 1}) {
				t.Fatalf("task result Result = %+v, want {v:1}", p.Result)
			}
			results++
		default:
			t.Fatalf("tasks payload is %T, want TaskEvent or TaskResultEvent", c.Payload)
		}
	}
	if starts != 1 {
		t.Fatalf("got %d task start chunks, want 1", starts)
	}
	if results != 1 {
		t.Fatalf("got %d task result chunks, want 1", results)
	}
}

// TestStreamCheckpointsTasksNotRejected pins that Stream() accepts the two new
// modes instead of rejecting them as unknown.
func TestStreamCheckpointsTasksNotRejected(t *testing.T) {
	cg := streamLinearGraph(t)
	if _, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"v": 0},
		StreamOptions{Modes: []StreamMode{StreamCheckpoints, StreamTasks}})); err != nil {
		t.Fatalf("Stream() rejected checkpoints/tasks modes: %v", err)
	}
}
