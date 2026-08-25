package graph

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestGetStateTasksPendingInterrupt mirrors tests/test_pregel.py:941-944: a
// paused run's snapshot carries one task per planned node, and the paused
// task's entry carries its pending interrupt. Scope per spec 1.4: id, name,
// path, interrupts.
func TestGetStateTasksPendingInterrupt(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"k1": "v1"}, nil
	})
	g.AddNode("n2", func(rt runtime.Runtime, _ map[string]any) (any, error) {
		Interrupt(rt, "need-approval") // pauses the run (panics GraphInterrupt)
		return map[string]any{"k2": "v2"}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", types.END)
	cg, err := g.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	ctx := context.Background()

	result, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(result.Interrupts) != 1 {
		t.Fatalf("expected a pause, got %+v", result.Interrupts)
	}

	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if len(snap.Tasks) != 1 {
		t.Fatalf("Tasks = %+v, want exactly the planned n2 task", snap.Tasks)
	}
	task := snap.Tasks[0]
	if task.Name != "n2" {
		t.Fatalf("Tasks[0].Name = %q, want n2", task.Name)
	}
	if task.ID == "" {
		t.Fatal("Tasks[0].ID must be the planned task ID (non-empty)")
	}
	if len(task.Interrupts) != 1 || task.Interrupts[0].Value != "need-approval" {
		t.Fatalf("Tasks[0].Interrupts = %+v, want the pending need-approval interrupt", task.Interrupts)
	}
	// The snapshot-level Interrupts projection stays consistent with Tasks.
	if len(snap.Interrupts) != 1 || snap.Interrupts[0].ID != task.Interrupts[0].ID {
		t.Fatalf("Interrupts = %+v, want the same interrupt as Tasks[0]", snap.Interrupts)
	}
}

// TestGetStateTasksBoundaryInterrupt: interrupt_before pauses stamp their
// interrupt write with the NODE NAME (resume.go:57-66), not a planned task
// ID — Tasks must still attach it to the planned task.
func TestGetStateTasksBoundaryInterrupt(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	g := NewStateGraph()
	g.AddNode("n1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"k1": "v1"}, nil
	})
	g.AddNode("n2", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"k2": "v2"}, nil
	})
	g.AddEdge(types.START, "n1")
	g.AddEdge("n1", "n2")
	g.AddEdge("n2", types.END)
	cg, err := g.Compile(WithCheckpointer(saver), WithInterruptBefore("n2"))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	ctx := context.Background()

	if _, err := cg.InvokeWithOptions(ctx, map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if len(snap.Tasks) != 1 || snap.Tasks[0].Name != "n2" {
		t.Fatalf("Tasks = %+v, want [n2]", snap.Tasks)
	}
	if len(snap.Tasks[0].Interrupts) != 1 || snap.Tasks[0].Interrupts[0].ID != interruptBeforeID+"n2" {
		t.Fatalf("Tasks[0].Interrupts = %+v, want the interrupt-before-n2 interrupt", snap.Tasks[0].Interrupts)
	}
}

// TestGetStateTasksCompletedRun: a finished run's final checkpoint plans no
// tasks (Python: state.tasks == () when next == ()).
func TestGetStateTasksCompletedRun(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := snapshotLinearGraph(t, saver, map[string]int{})
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	snap, err := cg.GetState(context.Background(), checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState: %v", err)
	}
	if len(snap.Tasks) != 0 {
		t.Fatalf("Tasks = %+v, want empty for a completed run", snap.Tasks)
	}
	if len(snap.Next) != 0 {
		t.Fatalf("Next = %+v, want empty for a completed run", snap.Next)
	}
}

// TestGetStateHistoryTasks: mid-history snapshots expose the task planned at
// that point (the n1-checkpoint plans n2).
func TestGetStateHistoryTasks(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	cg := snapshotLinearGraph(t, saver, map[string]int{})
	if _, err := cg.InvokeWithOptions(context.Background(), map[string]any{"k0": "v0"}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	snaps, err := cg.GetStateHistory(context.Background(), checkpoint.Config{ThreadID: "t1"}, checkpoint.ListOptions{})
	if err != nil {
		t.Fatalf("GetStateHistory: %v", err)
	}
	// 4 checkpoints (input + 3 supersteps), newest first: the step-0 loop
	// checkpoint (index 2) planned n2.
	if len(snaps) != 4 {
		t.Fatalf("history = %d snapshots, want 4", len(snaps))
	}
	if len(snaps[2].Tasks) != 1 || snaps[2].Tasks[0].Name != "n2" {
		t.Fatalf("snaps[2].Tasks = %+v, want [n2]", snaps[2].Tasks)
	}
	if snaps[2].Tasks[0].ID == "" {
		t.Fatal("snaps[2].Tasks[0].ID must be non-empty")
	}
}
