package checkpoint

import (
	"testing"

	"github.com/projanvil/langchain-golang/langchain/internal/agentruntime"
)

func TestMemorySaverGetPutDelete(t *testing.T) {
	saver := NewMemorySaver()

	if _, ok := saver.Get("thread-1"); ok {
		t.Fatal("expected no checkpoint for unknown thread")
	}

	cp := Checkpoint{
		Values:            map[string]any{"messages": []string{"hi"}},
		Next:              "tools",
		PendingInterrupts: []agentruntime.Interrupt{{Value: "confirm?", ID: "int-1"}},
	}
	saver.Put("thread-1", cp)

	got, ok := saver.Get("thread-1")
	if !ok {
		t.Fatal("expected checkpoint to be found")
	}
	if got.Next != "tools" || len(got.PendingInterrupts) != 1 {
		t.Fatalf("unexpected checkpoint: %+v", got)
	}

	saver.Delete("thread-1")
	if _, ok := saver.Get("thread-1"); ok {
		t.Fatal("expected checkpoint to be gone after Delete")
	}
}

func TestMemorySaverZeroValue(t *testing.T) {
	var saver MemorySaver
	if _, ok := saver.Get("x"); ok {
		t.Fatal("expected zero-value MemorySaver to report no checkpoint")
	}
	saver.Put("x", Checkpoint{Next: "n"})
	got, ok := saver.Get("x")
	if !ok || got.Next != "n" {
		t.Fatalf("expected checkpoint to be saved on zero-value MemorySaver, got %+v ok=%v", got, ok)
	}
}

func TestMemorySaverIndependentThreads(t *testing.T) {
	saver := NewMemorySaver()
	saver.Put("a", Checkpoint{Next: "node-a"})
	saver.Put("b", Checkpoint{Next: "node-b"})

	a, _ := saver.Get("a")
	b, _ := saver.Get("b")
	if a.Next != "node-a" || b.Next != "node-b" {
		t.Fatalf("expected independent per-thread checkpoints, got a=%+v b=%+v", a, b)
	}
}
