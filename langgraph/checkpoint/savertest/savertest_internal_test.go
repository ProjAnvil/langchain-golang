package savertest

import (
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
)

func TestTupleIDs(t *testing.T) {
	if got := tupleIDs(nil); len(got) != 0 {
		t.Fatalf("tupleIDs(nil) = %v, want empty", got)
	}
	tups := []checkpoint.Tuple{
		{Checkpoint: checkpoint.Checkpoint{ID: "a"}},
		{Checkpoint: checkpoint.Checkpoint{ID: "b"}},
		{Checkpoint: checkpoint.Checkpoint{ID: "c"}},
	}
	got := tupleIDs(tups)
	if len(got) != len(tups) {
		t.Fatalf("tupleIDs returned %d ids, want %d", len(got), len(tups))
	}
	for i, want := range []string{"a", "b", "c"} {
		if got[i] != want {
			t.Fatalf("tupleIDs[%d] = %q, want %q", i, got[i], want)
		}
	}
}
