package sqlite_test

// SQL-saver variant of the #9275 messages-delta-channel ID tests (see
// langgraph/graph/message_ids_test.go): a task write of an empty-ID message
// must survive the sqlite serde round-trip with a stable, non-empty ID that a
// targeted RemoveMessage update can address.

import (
	"path/filepath"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/serde"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint/sqlite"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestMessagesDeltaTaskWritesIDsSurviveSQLRoundTrip runs the task-writes path
// through a sqlite saver: the "messages" DeltaChannel (snapshotFrequency=100)
// stores every value as per-task pending writes, so GetState replays them
// through the serde-typed []messages.Message registry entries.
func TestMessagesDeltaTaskWritesIDsSurviveSQLRoundTrip(t *testing.T) {
	saver, err := sqlite.New(filepath.Join(t.TempDir(), "checkpoints.db"), serde.NewJSONSerializer())
	if err != nil {
		t.Fatalf("sqlite.New() error = %v", err)
	}

	orig := []messages.Message{messages.Human("hi")}
	g := graph.NewStateGraph()
	g.AddChannel("messages",
		channels.NewMessagesDeltaChannel(func() any { return []messages.Message{} }, 100))
	g.AddNode("a", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"messages": orig}, nil
	})
	g.AddEdge(types.START, "a")
	g.AddEdge("a", types.END)
	cg, err := g.Compile(graph.WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	ctx := t.Context()
	if _, err := cg.InvokeWithOptions(ctx, map[string]any{}, graph.Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}

	readIDs := func() []string {
		t.Helper()
		snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
		if err != nil {
			t.Fatalf("GetState() error = %v", err)
		}
		msgs, ok := snap.Values["messages"].([]messages.Message)
		if !ok {
			t.Fatalf("messages = %#v, want []messages.Message (typed serde round-trip)", snap.Values["messages"])
		}
		ids := make([]string, len(msgs))
		for i, m := range msgs {
			ids[i] = m.ID
		}
		return ids
	}

	ids1 := readIDs()
	if len(ids1) != 1 || ids1[0] == "" {
		t.Fatalf("message IDs after sqlite round-trip = %v, want one non-empty ID", ids1)
	}
	if ids2 := readIDs(); len(ids2) != 1 || ids2[0] != ids1[0] {
		t.Fatalf("message ID unstable across reads: %v then %v", ids1, ids2)
	}

	// A targeted RemoveMessage by the persisted ID must work through sqlite.
	if _, err := cg.UpdateState(ctx, checkpoint.Config{ThreadID: "t1"},
		map[string]any{"messages": []messages.RemoveMessage{{ID: ids1[0]}}}, "a"); err != nil {
		t.Fatalf("UpdateState(RemoveMessage) error = %v", err)
	}
	snap, err := cg.GetState(ctx, checkpoint.Config{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("GetState() after remove error = %v", err)
	}
	if v, ok := snap.Values["messages"]; ok {
		if msgs, ok := v.([]messages.Message); ok && len(msgs) != 0 {
			t.Fatalf("messages after RemoveMessage = %v, want none", msgs)
		} else if list, ok := v.([]any); ok && len(list) != 0 {
			t.Fatalf("messages after RemoveMessage = %v, want none", list)
		}
	}

	if orig[0].ID != "" {
		t.Fatalf("caller's message mutated: ID = %q, want empty", orig[0].ID)
	}
}
