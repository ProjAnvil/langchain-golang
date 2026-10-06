package graph

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// TestStreamMessagesFiltersInternalCalls drives two nodes fanning identical
// model events into their installed managers: node "internal" marks its
// emission context with callbacks.WithInternalCall (middleware bookkeeping
// call), node "visible" does not. Only the visible node's chunks may surface
// in messages mode (Python langchain #39252 InternalCallTransformer).
func TestStreamMessagesFiltersInternalCalls(t *testing.T) {
	emit := func(ctx context.Context, id string) {
		manager, ok := callbacks.ManagerFromContext(ctx)
		if !ok {
			t.Errorf("ManagerFromContext() ok = false for %s", id)
			return
		}
		chunk := messages.AI("tok-" + id)
		chunk.ID = id
		_ = manager.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelStream, Chunk: chunk})
	}

	g := NewStateGraph()
	g.AddNode("internal", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		emit(callbacks.WithInternalCall(ctx), "internal-run")
		return map[string]any{"a": 1}, nil
	})
	g.AddNode("visible", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		emit(ctx, "visible-run")
		return map[string]any{"b": 2}, nil
	})
	g.AddEdge(types.START, "internal")
	g.AddEdge("internal", "visible")
	g.AddEdge("visible", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	var got []MessageChunk
	for chunk := range cg.Stream(t.Context(), map[string]any{}, StreamOptions{Modes: []StreamMode{StreamMessages}}) {
		got = append(got, messageChunkPayload(t, chunk))
	}
	if len(got) != 1 {
		t.Fatalf("expected exactly one messages chunk (internal call filtered), got %d: %+v", len(got), got)
	}
	if got[0].Message.ID != "visible-run" {
		t.Fatalf("surviving chunk = %+v, want visible-run", got[0])
	}
}

// TestStreamMessagesInternalCallStillUpdatesValues ensures the filter only
// affects the messages projection: node outputs and updates chunks are
// delivered for internal-marked nodes exactly like any other node.
func TestStreamMessagesInternalCallStillUpdatesValues(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("internal", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		manager, ok := callbacks.ManagerFromContext(ctx)
		if ok {
			chunk := messages.AI("hidden")
			chunk.ID = "internal-run"
			_ = manager.Emit(callbacks.WithInternalCall(ctx), callbacks.Event{
				Kind: callbacks.EventChatModelStream, Chunk: chunk,
			})
		}
		return map[string]any{"a": 1}, nil
	})
	g.AddEdge(types.START, "internal")
	g.AddEdge("internal", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	sawMessages, sawUpdates, sawValues := 0, 0, 0
	ch := cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes: []StreamMode{StreamMessages, StreamUpdates, StreamValues},
	})
	for chunk := range ch {
		switch chunk.Mode {
		case StreamMessages:
			sawMessages++
		case StreamUpdates:
			sawUpdates++
		case StreamValues:
			sawValues++
		}
	}
	if sawMessages != 0 {
		t.Fatalf("internal call leaked into messages mode: %d chunks", sawMessages)
	}
	if sawUpdates == 0 || sawValues == 0 {
		t.Fatalf("updates/values must be unaffected: updates=%d values=%d", sawUpdates, sawValues)
	}
}
