package graph

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// messageChunkPayload extracts the MessageChunk payload from a stream chunk.
func messageChunkPayload(t *testing.T, c StreamChunk) MessageChunk {
	t.Helper()
	mc, ok := c.Payload.(MessageChunk)
	if !ok {
		t.Fatalf("chunk payload type = %T, want MessageChunk", c.Payload)
	}
	return mc
}

// TestStreamMessagesChunksAndMetadata drives a node that fans model events
// into the manager installed in its context: two streamed token chunks plus a
// final end message whose ID was already streamed (deduped, Python parity) and
// an end message with an unseen ID (emitted).
func TestStreamMessagesChunksAndMetadata(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("model", func(ctx context.Context, _ map[string]any) (any, error) {
		manager, ok := callbacks.ManagerFromContext(ctx)
		if !ok {
			t.Errorf("ManagerFromContext() ok = false, want an installed manager under StreamMessages")
			return nil, nil
		}
		chunk1 := messages.AI("Hel")
		chunk1.ID = "run-1"
		chunk2 := messages.AI("lo")
		chunk2.ID = "run-1"
		final := messages.AI("Hello")
		final.ID = "run-1" // already streamed: must be deduped
		other := messages.AI("other")
		other.ID = "run-2" // unseen: must be emitted
		for _, event := range []callbacks.Event{
			{Kind: callbacks.EventChatModelStream, Chunk: chunk1},
			{Kind: callbacks.EventChatModelStream, Chunk: chunk2},
			{Kind: callbacks.EventChatModelEnd, Output: final},
			{Kind: callbacks.EventChatModelEnd, Output: other},
		} {
			if err := manager.Emit(ctx, event); err != nil {
				t.Errorf("Emit(%s) error = %v", event.Kind, err)
			}
		}
		return map[string]any{"done": true}, nil
	})
	g.AddEdge(types.START, "model")
	g.AddEdge("model", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"in": 1}, StreamOptions{
		Modes: []StreamMode{StreamMessages},
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("len(chunks) = %d, want 3 (two token chunks + unseen end message; seen end message deduped): %#v", len(chunks), chunks)
	}

	wantContents := []string{"Hel", "lo", "other"}
	wantIDs := []string{"run-1", "run-1", "run-2"}
	// Step 0 matches the debug mode's first-superstep numbering (rs.step
	// starts at -1, Python parity).
	wantMetadata := map[string]any{
		"langgraph_node":          "model",
		"langgraph_step":          0,
		"langgraph_checkpoint_ns": "",
	}
	for i, c := range chunks {
		if c.Mode != StreamMessages {
			t.Errorf("chunks[%d].Mode = %q, want %q", i, c.Mode, StreamMessages)
		}
		if c.Namespace != "" {
			t.Errorf("chunks[%d].Namespace = %q, want root %q", i, c.Namespace, "")
		}
		mc := messageChunkPayload(t, c)
		if mc.Message.Content != wantContents[i] || mc.Message.ID != wantIDs[i] {
			t.Errorf("chunks[%d] message = {Content: %q, ID: %q}, want {Content: %q, ID: %q}",
				i, mc.Message.Content, mc.Message.ID, wantContents[i], wantIDs[i])
		}
		if !reflect.DeepEqual(mc.Metadata, wantMetadata) {
			t.Errorf("chunks[%d] metadata = %v, want %v", i, mc.Metadata, wantMetadata)
		}
	}
}

// TestStreamMessagesLLMStringChunk verifies that a legacy LLM string token is
// wrapped into an AI message chunk.
func TestStreamMessagesLLMStringChunk(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("llm", func(ctx context.Context, _ map[string]any) (any, error) {
		manager, ok := callbacks.ManagerFromContext(ctx)
		if !ok {
			t.Errorf("ManagerFromContext() ok = false, want an installed manager under StreamMessages")
			return nil, nil
		}
		if err := manager.Emit(ctx, callbacks.Event{Kind: callbacks.EventLLMStream, Chunk: "tok"}); err != nil {
			t.Errorf("Emit() error = %v", err)
		}
		return nil, nil
	})
	g.AddEdge(types.START, "llm")
	g.AddEdge("llm", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"in": 1}, StreamOptions{
		Modes: []StreamMode{StreamMessages},
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1: %#v", len(chunks), chunks)
	}
	mc := messageChunkPayload(t, chunks[0])
	if mc.Message.Role != messages.RoleAI || mc.Message.Content != "tok" {
		t.Errorf("message = {Role: %q, Content: %q}, want {Role: %q, Content: %q}",
			mc.Message.Role, mc.Message.Content, messages.RoleAI, "tok")
	}
}

// TestStreamCustomMode verifies that payloads written through the StreamWriter
// flow to the chunk stream, namespaced with the emitting node — at the root
// and inside a subgraph.
func TestStreamCustomMode(t *testing.T) {
	child := NewStateGraph()
	child.AddNode("inner", func(ctx context.Context, _ map[string]any) (any, error) {
		writer := StreamWriterFromContext(ctx)
		if writer == nil {
			t.Errorf("StreamWriterFromContext() = nil, want a writer under StreamCustom")
			return nil, nil
		}
		writer("inner progress")
		return nil, nil
	})
	child.AddEdge(types.START, "inner")
	child.AddEdge("inner", types.END)
	compiledChild, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddNode("worker", func(ctx context.Context, _ map[string]any) (any, error) {
		writer := StreamWriterFromContext(ctx)
		if writer == nil {
			t.Errorf("StreamWriterFromContext() = nil, want a writer under StreamCustom")
			return nil, nil
		}
		writer("progress: 50%")
		return map[string]any{"done": true}, nil
	})
	g.AddSubgraph("sub", compiledChild)
	g.AddEdge(types.START, "worker")
	g.AddEdge("worker", "sub")
	g.AddEdge("sub", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"in": 1}, StreamOptions{
		Modes:     []StreamMode{StreamCustom},
		Subgraphs: true,
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("len(chunks) = %d, want 2: %#v", len(chunks), chunks)
	}
	if chunks[0].Mode != StreamCustom || chunks[0].Namespace != "" || chunks[0].Payload != "progress: 50%" {
		t.Errorf("chunks[0] = %#v, want {Mode: custom, Namespace: \"\", Payload: \"progress: 50%%\"}", chunks[0])
	}
	if chunks[1].Mode != StreamCustom || chunks[1].Namespace != "sub" || chunks[1].Payload != "inner progress" {
		t.Errorf("chunks[1] = %#v, want {Mode: custom, Namespace: \"sub\", Payload: \"inner progress\"}", chunks[1])
	}
}

// TestStreamMessagesCustomInert verifies both modes leave no trace when not
// requested: no manager and no writer are installed, on both the streaming
// (other modes) and plain Invoke paths.
func TestStreamMessagesCustomInert(t *testing.T) {
	var managerOK bool
	var writer StreamWriter
	probe := func(ctx context.Context, _ map[string]any) (any, error) {
		_, managerOK = callbacks.ManagerFromContext(ctx)
		writer = StreamWriterFromContext(ctx)
		return nil, nil
	}

	g := NewStateGraph()
	g.AddNode("probe", probe)
	g.AddEdge(types.START, "probe")
	g.AddEdge("probe", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := collectStream(t, cg.Stream(context.Background(), map[string]any{"in": 1}, StreamOptions{
		Modes: []StreamMode{StreamValues},
	})); err != nil {
		t.Fatalf("Stream(values) error = %v", err)
	}
	if managerOK {
		t.Errorf("ManagerFromContext() ok = true under values-only stream, want false")
	}
	if writer != nil {
		t.Errorf("StreamWriterFromContext() non-nil under values-only stream, want nil")
	}

	managerOK = false
	writer = nil
	if _, err := cg.Invoke(context.Background(), map[string]any{"in": 1}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if managerOK {
		t.Errorf("ManagerFromContext() ok = true under Invoke, want false")
	}
	if writer != nil {
		t.Errorf("StreamWriterFromContext() non-nil under Invoke, want nil")
	}
}
