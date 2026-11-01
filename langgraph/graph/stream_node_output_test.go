package graph

// Tests for streaming node-RETURNED messages in messages mode (#9463):
// Python's StreamMessagesHandler scans each node's return value at
// on_chain_end (`_find_and_emit_messages`, pregel/_messages.py) and emits
// messages not yet seen (deduped by ID against everything the run has
// streamed — model chunks and earlier node outputs — with the node's input
// messages marked seen at on_chain_start). The Go port emits from the task
// outcomes the run loop already computes, through the same per-node metadata
// the chat-model bridge uses.

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// nodeOutputMessagesGraph builds START -> respond -> END over a "messages"
// key backed by a messages delta channel (the canonical messages state),
// whose node returns nodeResult.
func nodeOutputMessagesGraph(t *testing.T, nodeResult func(runtime.Runtime, map[string]any) (any, error)) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddChannel("messages",
		channels.NewMessagesDeltaChannel(func() any { return []messages.Message{} }, 100))
	g.AddNode("respond", nodeResult)
	g.AddEdge(types.START, "respond")
	g.AddEdge("respond", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return cg
}

// messageContents maps messages chunks to their contents, failing on any
// non-messages chunk.
func messageContents(t *testing.T, chunks []StreamChunk) []string {
	t.Helper()
	out := make([]string, 0, len(chunks))
	for i, c := range chunks {
		mc := messageChunkPayload(t, c)
		if c.Mode != StreamMessages {
			t.Errorf("chunks[%d].Mode = %q, want %q", i, c.Mode, StreamMessages)
		}
		out = append(out, mc.Message.Content)
	}
	return out
}

// TestStreamMessagesNodeReturnedMessage verifies the top-level shape of
// #9463: a node that RETURNS a message (no model ever streams it) still
// surfaces it in messages mode, exactly like Python's
// test_subgraph_streaming_node_output_messages_sync ("hi there" from a
// returning respond node).
func TestStreamMessagesNodeReturnedMessage(t *testing.T) {
	ai := messages.AI("hi there")
	ai.ID = "ai1"
	cg := nodeOutputMessagesGraph(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{ai}}, nil
	})

	chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes: []StreamMode{StreamMessages},
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if got := messageContents(t, chunks); !reflect.DeepEqual(got, []string{"hi there"}) {
		t.Fatalf("message contents = %v, want [hi there]: %#v", got, chunks)
	}
	mc := messageChunkPayload(t, chunks[0])
	if mc.Message.ID != "ai1" {
		t.Errorf("message ID = %q, want the node's own %q", mc.Message.ID, "ai1")
	}
	wantMetadata := map[string]any{
		"langgraph_node":          "respond",
		"langgraph_step":          0,
		"langgraph_checkpoint_ns": "",
	}
	if !reflect.DeepEqual(mc.Metadata, wantMetadata) {
		t.Errorf("metadata = %v, want %v", mc.Metadata, wantMetadata)
	}
}

// TestStreamMessagesNodeReturnedUnstampedID verifies a message returned
// without an ID on a messages delta channel is stamped by the #9275 hook
// before emission, so the streamed chunk carries the ID the checkpoint will
// persist.
func TestStreamMessagesNodeReturnedUnstampedID(t *testing.T) {
	cg := nodeOutputMessagesGraph(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{messages.AI("no id")}}, nil
	})

	chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes: []StreamMode{StreamMessages},
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1: %#v", len(chunks), chunks)
	}
	mc := messageChunkPayload(t, chunks[0])
	if mc.Message.ID == "" {
		t.Error("message ID = \"\", want a #9275-stamped non-empty ID on the streamed chunk")
	}
}

// TestStreamMessagesNodeOutputDedupedByRunScopedSeen verifies the dedup
// window of #9463 against Python's single shared handler: a message whose ID
// the run already streamed as model chunks (the final message of a streaming
// model) is not re-emitted from the node's return value, and a node echoing
// its input's messages does not re-emit them either.
func TestStreamMessagesNodeOutputDedupedByRunScopedSeen(t *testing.T) {
	g := NewStateGraph()
	g.AddChannel("messages",
		channels.NewMessagesDeltaChannel(func() any { return []messages.Message{} }, 100))
	g.AddNode("model", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		manager, ok := callbacks.ManagerFromContext(ctx)
		if !ok {
			t.Errorf("ManagerFromContext() ok = false, want an installed manager under StreamMessages")
			return nil, nil
		}
		chunk := messages.AI("Hel")
		chunk.ID = "run-1"
		if err := manager.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelStream, Chunk: chunk}); err != nil {
			t.Errorf("Emit() error = %v", err)
		}
		// The final message carries the same ID the chunks streamed under,
		// so the node-output scan must skip it.
		final := messages.AI("Hello")
		final.ID = "run-1"
		return map[string]any{"messages": []messages.Message{final}}, nil
	})
	g.AddNode("echo", func(_ runtime.Runtime, state map[string]any) (any, error) {
		// Echo the input's messages verbatim: every ID was already seen.
		return map[string]any{"messages": state["messages"]}, nil
	})
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "echo")
	g.AddEdge("echo", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes: []StreamMode{StreamMessages},
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if got := messageContents(t, chunks); !reflect.DeepEqual(got, []string{"Hel"}) {
		t.Fatalf("message contents = %v, want just the streamed chunk [Hel] (final message and echo deduped): %#v", got, chunks)
	}
}

// TestStreamMessagesNodeOutputGraphStreamedInNode is the direct Go analogue
// of Python's #9463 repro: a node explicitly streams a subgraph with
// messages mode and consumes its iterator; a node of that subgraph RETURNS a
// message, which must reach the consuming node (previously only chat-model
// events were bridged, so returned messages were dropped).
func TestStreamMessagesNodeOutputGraphStreamedInNode(t *testing.T) {
	ai := messages.AI("hi there")
	ai.ID = "ai1"
	subgraph := nodeOutputMessagesGraph(t, func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{ai}}, nil
	})

	var inNode []string
	g := NewStateGraph()
	g.AddNode("parent", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		for c := range subgraph.Stream(ctx, map[string]any{}, StreamOptions{
			Modes: []StreamMode{StreamMessages},
		}) {
			inNode = append(inNode, messageChunkPayload(t, c).Message.Content)
		}
		return nil, nil
	})
	g.AddEdge(types.START, "parent")
	g.AddEdge("parent", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	if _, err := cg.Invoke(t.Context(), map[string]any{}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if !reflect.DeepEqual(inNode, []string{"hi there"}) {
		t.Fatalf("messages consumed inside the node = %v, want [hi there]", inNode)
	}
}

// TestStreamMessagesNodeOutputFromSubgraph verifies the AddSubgraph path: a
// subgraph node's RETURNED message streams under the subgraph namespace when
// Subgraphs is set, with the child's checkpoint namespace in the metadata.
func TestStreamMessagesNodeOutputFromSubgraph(t *testing.T) {
	ai := messages.AI("inner reply")
	ai.ID = "inner-1"
	child := NewStateGraph()
	child.AddChannel("messages",
		channels.NewMessagesDeltaChannel(func() any { return []messages.Message{} }, 100))
	child.AddNode("respond", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{ai}}, nil
	})
	child.AddEdge(types.START, "respond")
	child.AddEdge("respond", types.END)
	compiledChild, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddSubgraph("sub", compiledChild)
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes:     []StreamMode{StreamMessages},
		Subgraphs: true,
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1: %#v", len(chunks), chunks)
	}
	c := chunks[0]
	if c.Mode != StreamMessages || c.Namespace != "sub" {
		t.Errorf("chunk = {Mode: %q, Namespace: %q}, want {Mode: %q, Namespace: %q}", c.Mode, c.Namespace, StreamMessages, "sub")
	}
	mc := messageChunkPayload(t, c)
	if mc.Message.Content != "inner reply" || mc.Message.ID != "inner-1" {
		t.Errorf("message = {Content: %q, ID: %q}, want {Content: %q, ID: %q}", mc.Message.Content, mc.Message.ID, "inner reply", "inner-1")
	}
	wantMetadata := map[string]any{
		"langgraph_node":          "respond",
		"langgraph_step":          0,
		"langgraph_checkpoint_ns": "sub",
	}
	if !reflect.DeepEqual(mc.Metadata, wantMetadata) {
		t.Errorf("metadata = %v, want %v", mc.Metadata, wantMetadata)
	}
}
