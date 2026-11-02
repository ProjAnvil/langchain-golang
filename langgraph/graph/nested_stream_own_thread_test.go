package graph

// Regression (v0.12.0..v0.12.4): a node that STREAMS a child graph carrying
// its own ThreadID must still receive the child's chunks. Stream installed
// its emitter into ctx before calling run, so stripNestedInvokeCarriers
// (#9458) could not tell the run's own emitter from an inherited one and
// stripped it — the nested Stream yielded zero chunks and ended cleanly.

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
)

func nestedStreamReducer(existing, update any) (any, error) {
	return channels.AppendSliceReducer(existing, update)
}

// newNestedStreamGraphs builds the two graphs used by the nested-Stream
// tests: a single-node child and a single-node parent whose only node runs
// the child. parentRun receives the parent's own chunk count.
func newNestedStreamGraphs(t *testing.T, innerThread string, innerChunks *int, parentChunks *int) (*CompiledGraph, *CompiledGraph) {
	t.Helper()

	b := NewStateGraph()
	b.AddReducer("messages", nestedStreamReducer)
	b.AddNode("bnode", func(rt runtime.Runtime, state map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{messages.AI("from-b")}}, nil
	})
	b.SetEntryPoint("bnode")
	b.SetFinishPoint("bnode")
	bg, err := b.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatal(err)
	}

	a := NewStateGraph()
	a.AddReducer("messages", nestedStreamReducer)
	a.AddNode("anode", func(rt runtime.Runtime, state map[string]any) (any, error) {
		for chunk, cerr := range bg.Stream(rt, map[string]any{"messages": []messages.Message{messages.Human("hi")}},
			StreamOptions{
				Options: Options{ThreadID: innerThread},
				Modes:   []StreamMode{StreamUpdates, StreamValues, StreamCustom},
			}) {
			if cerr != nil {
				t.Errorf("inner stream err: %v", cerr)
				break
			}
			_ = chunk
			(*innerChunks)++
		}
		return map[string]any{}, nil
	})
	a.SetEntryPoint("anode")
	a.SetFinishPoint("anode")
	ag, err := a.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatal(err)
	}
	_ = parentChunks
	return ag, bg
}

func TestNestedStreamOwnThreadStillYields(t *testing.T) {
	inner, outer := 0, 0
	ag, _ := newNestedStreamGraphs(t, "inner-thread-1", &inner, &outer)

	chunks := 0
	for chunk, err := range ag.Stream(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("go")}},
		StreamOptions{
			Options: Options{ThreadID: "outer-thread-1"},
			Modes:   []StreamMode{StreamUpdates, StreamValues},
		}) {
		if err != nil {
			t.Fatalf("outer stream err: %v", err)
		}
		_ = chunk
		chunks++
	}
	if inner == 0 {
		t.Error("inner stream yielded ZERO chunks — nested Stream lost its own emitter")
	}
	if chunks == 0 {
		t.Error("outer stream yielded ZERO chunks")
	}
}

// TestNestedStreamSameThreadShadowsCallerEmitter pins the shadow contract:
// when a node STREAMS a child on the caller's own thread, the child's chunks
// go to the node's own consumer (the Stream call it holds), not to the
// caller's stream — the child's emitter shadows the caller's for that run.
func TestNestedStreamSameThreadShadowsCallerEmitter(t *testing.T) {
	inner := 0
	ag, _ := newNestedStreamGraphs(t, "outer-thread-1", &inner, nil)

	chunks := 0
	for chunk, err := range ag.Stream(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("go")}},
		StreamOptions{
			Options: Options{ThreadID: "outer-thread-1"},
			Modes:   []StreamMode{StreamUpdates, StreamValues},
		}) {
		if err != nil {
			t.Fatalf("outer stream err: %v", err)
		}
		if _, ok := chunk.Payload.(map[string]any); ok {
			// Count only state-bearing chunks from the anode task itself; the
			// child's payloads must never appear on this channel.
			chunks++
		}
	}
	if inner == 0 {
		t.Fatal("inner stream yielded ZERO chunks — same-thread nested Stream must deliver to its own consumer")
	}
	for chunk, err := range ag.Stream(context.Background(),
		map[string]any{"messages": []messages.Message{messages.Human("again")}},
		StreamOptions{
			Options: Options{ThreadID: "outer-thread-1"},
			Modes:   []StreamMode{StreamUpdates},
		}) {
		if err != nil {
			t.Fatalf("second outer stream err: %v", err)
		}
		updates, _ := chunk.Payload.(map[string]any)
		for node, v := range updates {
			if node == "bnode" {
				t.Fatalf("child node payload leaked into the caller's stream: %v", v)
			}
		}
	}
}
