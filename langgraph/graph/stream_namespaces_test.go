package graph

// Tests for stream namespace isolation of nested invokes (#9458, Python's
// #5674 shape): a graph invoked inside a node WITH ITS OWN ThreadID (not via
// AddSubgraph) starts its namespace at the root, so it must not adopt the
// caller's stream carriers — its messages/custom chunks stay out of the
// caller's stream, whatever Subgraphs says. A same-thread (or thread-less)
// nested invoke keeps streaming as part of the caller's namespace, and the
// AddSubgraph path keeps its child-namespace streaming (regression pins).

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// tokenEmittingNode is a node body that fans one chat-model stream chunk
// through the callbacks manager in its context, recording whether it found a
// non-empty (live) manager.
type tokenEmitObservation struct {
	ran          bool
	managerOK    bool
	managerEmpty bool
}

func tokenEmittingNode(obs *tokenEmitObservation) func(runtime.Runtime, map[string]any) (any, error) {
	return func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		obs.ran = true
		manager, ok := callbacks.ManagerFromContext(ctx)
		obs.managerOK, obs.managerEmpty = ok, manager.Empty()
		if ok {
			chunk := messages.AI("child token")
			chunk.ID = "child-1"
			if err := manager.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelStream, Chunk: chunk}); err != nil {
				return nil, err
			}
		}
		return nil, nil
	}
}

// TestStreamNestedInvokeFreshThreadDoesNotLeak verifies #9458's core rule: a
// child graph invoked inside a node with its own ThreadID must not stream
// into the caller's messages stream — not even with Subgraphs:true, matching
// Python's "drops the messages/tools handlers it inherits, whatever their
// subgraphs" (the child's namespace starts at the root, so the caller's
// emitter would misattribute its chunks).
func TestStreamNestedInvokeFreshThreadDoesNotLeak(t *testing.T) {
	var childObs tokenEmitObservation
	child := NewStateGraph()
	child.AddNode("child_model", tokenEmittingNode(&childObs))
	child.AddEdge(types.START, "child_model")
	child.AddEdge("child_model", types.END)
	compiledChild, err := child.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddNode("outer", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		if _, err := compiledChild.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "child-thread"}); err != nil {
			return nil, err
		}
		return nil, nil
	})
	g.AddEdge(types.START, "outer")
	g.AddEdge("outer", types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Modes:     []StreamMode{StreamMessages, StreamCustom},
		Subgraphs: true,
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if !childObs.ran {
		t.Fatal("child graph never ran; the test is vacuous")
	}
	if len(chunks) != 0 {
		t.Fatalf("len(chunks) = %d, want 0 (fresh-thread child must not stream into the caller's stream): %#v", len(chunks), chunks)
	}
}

// TestStreamNestedInvokeSameThreadStillStreams pins the opposite rule: a
// nested invoke WITHOUT its own thread identity stays part of the caller's
// stream (Python keeps the inherited handlers when the child runs within the
// caller's namespace), so its token arrives under the caller's root namespace
// with the child node's name in the metadata.
func TestStreamNestedInvokeSameThreadStillStreams(t *testing.T) {
	var childObs tokenEmitObservation
	child := NewStateGraph()
	child.AddNode("child_model", tokenEmittingNode(&childObs))
	child.AddEdge(types.START, "child_model")
	child.AddEdge("child_model", types.END)
	compiledChild, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddNode("outer", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		if _, err := compiledChild.Invoke(ctx, map[string]any{}); err != nil {
			return nil, err
		}
		return nil, nil
	})
	g.AddEdge(types.START, "outer")
	g.AddEdge("outer", types.END)
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
	if !childObs.ran {
		t.Fatal("child graph never ran; the test is vacuous")
	}
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1 (a same-namespace nested invoke streams into the caller): %#v", len(chunks), chunks)
	}
	c := chunks[0]
	if c.Mode != StreamMessages || c.Namespace != "" {
		t.Errorf("chunk = {Mode: %q, Namespace: %q}, want {Mode: %q, Namespace: \"\"}", c.Mode, c.Namespace, StreamMessages)
	}
	mc := messageChunkPayload(t, c)
	wantMetadata := map[string]any{
		"langgraph_node":          "child_model",
		"langgraph_step":          0,
		"langgraph_checkpoint_ns": "",
	}
	if !reflect.DeepEqual(mc.Metadata, wantMetadata) {
		t.Errorf("metadata = %v, want %v", mc.Metadata, wantMetadata)
	}
}

// TestStreamNestedInvokeOwnCheckpointIDDoesNotLeak isolates the CheckpointID
// half of #9458's drop rule: a child invoked on the CALLER'S OWN thread (so
// the ThreadID clause cannot fire) but pinned to a CheckpointID of its own
// starts its namespace at the root and must not stream into the caller —
// Python's "own thread_id, checkpoint_ns or checkpoint_id" (1f0cf816d).
func TestStreamNestedInvokeOwnCheckpointIDDoesNotLeak(t *testing.T) {
	var childObs tokenEmitObservation
	child := NewStateGraph()
	child.AddNode("child_model", tokenEmittingNode(&childObs))
	child.AddEdge(types.START, "child_model")
	child.AddEdge("child_model", types.END)
	// The child needs its own saver and a pre-existing checkpoint to pin to;
	// a SEPARATE saver keeps it decoupled from the parent's thread history
	// while sharing the thread ID string.
	compiledChild, err := child.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}
	if _, err := compiledChild.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "shared"}); err != nil {
		t.Fatalf("child warm-up Invoke() error = %v", err)
	}
	snap, err := compiledChild.GetState(t.Context(), checkpoint.Config{ThreadID: "shared"})
	if err != nil {
		t.Fatalf("child GetState() error = %v", err)
	}
	if snap.Config.CheckpointID == "" {
		t.Fatal("child warm-up produced no checkpoint to pin; the test is vacuous")
	}

	g := NewStateGraph()
	g.AddNode("outer", func(ctx runtime.Runtime, _ map[string]any) (any, error) {
		opts := Options{ThreadID: "shared", CheckpointID: snap.Config.CheckpointID}
		if _, err := compiledChild.InvokeWithOptions(ctx, map[string]any{"turn": 2}, opts); err != nil {
			return nil, err
		}
		return nil, nil
	})
	g.AddEdge(types.START, "outer")
	g.AddEdge("outer", types.END)
	cg, err := g.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
		Options:   Options{ThreadID: "shared"},
		Modes:     []StreamMode{StreamMessages},
		Subgraphs: true,
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if !childObs.ran {
		t.Fatal("child graph never ran; the test is vacuous")
	}
	if len(chunks) != 0 {
		t.Fatalf("len(chunks) = %d, want 0 (a child pinned to its own checkpoint must not stream into the caller's stream): %#v", len(chunks), chunks)
	}
}

// TestStreamNestedInvokeAddSubgraphPinStillStreams is the time-travel-pin
// regression pin for the CheckpointID clause: an AddSubgraph child re-entered
// by a RESUMING parent carries a pinned CheckpointID (its pause position,
// recorded in the parent's Metadata.Parents) — invokeSubgraph sets its
// checkpointNS first, so the #9458 drop never fires, and the resumed child's
// token still streams under the subgraph namespace.
func TestStreamNestedInvokeAddSubgraphPinStillStreams(t *testing.T) {
	var childObs tokenEmitObservation
	child := NewStateGraph()
	child.AddNode("child_a", func(runtime.Runtime, map[string]any) (any, error) {
		return map[string]any{"a": "done"}, nil
	})
	child.AddNode("child_b", tokenEmittingNode(&childObs))
	child.AddEdge(types.START, "child_a")
	child.AddEdge("child_a", "child_b")
	child.AddEdge("child_b", types.END)
	compiledChild, err := child.Compile(WithInterruptBefore("child_b"))
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddSubgraph("sub", compiledChild)
	g.AddEdge(types.START, "sub")
	g.AddEdge("sub", types.END)
	cg, err := g.Compile(WithCheckpointer(checkpoint.NewMemorySaver()))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	// First run pauses inside the child at its interrupt_before boundary.
	res, err := cg.InvokeWithOptions(t.Context(), map[string]any{}, Options{ThreadID: "pin-sub"})
	if err != nil {
		t.Fatalf("InvokeWithOptions() error = %v", err)
	}
	if len(res.Interrupts) == 0 {
		t.Fatal("first run did not pause inside the subgraph; the test is vacuous")
	}

	// The resume re-enters the child PINNED to its recorded pause position
	// (opts.CheckpointID set alongside its checkpointNS): child_b runs and
	// its token must still stream under the subgraph namespace.
	chunks, err := collectStream(t, cg.Stream(t.Context(), nil, StreamOptions{
		Options:   Options{ThreadID: "pin-sub"},
		Modes:     []StreamMode{StreamMessages},
		Subgraphs: true,
	}))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1 (the pinned subgraph child still streams): %#v", len(chunks), chunks)
	}
	if chunks[0].Namespace != "sub" {
		t.Errorf("chunk Namespace = %q, want the subgraph namespace %q", chunks[0].Namespace, "sub")
	}
}

// TestStreamNestedInvokeAddSubgraphStillStreams is the regression pin for
// #9458: the AddSubgraph path (a subgraph TASK of the caller, sharing its
// thread) keeps streaming through its child emitter under the subgraph
// namespace, with or without a checkpointer.
func TestStreamNestedInvokeAddSubgraphStillStreams(t *testing.T) {
	for _, tc := range []struct {
		name        string
		checkpoint  bool
		subgraphsOn bool
	}{
		{"checkpointed, subgraphs on", true, true},
		{"uncheckpointed, subgraphs on", false, true},
		{"subgraphs off strips", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var childObs tokenEmitObservation
			child := NewStateGraph()
			child.AddNode("child_model", tokenEmittingNode(&childObs))
			child.AddEdge(types.START, "child_model")
			child.AddEdge("child_model", types.END)

			var compileOpts []CompileOption
			var streamOpts Options
			if tc.checkpoint {
				compileOpts = append(compileOpts, WithCheckpointer(checkpoint.NewMemorySaver()))
				streamOpts = Options{ThreadID: "parent-thread"}
			}
			compiledChild, err := child.Compile()
			if err != nil {
				t.Fatalf("child Compile() error = %v", err)
			}

			g := NewStateGraph()
			g.AddSubgraph("sub", compiledChild)
			g.AddEdge(types.START, "sub")
			g.AddEdge("sub", types.END)
			cg, err := g.Compile(compileOpts...)
			if err != nil {
				t.Fatalf("Compile() error = %v", err)
			}

			chunks, err := collectStream(t, cg.Stream(t.Context(), map[string]any{}, StreamOptions{
				Options:   streamOpts,
				Modes:     []StreamMode{StreamMessages},
				Subgraphs: tc.subgraphsOn,
			}))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			if !childObs.ran {
				t.Fatal("child graph never ran; the test is vacuous")
			}
			if !tc.subgraphsOn {
				if len(chunks) != 0 {
					t.Fatalf("len(chunks) = %d, want 0 with Subgraphs=false", len(chunks))
				}
				return
			}
			if len(chunks) != 1 {
				t.Fatalf("len(chunks) = %d, want 1: %#v", len(chunks), chunks)
			}
			if chunks[0].Namespace != "sub" {
				t.Errorf("chunk Namespace = %q, want the subgraph namespace %q", chunks[0].Namespace, "sub")
			}
		})
	}
}
