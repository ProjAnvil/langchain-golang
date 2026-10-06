package graph

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// subgraphDeltaGraph builds parent -> child, where the child owns an "items"
// delta channel (freq=100, sentinel-only storage).
func subgraphDeltaGraph(t *testing.T, saver checkpoint.Saver) (*CompiledGraph, *CompiledGraph) {
	t.Helper()
	child := NewStateGraph()
	child.AddChannel("items", channels.NewDeltaChannel(appendInts, func() any { return []int{} }, 100))
	child.AddNode("cn", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"items": []int{5}, "cdone": true}, nil
	})
	child.AddEdge(types.START, "cn")
	child.AddEdge("cn", types.END)
	// The child compiles against the SAME saver the parent checkpoints into:
	// subgraph runs namespace under the parent's thread, and the child-side
	// GetState read path resolves the saver from its own compile config (the
	// caller-resolved saver, Python #8538).
	childCompiled, err := child.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("compile child: %v", err)
	}

	parent := NewStateGraph()
	parent.AddSubgraph("run_child", childCompiled)
	parent.AddEdge(types.START, "run_child")
	parent.AddEdge("run_child", types.END)
	parentCompiled, err := parent.Compile(WithCheckpointer(saver))
	if err != nil {
		t.Fatalf("compile parent: %v", err)
	}
	return parentCompiled, childCompiled
}

// TestSubgraphDeltaReadViaChildGraph verifies the SUPPORTED read path for
// subgraph sentinel delta channels: reading through the child graph wired to
// the same saver (the caller-resolved saver, Python #8538) reconstructs the
// accumulated value, including on the run/resume path (Task 9 hydration).
func TestSubgraphDeltaReadViaChildGraph(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	parent, child := subgraphDeltaGraph(t, saver)
	ctx := t.Context()

	if _, err := parent.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("parent invoke: %v", err)
	}

	// Locate the child namespace's latest checkpoint and read it through the
	// child graph wired to the same saver.
	childNSs := rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 1 {
		t.Fatalf("child namespaces = %v, want exactly one", childNSs)
	}
	childNS := childNSs[0]
	snap, err := child.GetState(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: childNS})
	if err != nil {
		t.Fatalf("child GetState: %v", err)
	}
	if want := []int{5}; !reflect.DeepEqual(snap.Values["items"], any(want)) {
		t.Fatalf("child items = %v, want %v (subgraph delta reconstruction through the caller-resolved saver)", snap.Values["items"], want)
	}
}

// TestSubgraphDeltaReadViaParentGraphIsDocumentedLimitation pins the known
// boundary: reading a child namespace THROUGH the parent graph unwraps delta
// snapshot blobs (proto-independent) but does not reconstruct sentinel-only
// child delta channels — the parent does not know the child's channel protos.
// Documented in DIVERGENCES.md; read sentinel channels through the child
// graph compiled with the shared saver instead (see
// TestSubgraphDeltaReadViaChildGraph). Python #8538's run-path half (hydrate
// subgraph delta channels with the caller-resolved saver) is covered by the
// run path using the parent-context saver.
func TestSubgraphDeltaReadViaParentGraphIsDocumentedLimitation(t *testing.T) {
	saver := checkpoint.NewMemorySaver()
	parent, _ := subgraphDeltaGraph(t, saver)
	ctx := t.Context()

	if _, err := parent.InvokeWithOptions(ctx, map[string]any{}, Options{ThreadID: "t1"}); err != nil {
		t.Fatalf("parent invoke: %v", err)
	}
	childNSs := rootChildNamespaces(t, saver, "t1")
	if len(childNSs) != 1 {
		t.Fatalf("child namespaces = %v, want exactly one", childNSs)
	}
	snap, err := parent.GetState(ctx, checkpoint.Config{ThreadID: "t1", CheckpointNS: childNSs[0]})
	if err != nil {
		t.Fatalf("parent GetState(childNS): %v", err)
	}
	// Non-delta child state projects through the parent read.
	if snap.Values["cdone"] != true {
		t.Fatalf("cdone = %v, want true (non-delta child values must project)", snap.Values["cdone"])
	}
	// The sentinel delta channel does not (documented limitation).
	if _, ok := snap.Values["items"]; ok {
		t.Fatalf("items = %v unexpectedly reconstructed through parent protos; if this now works, update DIVERGENCES.md", snap.Values["items"])
	}
}
