package graph

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func joinNoop(_ context.Context, _ map[string]any) (any, error) { return nil, nil }

// joinBuilderBase returns a builder with nodes a, b, c registered and a as
// entry point, so AddJoinEdge validation failures are the only Compile errors.
func joinBuilderBase() *StateGraph {
	g := NewStateGraph()
	g.AddNode("a", joinNoop)
	g.AddNode("b", joinNoop)
	g.AddNode("c", joinNoop)
	g.AddEdge(types.START, "a")
	return g
}

// TestAddJoinEdgeValidation covers the call-time (count/dup/reserved-name)
// and Compile-time (node existence, duplicate join channel) checks. Python
// accepts a single-element start tuple and silently set-dedups (state.py:956);
// Go deliberately tightens both to errors (spec M6 documented divergence), and
// rejects END as a join child where Python allows it (state.py:963-964).
func TestAddJoinEdgeValidation(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(g *StateGraph)
		wantErr string
	}{
		{"zero parents", func(g *StateGraph) { g.AddJoinEdge(nil, "c") }, "at least 2"},
		{"single parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a"}, "c") }, "at least 2"},
		{"duplicate parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "a"}, "c") }, "duplicate join parent"},
		{"START parent", func(g *StateGraph) { g.AddJoinEdge([]string{types.START, "a"}, "c") }, "invalid join parent"},
		{"END parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", types.END}, "c") }, "invalid join parent"},
		{"START child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, types.START) }, "invalid join child"},
		{"END child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, types.END) }, "invalid join child"},
		{"unknown parent", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "zzz"}, "c") }, "not a registered node"},
		{"unknown child", func(g *StateGraph) { g.AddJoinEdge([]string{"a", "b"}, "zzz") }, "not a registered node"},
		{"duplicate join edge", func(g *StateGraph) {
			g.AddJoinEdge([]string{"a", "b"}, "c")
			g.AddJoinEdge([]string{"a", "b"}, "c")
		}, "duplicate join channel"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := joinBuilderBase()
			tc.mutate(g)
			_, err := g.Compile()
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Compile() error = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

// TestCompileJoinRegistersBarrierChannel: Compile registers a Barrier
// prototype under "join:a+b:c" and the parent->barrier index, without
// polluting the builder's own channelProtos (Compile stays re-entrant).
func TestCompileJoinRegistersBarrierChannel(t *testing.T) {
	g := joinBuilderBase()
	g.AddJoinEdge([]string{"a", "b"}, "c")
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	proto, ok := cg.channelProtos["join:a+b:c"]
	if !ok {
		t.Fatal("channelProtos missing join:a+b:c")
	}
	b, ok := proto.(*channels.Barrier)
	if !ok {
		t.Fatalf("join:a+b:c proto = %T, want *channels.Barrier", proto)
	}
	if got := b.Names(); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Names() = %v, want [a b]", got)
	}
	if got := cg.joinsByParent["a"]; len(got) != 1 || got[0] != "join:a+b:c" {
		t.Fatalf("joinsByParent[a] = %v", got)
	}
	if got := cg.joinsByParent["b"]; len(got) != 1 || got[0] != "join:a+b:c" {
		t.Fatalf("joinsByParent[b] = %v", got)
	}
	if _, polluted := g.channelProtos["join:a+b:c"]; polluted {
		t.Fatal("builder channelProtos polluted by Compile")
	}
	if _, err := g.Compile(); err != nil {
		t.Fatalf("second Compile() error = %v (join registration must be re-entrant)", err)
	}
}
