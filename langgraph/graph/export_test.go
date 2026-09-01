package graph

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// noopNode is shared with run_errors_test.go.

// TestGetGraphLinearMermaid golden-tests the Python-compatible Mermaid
// export of a linear START -> a -> b -> END graph (see
// langchain_core/runnables/graph_mermaid.py draw_mermaid).
func TestGetGraphLinearMermaid(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	want := `---
config:
  flowchart:
    curve: linear
---
graph TD;
	__end__([<p>__end__</p>]):::last
	__start__([<p>__start__</p>]):::first
	a(a)
	b(b)
	__start__ --> a;
	a --> b;
	b --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
`
	// The builder and the compiled graph must export identically.
	if got := g.DrawMermaid(); got != want {
		t.Fatalf("StateGraph.DrawMermaid() =\n%q\nwant\n%q", got, want)
	}
	if got := cg.DrawMermaid(); got != want {
		t.Fatalf("CompiledGraph.DrawMermaid() =\n%q\nwant\n%q", got, want)
	}
}

// TestGetGraphConditionalEdgesWithPathMap covers dashed conditional edges
// whose labels come from the path map: a key equal to its target node draws
// no label, a differing key (typically the types.END mapping) draws one.
func TestGetGraphConditionalEdgesWithPathMap(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("model", noopNode)
	g.AddNode("tools", noopNode)
	g.AddEdge(types.START, "model")
	g.AddConditionalEdgesWithPathMap("model", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To(types.END), nil
	}, map[string]string{
		"tools": "tools",
		"end":   types.END,
	})
	g.AddEdge("tools", "model")

	mermaid := g.DrawMermaid()
	for _, want := range []string{
		"\tmodel -.-> tools;\n",
		"\tmodel -. &nbsp;end&nbsp; .-> __end__;\n",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, mermaid)
		}
	}

	// The path map is visualization metadata only: runtime routing is the
	// router's job, exactly like AddConditionalEdges.
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	if _, err := cg.Invoke(context.Background(), map[string]any{}); err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
}

// TestGetGraphConditionalEntryPoint covers a conditional entry point
// exported as dashed edges out of START.
func TestGetGraphConditionalEntryPoint(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.SetConditionalEntryPointWithPathMap(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("a"), nil
	}, map[string]string{
		"a":   "a",
		"bee": "b",
	})
	g.AddEdge("a", types.END)
	g.AddEdge("b", types.END)

	mermaid := g.DrawMermaid()
	for _, want := range []string{
		"\t__start__ -.-> a;\n",
		"\t__start__ -. &nbsp;bee&nbsp; .-> b;\n",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, mermaid)
		}
	}
}

// TestGetGraphJoinEdges covers waiting edges: each parent draws one solid
// edge into the join child.
func TestGetGraphJoinEdges(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("p1", noopNode)
	g.AddNode("p2", noopNode)
	g.AddNode("c", noopNode)
	g.AddEdge(types.START, "p1")
	// Only one START edge is allowed (single entry point), so p2 hangs off
	// p1; the join still waits on both parents.
	g.AddEdge("p1", "p2")
	g.AddJoinEdge([]string{"p1", "p2"}, "c")
	g.AddEdge("c", types.END)

	mermaid := g.DrawMermaid()
	for _, want := range []string{
		"\tp1 --> c;\n",
		"\tp2 --> c;\n",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, mermaid)
		}
	}
}

// TestAddConditionalEdgesWithPathMapValidation covers the error paths:
// unknown path map targets fail Compile; nil routers and duplicate
// registration fail like AddConditionalEdges.
func TestAddConditionalEdgesWithPathMapValidation(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddConditionalEdgesWithPathMap("a", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("missing"), nil
	}, map[string]string{"missing": "missing"})
	if _, err := g.Compile(); err == nil || !strings.Contains(err.Error(), `"missing"`) {
		t.Fatalf("Compile() error = %v, want unknown path map target", err)
	}

	g = NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddConditionalEdgesWithPathMap("a", nil, map[string]string{"a": "a"})
	if _, err := g.Compile(); err == nil {
		t.Fatal("Compile() expected nil router error")
	}

	g = NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	router := func(_ runtime.Runtime, _ map[string]any) ([]any, error) { return To("a"), nil }
	g.AddConditionalEdges("a", router)
	g.AddConditionalEdgesWithPathMap("a", router, map[string]string{"a": "a"})
	if _, err := g.Compile(); err == nil {
		t.Fatal("Compile() expected duplicate conditional edge error")
	}
}

// TestSetConditionalEntryPointWithPathMapValidation covers unknown entry
// path map targets failing Compile.
func TestSetConditionalEntryPointWithPathMapValidation(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.SetConditionalEntryPointWithPathMap(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("missing"), nil
	}, map[string]string{"missing": "missing"})
	if _, err := g.Compile(); err == nil || !strings.Contains(err.Error(), `"missing"`) {
		t.Fatalf("Compile() error = %v, want unknown path map target", err)
	}
}

// TestGetGraphRouterWithoutPathMap covers the best-effort router probing
// divergence: a router registered without a path map is probed once with an
// empty state, and the targets it returns draw as dashed edges (Python
// discovers them with a full dry-run simulation; this port makes a single
// best-effort call). Disable with WithRouterProbing(false).
func TestGetGraphRouterWithoutPathMap(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.AddEdge(types.START, "a")
	g.AddConditionalEdges("a", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("b", "b", types.END), nil // duplicates and END are handled
	})
	g.AddEdge("b", types.END)

	mermaid := g.DrawMermaid()
	for _, want := range []string{
		"\ta -.-> b;\n",
		"\ta -.-> __end__;\n",
		"\t__start__ --> a;\n",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, mermaid)
		}
	}
	if strings.Count(mermaid, "a -.-> b;") != 1 {
		t.Errorf("probed edge a->b drawn more than once:\n%s", mermaid)
	}

	// Probing can be disabled, restoring the pure static export.
	graph := g.GetGraph(WithRouterProbing(false))
	for _, edge := range graph.Edges {
		if edge.Conditional {
			t.Fatalf("WithRouterProbing(false) still drew conditional edge %#v", edge)
		}
	}
}

// TestGetGraphRouterProbeFailure covers routers that fail under probing —
// returning an error or panicking (e.g. reading state keys absent from the
// empty probe state): their edges are omitted from the drawing.
func TestGetGraphRouterProbeFailure(t *testing.T) {
	errRouter := NewStateGraph()
	errRouter.AddNode("a", noopNode)
	errRouter.AddNode("b", noopNode)
	errRouter.AddEdge(types.START, "a")
	errRouter.AddConditionalEdges("a", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return nil, fmt.Errorf("cannot route without input")
	})
	errRouter.AddEdge("b", types.END)
	if mermaid := errRouter.DrawMermaid(); strings.Contains(mermaid, ".->") {
		t.Fatalf("erroring router drew conditional edges:\n%s", mermaid)
	}

	panicRouter := NewStateGraph()
	panicRouter.AddNode("a", noopNode)
	panicRouter.AddNode("b", noopNode)
	panicRouter.AddEdge(types.START, "a")
	panicRouter.AddConditionalEdges("a", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		panic("router read a missing state key")
	})
	panicRouter.AddEdge("b", types.END)
	if mermaid := panicRouter.DrawMermaid(); strings.Contains(mermaid, ".->") {
		t.Fatalf("panicking router drew conditional edges:\n%s", mermaid)
	}
}

// TestGetGraphRouterProbeSendAndEntry covers probing *types.Send targets and
// a conditional entry point without a path map.
func TestGetGraphRouterProbeSendAndEntry(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.SetConditionalEntryPoint(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("a"), nil
	})
	g.AddConditionalEdges("a", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return []any{&types.Send{Node: "b", Arg: map[string]any{}}}, nil
	})
	g.AddEdge("b", types.END)

	mermaid := g.DrawMermaid()
	for _, want := range []string{
		"\t__start__ -.-> a;\n",
		"\ta -.-> b;\n",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, mermaid)
		}
	}
}

// TestGetGraphStructure checks the exported runnables.Graph shape itself:
// START/END bookend the node list as FirstNode/LastNode, and edges carry
// the conditional flag.
func TestGetGraphStructure(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddEdge(types.START, "a")
	g.AddConditionalEdgesWithPathMap("a", func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To(types.END), nil
	}, map[string]string{"end": types.END})

	graph := g.GetGraph()
	if graph.FirstNode != types.START || graph.LastNode != types.END {
		t.Fatalf("FirstNode/LastNode = %q/%q", graph.FirstNode, graph.LastNode)
	}
	names := map[string]bool{}
	for _, node := range graph.Nodes {
		names[node.ID] = true
	}
	for _, want := range []string{types.START, types.END, "a"} {
		if !names[want] {
			t.Fatalf("missing node %q in %#v", want, graph.Nodes)
		}
	}
	var conditional, entry int
	for _, edge := range graph.Edges {
		if edge.Conditional {
			conditional++
			if edge.Source != "a" || edge.Target != types.END || edge.Label != "end" {
				t.Fatalf("conditional edge = %#v", edge)
			}
		}
		if edge.Source == types.START {
			entry++
		}
	}
	if conditional != 1 || entry != 1 {
		t.Fatalf("edges = %#v", graph.Edges)
	}
}

// TestDrawMermaidWithStylesFalse covers the mermaid option passthrough:
// with_styles=false emits only `graph TD;` plus the edge lines (Python's
// simple form used across its test suite).
func TestDrawMermaidWithStylesFalse(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)

	want := "graph TD;\n" +
		"\t__start__ --> a;\n" +
		"\ta --> b;\n" +
		"\tb --> __end__;\n"
	if got := g.DrawMermaid(runnables.WithStyles(false)); got != want {
		t.Fatalf("DrawMermaid(WithStyles(false)) =\n%q\nwant\n%q", got, want)
	}
}

// TestGetGraphInterruptMetadata covers the __interrupt node metadata
// exported from the compiled graph's interrupt boundaries.
func TestGetGraphInterruptMetadata(t *testing.T) {
	g := NewStateGraph()
	g.AddNode("a", noopNode)
	g.AddNode("b", noopNode)
	g.AddEdge(types.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", types.END)

	cg, err := g.Compile(WithInterruptBefore("a"), WithInterruptAfter("a", "b"))
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	graph := cg.GetGraph()
	interrupts := map[string]string{}
	for _, node := range graph.Nodes {
		if v, ok := node.Metadata["__interrupt"]; ok {
			interrupts[node.ID], _ = v.(string)
		}
	}
	if interrupts["a"] != "before,after" || interrupts["b"] != "after" || len(interrupts) != 2 {
		t.Fatalf("__interrupt metadata = %v", interrupts)
	}

	mermaid := cg.DrawMermaid()
	for _, want := range []string{
		"a(a<hr/><small><em>__interrupt = before,after</em></small>)",
		"b(b<hr/><small><em>__interrupt = after</em></small>)",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, mermaid)
		}
	}

	// The builder export has no interrupt metadata (boundaries are compile
	// options, mirroring Python where they attach at compile time).
	for _, node := range g.GetGraph().Nodes {
		if _, ok := node.Metadata["__interrupt"]; ok {
			t.Fatalf("StateGraph.GetGraph() unexpectedly carries __interrupt on %q", node.ID)
		}
	}
}

// compileSubgraph is a test helper building START -> x -> y -> END.
func compileSubgraph(t *testing.T, nameX, nameY string) *CompiledGraph {
	t.Helper()
	g := NewStateGraph()
	g.AddNode(nameX, noopNode)
	g.AddNode(nameY, noopNode)
	g.AddEdge(types.START, nameX)
	g.AddEdge(nameX, nameY)
	g.AddEdge(nameY, types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("subgraph Compile() error = %v", err)
	}
	return cg
}

// TestGetGraphXray golden-tests subgraph expansion (WithXrayDepth(-1), the
// Python `xray=True` equivalent): the subgraph node is replaced by its
// trimmed inner graph under the `parent:child` prefix, parent edges are
// rewired to the inner first/last nodes, and Mermaid renders a subgraph
// block (shape mirrors Python's test_nested_graph_xray snapshot).
func TestGetGraphXray(t *testing.T) {
	child := compileSubgraph(t, "x", "y")

	g := NewStateGraph()
	g.AddNode("pre", noopNode)
	g.AddSubgraph("child", child)
	g.AddNode("post", noopNode)
	g.AddEdge(types.START, "pre")
	g.AddEdge("pre", "child")
	g.AddEdge("child", "post")
	g.AddEdge("post", types.END)

	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	// Depth 0 (default): the subgraph stays a single node.
	if mermaid := cg.DrawMermaid(); strings.Contains(mermaid, "subgraph child") {
		t.Fatalf("default GetGraph expanded the subgraph:\n%s", mermaid)
	}

	want := `---
config:
  flowchart:
    curve: linear
---
graph TD;
	__end__([<p>__end__</p>]):::last
	__start__([<p>__start__</p>]):::first
	post(post)
	pre(pre)
	__start__ --> pre;
	child\3ay --> post;
	post --> __end__;
	pre --> child\3ax;
	subgraph child
	child\3ax(x)
	child\3ay(y)
	child\3ax --> child\3ay;
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
`
	if got := cg.GetGraph(WithXrayDepth(-1)).DrawMermaid(); got != want {
		t.Errorf("CompiledGraph xray mermaid =\n%q\nwant\n%q", got, want)
	}
	// The builder exports identically (Compile copies the subgraph registry).
	if got := g.GetGraph(WithXrayDepth(-1)).DrawMermaid(); got != want {
		t.Errorf("StateGraph xray mermaid =\n%q\nwant\n%q", got, want)
	}
}

// TestGetGraphXrayDepth covers bounded expansion: depth 1 expands direct
// children only, while -1 recurses into nested subgraphs, accumulating
// `a:b:c` prefixes.
func TestGetGraphXrayDepth(t *testing.T) {
	inner := compileSubgraph(t, "inner1", "inner2")

	middle := NewStateGraph()
	middle.AddNode("mid", noopNode)
	middle.AddSubgraph("inner", inner)
	middle.AddEdge(types.START, "mid")
	middle.AddEdge("mid", "inner")
	middle.AddEdge("inner", types.END)
	middleCG, err := middle.Compile()
	if err != nil {
		t.Fatalf("middle Compile() error = %v", err)
	}

	outer := NewStateGraph()
	outer.AddSubgraph("middle", middleCG)
	outer.AddEdge(types.START, "middle")
	outer.AddEdge("middle", types.END)
	outerCG, err := outer.Compile()
	if err != nil {
		t.Fatalf("outer Compile() error = %v", err)
	}

	depth1 := outerCG.GetGraph(WithXrayDepth(1)).DrawMermaid()
	if !strings.Contains(depth1, "subgraph middle") || !strings.Contains(depth1, "middle\\3ainner(inner)") {
		t.Errorf("depth=1 should expand one level, keeping the inner node:\n%s", depth1)
	}
	if strings.Contains(depth1, "middle\\3ainner\\3a") {
		t.Errorf("depth=1 expanded the nested subgraph:\n%s", depth1)
	}

	full := outerCG.GetGraph(WithXrayDepth(-1)).DrawMermaid()
	for _, want := range []string{"subgraph middle", "subgraph inner", "middle\\3ainner\\3ainner1(inner1)"} {
		if !strings.Contains(full, want) {
			t.Errorf("xray=-1 missing %q:\n%s", want, full)
		}
	}
}

// TestGetGraphXrayUnregisteredNode confirms nodes added via plain AddNode
// are never expanded, and that a subgraph that cannot trim to unique
// first/last inner nodes still expands with its sentinel nodes kept
// (mirroring Python's conditional-entry-point subgraphs).
func TestGetGraphXrayUnregisteredNode(t *testing.T) {
	// Child with a conditional entry point: its __start__ has two outgoing
	// edges, so trim_first_node keeps it (Python parity).
	child := NewStateGraph()
	child.AddNode("slow", noopNode)
	child.AddNode("fast", noopNode)
	child.SetConditionalEntryPointWithPathMap(func(_ runtime.Runtime, _ map[string]any) ([]any, error) {
		return To("slow"), nil
	}, map[string]string{"slow": "slow", "fast": "fast"})
	child.AddEdge("slow", types.END)
	child.AddEdge("fast", types.END)
	childCG, err := child.Compile()
	if err != nil {
		t.Fatalf("child Compile() error = %v", err)
	}

	g := NewStateGraph()
	g.AddNode("plain", noopNode) // not a registered subgraph: never expands
	g.AddSubgraph("child", childCG)
	g.AddEdge(types.START, "plain")
	g.AddEdge("plain", "child")
	g.AddEdge("child", types.END)

	mermaid := g.GetGraph(WithXrayDepth(-1)).DrawMermaid()
	for _, want := range []string{
		"\tplain(plain)\n",
		"\tsubgraph child\n",
		"child\\3a__start__ -.-> child\\3afast;",
		"child\\3a__end__ --> __end__;",
	} {
		if !strings.Contains(mermaid, want) {
			t.Errorf("xray mermaid missing %q:\n%s", want, mermaid)
		}
	}
}
