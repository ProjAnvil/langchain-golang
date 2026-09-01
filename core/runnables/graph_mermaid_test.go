package runnables

import (
	"strings"
	"testing"
)

// TestDrawMermaidWithStylesFalse pins the Python with_styles=False shape
// (graph_mermaid.py:103-111): only `graph TD;` plus edge lines — no
// frontmatter, no node declarations, no classDef styles.
func TestDrawMermaidWithStylesFalse(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}, {ID: "c", Name: "c"}},
		Edges: []GraphEdge{
			{Source: "a", Target: "b"},
			{Source: "b", Target: "c", Label: "next", Conditional: true},
		},
	}
	want := "graph TD;\n" +
		"\ta --> b;\n" +
		"\tb -. &nbsp;next&nbsp; .-> c;\n"
	if got := graph.DrawMermaid(WithStyles(false)); got != want {
		t.Fatalf("DrawMermaid(WithStyles(false)) =\n%q\nwant\n%q", got, want)
	}
}

// TestDrawMermaidFrontmatterConfig golden-tests the user frontmatter merge
// against Python's snapshot (test_graph.ambr
// test_graph_mermaid_frontmatter_config): the curve merges into
// config.flowchart, other user keys are emitted verbatim (keys sorted like
// PyYAML), and values containing '#' are single-quoted.
func TestDrawMermaidFrontmatterConfig(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{
			{ID: "__start__", Name: "__start__"},
			{ID: "my_node", Name: "my_node"},
		},
		Edges:     []GraphEdge{{Source: "__start__", Target: "my_node"}},
		FirstNode: "__start__",
		LastNode:  "my_node",
	}
	got := graph.DrawMermaid(WithFrontmatterConfig(map[string]any{
		"config": map[string]any{
			"theme":          "neutral",
			"look":           "handDrawn",
			"themeVariables": map[string]any{"primaryColor": "#e2e2e2"},
		},
	}))
	want := `---
config:
  flowchart:
    curve: linear
  look: handDrawn
  theme: neutral
  themeVariables:
    primaryColor: '#e2e2e2'
---
graph TD;
	__start__([<p>__start__</p>]):::first
	my_node([my_node]):::last
	__start__ --> my_node;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
`
	if got != want {
		t.Fatalf("DrawMermaid(WithFrontmatterConfig) =\n%q\nwant\n%q", got, want)
	}
}

// TestDrawMermaidCurveStyle checks the curve option lands in
// config.flowchart.curve and overrides a user-supplied curve (Python merge
// order: user's flowchart config first, curve last).
func TestDrawMermaidCurveStyle(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}},
		Edges: []GraphEdge{{Source: "a", Target: "b"}},
	}
	got := graph.DrawMermaid(
		WithCurveStyle("basis"),
		WithFrontmatterConfig(map[string]any{
			"config": map[string]any{"flowchart": map[string]any{"curve": "linear", "htmlLabels": true}},
		}),
	)
	for _, want := range []string{"    curve: basis\n", "    htmlLabels: true\n"} {
		if !strings.Contains(got, want) {
			t.Errorf("DrawMermaid(WithCurveStyle) missing %q:\n%s", want, got)
		}
	}
}

// TestDrawMermaidNodeStyles checks WithNodeStyles replaces the three
// classDef lines (graph_mermaid.py _generate_mermaid_graph_styles).
func TestDrawMermaidNodeStyles(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}},
	}
	got := graph.DrawMermaid(WithNodeStyles(NodeStyles{
		Default: "fill:#fff",
		First:   "fill:#000",
		Last:    "fill:#111",
	}))
	for _, want := range []string{
		"\tclassDef default fill:#fff\n",
		"\tclassDef first fill:#000\n",
		"\tclassDef last fill:#111\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("DrawMermaid(WithNodeStyles) missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "fill:#f2f0ff") {
		t.Errorf("default classDef not replaced:\n%s", got)
	}
}

// subgraphTestGraph mirrors the Python test_graph_mermaid[...] fixture: two
// nesting levels of ":"-prefixed nodes plus top-level nodes and edges.
func subgraphTestGraph() Graph {
	return Graph{
		Nodes: []GraphNode{
			{ID: "__start__", Name: "__start__"},
			{ID: "parent_1", Name: "parent_1"},
			{ID: "parent_2", Name: "parent_2"},
			{ID: "__end__", Name: "__end__"},
			{ID: "child:child_1:grandchild_1", Name: "child:child_1:grandchild_1"},
			{ID: "child:child_1:grandchild_2", Name: "child:child_1:grandchild_2", Metadata: map[string]any{"__interrupt": "before"}},
			{ID: "child:child_2", Name: "child:child_2"},
		},
		Edges: []GraphEdge{
			{Source: "__start__", Target: "parent_1"},
			{Source: "child:child_2", Target: "parent_2"},
			{Source: "parent_1", Target: "child:child_1:grandchild_1"},
			{Source: "parent_2", Target: "__end__"},
			{Source: "child:child_1:grandchild_2", Target: "child:child_2"},
			{Source: "child:child_1:grandchild_1", Target: "child:child_1:grandchild_2"},
		},
		FirstNode: "__start__",
		LastNode:  "__end__",
	}
}

// TestDrawMermaidSubgraphGrouping golden-tests mermaid subgraph blocks for
// ":"-prefixed node IDs (graph_mermaid.py:113-251): edges are grouped by
// common prefix, subgraph blocks nest, and node declarations inside blocks
// use the last ":"-separated segment as label. Ordering follows Go's sorted
// normalization (documented divergence from Python's insertion order).
func TestDrawMermaidSubgraphGrouping(t *testing.T) {
	want := `---
config:
  flowchart:
    curve: linear
---
graph TD;
	__end__([<p>__end__</p>]):::last
	__start__([<p>__start__</p>]):::first
	parent_1(parent_1)
	parent_2(parent_2)
	__start__ --> parent_1;
	child\3achild_2 --> parent_2;
	parent_1 --> child\3achild_1\3agrandchild_1;
	parent_2 --> __end__;
	subgraph child
	child\3achild_2(child_2)
	child\3achild_1\3agrandchild_2 --> child\3achild_2;
	subgraph child_1
	child\3achild_1\3agrandchild_1(grandchild_1)
	child\3achild_1\3agrandchild_2(grandchild_2<hr/><small><em>__interrupt = before</em></small>)
	child\3achild_1\3agrandchild_1 --> child\3achild_1\3agrandchild_2;
	end
	end
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
`
	if got := subgraphTestGraph().DrawMermaid(); got != want {
		t.Fatalf("DrawMermaid() =\n%q\nwant\n%q", got, want)
	}
}

// TestDrawMermaidSubgraphNoStyles: with_styles=false still emits subgraph
// blocks (Python only gates node declarations, empty subgraphs and classDef
// on with_styles).
func TestDrawMermaidSubgraphNoStyles(t *testing.T) {
	want := `graph TD;
	__start__ --> parent_1;
	child\3achild_2 --> parent_2;
	parent_1 --> child\3achild_1\3agrandchild_1;
	parent_2 --> __end__;
	subgraph child
	child\3achild_1\3agrandchild_2 --> child\3achild_2;
	subgraph child_1
	child\3achild_1\3agrandchild_1 --> child\3achild_1\3agrandchild_2;
	end
	end
`
	if got := subgraphTestGraph().DrawMermaid(WithStyles(false)); got != want {
		t.Fatalf("DrawMermaid(WithStyles(false)) =\n%q\nwant\n%q", got, want)
	}
}

// TestDrawMermaidSubgraphEdgeCases covers the self-loop single-edge special
// case (graph_mermaid.py:171: a group whose only edge is a self-loop renders
// its edge without a subgraph wrapper) and the duplicate-subgraph-name
// divergence (Python raises ValueError; Go silently renders the edges
// inline).
func TestDrawMermaidSubgraphEdgeCases(t *testing.T) {
	// The self-loop edge's group prefix is its full node ID, so it renders
	// inline inside the parent subgraph block, without its own wrapper.
	selfLoop := Graph{
		Nodes: []GraphNode{{ID: "child:a", Name: "child:a"}, {ID: "child:b", Name: "child:b"}},
		Edges: []GraphEdge{
			{Source: "child:a", Target: "child:a"},
			{Source: "child:a", Target: "child:b"},
		},
	}
	got := selfLoop.DrawMermaid()
	if strings.Count(got, "subgraph") != 1 {
		t.Fatalf("self-loop must not emit its own subgraph block:\n%s", got)
	}
	for _, want := range []string{`child\3aa --> child\3aa;`, `child\3aa --> child\3ab;`} {
		if !strings.Contains(got, want) {
			t.Errorf("edge %q missing:\n%s", want, got)
		}
	}

	// Two nested subgraphs whose last ":"-segment collides ("p1:x" and
	// "p2:x"): the second block is skipped and its edges render inline.
	dup := Graph{
		Nodes: []GraphNode{
			{ID: "p1:a", Name: "a"}, {ID: "p1:b", Name: "b"},
			{ID: "p1:x:a", Name: "a"}, {ID: "p1:x:b", Name: "b"},
			{ID: "p2:a", Name: "a"}, {ID: "p2:b", Name: "b"},
			{ID: "p2:x:a", Name: "a"}, {ID: "p2:x:b", Name: "b"},
		},
		Edges: []GraphEdge{
			{Source: "p1:a", Target: "p1:b"},
			{Source: "p1:x:a", Target: "p1:x:b"},
			{Source: "p2:a", Target: "p2:b"},
			{Source: "p2:x:a", Target: "p2:x:b"},
		},
	}
	got = dup.DrawMermaid(WithStyles(false))
	if strings.Count(got, "subgraph x") != 1 {
		t.Fatalf("duplicate subgraph name must render one block:\n%s", got)
	}
	for _, want := range []string{
		`p1\3aa --> p1\3ab;`, `p1\3ax\3aa --> p1\3ax\3ab;`,
		`p2\3aa --> p2\3ab;`, `p2\3ax\3aa --> p2\3ax\3ab;`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("duplicate-subgraph edge %q missing:\n%s", want, got)
		}
	}
}

func TestGraphExtend(t *testing.T) {
	child := Graph{
		Nodes:     []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}},
		Edges:     []GraphEdge{{Source: "a", Target: "b"}},
		FirstNode: "a",
		LastNode:  "b",
	}
	parent := Graph{Nodes: []GraphNode{{ID: "root", Name: "root"}}}
	first, last := parent.Extend(child, "sub")
	if first != "sub:a" || last != "sub:b" {
		t.Fatalf("Extend returned (%q, %q), want (sub:a, sub:b)", first, last)
	}
	if len(parent.Nodes) != 3 {
		t.Fatalf("parent nodes: %#v", parent.Nodes)
	}
	if len(parent.Edges) != 1 || parent.Edges[0].Source != "sub:a" || parent.Edges[0].Target != "sub:b" {
		t.Fatalf("parent edges: %#v", parent.Edges)
	}

	// Empty prefix keeps IDs unchanged; a child without a unique first/last
	// reports "".
	flat := Graph{}
	first, last = flat.Extend(Graph{
		Nodes: []GraphNode{{ID: "x", Name: "x"}, {ID: "y", Name: "y"}},
	}, "")
	if first != "" || last != "" {
		t.Fatalf("Extend without unique first/last returned (%q, %q)", first, last)
	}
	if flat.Nodes[0].ID != "x" || flat.Nodes[1].ID != "y" {
		t.Fatalf("unprefixed nodes: %#v", flat.Nodes)
	}
}

func TestGraphFirstLastNodeID(t *testing.T) {
	// Explicit fields win over edge inference.
	graph := Graph{
		Nodes:     []GraphNode{{ID: "a"}, {ID: "b"}, {ID: "c"}},
		Edges:     []GraphEdge{{Source: "a", Target: "b"}, {Source: "b", Target: "c"}},
		FirstNode: "explicit_first",
		LastNode:  "explicit_last",
	}
	if got := graph.FirstNodeID(); got != "explicit_first" {
		t.Fatalf("FirstNodeID = %q", got)
	}
	if got := graph.LastNodeID(); got != "explicit_last" {
		t.Fatalf("LastNodeID = %q", got)
	}

	// Inference: unique root/terminal.
	graph.FirstNode, graph.LastNode = "", ""
	if got := graph.FirstNodeID(); got != "a" {
		t.Fatalf("inferred FirstNodeID = %q", got)
	}
	if got := graph.LastNodeID(); got != "c" {
		t.Fatalf("inferred LastNodeID = %q", got)
	}

	// Ambiguous roots/terminals report "".
	graph.Nodes = append(graph.Nodes, GraphNode{ID: "d"})
	if got := graph.FirstNodeID(); got != "" {
		t.Fatalf("ambiguous FirstNodeID = %q", got)
	}
	if got := graph.LastNodeID(); got != "" {
		t.Fatalf("ambiguous LastNodeID = %q", got)
	}
}

// TestGraphTrimFirstLastNode mirrors Python graph.py:483-507: the first/last
// node is removed only when it exists, has exactly one outgoing/incoming
// edge, and removing it leaves a unique first/last node.
func TestGraphTrimFirstLastNode(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "__start__"}, {ID: "a"}, {ID: "b"}, {ID: "__end__"}},
		Edges: []GraphEdge{
			{Source: "__start__", Target: "a"},
			{Source: "a", Target: "b"},
			{Source: "b", Target: "__end__"},
		},
		FirstNode: "__start__",
		LastNode:  "__end__",
	}
	graph.TrimFirstNode()
	graph.TrimLastNode()
	if len(graph.Nodes) != 2 || graph.Nodes[0].ID != "a" || graph.Nodes[1].ID != "b" {
		t.Fatalf("trimmed nodes: %#v", graph.Nodes)
	}
	if len(graph.Edges) != 1 || graph.Edges[0].Source != "a" || graph.Edges[0].Target != "b" {
		t.Fatalf("trimmed edges: %#v", graph.Edges)
	}
	if graph.FirstNode != "" || graph.LastNode != "" {
		t.Fatalf("stale first/last fields: %#v", graph)
	}

	// A first node with two outgoing edges is kept.
	graph = Graph{
		Nodes: []GraphNode{{ID: "s"}, {ID: "a"}, {ID: "b"}},
		Edges: []GraphEdge{{Source: "s", Target: "a"}, {Source: "s", Target: "b"}},
	}
	graph.TrimFirstNode()
	if len(graph.Nodes) != 3 {
		t.Fatalf("fan-out first node must be kept: %#v", graph.Nodes)
	}

	// A last node with two incoming edges is kept.
	graph = Graph{
		Nodes: []GraphNode{{ID: "a"}, {ID: "b"}, {ID: "e"}},
		Edges: []GraphEdge{{Source: "a", Target: "e"}, {Source: "b", Target: "e"}},
	}
	graph.TrimLastNode()
	if len(graph.Nodes) != 3 {
		t.Fatalf("join last node must be kept: %#v", graph.Nodes)
	}
}
