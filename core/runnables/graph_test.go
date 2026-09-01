package runnables

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestRunnableGraphRouter(t *testing.T) {
	add := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input + 1, nil
	}, schema.Integer(""), schema.Integer(""))
	router := NewRouter(map[string]Runnable[int, int]{"add": add})

	graph := GetGraph(router)
	if len(graph.Nodes) != 2 {
		t.Fatalf("router nodes: %#v", graph.Nodes)
	}
	if graph.Nodes[0].Type != "Router" {
		t.Fatalf("root node: %#v", graph.Nodes[0])
	}
	if len(graph.Edges) != 1 || graph.Edges[0].Label != "add" {
		t.Fatalf("router edges: %#v", graph.Edges)
	}
	if graph.Edges[0].Source != "router" || graph.Edges[0].Target != "route.add.add" {
		t.Fatalf("router edge: %#v", graph.Edges[0])
	}
}

func TestRunnableGraphBranch(t *testing.T) {
	cond := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input > 0, nil
	}, schema.Integer(""), schema.Boolean(""))
	body := NewFunc(func(_ context.Context, input int, _ ...Option) (string, error) {
		return "positive", nil
	}, schema.Integer(""), schema.String(""))
	def := NewFunc(func(_ context.Context, input int, _ ...Option) (string, error) {
		return "negative", nil
	}, schema.Integer(""), schema.String(""))
	branch, err := NewBranch([]BranchCase[int, string]{{Condition: cond, Runnable: body}}, def)
	if err != nil {
		t.Fatalf("new branch: %v", err)
	}

	graph := GetGraph(branch)
	if graph.Nodes[0].Type != "Branch" {
		t.Fatalf("root node: %#v", graph.Nodes[0])
	}
	// Root fans out to the condition and the default; the condition links to
	// its body with a "true" edge.
	wantEdges := map[string]bool{
		"branch-->case.0.condition.condition_0:condition_0":          false,
		"branch-->default.default:default":                           false,
		"case.0.condition.condition_0-->case.0.runnable.case_0:true": false,
	}
	for _, edge := range graph.Edges {
		key := edge.Source + "-->" + edge.Target + ":" + edge.Label
		if _, ok := wantEdges[key]; ok {
			wantEdges[key] = true
		}
	}
	for key, found := range wantEdges {
		if !found {
			t.Fatalf("missing edge %q in %#v", key, graph.Edges)
		}
	}
}

func TestRunnableGraphWithFallbacks(t *testing.T) {
	primary := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	fallback := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	runnable, err := NewWithFallbacks[string, string](primary, fallback)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}

	graph := GetGraph(runnable)
	if graph.Nodes[0].Type != "WithFallbacks" {
		t.Fatalf("root node: %#v", graph.Nodes[0])
	}
	if len(graph.Edges) != 2 || graph.Edges[0].Label != "primary" || graph.Edges[1].Label != "fallback_1" {
		t.Fatalf("fallback edges: %#v", graph.Edges)
	}
}

func TestGetGraphNilAndTypeNames(t *testing.T) {
	graph := GetGraph(nil)
	if len(graph.Nodes) != 1 || graph.Nodes[0].Type != "nil" || graph.Nodes[0].ID != "nil" {
		t.Fatalf("nil graph: %#v", graph.Nodes)
	}

	if got := runnableTypeName(nil); got != "nil" {
		t.Fatalf("nil type name: %q", got)
	}
	// Pointer runnables report the element type name.
	leaf := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	if got := runnableTypeName(&leaf); got != "Func[string,string]" {
		t.Fatalf("pointer type name: %q", got)
	}
	// Unnamed types fall back to their full type string.
	if got := runnableTypeName(struct{ x int }{}); !strings.Contains(got, "struct") {
		t.Fatalf("unnamed type name: %q", got)
	}
	// IDs are sanitized for graph rendering.
	graph = GetGraph(leaf)
	if graph.Nodes[0].ID != "runnable" {
		t.Fatalf("leaf node: %#v", graph.Nodes[0])
	}
}

func TestGraphSanitizeAndMermaidEscapes(t *testing.T) {
	if got := sanitizeGraphID("  "); got != "node" {
		t.Fatalf("blank id: %q", got)
	}
	if got := sanitizeGraphID("a b/c"); got != "a_b_c" {
		t.Fatalf("sanitized id: %q", got)
	}

	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "A", Type: "T"}},
		Edges: []GraphEdge{{Source: "a", Target: "a"}},
	}
	ascii := graph.DrawASCII()
	if !strings.Contains(ascii, "| A |") {
		t.Fatalf("ASCII node box: %q", ascii)
	}
	mermaid := graph.DrawMermaid()
	if !strings.Contains(mermaid, "a --> a;") {
		t.Fatalf("Mermaid unlabeled edge: %q", mermaid)
	}
}

// TestToSafeMermaidID pins the Python `_to_safe_id` equivalent
// (langchain_core/runnables/graph_mermaid.py:255): [a-zA-Z0-9_-] pass
// through, every other character becomes `\` + lowercase hex codepoint.
func TestToSafeMermaidID(t *testing.T) {
	cases := map[string]string{
		"plain_Name-1": "plain_Name-1",
		"a b":          `a\20b`,
		"a.b":          `a\2eb`,
		"中":            `\4e2d`,
	}
	for in, want := range cases {
		if got := toSafeMermaidID(in); got != want {
			t.Errorf("toSafeMermaidID(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestDrawMermaidPythonCompat golden-tests the Python-compatible Mermaid
// output: frontmatter, `id(label)` nodes, first/last rounded nodes with
// :::first/:::last classes, dashed conditional edges with &nbsp; labels,
// and the trailing classDef styles (graph_mermaid.py draw_mermaid).
func TestDrawMermaidPythonCompat(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{
			{ID: "__start__", Name: "__start__"},
			{ID: "a", Name: "a"},
			{ID: "__end__", Name: "__end__"},
		},
		Edges: []GraphEdge{
			{Source: "__start__", Target: "a"},
			{Source: "a", Target: "__end__", Label: "done", Conditional: true},
		},
		FirstNode: "__start__",
		LastNode:  "__end__",
	}
	// __start__/__end__ start and end with "_" (a markdown special char), so
	// Python wraps their labels in <p>...</p> — this is the real langgraph
	// draw_mermaid output shape.
	want := `---
config:
  flowchart:
    curve: linear
---
graph TD;
	__end__([<p>__end__</p>]):::last
	__start__([<p>__start__</p>]):::first
	a(a)
	__start__ --> a;
	a -. &nbsp;done&nbsp; .-> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
`
	if got := graph.DrawMermaid(); got != want {
		t.Fatalf("DrawMermaid() =\n%q\nwant\n%q", got, want)
	}
}

// TestDrawMermaidLabels covers the remaining label rules: solid labeled
// edges, unlabeled conditional edges, metadata blocks on node labels,
// markdown-special node names, and >9-word edge label wrapping.
func TestDrawMermaidLabels(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{
			{ID: "a", Name: "a", Metadata: map[string]any{"key": "value"}},
			{ID: "b", Name: "*bold*"},
			{ID: "c", Name: "c"},
			{ID: "d", Name: "d"},
		},
		Edges: []GraphEdge{
			{Source: "a", Target: "b", Label: "solid"},
			{Source: "b", Target: "c", Conditional: true},
			{Source: "c", Target: "d", Label: "one two three four five six seven eight nine ten"},
		},
	}
	got := graph.DrawMermaid()
	for _, want := range []string{
		"\ta(a<hr/><small><em>key = value</em></small>)\n",
		"\tb(<p>*bold*</p>)\n",
		"\ta -- &nbsp;solid&nbsp; --> b;\n",
		"\tb -.-> c;\n",
		"\tc -- &nbsp;one two three four five six seven eight nine&nbsp<br>&nbspten&nbsp; --> d;\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("DrawMermaid() missing %q:\n%s", want, got)
		}
	}
}

func TestGraphMarshalWithMetadata(t *testing.T) {
	def := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	configurable, err := NewConfigurableAlternatives[string, string]("model", "default", def, nil)
	if err != nil {
		t.Fatalf("new configurable: %v", err)
	}
	data, err := GetGraph(configurable).MarshalJSONStable()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// Node metadata survives normalization into the stable JSON form.
	if !strings.Contains(string(data), `"field":"model"`) {
		t.Fatalf("JSON missing metadata: %s", data)
	}
}
