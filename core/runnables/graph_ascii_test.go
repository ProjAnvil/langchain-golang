package runnables

import "testing"

// TestDrawASCIIChain golden-tests the box-drawing format for a linear chain:
// boxes of height 3 and width len(" name ")+2 (Python VertexViewer), one
// edge channel row per crossing edge, `*` for plain edges.
func TestDrawASCIIChain(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}, {ID: "c", Name: "c"}},
		Edges: []GraphEdge{{Source: "a", Target: "b"}, {Source: "b", Target: "c"}},
	}
	want := `+---+
| a |
+---+
  *
+---+
| b |
+---+
  *
+---+
| c |
+---+`
	if got := graph.DrawASCII(); got != want {
		t.Fatalf("DrawASCII() =\n%s\nwant\n%s", got, want)
	}
}

// TestDrawASCIIFanout covers one source fanning out to two targets: the gap
// gets one channel row per crossing edge.
func TestDrawASCIIFanout(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}, {ID: "c", Name: "c"}},
		Edges: []GraphEdge{{Source: "a", Target: "b"}, {Source: "a", Target: "c"}},
	}
	want := `+---+
| a |
+---+
  *
  ********
+---+  +---+
| b |  | c |
+---+  +---+`
	if got := graph.DrawASCII(); got != want {
		t.Fatalf("DrawASCII() =\n%s\nwant\n%s", got, want)
	}
}

// TestDrawASCIIConditionalEdge: conditional edges draw with '.' (Python
// graph_ascii.py:350).
func TestDrawASCIIConditionalEdge(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}},
		Edges: []GraphEdge{{Source: "a", Target: "b", Conditional: true}},
	}
	want := `+---+
| a |
+---+
  .
+---+
| b |
+---+`
	if got := graph.DrawASCII(); got != want {
		t.Fatalf("DrawASCII() =\n%s\nwant\n%s", got, want)
	}
}

// TestDrawASCIILongEdge: an edge spanning more than one layer routes around
// the intermediate boxes via a right-side margin column instead of crossing
// them.
func TestDrawASCIILongEdge(t *testing.T) {
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}, {ID: "b", Name: "b"}, {ID: "c", Name: "c"}},
		Edges: []GraphEdge{
			{Source: "a", Target: "b"},
			{Source: "a", Target: "c"},
			{Source: "b", Target: "c"},
		},
	}
	want := `+---+
| a |
+---+
  *
  ******
+---+  *
| b |  *
+---+  *
  ******
  *
+---+
| c |
+---+`
	if got := graph.DrawASCII(); got != want {
		t.Fatalf("DrawASCII() =\n%s\nwant\n%s", got, want)
	}
}

// TestDrawASCIITrivial covers the degenerate cases: an empty graph renders
// an empty string, a single node renders just its box, and self-loop edges
// are skipped.
func TestDrawASCIITrivial(t *testing.T) {
	if got := (Graph{}).DrawASCII(); got != "" {
		t.Fatalf("empty graph DrawASCII() = %q", got)
	}
	graph := Graph{
		Nodes: []GraphNode{{ID: "a", Name: "a"}},
		Edges: []GraphEdge{{Source: "a", Target: "a"}},
	}
	want := `+---+
| a |
+---+`
	if got := graph.DrawASCII(); got != want {
		t.Fatalf("DrawASCII() =\n%s\nwant\n%s", got, want)
	}
}
