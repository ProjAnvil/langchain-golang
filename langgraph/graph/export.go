package graph

import (
	"fmt"
	"maps"
	"slices"

	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Graph export and Mermaid visualization, mirroring Python's
// `Pregel.get_graph()` + `Graph.draw_mermaid()` (langgraph/pregel/main.py:845,
// langchain_core/runnables/graph_mermaid.py).
//
// Documented divergences from Python:
//
//   - Python discovers conditional-edge targets by dry-running the graph
//     with a mock state; this port makes a single best-effort probe instead:
//     routers registered via plain AddConditionalEdges/SetConditionalEntryPoint
//     are called once with an empty state (runtime.NewRuntime(nil) +
//     map[string]any{}), and the returned targets draw as dashed edges. A
//     router that errors or panics — e.g. because it reads state keys absent
//     from the empty probe state — is omitted from the drawing, as is every
//     router when probing is disabled (WithRouterProbing(false)). Register
//     routers via AddConditionalEdgesWithPathMap/
//     SetConditionalEntryPointWithPathMap (Python's path_map) for statically
//     enumerated, compile-time validated edges.
//   - Subgraph nodes (StateGraph.AddSubgraph) draw as a single node by
//     default; GetGraph(WithXrayDepth(n)) expands them into ":"-prefixed
//     inner graphs (Python's xray), with Mermaid subgraph blocks rendered by
//     runnables.Graph.DrawMermaid.
//   - PNG rendering is not supported (runnables.Graph.DrawPNG errors).

// AddConditionalEdgesWithPathMap is AddConditionalEdges plus a static path
// map, mirroring Python's `add_conditional_edges(source, path, path_map)`:
// each key is a label for the edge drawn to the node named by its value
// (types.END is allowed as a value). Runtime routing is unchanged — the
// router's return values are used directly, exactly like
// AddConditionalEdges — but the path map makes the router's targets
// statically enumerable for graph export (GetGraph/DrawMermaid) and they
// are validated at Compile time (targets must be registered nodes or
// types.END).
func (g *StateGraph) AddConditionalEdgesWithPathMap(from string, router ConditionalEdge, pathMap map[string]string) *StateGraph {
	if len(pathMap) == 0 {
		g.setErr(fmt.Errorf("graph: conditional edge path map for %q must not be empty", from))
		return g
	}
	prev := g.err
	g.AddConditionalEdges(from, router)
	if g.err == prev {
		g.pathMaps[from] = maps.Clone(pathMap)
	}
	return g
}

// SetConditionalEntryPointWithPathMap is SetConditionalEntryPoint plus a
// static path map (see AddConditionalEdgesWithPathMap): the entry router's
// targets become statically enumerable, so GetGraph draws dashed edges out
// of types.START and Compile validates the targets.
func (g *StateGraph) SetConditionalEntryPointWithPathMap(router ConditionalEdge, pathMap map[string]string) *StateGraph {
	if len(pathMap) == 0 {
		g.setErr(fmt.Errorf("graph: conditional entry point path map must not be empty"))
		return g
	}
	prev := g.err
	g.SetConditionalEntryPoint(router)
	if g.err == prev {
		g.entryPathMap = maps.Clone(pathMap)
	}
	return g
}

// GetGraphOptions configures GetGraph.
type GetGraphOptions struct {
	// XrayDepth controls subgraph expansion (Python's xray): 0 (default)
	// draws subgraph nodes as single nodes, a positive depth expands that
	// many nesting levels, and a negative depth expands recursively without
	// bound (Python's xray=True).
	XrayDepth int
	// ProbeRouters (default true) enables best-effort target discovery for
	// conditional routers registered without a path map: each such router is
	// called once with an empty state and its returned targets draw as
	// dashed edges. Routers that error or panic are omitted.
	ProbeRouters bool
}

// GetGraphOption mutates GetGraphOptions; see GetGraph.
type GetGraphOption func(*GetGraphOptions)

// WithXrayDepth sets the subgraph expansion depth (Python's xray): depth < 0
// expands all nesting levels, depth > 0 expands that many, 0 (default)
// draws subgraph nodes as single nodes.
func WithXrayDepth(depth int) GetGraphOption {
	return func(o *GetGraphOptions) { o.XrayDepth = depth }
}

// WithRouterProbing toggles best-effort router probing (default true).
func WithRouterProbing(probe bool) GetGraphOption {
	return func(o *GetGraphOptions) { o.ProbeRouters = probe }
}

func defaultGetGraphOptions() GetGraphOptions {
	return GetGraphOptions{ProbeRouters: true}
}

// GetGraph exports the graph's static structure as a core/runnables.Graph,
// mirroring Python's `get_graph()` returning a langchain_core Graph: all
// registered nodes plus the types.START/types.END sentinels (drawn as the
// first/last nodes), static edges and join edges (one solid edge per join
// parent into its child), and conditional edges (dashed, labeled by
// path-map key unless key == target). See the package comment above for the
// divergences around router probing and subgraph expansion.
func (g *StateGraph) GetGraph(opts ...GetGraphOption) runnables.Graph {
	options := defaultGetGraphOptions()
	for _, opt := range opts {
		opt(&options)
	}
	joins := make([]joinMeta, 0, len(g.joinEdges))
	for _, je := range g.joinEdges {
		joins = append(joins, joinMeta{parents: je.parents, child: je.child})
	}
	out := exportGraph(exportParams{
		nodes:        g.nodes,
		edges:        g.edges,
		conditional:  g.conditional,
		pathMaps:     g.pathMaps,
		joins:        joins,
		entry:        g.entry,
		entryRouter:  g.entryRouter,
		entryPathMap: g.entryPathMap,
		probe:        options.ProbeRouters,
	})
	expandSubgraphs(&out, g.subgraphs, options)
	return out
}

// GetGraph exports the compiled graph's static structure; it matches the
// builder's StateGraph.GetGraph (Compile preserves the metadata) and
// additionally marks nodes registered via WithInterruptBefore/
// WithInterruptAfter with `__interrupt` metadata (before/after/before,after),
// mirroring Python (pregel/_draw.py:225-230).
func (g *CompiledGraph) GetGraph(opts ...GetGraphOption) runnables.Graph {
	options := defaultGetGraphOptions()
	for _, opt := range opts {
		opt(&options)
	}
	out := exportGraph(exportParams{
		nodes:           g.nodes,
		edges:           g.edges,
		conditional:     g.conditional,
		pathMaps:        g.pathMaps,
		joins:           g.joins,
		entry:           g.entry,
		entryRouter:     g.entryRouter,
		entryPathMap:    g.entryPathMap,
		interruptBefore: g.interruptBefore,
		interruptAfter:  g.interruptAfter,
		probe:           options.ProbeRouters,
	})
	expandSubgraphs(&out, g.subgraphs, options)
	return out
}

// DrawMermaid renders the graph as a Python-compatible Mermaid flowchart
// (see runnables.Graph.DrawMermaid), passing through any mermaid options
// (e.g. runnables.WithStyles(false)). Shorthand for
// GetGraph().DrawMermaid(opts...); use GetGraph(WithXrayDepth(...)) first to
// expand subgraphs.
func (g *StateGraph) DrawMermaid(opts ...runnables.MermaidOption) string {
	return g.GetGraph().DrawMermaid(opts...)
}

// DrawMermaid renders the compiled graph as a Python-compatible Mermaid
// flowchart, passing through any mermaid options. Shorthand for
// GetGraph().DrawMermaid(opts...).
func (g *CompiledGraph) DrawMermaid(opts ...runnables.MermaidOption) string {
	return g.GetGraph().DrawMermaid(opts...)
}

// exportParams carries the builder/compiled metadata exportGraph assembles
// into a runnables.Graph.
type exportParams struct {
	nodes        map[string]NodeFunc
	edges        map[string][]string
	conditional  map[string]ConditionalEdge
	pathMaps     map[string]map[string]string
	joins        []joinMeta
	entry        string
	entryRouter  ConditionalEdge
	entryPathMap map[string]string
	// interruptBefore/interruptAfter are the compiled interrupt boundaries
	// (nil on the builder): nodes in either get `__interrupt` metadata.
	interruptBefore map[string]bool
	interruptAfter  map[string]bool
	// probe enables best-effort target discovery for routers without a
	// path map.
	probe bool
}

// exportGraph assembles the runnables.Graph shared by both GetGraph
// implementations. Edge/node ordering is irrelevant here:
// runnables.Graph.DrawMermaid normalizes (sorts) for deterministic output.
func exportGraph(p exportParams) runnables.Graph {
	out := runnables.Graph{FirstNode: types.START, LastNode: types.END}
	out.Nodes = append(out.Nodes,
		runnables.GraphNode{ID: types.START, Name: types.START, Type: "start"},
		runnables.GraphNode{ID: types.END, Name: types.END, Type: "end"},
	)
	for _, name := range slices.Sorted(maps.Keys(p.nodes)) {
		out.Nodes = append(out.Nodes, runnables.GraphNode{
			ID:       name,
			Name:     name,
			Type:     "node",
			Metadata: interruptMetadata(name, p.interruptBefore, p.interruptAfter),
		})
	}
	addEdge := func(source, target, label string, conditional bool) {
		out.Edges = append(out.Edges, runnables.GraphEdge{Source: source, Target: target, Label: label, Conditional: conditional})
	}
	if p.entry != "" {
		addEdge(types.START, p.entry, "", false)
	}
	for _, from := range slices.Sorted(maps.Keys(p.edges)) {
		for _, to := range p.edges[from] {
			addEdge(from, to, "", false)
		}
	}
	// Join (waiting) edges draw as one solid edge per parent into the child.
	for _, jm := range p.joins {
		for _, parent := range jm.parents {
			addEdge(parent, jm.child, "", false)
		}
	}
	for _, from := range slices.Sorted(maps.Keys(p.pathMaps)) {
		addPathMapEdges(addEdge, from, p.pathMaps[from])
	}
	if p.entryRouter != nil {
		addPathMapEdges(addEdge, types.START, p.entryPathMap)
	}
	if p.probe {
		seen := edgeSet(out.Edges)
		for _, from := range slices.Sorted(maps.Keys(p.conditional)) {
			if len(p.pathMaps[from]) > 0 {
				continue
			}
			addProbedEdges(&out, seen, from, p.conditional[from], p.nodes)
		}
		if p.entryRouter != nil && len(p.entryPathMap) == 0 {
			addProbedEdges(&out, seen, types.START, p.entryRouter, p.nodes)
		}
	}
	return out
}

// interruptMetadata returns the `__interrupt` node metadata for name,
// mirroring Python's before/after/before,after marking
// (pregel/_draw.py:225-230). Nil when the node has no boundary.
func interruptMetadata(name string, before, after map[string]bool) map[string]any {
	b, a := before[name], after[name]
	var value string
	switch {
	case b && a:
		value = "before,after"
	case b:
		value = "before"
	case a:
		value = "after"
	default:
		return nil
	}
	return map[string]any{"__interrupt": value}
}

// edgeSet indexes edges by source/target for dedup (Python's add_edge
// dedups on the pair).
func edgeSet(edges []runnables.GraphEdge) map[[2]string]bool {
	seen := make(map[[2]string]bool, len(edges))
	for _, edge := range edges {
		seen[[2]string{edge.Source, edge.Target}] = true
	}
	return seen
}

// addProbedEdges calls router once with an empty state (best effort — see
// the package comment) and draws each discovered target as a dashed edge,
// deduplicated against the edges already present (path-map, static, and
// previously probed). Targets that are neither registered nodes nor
// types.END are skipped. A router that errors or panics draws nothing.
func addProbedEdges(out *runnables.Graph, seen map[[2]string]bool, from string, router ConditionalEdge, nodes map[string]NodeFunc) {
	targets, ok := probeRouter(router)
	if !ok {
		return
	}
	for _, target := range targets {
		if target != types.END {
			if _, exists := nodes[target]; !exists {
				continue
			}
		}
		key := [2]string{from, target}
		if seen[key] {
			continue
		}
		seen[key] = true
		out.Edges = append(out.Edges, runnables.GraphEdge{Source: from, Target: target, Conditional: true})
	}
}

// probeRouter invokes router with an empty runtime and state, translating
// its return into target names: strings (types.END included) and
// *types.Send nodes. ok is false when the router errors, panics, or returns
// an element of any other type.
func probeRouter(router ConditionalEdge) (targets []string, ok bool) {
	defer func() {
		if recover() != nil {
			targets, ok = nil, false
		}
	}()
	out, err := router(runtime.NewRuntime(nil), map[string]any{})
	if err != nil {
		return nil, false
	}
	for _, item := range out {
		switch v := item.(type) {
		case string:
			targets = append(targets, v)
		case *types.Send:
			targets = append(targets, v.Node)
		default:
			return nil, false
		}
	}
	return targets, true
}

// expandSubgraphs replaces each registered subgraph node with its child's
// own exported graph under the `parent:child` prefix, mirroring Python's
// xray handling (pregel/_draw.py:258-275): the child's first/last sentinel
// nodes are trimmed (core TrimFirstNode/TrimLastNode), the trimmed graph is
// merged in via Graph.Extend, and parent edges sourced/targeted at the
// subgraph node are rewired to the inner last/first node. Children are
// expanded recursively with depth-1 (a negative depth stays negative, the
// Python xray=True equivalent). A child that does not resolve to unique
// first/last nodes is left as a single node, exactly like Python.
func expandSubgraphs(out *runnables.Graph, subgraphs map[string]*CompiledGraph, options GetGraphOptions) {
	if options.XrayDepth == 0 {
		return
	}
	childOptions := options
	if options.XrayDepth > 0 {
		childOptions.XrayDepth--
	}
	for _, name := range slices.Sorted(maps.Keys(subgraphs)) {
		sub := subgraphs[name].GetGraph(func(o *GetGraphOptions) { *o = childOptions })
		if len(sub.Nodes) <= 1 || !graphHasNode(*out, name) || sub.FirstNodeID() == "" || sub.LastNodeID() == "" {
			continue
		}
		sub.TrimFirstNode()
		sub.TrimLastNode()
		removeGraphNode(out, name)
		first, last := out.Extend(sub, name)
		for i := range out.Edges {
			if out.Edges[i].Source == name {
				out.Edges[i].Source = last
			}
			if out.Edges[i].Target == name {
				out.Edges[i].Target = first
			}
		}
	}
}

// graphHasNode reports whether id is a node of g.
func graphHasNode(g runnables.Graph, id string) bool {
	for _, node := range g.Nodes {
		if node.ID == id {
			return true
		}
	}
	return false
}

// removeGraphNode drops the node id from g.Nodes only — edges touching it
// stay (they are rewired by expandSubgraphs, mirroring Python's
// graph.nodes.pop(name)). FirstNode/LastNode are cleared if they name it.
func removeGraphNode(g *runnables.Graph, id string) {
	nodes := make([]runnables.GraphNode, 0, len(g.Nodes))
	for _, node := range g.Nodes {
		if node.ID != id {
			nodes = append(nodes, node)
		}
	}
	g.Nodes = nodes
	if g.FirstNode == id {
		g.FirstNode = ""
	}
	if g.LastNode == id {
		g.LastNode = ""
	}
}

// addPathMapEdges draws one dashed edge per path-map entry, labeled by the
// key unless key == target (Python draws no label for the common
// `{"node": "node"}` self-describing mapping).
func addPathMapEdges(addEdge func(source, target, label string, conditional bool), from string, pathMap map[string]string) {
	for _, key := range slices.Sorted(maps.Keys(pathMap)) {
		target := pathMap[key]
		label := key
		if label == target {
			label = ""
		}
		addEdge(from, target, label, true)
	}
}
