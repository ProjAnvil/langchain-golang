package runnables

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// GraphNode describes one runnable node in a runnable graph.
type GraphNode struct {
	ID       string
	Name     string
	Type     string
	Metadata map[string]any
}

// GraphEdge describes a directed edge between runnable graph nodes.
type GraphEdge struct {
	Source string
	Target string
	Label  string
	// Conditional marks edges produced by a dynamic router (drawn dashed in
	// Mermaid), mirroring Python's Edge.conditional.
	Conditional bool
}

// Graph is a lightweight representation of a runnable composition graph.
type Graph struct {
	Nodes []GraphNode
	Edges []GraphEdge
	// FirstNode/LastNode optionally name the node IDs drawn with Mermaid's
	// rounded first/last shapes and :::first/:::last classes (Python's
	// draw_mermaid first_node/last_node). Empty means no such styling; core
	// combinators leave them empty, langgraph sets them to START/END.
	FirstNode string
	LastNode  string
}

// MarshalJSONStable returns a deterministic JSON representation.
func (g Graph) MarshalJSONStable() ([]byte, error) {
	normalized := g.normalized()
	return json.Marshal(normalized)
}

// DrawPNG explicitly reports that Go core does not render graph images without
// an external renderer.
func (g Graph) DrawPNG() ([]byte, error) {
	return nil, fmt.Errorf("graph PNG rendering is not supported; use DrawMermaid or DrawASCII")
}

// Extend adds all nodes and edges from another graph, prefixing every node ID
// with `prefix:` when prefix is non-empty, mirroring Python's Graph.extend
// (graph.py:386-422). It does not connect the graphs. It returns the
// (prefixed) first and last node IDs of the added subgraph, "" when the
// subgraph has no unique one. Python's all-UUID prefix reset has no Go
// equivalent (IDs here are plain names) and is not mirrored.
func (g *Graph) Extend(other Graph, prefix string) (first, last string) {
	prefixed := func(id string) string {
		if prefix == "" {
			return id
		}
		return prefix + ":" + id
	}
	for _, node := range other.Nodes {
		node.ID = prefixed(node.ID)
		node.Metadata = cloneGraphMetadata(node.Metadata)
		g.Nodes = append(g.Nodes, node)
	}
	for _, edge := range other.Edges {
		edge.Source = prefixed(edge.Source)
		edge.Target = prefixed(edge.Target)
		g.Edges = append(g.Edges, edge)
	}
	if id := other.FirstNodeID(); id != "" {
		first = prefixed(id)
	}
	if id := other.LastNodeID(); id != "" {
		last = prefixed(id)
	}
	return first, last
}

// FirstNodeID returns the unique root node's ID: the explicit FirstNode field
// when set, otherwise the single node that is not a target of any edge
// (Python's Graph.first_node). Returns "" when there is no unique root.
func (g Graph) FirstNodeID() string {
	if g.FirstNode != "" {
		return g.FirstNode
	}
	return g.uniqueBoundaryNode(nil, true)
}

// LastNodeID returns the unique terminal node's ID: the explicit LastNode
// field when set, otherwise the single node that is not a source of any edge
// (Python's Graph.last_node). Returns "" when there is no unique terminal.
func (g Graph) LastNodeID() string {
	if g.LastNode != "" {
		return g.LastNode
	}
	return g.uniqueBoundaryNode(nil, false)
}

// TrimFirstNode removes the first node if it exists, has exactly one
// outgoing edge, and removing it still leaves a unique first node
// (Python's Graph.trim_first_node, graph.py:483-494).
func (g *Graph) TrimFirstNode() {
	first := g.FirstNodeID()
	if first == "" {
		return
	}
	outgoing := 0
	for _, edge := range g.Edges {
		if edge.Source == first {
			outgoing++
		}
	}
	if outgoing != 1 {
		return
	}
	if g.uniqueBoundaryNode(map[string]bool{first: true}, true) == "" {
		return
	}
	g.removeNode(first)
}

// TrimLastNode removes the last node if it exists, has exactly one
// incoming edge, and removing it still leaves a unique last node
// (Python's Graph.trim_last_node, graph.py:496-507).
func (g *Graph) TrimLastNode() {
	last := g.LastNodeID()
	if last == "" {
		return
	}
	incoming := 0
	for _, edge := range g.Edges {
		if edge.Target == last {
			incoming++
		}
	}
	if incoming != 1 {
		return
	}
	if g.uniqueBoundaryNode(map[string]bool{last: true}, false) == "" {
		return
	}
	g.removeNode(last)
}

// uniqueBoundaryNode mirrors Python's _first_node/_last_node (graph.py:708-
// 741): root=true finds the single node that is not a target of any edge
// whose source is not excluded; root=false finds the single node that is not
// a source of any edge whose target is not excluded. Excluded nodes are never
// candidates.
func (g Graph) uniqueBoundaryNode(exclude map[string]bool, root bool) string {
	blocked := map[string]bool{}
	for _, edge := range g.Edges {
		if root {
			if !exclude[edge.Source] {
				blocked[edge.Target] = true
			}
		} else if !exclude[edge.Target] {
			blocked[edge.Source] = true
		}
	}
	found := ""
	count := 0
	for _, node := range g.Nodes {
		if exclude[node.ID] || blocked[node.ID] {
			continue
		}
		found = node.ID
		count++
	}
	if count == 1 {
		return found
	}
	return ""
}

// removeNode removes a node and every edge connected to it (Python's
// Graph.remove_node), clearing FirstNode/LastNode if they point at it.
func (g *Graph) removeNode(id string) {
	nodes := make([]GraphNode, 0, len(g.Nodes))
	for _, node := range g.Nodes {
		if node.ID != id {
			nodes = append(nodes, node)
		}
	}
	g.Nodes = nodes
	edges := make([]GraphEdge, 0, len(g.Edges))
	for _, edge := range g.Edges {
		if edge.Source != id && edge.Target != id {
			edges = append(edges, edge)
		}
	}
	g.Edges = edges
	if g.FirstNode == id {
		g.FirstNode = ""
	}
	if g.LastNode == id {
		g.LastNode = ""
	}
}

type graphProvider interface {
	Graph() Graph
}

// GetGraph returns a runnable graph for composed runnables. Runnables without
// graph support are represented as a single leaf node.
func GetGraph(runnable any) Graph {
	return graphForRunnable("runnable", runnable)
}

// Graph returns a sequence graph with edges from first terminals to second
// roots.
func (r Sequence[I, M, O]) Graph() Graph {
	graph := Graph{}
	first := graphForRunnable("first", r.First).prefixed("first")
	second := graphForRunnable("second", r.Second).prefixed("second")
	graph.append(first)
	graph.append(second)
	for _, source := range first.terminals() {
		for _, target := range second.roots() {
			graph.addEdge(source, target, "then")
		}
	}
	return graph
}

// Graph returns a parallel graph with one branch per step.
func (r Parallel[I]) Graph() Graph {
	graph := Graph{}
	root := graph.addNode("parallel", "Parallel", "Parallel", nil)
	for _, key := range sortedRunnableKeys(r.Steps) {
		child := graphForRunnable(key, r.Steps[key]).prefixed("step." + key)
		graph.append(child)
		for _, target := range child.roots() {
			graph.addEdge(root, target, key)
		}
	}
	return graph
}

// Graph returns a router graph with one edge per route key.
func (r Router[I, O]) Graph() Graph {
	graph := Graph{}
	root := graph.addNode("router", "Router", "Router", nil)
	for _, key := range sortedRunnableKeys(r.Runnables) {
		child := graphForRunnable(key, r.Runnables[key]).prefixed("route." + key)
		graph.append(child)
		for _, target := range child.roots() {
			graph.addEdge(root, target, key)
		}
	}
	return graph
}

// Graph returns a configurable alternatives graph with one edge per choice.
func (r ConfigurableAlternatives[I, O]) Graph() Graph {
	graph := Graph{}
	root := graph.addNode("configurable", "ConfigurableAlternatives", "ConfigurableAlternatives", map[string]any{
		"field":       r.Field,
		"default_key": r.DefaultKey,
	})
	def := graphForRunnable("default", r.Default).prefixed("default")
	graph.append(def)
	for _, target := range def.roots() {
		graph.addEdge(root, target, r.DefaultKey)
	}
	for _, key := range sortedRunnableKeys(r.Choices) {
		child := graphForRunnable(key, r.Choices[key]).prefixed("choice." + key)
		graph.append(child)
		for _, target := range child.roots() {
			graph.addEdge(root, target, key)
		}
	}
	return graph
}

// Graph returns a branch graph with condition and selected runnable branches.
func (r Branch[I, O]) Graph() Graph {
	graph := Graph{}
	root := graph.addNode("branch", "Branch", "Branch", nil)
	for i, item := range r.Cases {
		condition := graphForRunnable(fmt.Sprintf("condition_%d", i), item.Condition).prefixed(fmt.Sprintf("case.%d.condition", i))
		body := graphForRunnable(fmt.Sprintf("case_%d", i), item.Runnable).prefixed(fmt.Sprintf("case.%d.runnable", i))
		graph.append(condition)
		graph.append(body)
		for _, target := range condition.roots() {
			graph.addEdge(root, target, fmt.Sprintf("condition_%d", i))
		}
		for _, source := range condition.terminals() {
			for _, target := range body.roots() {
				graph.addEdge(source, target, "true")
			}
		}
	}
	def := graphForRunnable("default", r.Default).prefixed("default")
	graph.append(def)
	for _, target := range def.roots() {
		graph.addEdge(root, target, "default")
	}
	return graph
}

// Graph returns a fallback graph with ordered fallback edges.
func (r WithFallbacks[I, O]) Graph() Graph {
	graph := Graph{}
	root := graph.addNode("fallbacks", "WithFallbacks", "WithFallbacks", nil)
	runnables := append([]Runnable[I, O]{r.Runnable}, r.Fallbacks...)
	for i, runnable := range runnables {
		label := "primary"
		if i > 0 {
			label = fmt.Sprintf("fallback_%d", i)
		}
		child := graphForRunnable(label, runnable).prefixed(label)
		graph.append(child)
		for _, target := range child.roots() {
			graph.addEdge(root, target, label)
		}
	}
	return graph
}

func graphForRunnable(name string, runnable any) Graph {
	if runnable == nil {
		return Graph{Nodes: []GraphNode{{
			ID:   "nil",
			Name: name,
			Type: "nil",
		}}}
	}
	if provider, ok := runnable.(graphProvider); ok {
		graph := provider.Graph()
		if len(graph.Nodes) > 0 {
			return graph
		}
	}
	typ := runnableTypeName(runnable)
	return Graph{Nodes: []GraphNode{{
		ID:   sanitizeGraphID(name),
		Name: name,
		Type: typ,
	}}}
}

func (g *Graph) addNode(id string, name string, typ string, metadata map[string]any) string {
	id = sanitizeGraphID(id)
	g.Nodes = append(g.Nodes, GraphNode{
		ID:       id,
		Name:     name,
		Type:     typ,
		Metadata: cloneGraphMetadata(metadata),
	})
	return id
}

func (g *Graph) addEdge(source string, target string, label string) {
	g.Edges = append(g.Edges, GraphEdge{Source: source, Target: target, Label: label})
}

func (g *Graph) append(other Graph) {
	g.Nodes = append(g.Nodes, other.Nodes...)
	g.Edges = append(g.Edges, other.Edges...)
}

func (g Graph) prefixed(prefix string) Graph {
	prefix = sanitizeGraphID(prefix)
	if prefix == "" {
		return g
	}
	out := Graph{
		Nodes: make([]GraphNode, len(g.Nodes)),
		Edges: make([]GraphEdge, len(g.Edges)),
	}
	for i, node := range g.Nodes {
		node.ID = prefix + "." + sanitizeGraphID(node.ID)
		out.Nodes[i] = node
	}
	for i, edge := range g.Edges {
		edge.Source = prefix + "." + sanitizeGraphID(edge.Source)
		edge.Target = prefix + "." + sanitizeGraphID(edge.Target)
		out.Edges[i] = edge
	}
	return out
}

func (g Graph) roots() []string {
	targets := map[string]bool{}
	for _, edge := range g.Edges {
		targets[edge.Target] = true
	}
	out := []string{}
	for _, node := range g.Nodes {
		if !targets[node.ID] {
			out = append(out, node.ID)
		}
	}
	return out
}

func (g Graph) terminals() []string {
	sources := map[string]bool{}
	for _, edge := range g.Edges {
		sources[edge.Source] = true
	}
	out := []string{}
	for _, node := range g.Nodes {
		if !sources[node.ID] {
			out = append(out, node.ID)
		}
	}
	return out
}

func sanitizeGraphID(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "node"
	}
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-' || r == '.':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func runnableTypeName(value any) string {
	typ := reflect.TypeOf(value)
	if typ == nil {
		return "nil"
	}
	if typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if typ.Name() != "" {
		return typ.Name()
	}
	return typ.String()
}

func sortedRunnableKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func cloneGraphMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}

func (g Graph) normalized() Graph {
	out := Graph{
		Nodes:     make([]GraphNode, len(g.Nodes)),
		Edges:     make([]GraphEdge, len(g.Edges)),
		FirstNode: g.FirstNode,
		LastNode:  g.LastNode,
	}
	for i, node := range g.Nodes {
		node.Metadata = cloneGraphMetadata(node.Metadata)
		out.Nodes[i] = node
	}
	copy(out.Edges, g.Edges)
	sort.SliceStable(out.Nodes, func(i, j int) bool { return out.Nodes[i].ID < out.Nodes[j].ID })
	sort.SliceStable(out.Edges, func(i, j int) bool {
		left := out.Edges[i].Source + "\x00" + out.Edges[i].Target + "\x00" + out.Edges[i].Label
		right := out.Edges[j].Source + "\x00" + out.Edges[j].Target + "\x00" + out.Edges[j].Label
		return left < right
	})
	return out
}
