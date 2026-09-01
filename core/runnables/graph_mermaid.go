package runnables

import (
	"fmt"
	"sort"
	"strings"
)

// NodeStyles customizes the Mermaid classDef styles for the default, first
// and last node classes, mirroring Python's NodeStyles dataclass
// (graph.py:156-168).
type NodeStyles struct {
	Default string
	First   string
	Last    string
}

// defaultNodeStyles mirrors Python's NodeStyles defaults.
var defaultNodeStyles = NodeStyles{
	Default: "fill:#f2f0ff,line-height:1.2",
	First:   "fill-opacity:0",
	Last:    "fill:#bfb6fc",
}

// MermaidOptions configures DrawMermaid, mirroring the keyword arguments of
// Python's draw_mermaid (graph_mermaid.py:45-56).
type MermaidOptions struct {
	// WithStyles (default true) renders frontmatter, node declarations and
	// classDef styles. False emits only `graph TD;` plus edge lines
	// (subgraph blocks are still emitted, matching Python).
	WithStyles bool
	// CurveStyle (default "linear") merges into frontmatter
	// config.flowchart.curve.
	CurveStyle string
	// FrontmatterConfig is merged with the same semantics as Python: the
	// curve joins config.flowchart, all other user keys are emitted
	// verbatim.
	FrontmatterConfig map[string]any
	// NodeStyles optionally replaces the default/first/last classDef styles.
	NodeStyles *NodeStyles
}

// MermaidOption mutates MermaidOptions; see DrawMermaid.
type MermaidOption func(*MermaidOptions)

// WithStyles toggles styled output (Python's with_styles).
func WithStyles(with bool) MermaidOption {
	return func(o *MermaidOptions) { o.WithStyles = with }
}

// WithCurveStyle sets the Mermaid flowchart curve (Python's curve_style).
func WithCurveStyle(curve string) MermaidOption {
	return func(o *MermaidOptions) { o.CurveStyle = curve }
}

// WithFrontmatterConfig sets the user frontmatter config (Python's
// frontmatter_config); the merge happens at draw time.
func WithFrontmatterConfig(config map[string]any) MermaidOption {
	return func(o *MermaidOptions) { o.FrontmatterConfig = config }
}

// WithNodeStyles replaces the classDef node styles (Python's node_styles).
func WithNodeStyles(styles NodeStyles) MermaidOption {
	return func(o *MermaidOptions) { o.NodeStyles = &styles }
}

// DrawMermaid renders a Mermaid flowchart compatible with Python's
// langchain_core.runnables.graph_mermaid.draw_mermaid (defaults:
// with_styles=True, curve linear): frontmatter config, `id(label)` nodes,
// rounded `id([label]):::first`/`:::last` shapes for FirstNode/LastNode,
// solid ` --> ` / labeled ` -- &nbsp;label&nbsp; --> ` edges, dashed
// conditional edges (` -.-> ` / ` -. &nbsp;label&nbsp; .-> `), mermaid
// `subgraph` blocks for ":"-prefixed node IDs (graph_mermaid.py:113-251),
// and the trailing classDef default/first/last styles. Node IDs are
// sanitized with the `_to_safe_id` equivalent (see toSafeMermaidID). It is
// text-only and does not call remote rendering services.
//
// Documented divergences from Python: node/edge ordering is deterministic
// (sorted, via normalized) rather than insertion order, and a duplicate
// subgraph name (two prefixes sharing their last ":" segment) is silently
// rendered inline — Python raises ValueError, but this signature returns
// only a string.
func (g Graph) DrawMermaid(opts ...MermaidOption) string {
	options := MermaidOptions{WithStyles: true, CurveStyle: "linear"}
	for _, opt := range opts {
		opt(&options)
	}
	g = g.normalized()

	var b strings.Builder
	if options.WithStyles {
		b.WriteString("---\n")
		b.WriteString(frontmatterYAML(mergeFrontmatterConfig(options.FrontmatterConfig, options.CurveStyle)))
		b.WriteString("---\ngraph TD;\n")
	} else {
		b.WriteString("graph TD;\n")
	}

	// Group nodes by subgraph: IDs containing ":" belong to their deepest
	// prefix; the rest are regular nodes (graph_mermaid.py:113-122).
	subgraphNodes := map[string][]GraphNode{}
	subgraphPrefixes := []string{}
	regularNodes := []GraphNode{}
	for _, node := range g.Nodes {
		idx := strings.LastIndex(node.ID, ":")
		if idx < 0 {
			regularNodes = append(regularNodes, node)
			continue
		}
		prefix := node.ID[:idx]
		if _, ok := subgraphNodes[prefix]; !ok {
			subgraphPrefixes = append(subgraphPrefixes, prefix)
		}
		subgraphNodes[prefix] = append(subgraphNodes[prefix], node)
	}

	renderNode := func(node GraphNode) {
		// Python labels subgraph nodes with the last ":" segment of the name.
		name := node.Name
		if idx := strings.LastIndex(name, ":"); idx >= 0 {
			name = name[idx+1:]
		}
		label := name
		if isMarkdownSpecialName(label) {
			label = "<p>" + label + "</p>"
		}
		if len(node.Metadata) > 0 {
			pairs := make([]string, 0, len(node.Metadata))
			for _, key := range sortedRunnableKeys(node.Metadata) {
				pairs = append(pairs, fmt.Sprintf("%s = %v", key, node.Metadata[key]))
			}
			label += "<hr/><small><em>" + strings.Join(pairs, "\n") + "</em></small>"
		}
		id := toSafeMermaidID(node.ID)
		switch node.ID {
		case g.LastNode:
			// Checked first: Python's format_dict lets last_node overwrite
			// first_node when both name the same node.
			fmt.Fprintf(&b, "\t%s([%s]):::last\n", id, label)
		case g.FirstNode:
			fmt.Fprintf(&b, "\t%s([%s]):::first\n", id, label)
		default:
			fmt.Fprintf(&b, "\t%s(%s)\n", id, label)
		}
	}

	if options.WithStyles {
		for _, node := range regularNodes {
			renderNode(node)
		}
	}

	// Group edges by the common ":"-prefix of source and target
	// (graph_mermaid.py:158-165), keeping first-appearance order over the
	// sorted edge list for deterministic output.
	type edgeGroup struct {
		prefix string
		edges  []GraphEdge
	}
	groups := []*edgeGroup{}
	groupByPrefix := map[string]*edgeGroup{}
	for _, edge := range g.Edges {
		prefix := commonIDPrefix(edge.Source, edge.Target)
		group, ok := groupByPrefix[prefix]
		if !ok {
			group = &edgeGroup{prefix: prefix}
			groupByPrefix[prefix] = group
			groups = append(groups, group)
		}
		group.edges = append(group.edges, edge)
	}

	renderEdge := func(edge GraphEdge) {
		link := ""
		if edge.Label != "" {
			data := wrapMermaidLabelWords(edge.Label, mermaidWrapLabelWords)
			if edge.Conditional {
				link = " -. &nbsp;" + data + "&nbsp; .-> "
			} else {
				link = " -- &nbsp;" + data + "&nbsp; --> "
			}
		} else if edge.Conditional {
			link = " -.-> "
		} else {
			link = " --> "
		}
		fmt.Fprintf(&b, "\t%s%s%s;\n", toSafeMermaidID(edge.Source), link, toSafeMermaidID(edge.Target))
	}

	seenSubgraphs := map[string]bool{}
	var addSubgraph func(edges []GraphEdge, prefix string)
	addSubgraph = func(edges []GraphEdge, prefix string) {
		selfLoop := len(edges) == 1 && edges[0].Source == edges[0].Target
		wrap := prefix != "" && !selfLoop
		if wrap {
			name := prefix[strings.LastIndex(prefix, ":")+1:]
			if seenSubgraphs[name] {
				// Divergence: Python raises ValueError on duplicate subgraph
				// names; Go's string-only signature renders the edges inline
				// without a block instead.
				wrap = false
			} else {
				seenSubgraphs[name] = true
				b.WriteString("\tsubgraph " + name + "\n")
				if options.WithStyles {
					for _, node := range subgraphNodes[prefix] {
						renderNode(node)
					}
				}
			}
		}
		for _, edge := range edges {
			renderEdge(edge)
		}
		// Recurse into first-level nested subgraphs only
		// (graph_mermaid.py:215-221).
		for _, group := range groups {
			nested := group.prefix
			if nested == prefix || !strings.HasPrefix(nested, prefix+":") {
				continue
			}
			if strings.Contains(nested[len(prefix)+1:], ":") {
				continue
			}
			addSubgraph(group.edges, nested)
		}
		if wrap {
			b.WriteString("\tend\n")
		}
	}

	// Top-level edges (no common prefix), then each top-level subgraph.
	if group, ok := groupByPrefix[""]; ok {
		addSubgraph(group.edges, "")
	} else {
		addSubgraph(nil, "")
	}
	for _, group := range groups {
		if group.prefix == "" || strings.Contains(group.prefix, ":") {
			continue
		}
		addSubgraph(group.edges, group.prefix)
		seenSubgraphs[group.prefix] = true
	}

	// Empty subgraphs (nodes but no internal edges), with_styles only.
	if options.WithStyles {
		for _, prefix := range subgraphPrefixes {
			if strings.Contains(prefix, ":") || seenSubgraphs[prefix] {
				continue
			}
			seenSubgraphs[prefix] = true
			b.WriteString("\tsubgraph " + prefix + "\n")
			for _, node := range subgraphNodes[prefix] {
				renderNode(node)
			}
			b.WriteString("\tend\n")
		}
	}

	if options.WithStyles {
		styles := defaultNodeStyles
		if options.NodeStyles != nil {
			styles = *options.NodeStyles
		}
		b.WriteString("\tclassDef default " + styles.Default + "\n")
		b.WriteString("\tclassDef first " + styles.First + "\n")
		b.WriteString("\tclassDef last " + styles.Last + "\n")
	}
	return b.String()
}

// commonIDPrefix returns the shared leading ":"-separated segments of two
// node IDs (graph_mermaid.py:160-164).
func commonIDPrefix(source, target string) string {
	srcParts := strings.Split(source, ":")
	tgtParts := strings.Split(target, ":")
	n := min(len(srcParts), len(tgtParts))
	common := []string{}
	for i := 0; i < n; i++ {
		if srcParts[i] != tgtParts[i] {
			break
		}
		common = append(common, srcParts[i])
	}
	return strings.Join(common, ":")
}

// mergeFrontmatterConfig merges the curve into the user's frontmatter config
// with Python's semantics (graph_mermaid.py:91-101): the result is the user
// config with config.flowchart.curve set (curve wins over a user-supplied
// curve), all other keys untouched.
func mergeFrontmatterConfig(user map[string]any, curve string) map[string]any {
	out := make(map[string]any, len(user)+1)
	for key, value := range user {
		out[key] = value
	}
	userConfig, _ := user["config"].(map[string]any)
	config := make(map[string]any, len(userConfig)+1)
	for key, value := range userConfig {
		config[key] = value
	}
	userFlowchart, _ := userConfig["flowchart"].(map[string]any)
	flowchart := make(map[string]any, len(userFlowchart)+1)
	for key, value := range userFlowchart {
		flowchart[key] = value
	}
	flowchart["curve"] = curve
	config["flowchart"] = flowchart
	out["config"] = config
	return out
}

// frontmatterYAML is a minimal YAML emitter for frontmatter configs, matching
// the PyYAML dump style used by Python: sorted keys, 2-space indentation,
// `key: value` scalars, and single quotes for strings that need them (e.g.
// "#e2e2e2" -> '#e2e2e2'). It supports nested map[string]any, []any, and
// string/bool/int/float/nil scalars.
func frontmatterYAML(config map[string]any) string {
	var b strings.Builder
	emitYAMLMap(&b, config, 0)
	return b.String()
}

func emitYAMLMap(b *strings.Builder, values map[string]any, indent int) {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	pad := strings.Repeat(" ", indent)
	for _, key := range keys {
		switch value := values[key].(type) {
		case map[string]any:
			b.WriteString(pad + key + ":\n")
			emitYAMLMap(b, value, indent+2)
		case []any:
			b.WriteString(pad + key + ":\n")
			emitYAMLList(b, value, indent+2)
		default:
			b.WriteString(pad + key + ": " + yamlScalar(value) + "\n")
		}
	}
}

func emitYAMLList(b *strings.Builder, values []any, indent int) {
	pad := strings.Repeat(" ", indent)
	for _, value := range values {
		switch value := value.(type) {
		case map[string]any:
			b.WriteString(pad + "-\n")
			emitYAMLMap(b, value, indent+2)
		default:
			b.WriteString(pad + "- " + yamlScalar(value) + "\n")
		}
	}
}

// yamlScalar renders a scalar in PyYAML dump style: strings are plain unless
// they would be misparsed (then single-quoted, with embedded quotes doubled),
// bools and nil use YAML literals, numbers use Go's default formatting.
func yamlScalar(value any) string {
	switch value := value.(type) {
	case nil:
		return "null"
	case string:
		return yamlQuoteString(value)
	case bool:
		if value {
			return "true"
		}
		return "false"
	case float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%v", value)
	default:
		return yamlQuoteString(fmt.Sprintf("%v", value))
	}
}

// yamlQuoteString emits a plain string when possible and a single-quoted one
// otherwise, mirroring PyYAML's choice for the values frontmatter configs
// realistically hold.
func yamlQuoteString(s string) string {
	if !yamlNeedsQuotes(s) {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

func yamlNeedsQuotes(s string) bool {
	if s == "" || s != strings.TrimSpace(s) {
		return true
	}
	// YAML plain scalars cannot start with an indicator character.
	if strings.ContainsRune("-?:,[]{}#&*!|>'\"%@`", rune(s[0])) {
		return true
	}
	// '#' starts a comment and ': '/' :' are mapping syntax.
	if strings.Contains(s, "#") || strings.Contains(s, ": ") || strings.HasSuffix(s, ":") {
		return true
	}
	// Words PyYAML would resolve as non-strings.
	switch strings.ToLower(s) {
	case "true", "false", "null", "yes", "no", "on", "off", "~", ".nan", ".inf", "-.inf":
		return true
	}
	return false
}

// toSafeMermaidID converts a node ID into a Mermaid-compatible id, the Go
// equivalent of Python's `_to_safe_id` (graph_mermaid.py:255): characters in
// [a-zA-Z0-9_-] pass through unchanged, every other character becomes
// backslash + its lowercase hex codepoint.
func toSafeMermaidID(id string) string {
	var b strings.Builder
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			fmt.Fprintf(&b, "\\%x", r)
		}
	}
	return b.String()
}

// markdownSpecialChars mirrors Python's MARKDOWN_SPECIAL_CHARS
// (graph_mermaid.py:42): node names starting AND ending with one of these
// are wrapped in <p>...</p> so Mermaid treats them as markdown.
const markdownSpecialChars = "*_`"

func isMarkdownSpecialName(name string) bool {
	if name == "" {
		return false
	}
	first := name[0]
	last := name[len(name)-1]
	return strings.ContainsRune(markdownSpecialChars, rune(first)) &&
		strings.ContainsRune(markdownSpecialChars, rune(last))
}

// mermaidWrapLabelWords mirrors Python's wrap_label_n_words default
// (graph_mermaid.py:54).
const mermaidWrapLabelWords = 9

// wrapMermaidLabelWords inserts a "&nbsp<br>&nbsp" line break after every
// n words, mirroring Python's edge-label wrapping in draw_mermaid.
func wrapMermaidLabelWords(label string, n int) string {
	words := strings.Fields(label)
	if len(words) <= n {
		return label
	}
	chunks := make([]string, 0, (len(words)+n-1)/n)
	for i := 0; i < len(words); i += n {
		end := i + n
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}
	return strings.Join(chunks, "&nbsp<br>&nbsp")
}
