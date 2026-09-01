package runnables

import (
	"math"
	"strings"
	"unicode/utf8"
)

// This file ports Python's graph_ascii.py box-drawing renderer.
//
// Documented divergence: Python lays the graph out with grandalf's Sugiyama
// algorithm; Go uses a simpler self-contained layout — longest-path layering,
// in-layer ordering by node ID, and orthogonal (vertical/horizontal) edge
// routing with one channel row per edge in each inter-layer gap. Edges
// spanning more than one layer detour through a right-side margin column so
// they never cross boxes. The output is therefore NOT byte-identical to the
// Python snapshots; it follows the same conventions (height-3 boxes of width
// len(" name ")+2, '*' for plain edges, '.' for conditional edges, edges
// drawn before boxes) and aims for readable, hierarchy-correct output.

// asciiBoxHeight mirrors VertexViewer.HEIGHT (graph_ascii.py:34).
const asciiBoxHeight = 3

// asciiBoxGap is the number of blank columns between boxes in one layer.
const asciiBoxGap = 2

// DrawASCII renders the graph as an ASCII box drawing, the Go port of
// Python's Graph.draw_ascii (graph_ascii.py:247). See the file header for
// the layout divergence. Self-loop edges and edges referencing unknown nodes
// are skipped; an empty graph renders an empty string.
func (g Graph) DrawASCII() string {
	g = g.normalized()
	if len(g.Nodes) == 0 {
		return ""
	}

	nodes := make(map[string]GraphNode, len(g.Nodes))
	for _, node := range g.Nodes {
		nodes[node.ID] = node
	}
	edges := make([]GraphEdge, 0, len(g.Edges))
	for _, edge := range g.Edges {
		if edge.Source == edge.Target {
			continue
		}
		if _, ok := nodes[edge.Source]; !ok {
			continue
		}
		if _, ok := nodes[edge.Target]; !ok {
			continue
		}
		edges = append(edges, edge)
	}

	layers := asciiLayers(g.Nodes, edges)
	maxLayer := 0
	for _, layer := range layers {
		if layer > maxLayer {
			maxLayer = layer
		}
	}

	// Group nodes into layers (already sorted by ID via normalized).
	byLayer := make([][]GraphNode, maxLayer+1)
	for _, node := range g.Nodes {
		layer := layers[node.ID]
		byLayer[layer] = append(byLayer[layer], node)
	}

	// Count edges crossing each cut (between layer k and k+1); every crossing
	// edge gets its own channel row in that gap.
	crossing := make([][]GraphEdge, maxLayer)
	for _, edge := range edges {
		from, to := layers[edge.Source], layers[edge.Target]
		if to > from {
			for k := from; k < to; k++ {
				crossing[k] = append(crossing[k], edge)
			}
		} else {
			// Back edges (cycles) route through the gap below the source.
			k := from
			if k >= maxLayer {
				k = maxLayer - 1
			}
			if k >= 0 {
				crossing[k] = append(crossing[k], edge)
			}
		}
	}
	gapRows := make([]int, maxLayer)
	for k := range gapRows {
		gapRows[k] = max(1, len(crossing[k]))
	}

	// Y positions: each layer occupies asciiBoxHeight rows plus the gap.
	layerY := make([]int, maxLayer+1)
	for k := 1; k <= maxLayer; k++ {
		layerY[k] = layerY[k-1] + asciiBoxHeight + gapRows[k-1]
	}

	// X positions: boxes laid out left to right within each layer.
	type box struct {
		x, y, w int
		cx      int
		node    GraphNode
	}
	boxes := map[string]*box{}
	maxRight := 0
	for layer, members := range byLayer {
		cursor := 0
		for _, node := range members {
			text := " " + node.Name + " "
			w := utf8.RuneCountInString(text) + 2
			b := &box{x: cursor, y: layerY[layer], w: w, cx: cursor + w/2, node: node}
			boxes[node.ID] = b
			cursor += w + asciiBoxGap
		}
		if cursor-asciiBoxGap > maxRight {
			maxRight = cursor - asciiBoxGap
		}
	}

	// Channel row per crossing edge in each cut, in (sorted) edge order.
	channelRow := make(map[GraphEdge]map[int]int)
	for k, group := range crossing {
		for i, edge := range group {
			if channelRow[edge] == nil {
				channelRow[edge] = map[int]int{}
			}
			channelRow[edge][k] = layerY[k] + asciiBoxHeight + i
		}
	}

	// Multi-layer and back edges detour through per-edge right-margin columns.
	marginCol := map[GraphEdge]int{}
	margin := maxRight + 2
	for _, edge := range edges {
		from, to := layers[edge.Source], layers[edge.Target]
		if to != from+1 {
			marginCol[edge] = margin
			margin += 2
		}
	}

	canvas := newASCIICanvas()
	// Draw edges first so boxes overwrite any overlap (graph_ascii.py:327).
	for _, edge := range edges {
		char := '*'
		if edge.Conditional {
			char = '.'
		}
		rows := channelRow[edge]
		if rows == nil {
			continue // no gap to route through (e.g. single-layer cycle)
		}
		source, target := boxes[edge.Source], boxes[edge.Target]
		from, to := layers[edge.Source], layers[edge.Target]
		sourceBottom := source.y + asciiBoxHeight - 1
		targetTop := target.y
		if to == from+1 {
			// Adjacent layers: down to the edge's channel row, across, down.
			row := rows[from]
			canvas.line(source.cx, sourceBottom, source.cx, row, char)
			canvas.line(source.cx, row, target.cx, row, char)
			canvas.line(target.cx, row, target.cx, targetTop, char)
			continue
		}
		col := marginCol[edge]
		if to > from {
			// Down to the first cut's channel, out to the margin, down to the
			// last cut's channel, back across to the target center, and down.
			firstRow := rows[from]
			lastRow := rows[to-1]
			canvas.line(source.cx, sourceBottom, source.cx, firstRow, char)
			canvas.line(source.cx, firstRow, col, firstRow, char)
			canvas.line(col, firstRow, col, lastRow, char)
			canvas.line(col, lastRow, target.cx, lastRow, char)
			canvas.line(target.cx, lastRow, target.cx, targetTop, char)
			continue
		}
		// Back edge: out to the margin below the source, then into the
		// target's bottom edge (best-effort; cycles are rare).
		row := rows[min(from, maxLayer-1)]
		targetBottom := target.y + asciiBoxHeight - 1
		canvas.line(source.cx, sourceBottom, source.cx, row, char)
		canvas.line(source.cx, row, col, row, char)
		canvas.line(col, row, target.cx, row, char)
		canvas.line(target.cx, row, target.cx, targetBottom, char)
	}
	for _, b := range boxes {
		text := " " + b.node.Name + " "
		canvas.box(b.x, b.y, b.w, asciiBoxHeight)
		canvas.text(b.x+1, b.y+1, text)
	}
	return canvas.draw()
}

// asciiLayers assigns every node a layer via longest-path layering over the
// DAG; cycles are broken by ignoring back edges.
func asciiLayers(nodes []GraphNode, edges []GraphEdge) map[string]int {
	incoming := map[string][]string{}
	for _, edge := range edges {
		incoming[edge.Target] = append(incoming[edge.Target], edge.Source)
	}
	layers := make(map[string]int, len(nodes))
	visiting := map[string]bool{}
	var compute func(id string) int
	compute = func(id string) int {
		if layer, ok := layers[id]; ok {
			return layer
		}
		if visiting[id] {
			return 0 // cycle: treat the back edge as layer-neutral
		}
		visiting[id] = true
		layer := 0
		for _, source := range incoming[id] {
			if candidate := compute(source) + 1; candidate > layer {
				layer = candidate
			}
		}
		visiting[id] = false
		layers[id] = layer
		return layer
	}
	for _, node := range nodes {
		compute(node.ID)
	}
	return layers
}

// asciiCanvas is a sparse character canvas with the point/line/box/text
// primitives of Python's AsciiCanvas (graph_ascii.py:57-190).
type asciiCanvas struct {
	cells       map[[2]int]rune
	maxX, maxY  int
	initialized bool
}

func newASCIICanvas() *asciiCanvas {
	return &asciiCanvas{cells: map[[2]int]rune{}}
}

func (c *asciiCanvas) point(x, y int, char rune) {
	c.cells[[2]int{x, y}] = char
	if !c.initialized || x > c.maxX {
		c.maxX = x
	}
	if !c.initialized || y > c.maxY {
		c.maxY = y
	}
	c.initialized = true
}

// line draws a straight line with Python's AsciiCanvas.line algorithm.
func (c *asciiCanvas) line(x0, y0, x1, y1 int, char rune) {
	if x0 > x1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
	}
	dx, dy := x1-x0, y1-y0
	switch {
	case dx == 0 && dy == 0:
		c.point(x0, y0, char)
	case absInt(dx) >= absInt(dy):
		for x := x0; x <= x1; x++ {
			y := y0
			if dx != 0 {
				y = y0 + int(math.Round(float64(x-x0)*float64(dy)/float64(dx)))
			}
			c.point(x, y, char)
		}
	case y0 < y1:
		for y := y0; y <= y1; y++ {
			x := x0
			if dy != 0 {
				x = x0 + int(math.Round(float64(y-y0)*float64(dx)/float64(dy)))
			}
			c.point(x, y, char)
		}
	default:
		for y := y1; y <= y0; y++ {
			x := x1
			if dy != 0 {
				x = x1 + int(math.Round(float64(y-y1)*float64(dx)/float64(dy)))
			}
			c.point(x, y, char)
		}
	}
}

func (c *asciiCanvas) text(x, y int, text string) {
	for i, char := range []rune(text) {
		c.point(x+i, y, char)
	}
}

// box draws a rectangle with +-| corners and edges, Python's
// AsciiCanvas.box semantics (width/height include the border).
func (c *asciiCanvas) box(x0, y0, width, height int) {
	width--
	height--
	for x := x0; x < x0+width; x++ {
		c.point(x, y0, '-')
		c.point(x, y0+height, '-')
	}
	for y := y0; y < y0+height; y++ {
		c.point(x0, y, '|')
		c.point(x0+width, y, '|')
	}
	c.point(x0, y0, '+')
	c.point(x0+width, y0, '+')
	c.point(x0, y0+height, '+')
	c.point(x0+width, y0+height, '+')
}

// draw renders the canvas, trimming trailing spaces on each line.
func (c *asciiCanvas) draw() string {
	if !c.initialized {
		return ""
	}
	lines := make([]string, 0, c.maxY+1)
	for y := 0; y <= c.maxY; y++ {
		row := make([]rune, c.maxX+1)
		for x := range row {
			if char, ok := c.cells[[2]int{x, y}]; ok {
				row[x] = char
			} else {
				row[x] = ' '
			}
		}
		lines = append(lines, strings.TrimRight(string(row), " "))
	}
	return strings.Join(lines, "\n")
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
