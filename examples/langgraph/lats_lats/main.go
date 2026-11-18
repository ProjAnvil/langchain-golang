// Command lats_lats is the Go port of langgraph/examples/lats/lats.ipynb.
//
// LATS (Language Agent Tree Search) grows a tree of candidate answers: each
// node carries a reflection with a 0-10 score, UCT balances exploration and
// exploitation when selecting a leaf, and scores are backpropagated to the
// root. The graph is `start -> expand -> expand -> ...` with a conditional
// edge that stops when a solution is found or the tree grows too deep.
//
// The Go port keeps the tree data structure and the graph loop exactly, but
// the LLM reflection/expansion calls are deterministic offline stubs: scores
// improve monotonically and one candidate is marked solved at depth three.
package main

import (
	"context"
	"fmt"
	"log"
	"math"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// reflection is the Go form of the notebook's Reflection pydantic model.
type reflection struct {
	Text          string
	Score         int
	FoundSolution bool
}

func (r reflection) normalizedScore() float64 { return float64(r.Score) / 10.0 }

// treeNode is the notebook's Node: a search-tree node with UCT bookkeeping.
type treeNode struct {
	messages   []string
	reflection reflection
	parent     *treeNode
	children   []*treeNode
	value      float64
	visits     int
	depth      int
	isSolved   bool
}

func newTreeNode(messages []string, refl reflection, parent *treeNode) *treeNode {
	n := &treeNode{messages: messages, reflection: refl, parent: parent}
	if parent != nil {
		n.depth = parent.depth + 1
	} else {
		n.depth = 1
	}
	n.isSolved = refl.FoundSolution
	if n.isSolved {
		n.markTreeAsSolved()
	}
	n.backpropagate(refl.normalizedScore())
	if parent != nil {
		parent.children = append(parent.children, n)
	}
	return n
}

// uct returns the upper confidence bound used to select among siblings.
func (n *treeNode) uct() float64 {
	if n.parent == nil {
		return 0
	}
	if n.visits == 0 {
		return n.value
	}
	average := n.value / float64(n.visits)
	exploration := math.Sqrt(math.Log(float64(n.parent.visits)) / float64(n.visits))
	return average + exploration
}

// backpropagate updates this node and all ancestors with a reward.
func (n *treeNode) backpropagate(reward float64) {
	for node := n; node != nil; node = node.parent {
		node.visits++
		node.value = (node.value*float64(node.visits-1) + reward) / float64(node.visits)
	}
}

func (n *treeNode) markTreeAsSolved() {
	for p := n.parent; p != nil; p = p.parent {
		p.isSolved = true
	}
}

// height is the depth of the deepest leaf below this node.
func (n *treeNode) height() int {
	if len(n.children) == 0 {
		return 1
	}
	maxChild := 0
	for _, c := range n.children {
		if h := c.height(); h > maxChild {
			maxChild = h
		}
	}
	return 1 + maxChild
}

// selectLeaf walks from the root to a leaf choosing the highest-UCT child.
func selectLeaf(root *treeNode) *treeNode {
	node := root
	for len(node.children) > 0 {
		best := node.children[0]
		for _, c := range node.children[1:] {
			if c.uct() > best.uct() {
				best = c
			}
		}
		node = best
	}
	return node
}

// bestSolution returns the highest-valued solved terminal node, or the
// highest-valued node when none is solved.
func bestSolution(root *treeNode) *treeNode {
	best := root
	var walk func(*treeNode)
	walk = func(n *treeNode) {
		for _, c := range n.children {
			score := c.value
			if c.isSolved && len(c.children) == 0 {
				score += 1
			}
			bestScore := best.value
			if best.isSolved && len(best.children) == 0 {
				bestScore += 1
			}
			if score > bestScore {
				best = c
			}
			walk(c)
		}
	}
	walk(root)
	return best
}

// generateInitialResponse stands in for the initial-answer chain plus the
// first reflection: it creates the tree root.
func generateInitialResponse(_ runtime.Runtime, state map[string]any) (any, error) {
	question, _ := state["input"].(string)
	refl := reflection{Text: "initial candidate is incomplete", Score: 4}
	root := newTreeNode([]string{"Initial answer to: " + question}, refl, nil)
	fmt.Printf("start: created root (score=%d, height=%d)\n", root.reflection.Score, root.height())
	return map[string]any{"root": root}, nil
}

// expand stands in for the notebook's expansion chain: it selects the best
// leaf with UCT and grows three deterministically better children, one of
// which is marked solved once depth three is reached.
func expand(_ runtime.Runtime, state map[string]any) (any, error) {
	root, _ := state["root"].(*treeNode)
	if root == nil {
		return nil, fmt.Errorf("expand: missing root")
	}
	leaf := selectLeaf(root)
	fmt.Printf("expand: selected leaf at depth %d (visits=%d, value=%.2f)\n",
		leaf.depth, leaf.visits, leaf.value)
	for i := 0; i < 3; i++ {
		score := leaf.reflection.Score + 2 + i
		if score > 10 {
			score = 10
		}
		found := leaf.depth+1 >= 3 && i == 2
		refl := reflection{
			Text:          fmt.Sprintf("candidate %d critique at depth %d", i+1, leaf.depth+1),
			Score:         score,
			FoundSolution: found,
		}
		newTreeNode([]string{fmt.Sprintf("Candidate answer %d (depth %d, score %d)", i+1, leaf.depth+1, score)}, refl, leaf)
	}
	fmt.Printf("expand: tree height=%d solved=%v\n", root.height(), root.isSolved)
	return map[string]any{"root": root}, nil
}

// shouldLoop mirrors the notebook's should_loop router.
func shouldLoop(_ runtime.Runtime, state map[string]any) ([]any, error) {
	root, _ := state["root"].(*treeNode)
	if root == nil {
		return nil, fmt.Errorf("should_loop: missing root")
	}
	if root.isSolved {
		return graphpkg.To(types.END), nil
	}
	if root.height() > 5 {
		return graphpkg.To(types.END), nil
	}
	return graphpkg.To("expand"), nil
}

func main() {
	ctx := context.Background()

	g := graphpkg.NewStateGraph()
	g.AddNode("start", generateInitialResponse)
	g.AddNode("expand", expand)
	g.AddEdge(types.START, "start")
	g.AddConditionalEdges("start", shouldLoop)
	g.AddConditionalEdges("expand", shouldLoop)
	cg, err := g.Compile(graphpkg.WithRecursionLimit(50))
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	question := "Generate a table of the top 5 most common birds."
	result, err := cg.Invoke(ctx, map[string]any{"input": question})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	root, _ := result.Values["root"].(*treeNode)
	if root == nil {
		log.Fatal("no root in final state")
	}
	best := bestSolution(root)
	fmt.Println("--- best solution trajectory ---")
	for _, m := range best.messages {
		fmt.Println(" ", m)
	}
	fmt.Printf("root value=%.2f visits=%d, tree height=%d, solved=%v\n",
		root.value, root.visits, root.height(), root.isSolved)
}
