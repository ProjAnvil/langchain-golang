// Command streaming-subgraphs is the Go port of
// langgraph/examples/streaming-subgraphs.ipynb. It streams a parent graph that
// embeds a compiled subgraph in "updates" mode, first without subgraph chunks
// (Subgraphs false) and then with them (Subgraphs true), showing the namespace
// Go attaches to nested chunks.
package main

import (
	"context"
	"fmt"
	"log"

	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// buildSubgraph mirrors the notebook's SubgraphState graph: node 1 writes
// "bar", node 2 concatenates the shared "foo" key with it.
func buildSubgraph() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddNode("subgraph_node_1", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"bar": "bar"}, nil
	})
	g.AddNode("subgraph_node_2", func(_ runtime.Runtime, state map[string]any) (any, error) {
		foo, _ := state["foo"].(string)
		bar, _ := state["bar"].(string)
		return map[string]any{"foo": foo + bar}, nil
	})
	g.AddEdge(types.START, "subgraph_node_1")
	g.AddEdge("subgraph_node_1", "subgraph_node_2")
	g.AddEdge("subgraph_node_2", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile subgraph: %v", err)
	}
	return cg
}

func buildParent() *graphpkg.CompiledGraph {
	g := graphpkg.NewStateGraph()
	g.AddNode("node_1", func(_ runtime.Runtime, state map[string]any) (any, error) {
		foo, _ := state["foo"].(string)
		return map[string]any{"foo": "hi! " + foo}, nil
	})
	g.AddSubgraph("node_2", buildSubgraph())
	g.AddEdge(types.START, "node_1")
	g.AddEdge("node_1", "node_2")
	g.AddEdge("node_2", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile parent: %v", err)
	}
	return cg
}

func stream(ctx context.Context, cg *graphpkg.CompiledGraph, subgraphs bool) {
	opts := graphpkg.StreamOptions{Modes: []graphpkg.StreamMode{graphpkg.StreamUpdates}}
	if subgraphs {
		fmt.Println("--- stream_mode='updates', subgraphs=True ---")
		opts.Subgraphs = true
	} else {
		fmt.Println("--- stream_mode='updates' ---")
	}
	for chunk, err := range cg.Stream(ctx, map[string]any{"foo": "foo"}, opts) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		if subgraphs {
			// Python returns (namespace, chunk) tuples; Go carries the
			// namespace explicitly on StreamChunk.
			fmt.Printf("namespace=%q chunk=%v\n", chunk.Namespace, chunk.Payload)
		} else {
			fmt.Printf("%v\n", chunk.Payload)
		}
	}
}

func main() {
	ctx := context.Background()
	cg := buildParent()
	stream(ctx, cg, false)
	stream(ctx, cg, true)
}
