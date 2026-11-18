// Command configuration is the Go port of langgraph/examples/configuration.ipynb.
// It demonstrates reading per-invocation configuration (configurable.model and
// configurable.system_message) from runtime.Runtime inside a graph node.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func main() {
	ctx := context.Background()

	// modelReplies stands in for the notebook's models dict. Each stub returns
	// a deterministic reply that identifies the selected model.
	modelReplies := map[string]func(string) string{
		"anthropic": func(text string) string { return "[anthropic] " + text },
		"openai":    func(text string) string { return "[openai] " + text },
	}

	// callModel reads the "configurable" bag from the run context, exactly
	// where this SDK surfaces Python's config["configurable"].
	callModel := func(rt runtime.Runtime, state map[string]any) (any, error) {
		modelName := "anthropic"
		if v, ok := runtime.ValueFromRuntime(rt, "model"); ok {
			if name, ok := v.(string); ok && name != "" {
				modelName = name
			}
		}
		reply, ok := modelReplies[modelName]
		if !ok {
			return nil, fmt.Errorf("unknown model %q", modelName)
		}

		msgs, _ := state["messages"].([]messages.Message)
		if system, ok := runtime.ValueFromRuntime(rt, "system_message"); ok {
			if text, ok := system.(string); ok && text != "" {
				msgs = append([]messages.Message{messages.System(text)}, msgs...)
				fmt.Printf("  [config] prepended system message: %q\n", text)
			}
		}

		last := "..."
		if len(msgs) > 0 {
			last = messages.Text(msgs[len(msgs)-1])
		}
		fmt.Printf("  [config] model=%s (visible messages: %d)\n", modelName, len(msgs))
		return map[string]any{"messages": []messages.Message{messages.AI(reply(last))}}, nil
	}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", callModel)
	g.AddEdge(types.START, "model")
	g.AddEdge("model", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	input := func() map[string]any {
		return map[string]any{"messages": []messages.Message{messages.Human("hi")}}
	}

	// 1. No configuration: default model.
	fmt.Println("=== default config ===")
	res, err := cg.Invoke(ctx, input())
	if err != nil {
		log.Fatalf("invoke default: %v", err)
	}
	printReply(res)

	// 2. Select the openai model through configurable.model.
	fmt.Println("=== configurable: model=openai ===")
	cfgCtx := runtime.ContextWithValues(ctx, map[string]any{"model": "openai"})
	res, err = cg.Invoke(cfgCtx, input())
	if err != nil {
		log.Fatalf("invoke openai: %v", err)
	}
	printReply(res)

	// 3. Add a system message through configurable.system_message.
	fmt.Println("=== configurable: model=anthropic, system_message=respond in italian ===")
	cfgCtx = runtime.ContextWithValues(ctx, map[string]any{
		"model":          "anthropic",
		"system_message": "respond in italian",
	})
	res, err = cg.Invoke(cfgCtx, input())
	if err != nil {
		log.Fatalf("invoke system message: %v", err)
	}
	printReply(res)
}

func printReply(res graphpkg.Result) {
	msgs, _ := res.Values["messages"].([]messages.Message)
	if len(msgs) == 0 {
		fmt.Println("  final: (no messages)")
		return
	}
	fmt.Printf("  final: %s -> %s\n", msgs[len(msgs)-1].Role, messages.Text(msgs[len(msgs)-1]))
}
