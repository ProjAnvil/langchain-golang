// Command state-model is the Go port of langgraph/examples/state-model.ipynb.
// The Go SDK has no Pydantic schema, so validation and coercion are expressed
// as explicit helpers around plain map[string]any state.
package main

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/messages"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// validateStringA mirrors Pydantic requiring field `a` to be a string.
func validateStringA(state map[string]any) error {
	v, ok := state["a"]
	if !ok {
		return fmt.Errorf("validation error: field 'a' is required")
	}
	if _, ok := v.(string); !ok {
		return fmt.Errorf("validation error: 'a' must be a string, got %T (%v)", v, v)
	}
	return nil
}

func compileOrDie(g *graphpkg.StateGraph) *graphpkg.CompiledGraph {
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}
	return cg
}

func main() {
	ctx := context.Background()

	// --- 1. Input validation -------------------------------------------
	fmt.Println("=== 1. OverallState(a: str) input validation ===")
	g := graphpkg.NewStateGraph()
	g.AddNode("node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"a": "goodbye"}, nil
	})
	g.AddEdge(types.START, "node")
	g.AddEdge("node", types.END)
	cg := compileOrDie(g)

	valid := map[string]any{"a": "hello"}
	if err := validateStringA(valid); err != nil {
		fmt.Println("unexpected error:", err)
	} else {
		res, err := cg.Invoke(ctx, valid)
		if err != nil {
			log.Fatalf("invoke valid: %v", err)
		}
		fmt.Printf("  valid input -> %v\n", res.Values)
	}
	invalid := map[string]any{"a": 123}
	if err := validateStringA(invalid); err != nil {
		fmt.Printf("  invalid input -> %v\n", err)
	}

	// --- 2. Invalid node output ----------------------------------------
	fmt.Println("=== 2. bad_node sets `a` to an integer ===")
	g = graphpkg.NewStateGraph()
	g.AddNode("bad_node", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"a": 123}, nil
	})
	g.AddNode("ok_node", func(_ runtime.Runtime, state map[string]any) (any, error) {
		if err := validateStringA(state); err != nil {
			return nil, err
		}
		return map[string]any{"a": "goodbye"}, nil
	})
	g.AddEdge(types.START, "bad_node")
	g.AddEdge("bad_node", "ok_node")
	g.AddEdge("ok_node", types.END)
	cg = compileOrDie(g)
	if _, err := cg.Invoke(ctx, map[string]any{"a": "hello"}); err != nil {
		fmt.Printf("  invalid node output -> %v\n", err)
	}

	// --- 3. Nested state model -----------------------------------------
	fmt.Println("=== 3. Nested model (text, count, nested.value) ===")
	type nestedModel struct{ Value string }
	type complexState struct {
		Text   string
		Count  int
		Nested nestedModel
	}
	validateNested := func(state map[string]any) error {
		if _, ok := state["text"].(string); !ok {
			return fmt.Errorf("validation error: 'text' must be a string")
		}
		if _, ok := state["count"].(float64); !ok {
			if _, ok := state["count"].(int); !ok {
				return fmt.Errorf("validation error: 'count' must be an int")
			}
		}
		nested, ok := state["nested"].(map[string]any)
		if !ok {
			return fmt.Errorf("validation error: 'nested' must be an object")
		}
		if _, ok := nested["value"].(string); !ok {
			return fmt.Errorf("validation error: 'nested.value' must be a string")
		}
		return nil
	}
	g = graphpkg.NewStateGraph()
	g.AddNode("process", func(_ runtime.Runtime, state map[string]any) (any, error) {
		if err := validateNested(state); err != nil {
			return nil, err
		}
		text, _ := state["text"].(string)
		count, _ := state["count"].(int)
		fmt.Printf("  input state type: %T, nested type: %T\n", complexState{}, nestedModel{})
		return map[string]any{"text": text + " processed", "count": count + 1}, nil
	})
	g.AddEdge(types.START, "process")
	g.AddEdge("process", types.END)
	cg = compileOrDie(g)
	res, err := cg.Invoke(ctx, map[string]any{
		"text":   "hello",
		"count":  0,
		"nested": map[string]any{"value": "test"},
	})
	if err != nil {
		log.Fatalf("invoke nested: %v", err)
	}
	fmt.Printf("  processed -> %v\n", res.Values)

	// --- 4. Coercion ----------------------------------------------------
	fmt.Println("=== 4. Coercion of string inputs (number: int, flag: bool) ===")
	coerceInt := func(v any) (int, error) {
		switch x := v.(type) {
		case int:
			return x, nil
		case float64:
			return int(x), nil
		case string:
			n, err := strconv.Atoi(strings.TrimSpace(x))
			if err != nil {
				return 0, fmt.Errorf("validation error: cannot coerce %q to int", x)
			}
			return n, nil
		default:
			return 0, fmt.Errorf("validation error: cannot coerce %T to int", v)
		}
	}
	coerceBool := func(v any) (bool, error) {
		switch x := v.(type) {
		case bool:
			return x, nil
		case string:
			b, err := strconv.ParseBool(strings.TrimSpace(x))
			if err != nil {
				return false, fmt.Errorf("validation error: cannot coerce %q to bool", x)
			}
			return b, nil
		default:
			return false, fmt.Errorf("validation error: cannot coerce %T to bool", v)
		}
	}
	g = graphpkg.NewStateGraph()
	g.AddNode("inspect", func(_ runtime.Runtime, state map[string]any) (any, error) {
		n, err := coerceInt(state["number"])
		if err != nil {
			return nil, err
		}
		b, err := coerceBool(state["flag"])
		if err != nil {
			return nil, err
		}
		fmt.Printf("  number=%d (%T), flag=%t (%T)\n", n, n, b, b)
		return nil, nil
	})
	g.AddEdge(types.START, "inspect")
	g.AddEdge("inspect", types.END)
	cg = compileOrDie(g)
	if _, err := cg.Invoke(ctx, map[string]any{"number": "42", "flag": "true"}); err != nil {
		log.Fatalf("invoke coercion: %v", err)
	}
	if _, err := cg.Invoke(ctx, map[string]any{"number": "not-a-number", "flag": "true"}); err != nil {
		fmt.Printf("  expected coercion error -> %v\n", err)
	}

	// --- 5. Message list state -----------------------------------------
	fmt.Println("=== 5. ChatState(messages, context) ===")
	g = graphpkg.NewStateGraph()
	g.AddNode("add_message", func(_ runtime.Runtime, state map[string]any) (any, error) {
		msgs, _ := state["messages"].([]messages.Message)
		return map[string]any{"messages": append(msgs, messages.AI("Hello there!"))}, nil
	})
	g.AddEdge(types.START, "add_message")
	g.AddEdge("add_message", types.END)
	cg = compileOrDie(g)
	res, err = cg.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human("Hi")},
		"context":  "Customer support chat",
	})
	if err != nil {
		log.Fatalf("invoke chat: %v", err)
	}
	msgs, _ := res.Values["messages"].([]messages.Message)
	for i, m := range msgs {
		fmt.Printf("  Message %d: %s - %s\n", i, m.Role, messages.Text(m))
	}
}
