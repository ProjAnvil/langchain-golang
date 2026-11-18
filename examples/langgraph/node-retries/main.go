// Command node-retries is the Go port of langgraph/examples/node-retries.ipynb.
// It shows per-node RetryPolicy with a custom retry predicate and a larger
// attempt budget on a second node.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	graphpkg "github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// operationalError stands in for sqlite3.OperationalError from the notebook.
type operationalError struct{ msg string }

func (e *operationalError) Error() string { return e.msg }

func main() {
	ctx := context.Background()

	var queryAttempts, modelAttempts int

	// queryDatabase fails twice with a simulated operational error, then
	// succeeds. Only that error type is retried.
	queryDatabase := func(_ runtime.Runtime, _ map[string]any) (any, error) {
		queryAttempts++
		if queryAttempts <= 2 {
			return nil, &operationalError{msg: fmt.Sprintf("attempt %d: database is busy", queryAttempts)}
		}
		return map[string]any{
			"messages": []messages.Message{messages.AI("query result: 10 rows from Artist")},
		}, nil
	}

	// callModel succeeds on its first try but carries a 5-attempt budget.
	callModel := func(_ runtime.Runtime, _ map[string]any) (any, error) {
		modelAttempts++
		return map[string]any{
			"messages": []messages.Message{messages.AI("model reply")},
		}, nil
	}

	g := graphpkg.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNodeWithPolicies("query_database", queryDatabase, graphpkg.NodePolicies{
		Retry: &graphpkg.RetryPolicy{
			MaxAttempts:     4,
			InitialInterval: 10 * time.Millisecond,
			BackoffFactor:   2,
			NoJitter:        true,
			RetryOn: func(err error) bool {
				var opErr *operationalError
				return errors.As(err, &opErr)
			},
		},
	})
	g.AddNodeWithPolicies("model", callModel, graphpkg.NodePolicies{
		Retry: &graphpkg.RetryPolicy{
			MaxAttempts:     5,
			InitialInterval: 10 * time.Millisecond,
			NoJitter:        true,
		},
	})
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "query_database")
	g.AddEdge("query_database", types.END)
	cg, err := g.Compile()
	if err != nil {
		log.Fatalf("compile: %v", err)
	}

	res, err := cg.Invoke(ctx, map[string]any{
		"messages": []messages.Message{messages.Human("list artists")},
	})
	if err != nil {
		log.Fatalf("invoke: %v", err)
	}
	fmt.Printf("model attempts: %d (budget 5)\n", modelAttempts)
	fmt.Printf("query_database attempts: %d (2 transient failures retried)\n", queryAttempts)
	msgs, _ := res.Values["messages"].([]messages.Message)
	for i, m := range msgs {
		fmt.Printf("  [%d] %-5s %s\n", i, m.Role, messages.Text(m))
	}
}
