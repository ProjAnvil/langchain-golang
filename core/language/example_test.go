package language

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

// ExampleInvokeStructured calls InvokeStructured with a FakeChatModel (which
// does NOT implement StructuredCaller) and a schema, exercising the fallback
// JSON-decode-and-validate path. The model's response text must be JSON
// containing every key the schema marks required. A partner model that
// implements StructuredCaller (e.g. partners/openai.ChatModel) would take the
// native path instead.
func ExampleInvokeStructured() {
	// FakeChatModel returns its canned response for every Invoke call.
	model := NewFakeChatModel(
		WithResponses(messages.AI(`{"city": "Go Valley", "population": 7}`)),
	)

	// A schema requiring both "city" and "population".
	sch := schema.Object(map[string]schema.Schema{
		"city":       schema.String(""),
		"population": schema.Integer(""),
	}, "city", "population")

	msg, err := InvokeStructured(context.Background(), model, nil, sch)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(messages.Text(msg))
	// Output:
	// {"city": "Go Valley", "population": 7}
}
