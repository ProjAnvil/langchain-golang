package agents

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/chatmodels"
)

// ExampleCreateAgent_minimal builds a one-tool agent on a FakeChatModel,
// invokes it, and prints the final assistant message. The model is declared
// tool-calling-capable so CreateAgent can bind the "echo" tool, and in this
// minimal run it answers directly (no tool call), so the loop completes in a
// single model step. Everything is offline and deterministic. (A full
// model->tool->model loop is exercised in the test suite's
// TestCreateAgentToolLoop using a stateful fake; FakeChatModel returns a fresh
// copy from BindTools, so it cannot itself advance a multi-response sequence
// across the per-iteration re-bind CreateAgent performs.)
func ExampleCreateAgent_minimal() {
	model := language.NewFakeChatModel(
		language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
		language.WithResponses(messages.AI("hello from the agent")),
	)

	echo, err := coretools.NewSimple("echo", "echo its input",
		func(ctx context.Context, input string) (coretools.Result, error) {
			return coretools.Result{Content: "echo:" + input}, nil
		})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	agent, err := CreateAgent(model, []coretools.Tool{echo})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// out is the full message history: [Human, AI]. Print the assistant reply.
	fmt.Printf("%d messages; reply=%q\n", len(out), messages.Text(out[len(out)-1]))
	// Output:
	// 2 messages; reply="hello from the agent"
}

// ExampleCreateAgent_modelString resolves the model from a "provider:model"
// string instead of passing a concrete ChatModel positionally. It registers a
// fake factory under the provider name "example-fake" (in real code you would
// import a partner package whose init registers a real factory, e.g.
// partners/openai for "openai"); CreateAgent is then called with a nil
// positional model and WithAgentModel supplies the model instead.
func ExampleCreateAgent_modelString() {
	chatmodels.RegisterProvider("example-fake",
		func(model string, opts map[string]any) (language.ChatModel, error) {
			return language.NewFakeChatModel(
				language.WithResponses(messages.AI("resolved model: " + model)),
			), nil
		})

	agent, err := CreateAgent(nil, nil, WithAgentModel("example-fake:gpt-x"))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("ping")})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(messages.Text(out[len(out)-1]))
	// Output:
	// resolved model: gpt-x
}
