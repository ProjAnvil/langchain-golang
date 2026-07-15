package agents

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
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

// ExampleCreateAgent_subagent shows the canonical subagent (agent-as-tool)
// pattern: a supervisor delegates to a named inner agent via a hand-rolled
// tool whose body calls the inner agent's InvokeWithState. There is no
// AgentAsTool helper — this user-authored tool mirrors Python's
// langchain.agents pattern exactly.
func ExampleCreateAgent_subagent() {
	ctx := context.Background()

	// Inner named agent. The name is what makes the nested run distinguishable:
	// inside it, NameFromContext returns "weather_agent", not "supervisor".
	inner := language.NewFakeChatModel(
		language.WithCapabilities(language.ChatModelCapabilities{ToolCalling: true}),
		language.WithResponses(messages.AI("sunny")),
	)
	weather, err := CreateAgent(inner, nil, WithAgentName("weather_agent"))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// The subagent tool: delegate, run the inner agent, return its final text.
	// Go equivalent of Python's:
	//   @tool
	//   def call_weather(city):
	//       return weather.invoke({"messages":[HumanMessage(...)]})["messages"][-1].text
	callWeather, err := coretools.NewFunc(
		"call_weather", "Call the weather agent.",
		schema.Object(map[string]schema.Schema{"city": schema.String("city")}, "city"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			city, _ := input["city"].(string)
			state, err := weather.InvokeWithState(ctx, []messages.Message{messages.Human("weather in " + city)})
			if err != nil {
				return coretools.Result{}, err
			}
			msgs, _ := state["messages"].([]messages.Message)
			for i := len(msgs) - 1; i >= 0; i-- {
				if msgs[i].Role == messages.RoleAI {
					return coretools.Result{Content: messages.Text(msgs[i])}, nil
				}
			}
			return coretools.Result{}, fmt.Errorf("weather agent produced no output")
		},
	)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	// Supervisor: one tool call delegating to the weather agent, then done.
	// Uses sequenceModel (not FakeChatModel) here because CreateAgent re-binds
	// tools on every model iteration and FakeChatModel.BindTools returns a fresh
	// copy whose response index is copied from the original at its current (0)
	// value — the copy's advances never write back, so a FakeChatModel supervisor
	// would never advance past its first (tool-call) response and would loop to
	// the recursion limit. sequenceModel.BindTools returns the same receiver, so
	// its response cursor advances across iterations. (See the doc comment on
	// ExampleCreateAgent_minimal for the same FakeChatModel limitation. The inner
	// agent above is fine as a FakeChatModel: it has no tools, so it is never
	// re-bound, and a single Invoke on the original advances its index normally.)
	supModel := &sequenceModel{responses: []messages.Message{
		{Role: messages.RoleAI, ToolCalls: []messages.ToolCall{
			{ID: "c1", Name: "call_weather", Args: map[string]any{"city": "SF"}},
		}},
		messages.AI("The weather is sunny"),
	}}
	supervisor, err := CreateAgent(supModel, []coretools.Tool{callWeather}, WithAgentName("supervisor"))
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	out, err := supervisor.Invoke(ctx, []messages.Message{messages.Human("weather?")})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	for _, m := range out {
		if m.Role == messages.RoleTool {
			fmt.Println("subagent replied:", messages.Text(m))
		}
	}
	// Output:
	// subagent replied: sunny
}
