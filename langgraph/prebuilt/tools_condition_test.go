package prebuilt

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// route runs the router directly, mirroring Python's direct tools_condition(state) calls.
func route(t *testing.T, router graph.ConditionalEdge, state map[string]any) []any {
	t.Helper()
	out, err := router(runtime.NewRuntime(context.Background()), state)
	if err != nil {
		t.Fatalf("router error = %v", err)
	}
	return out
}

// Mirrors Python tools_condition returning "tools" when the last AIMessage has
// tool calls (tool_node.py:1657-1658).
func TestToolsConditionRoutesToTools(t *testing.T) {
	state := map[string]any{"messages": []messages.Message{
		messages.Human("hi"),
		aiWithCalls(messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{"x": "hi"}}),
	}}
	got := route(t, ToolsCondition(), state)
	if len(got) != 1 || got[0] != "tools" {
		t.Fatalf("route = %v, want [tools]", got)
	}
}

// Mirrors Python tools_condition returning "__end__" when the last message has
// no tool calls (tool_node.py:1659).
func TestToolsConditionRoutesToEnd(t *testing.T) {
	state := map[string]any{"messages": []messages.Message{
		messages.Human("hi"),
		messages.AI("done"),
	}}
	got := route(t, ToolsCondition(), state)
	if len(got) != 1 || got[0] != types.END {
		t.Fatalf("route = %v, want [%s]", got, types.END)
	}
}

// Python inspects state[-1] only: an AI message with tool calls followed by a
// ToolMessage routes to "__end__". (This is why ToolsCondition does not reuse
// tools.PendingToolCalls, which scans backwards for any AI message.)
func TestToolsConditionLastMessageNotAIRoutesToEnd(t *testing.T) {
	toolMsg := messages.Tool("call-1", "echo:hi")
	toolMsg.Name = "echo"
	state := map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{"x": "hi"}}),
		toolMsg,
	}}
	got := route(t, ToolsCondition(), state)
	if len(got) != 1 || got[0] != types.END {
		t.Fatalf("route = %v, want [%s]", got, types.END)
	}
}

// Mirrors the custom messages_key usage in
// langgraph/libs/prebuilt/tests/test_tool_node.py:1545-1551
// (partial(tools_condition, messages_key="subgraph_messages")).
func TestToolsConditionCustomMessagesKey(t *testing.T) {
	state := map[string]any{"subgraph_messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{"x": "hi"}}),
	}}
	got := route(t, ToolsCondition("subgraph_messages"), state)
	if len(got) != 1 || got[0] != "tools" {
		t.Fatalf("route = %v, want [tools]", got)
	}
}

// Python raises ValueError("No messages found in input state to tool_edge")
// (tool_node.py:1655) for a missing key or an empty list; the Go error text is
// adapted (lowercased, names tools_condition instead of tool_edge, and adds the
// offending key). Erroring on a wrong-typed messages value is a deliberate Go
// divergence: Python would return __end__ for it.
func TestToolsConditionNoMessagesErrors(t *testing.T) {
	cases := map[string]map[string]any{
		"missing key": {},
		"wrong type":  {"messages": "not a slice"},
		"empty list":  {"messages": []messages.Message{}},
	}
	for name, state := range cases {
		if _, err := ToolsCondition()(runtime.NewRuntime(context.Background()), state); err == nil ||
			!strings.Contains(err.Error(), "no messages found in input state to tools_condition") {
			t.Errorf("%s: error = %v, want a 'no messages found in input state to tools_condition' error", name, err)
		}
	}
}

func TestToolsConditionEmptyKeyPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("ToolsCondition(\"\") did not panic")
		}
	}()
	ToolsCondition("")
}

// End-to-end: model -> conditional(ToolsCondition) -> tools/END, the canonical
// ReAct wiring from Python's tools_condition docstring (tool_node.py:1611-1633).
func TestToolsConditionEndToEnd(t *testing.T) {
	echo := funcTool(t, "echo", func(_ context.Context, args map[string]any) (tools.Result, error) {
		return tools.Result{Content: "echo:" + args["x"].(string)}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{"x": "hi"}},
	)))
	g.AddNode("tools", ToolNode(toolNode))
	g.AddEdge(types.START, "model")
	g.AddConditionalEdges("model", ToolsCondition())
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := compiled.Invoke(context.Background(), map[string]any{
		"messages": []messages.Message{messages.Human("run the tool")},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 3 || msgs[2].Content != "echo:hi" || msgs[2].ToolCallID != "call-1" {
		t.Fatalf("messages = %+v, want human + ai + tool result echo:hi", msgs)
	}
}
