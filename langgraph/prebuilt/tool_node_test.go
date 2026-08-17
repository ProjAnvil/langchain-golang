package prebuilt

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

func funcTool(t *testing.T, name string, fn func(context.Context, map[string]any) (tools.Result, error)) tools.Tool {
	t.Helper()
	tool, err := tools.NewFunc(name, "", schema.Object(nil), fn)
	if err != nil {
		t.Fatalf("NewFunc(%q) error = %v", name, err)
	}
	return tool
}

func aiWithCalls(calls ...messages.ToolCall) messages.Message {
	msg := messages.AI("")
	msg.ToolCalls = calls
	return msg
}

// modelNodeStub returns a node that appends msg to the "messages" key,
// standing in for a chat model.
func modelNodeStub(msg messages.Message) graph.NodeFunc {
	return func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"messages": []messages.Message{msg}}, nil
	}
}

func stateMessages(t *testing.T, values map[string]any, key string) []messages.Message {
	t.Helper()
	msgs, ok := values[key].([]messages.Message)
	if !ok {
		t.Fatalf("state[%q] = %T, want []messages.Message", key, values[key])
	}
	return msgs
}

// TestToolNodeExecutesPendingToolCalls builds the canonical
// model -> tools graph: the model stub writes an AI message with two tool
// calls, and the prebuilt ToolNode must execute both (concurrently, via
// langchain/tools.ToolNode) and append one ToolMessage per call, in call
// order, under the default "messages" key.
func TestToolNodeExecutesPendingToolCalls(t *testing.T) {
	echo := funcTool(t, "echo", func(_ context.Context, args map[string]any) (tools.Result, error) {
		return tools.Result{Content: "echo:" + args["x"].(string)}, nil
	})
	upper := funcTool(t, "upper", func(_ context.Context, args map[string]any) (tools.Result, error) {
		return tools.Result{Content: strings.ToUpper(args["x"].(string))}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo, upper})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{"x": "hi"}},
		messages.ToolCall{ID: "call-2", Name: "upper", Args: map[string]any{"x": "hey"}},
	)))
	g.AddNode("tools", ToolNode(toolNode))
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "tools")
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := compiled.Invoke(context.Background(), map[string]any{
		"messages": []messages.Message{messages.Human("run the tools")},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 4 {
		t.Fatalf("len(messages) = %d, want 4 (human, ai, 2 tool results): %+v", len(msgs), msgs)
	}
	for i, want := range []struct{ id, name, content string }{
		{"call-1", "echo", "echo:hi"},
		{"call-2", "upper", "HEY"},
	} {
		got := msgs[2+i]
		if got.Role != messages.RoleTool || got.ToolCallID != want.id || got.Name != want.name || got.Content != want.content {
			t.Errorf("messages[%d] = %+v, want tool message id=%q name=%q content=%q", 2+i, got, want.id, want.name, want.content)
		}
	}
}

// TestToolNodeWithMessagesKey verifies the WithMessagesKey variant: tool
// results land under the configured key instead of "messages".
func TestToolNodeWithMessagesKey(t *testing.T) {
	echo := funcTool(t, "echo", func(_ context.Context, args map[string]any) (tools.Result, error) {
		return tools.Result{Content: "ok"}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddReducer("chat_history", channels.MessagesReducer)
	g.AddNode("model", func(_ runtime.Runtime, _ map[string]any) (any, error) {
		return map[string]any{"chat_history": []messages.Message{aiWithCalls(
			messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{}},
		)}}, nil
	})
	g.AddNode("tools", ToolNode(toolNode, WithMessagesKey("chat_history")))
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "tools")
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := compiled.Invoke(context.Background(), map[string]any{
		"chat_history": []messages.Message{messages.Human("hi")},
	})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := stateMessages(t, res.Values, "chat_history")
	if len(msgs) != 3 {
		t.Fatalf("len(chat_history) = %d, want 3: %+v", len(msgs), msgs)
	}
	if msgs[2].Role != messages.RoleTool || msgs[2].ToolCallID != "call-1" || msgs[2].Content != "ok" {
		t.Fatalf("chat_history[2] = %+v, want echo tool result", msgs[2])
	}
}

// TestToolNodeToolErrorBecomesErrorMessage verifies the default error
// handling: a failing tool produces an error ToolMessage (handled by
// langchain/tools.DefaultHandleToolErrors) instead of failing the graph run.
func TestToolNodeToolErrorBecomesErrorMessage(t *testing.T) {
	failing := funcTool(t, "failing", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{}, errors.New("boom")
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{failing})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "failing", Args: map[string]any{}},
	)))
	g.AddNode("tools", ToolNode(toolNode))
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "tools")
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := compiled.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 2 {
		t.Fatalf("len(messages) = %d, want 2: %+v", len(msgs), msgs)
	}
	got := msgs[1]
	if got.Role != messages.RoleTool || got.ToolCallID != "call-1" {
		t.Fatalf("messages[1] = %+v, want tool message for call-1", got)
	}
	if !strings.Contains(got.Content, "boom") {
		t.Fatalf("error tool message content = %q, want it to mention the tool error", got.Content)
	}
	if got.ResponseMetadata["status"] != "error" {
		t.Fatalf("error tool message metadata = %+v, want status=error", got.ResponseMetadata)
	}
}

// commandGraph builds a model -> tools graph where tools has a static edge to
// "fallback" and a "done" node is reachable only via a tool's Command.Goto.
// It returns the compiled graph and per-node run counters.
func commandGraph(t *testing.T, toolNode *tools.ToolNode, aiMsg messages.Message) (*graph.CompiledGraph, map[string]*int) {
	t.Helper()
	runs := map[string]*int{"done": new(int), "fallback": new(int)}
	counting := func(name string) graph.NodeFunc {
		return func(_ runtime.Runtime, _ map[string]any) (any, error) {
			*runs[name]++
			return nil, nil
		}
	}
	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiMsg))
	g.AddNode("tools", ToolNode(toolNode))
	g.AddNode("done", counting("done"))
	g.AddNode("fallback", counting("fallback"))
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "tools")
	g.AddEdge("tools", "fallback") // overridden when a tool returns Command.Goto
	g.AddEdge("done", types.END)
	g.AddEdge("fallback", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	return compiled, runs
}

// TestToolNodeCommandPassthrough verifies that a tool placing a
// *types.Command in Result.Artifact has the command surfaced as the node's
// result: the graph applies the command's Update and routes via its Goto
// instead of the static edge.
func TestToolNodeCommandPassthrough(t *testing.T) {
	navigator := funcTool(t, "navigate", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{
			Content: "navigated",
			Artifact: &types.Command{
				Update: map[string]any{"route": "took-command"},
				Goto:   graph.To("done"),
			},
		}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{navigator})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	compiled, runs := commandGraph(t, toolNode, aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "navigate", Args: map[string]any{}},
	))

	res, err := compiled.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["route"] != "took-command" {
		t.Fatalf("state[route] = %v, want the command update applied", res.Values["route"])
	}
	if *runs["done"] != 1 || *runs["fallback"] != 0 {
		t.Fatalf("runs: done=%d fallback=%d, want done=1 fallback=0 (Command.Goto overrides static edge)", *runs["done"], *runs["fallback"])
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 2 || msgs[1].Content != "navigated" {
		t.Fatalf("messages = %+v, want the tool result message appended alongside the command", msgs)
	}
}

// TestToolNodeMergesMultipleCommands verifies that when several tools in one
// batch return Commands, their Update maps merge and their Goto lists
// concatenate.
func TestToolNodeMergesMultipleCommands(t *testing.T) {
	mkTool := func(name, key string, gotoNode string) tools.Tool {
		return funcTool(t, name, func(context.Context, map[string]any) (tools.Result, error) {
			return tools.Result{
				Content:  name + " done",
				Artifact: &types.Command{Update: map[string]any{key: name}, Goto: graph.To(gotoNode)},
			}, nil
		})
	}
	toolNode, err := tools.NewToolNode([]tools.Tool{mkTool("toolA", "a", "done"), mkTool("toolB", "b", "extra")})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	extraRuns := 0
	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "toolA", Args: map[string]any{}},
		messages.ToolCall{ID: "call-2", Name: "toolB", Args: map[string]any{}},
	)))
	g.AddNode("tools", ToolNode(toolNode))
	g.AddNode("done", func(_ runtime.Runtime, _ map[string]any) (any, error) { return nil, nil })
	g.AddNode("extra", func(_ runtime.Runtime, _ map[string]any) (any, error) { extraRuns++; return nil, nil })
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "tools")
	g.AddEdge("tools", types.END) // overridden by the merged command's Goto
	g.AddEdge("done", types.END)
	g.AddEdge("extra", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := compiled.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if res.Values["a"] != "toolA" || res.Values["b"] != "toolB" {
		t.Fatalf("merged update = %v, want both command updates applied", res.Values)
	}
	if extraRuns != 1 {
		t.Fatalf("extra runs = %d, want 1 (concatenated Goto lists both routed)", extraRuns)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 3 {
		t.Fatalf("len(messages) = %d, want 3 (ai + 2 tool results): %+v", len(msgs), msgs)
	}
}

// TestToolNodeCommandWithHandledErrorInBatch verifies the interplay of a
// Command-returning tool and a failing (but handled) tool in the same batch:
// the error becomes an error ToolMessage while the command still drives
// routing.
func TestToolNodeCommandWithHandledErrorInBatch(t *testing.T) {
	failing := funcTool(t, "failing", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{}, errors.New("boom")
	})
	navigator := funcTool(t, "navigate", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{
			Content:  "navigated",
			Artifact: &types.Command{Goto: graph.To("done")},
		}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{failing, navigator})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	compiled, runs := commandGraph(t, toolNode, aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "failing", Args: map[string]any{}},
		messages.ToolCall{ID: "call-2", Name: "navigate", Args: map[string]any{}},
	))

	res, err := compiled.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if *runs["done"] != 1 || *runs["fallback"] != 0 {
		t.Fatalf("runs: done=%d fallback=%d, want the command's Goto honored despite the sibling error", *runs["done"], *runs["fallback"])
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 3 {
		t.Fatalf("len(messages) = %d, want 3: %+v", len(msgs), msgs)
	}
	if msgs[1].ResponseMetadata["status"] != "error" || !strings.Contains(msgs[1].Content, "boom") {
		t.Fatalf("messages[1] = %+v, want the handled error tool message", msgs[1])
	}
	if msgs[2].Content != "navigated" {
		t.Fatalf("messages[2] = %+v, want the command tool's result message", msgs[2])
	}
}

// TestToolNodeMissingMessagesKey verifies a descriptive error when the state
// has no value under the configured messages key.
func TestToolNodeMissingMessagesKey(t *testing.T) {
	echo := funcTool(t, "echo", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{Content: "ok"}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddNode("tools", ToolNode(toolNode))
	g.AddEdge(types.START, "tools")
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = compiled.Invoke(context.Background(), nil)
	if err == nil || !strings.Contains(err.Error(), `"messages"`) {
		t.Fatalf("Invoke() error = %v, want a descriptive error naming the missing %q key", err, "messages")
	}
}

// TestToolNodeWrongTypeMessagesKey verifies a descriptive error when the
// value under the messages key is not a []messages.Message.
func TestToolNodeWrongTypeMessagesKey(t *testing.T) {
	echo := funcTool(t, "echo", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{Content: "ok"}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddNode("tools", ToolNode(toolNode))
	g.AddEdge(types.START, "tools")
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	_, err = compiled.Invoke(context.Background(), map[string]any{"messages": "not-a-message-list"})
	if err == nil || !strings.Contains(err.Error(), "[]messages.Message") {
		t.Fatalf("Invoke() error = %v, want a descriptive error naming the expected type", err)
	}
}

// TestToolNodeNoToolCalls verifies that a last AI message without tool calls
// produces a nil update (the graph state is left untouched).
func TestToolNodeNoToolCalls(t *testing.T) {
	echo := funcTool(t, "echo", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{Content: "ok"}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	node := ToolNode(toolNode)
	state := map[string]any{"messages": []messages.Message{messages.AI("no calls here")}}
	update, err := node(runtime.NewRuntime(context.Background()), state)
	if err != nil {
		t.Fatalf("ToolNode() error = %v", err)
	}
	if update != nil {
		t.Fatalf("ToolNode() update = %v, want nil when there are no tool calls", update)
	}
}

// TestToolNodeWithCheckpointerInterruptBefore verifies the adapter composes
// with checkpointing: interrupt_before pauses the run ahead of the tools
// node, and resuming executes the tools exactly once.
func TestToolNodeWithCheckpointerInterruptBefore(t *testing.T) {
	toolRuns := 0
	echo := funcTool(t, "echo", func(context.Context, map[string]any) (tools.Result, error) {
		toolRuns++
		return tools.Result{Content: "ok"}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "echo", Args: map[string]any{}},
	)))
	g.AddNode("tools", ToolNode(toolNode))
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "tools")
	g.AddEdge("tools", types.END)
	compiled, err := g.Compile(
		graph.WithCheckpointer(checkpoint.NewMemorySaver()),
		graph.WithInterruptBefore("tools"),
	)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}

	res, err := compiled.InvokeWithOptions(context.Background(), nil, graph.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	if len(res.Interrupts) == 0 {
		t.Fatal("expected the run to pause before the tools node")
	}
	if toolRuns != 0 {
		t.Fatalf("tool ran %d times before the interrupt, want 0", toolRuns)
	}

	res, err = compiled.InvokeWithOptions(context.Background(), nil, graph.Options{ThreadID: "t1"})
	if err != nil {
		t.Fatalf("resume Invoke() error = %v", err)
	}
	if len(res.Interrupts) != 0 {
		t.Fatalf("expected no interrupts after resume, got %+v", res.Interrupts)
	}
	if toolRuns != 1 {
		t.Fatalf("tool ran %d times after resume, want exactly 1", toolRuns)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 2 || msgs[1].Role != messages.RoleTool || msgs[1].Content != "ok" {
		t.Fatalf("messages after resume = %+v, want the echo tool result appended", msgs)
	}
}

// TestToolNodeNilNodePanics verifies the construction-time guard: a nil
// *tools.ToolNode is a programming error and must panic immediately.
func TestToolNodeNilNodePanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("ToolNode(nil) did not panic")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "non-nil") {
			t.Fatalf("ToolNode(nil) panic = %v, want a message about the nil node", r)
		}
	}()
	ToolNode(nil)
}

// TestToolNodeEmptyMessagesKeyPanics verifies the construction-time guard:
// an option that sets an empty messages key is a programming error and must
// panic immediately.
func TestToolNodeEmptyMessagesKeyPanics(t *testing.T) {
	echo := funcTool(t, "echo", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{Content: "ok"}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{echo})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("ToolNode(node, WithMessagesKey(\"\")) did not panic")
		}
		msg, ok := r.(string)
		if !ok || !strings.Contains(msg, "messages key") {
			t.Fatalf("panic = %v, want a message about the empty messages key", r)
		}
	}()
	ToolNode(toolNode, WithMessagesKey(""))
}

// TestToolNodeUnhandledToolErrorPropagates verifies that when the underlying
// tools.ToolNode declines to handle a tool error (HandleToolErrors returns
// handled=false), the error surfaces as the node error instead of an error
// ToolMessage.
func TestToolNodeUnhandledToolErrorPropagates(t *testing.T) {
	failing := funcTool(t, "failing", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{}, errors.New("boom")
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{failing},
		tools.WithHandleToolErrors(func(error) (string, bool) { return "", false }),
	)
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	node := ToolNode(toolNode)
	state := map[string]any{"messages": []messages.Message{aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "failing", Args: map[string]any{}},
	)}}
	update, err := node(runtime.NewRuntime(context.Background()), state)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("ToolNode() error = %v, want the unhandled tool error propagated", err)
	}
	if update != nil {
		t.Fatalf("ToolNode() update = %v, want nil on error", update)
	}
}

// TestToolNodeCommandConflictLastWins verifies the conflict semantics of the
// merged command: when several tools in one batch return commands, the last
// command in call order wins for conflicting Update keys, Graph, and Resume,
// while the Goto lists concatenate in call order.
func TestToolNodeCommandConflictLastWins(t *testing.T) {
	first := funcTool(t, "first", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{
			Content:  "first done",
			Artifact: &types.Command{Update: map[string]any{"shared": "first", "only-first": 1}, Goto: graph.To("n1")},
		}, nil
	})
	second := funcTool(t, "second", func(context.Context, map[string]any) (tools.Result, error) {
		return tools.Result{
			Content: "second done",
			Artifact: &types.Command{
				Graph:  types.ParentGraph,
				Update: map[string]any{"shared": "second"},
				Resume: "resume-value",
				Goto:   graph.To("n2"),
			},
		}, nil
	})
	toolNode, err := tools.NewToolNode([]tools.Tool{first, second})
	if err != nil {
		t.Fatalf("NewToolNode() error = %v", err)
	}

	node := ToolNode(toolNode)
	state := map[string]any{"messages": []messages.Message{aiWithCalls(
		messages.ToolCall{ID: "call-1", Name: "first", Args: map[string]any{}},
		messages.ToolCall{ID: "call-2", Name: "second", Args: map[string]any{}},
	)}}
	out, err := node(runtime.NewRuntime(context.Background()), state)
	if err != nil {
		t.Fatalf("ToolNode() error = %v", err)
	}
	cmd, ok := out.(*types.Command)
	if !ok {
		t.Fatalf("ToolNode() result = %T, want *types.Command", out)
	}
	if cmd.Graph != types.ParentGraph {
		t.Errorf("merged Graph = %q, want %q from the last command", cmd.Graph, types.ParentGraph)
	}
	if cmd.Resume != "resume-value" {
		t.Errorf("merged Resume = %v, want %q from the last command", cmd.Resume, "resume-value")
	}
	if cmd.Update["shared"] != "second" {
		t.Errorf("merged Update[shared] = %v, want the last command's value winning the conflict", cmd.Update["shared"])
	}
	if cmd.Update["only-first"] != 1 {
		t.Errorf("merged Update[only-first] = %v, want the non-conflicting key kept", cmd.Update["only-first"])
	}
	if len(cmd.Goto) != 2 || cmd.Goto[0] != "n1" || cmd.Goto[1] != "n2" {
		t.Errorf("merged Goto = %v, want both destinations concatenated in call order", cmd.Goto)
	}
	resultMsgs, ok := cmd.Update["messages"].([]messages.Message)
	if !ok || len(resultMsgs) != 2 {
		t.Fatalf("merged Update[messages] = %v, want the two tool result messages", cmd.Update["messages"])
	}
	if resultMsgs[0].Content != "first done" || resultMsgs[1].Content != "second done" {
		t.Errorf("result messages = %+v, want both tool results in call order", resultMsgs)
	}
}

// TestMergeCommandSeedFromFirstCommand verifies mergeCommand's initialization
// path: folding into a nil accumulator seeds Graph and Resume from the first
// command and starts a fresh Update map that does not alias the input.
func TestMergeCommandSeedFromFirstCommand(t *testing.T) {
	cmd := &types.Command{
		Graph:  types.ParentGraph,
		Update: map[string]any{"k": "v"},
		Resume: 42,
		Goto:   graph.To("n1"),
	}
	merged := mergeCommand(nil, cmd)
	if merged.Graph != types.ParentGraph || merged.Resume != 42 {
		t.Errorf("merged = %+v, want Graph and Resume seeded from the first command", merged)
	}
	if merged.Update["k"] != "v" {
		t.Errorf("merged.Update = %v, want the first command's update merged in", merged.Update)
	}
	if len(merged.Goto) != 1 || merged.Goto[0] != "n1" {
		t.Errorf("merged.Goto = %v, want the first command's destinations", merged.Goto)
	}
	// Folding a second command with empty Graph/Resume must not clear the
	// values seeded by the first.
	merged = mergeCommand(merged, &types.Command{Update: map[string]any{"k2": "v2"}})
	if merged.Graph != types.ParentGraph || merged.Resume != 42 {
		t.Errorf("merged after empty fields = %+v, want Graph and Resume preserved", merged)
	}
	if merged.Update["k"] != "v" || merged.Update["k2"] != "v2" {
		t.Errorf("merged.Update = %v, want both commands' updates present", merged.Update)
	}
}
