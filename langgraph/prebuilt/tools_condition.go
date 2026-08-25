package prebuilt

import (
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// ToolsCondition returns a graph.ConditionalEdge implementing Python's
// `langgraph.prebuilt.tools_condition` (prebuilt/tool_node.py:1582): if the
// last message in state[messagesKey] is an AI message with tool calls, route
// to the "tools" node; otherwise route to types.END ("__end__"). This is the
// standard routing function for ReAct-style tool-calling loops:
//
//	g.AddConditionalEdges("model", prebuilt.ToolsCondition())
//
// messagesKey defaults to "messages"; pass a custom key for graphs that store
// the conversation elsewhere (Python's `messages_key` argument, e.g.
// test_tool_node.py:1548's messages_key="subgraph_messages").
//
// Like Python, only the LAST message is inspected (state[-1]): an AI message
// with tool calls followed by a ToolMessage routes to END. This deliberately
// differs from tools.PendingToolCalls (langchain/tools/tool_node.go:174),
// which scans backwards for the most recent AI message.
//
// The error text is adapted from Python's ValueError("No messages found in
// input state to tool_edge") (tool_node.py:1655): lowercased, naming
// tools_condition, and including the offending key. A present but wrong-typed
// messages value is a deliberate divergence — Python returns __end__ for it,
// Go returns an error.
//
// ToolsCondition panics on an empty messagesKey, matching ToolNode's
// construction-time panic for programming errors.
func ToolsCondition(messagesKey ...string) graph.ConditionalEdge {
	key := defaultMessagesKey
	if len(messagesKey) > 0 {
		key = messagesKey[0]
	}
	if key == "" {
		panic("prebuilt: ToolsCondition messages key must not be empty")
	}
	return func(_ runtime.Runtime, state map[string]any) ([]any, error) {
		raw, ok := state[key]
		if !ok {
			return nil, fmt.Errorf("prebuilt: no messages found in input state to tools_condition: missing key %q", key)
		}
		msgs, ok := raw.([]messages.Message)
		if !ok || len(msgs) == 0 {
			return nil, fmt.Errorf("prebuilt: no messages found in input state to tools_condition: key %q holds %v", key, raw)
		}
		last := msgs[len(msgs)-1]
		if last.Role == messages.RoleAI && len(last.ToolCalls) > 0 {
			return []any{"tools"}, nil
		}
		return []any{types.END}, nil
	}
}
