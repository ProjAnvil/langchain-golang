package prebuilt

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// defaultMessagesKey is the state key ToolNode reads and writes when
// WithMessagesKey is not used, matching Python's `messages_key="messages"`.
const defaultMessagesKey = "messages"

// toolNodeConfig holds the configuration assembled from ToolNodeOptions.
type toolNodeConfig struct {
	messagesKey string
}

// ToolNodeOption configures the NodeFunc returned by ToolNode.
type ToolNodeOption func(*toolNodeConfig)

// WithMessagesKey sets the state key the returned node reads the message
// list from and writes tool result messages to (default "messages"),
// mirroring Python's `messages_key` argument.
func WithMessagesKey(key string) ToolNodeOption {
	return func(c *toolNodeConfig) { c.messagesKey = key }
}

// ToolNode returns a graph.NodeFunc that runs the tool calls of the last AI
// message in state[messagesKey] through node (a langchain/tools.ToolNode)
// and returns map[string]any{messagesKey: resultMessages}.
//
// The messages key needs an append reducer registered on the graph —
// graph.AddReducer(key, channels.MessagesReducer) — otherwise the default
// LastValue channel replaces the message history with each update.
//
// Behavior:
//
//   - Execution and error handling are delegated to node unchanged (no
//     behavior fork): calls run concurrently, and tool errors are handled per
//     the node's HandleToolErrors (by default, converted into error
//     ToolMessages).
//   - The full graph state is passed as ToolCallRequest.State, so tools and
//     wrappers see the same read-only context Python's InjectedState
//     provides.
//   - A tool signals graph control flow by placing a *types.Command in its
//     Result.Artifact (surfaced via ToolNode.InvokeToolCallsFull). When any
//     tool in the batch returned a Command, the node's result is a single
//     merged *types.Command: the messages update is always present in its
//     Update map, the individual commands' Update maps merge into it, and
//     their Goto lists concatenate (so the graph routes to every requested
//     destination). When several commands conflict — the same Update key,
//     Graph, or Resume — the last one in call order wins; tools in one batch
//     are expected not to conflict.
//   - A missing state[messagesKey], or a value that is not a
//     []messages.Message, is a descriptive error. A last AI message without
//     tool calls yields a nil update (state untouched).
//
// Divergences from Python's `langgraph.prebuilt.ToolNode` (matching
// langchain/tools.ToolNode's scope): no Send-per-tool-call dispatch (Go
// executes the calls concurrently within one node) and no reflection-based
// argument injection (Go's ToolCallRequest.State is explicit).
//
// ToolNode panics if node is nil or an option sets an empty messages key:
// both are programming errors best caught at graph construction time.
func ToolNode(node *tools.ToolNode, opts ...ToolNodeOption) graph.NodeFunc {
	if node == nil {
		panic("prebuilt: ToolNode requires a non-nil *tools.ToolNode")
	}
	cfg := toolNodeConfig{messagesKey: defaultMessagesKey}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.messagesKey == "" {
		panic("prebuilt: ToolNode messages key must not be empty")
	}

	return func(ctx context.Context, state map[string]any) (any, error) {
		raw, ok := state[cfg.messagesKey]
		if !ok {
			return nil, fmt.Errorf("prebuilt: ToolNode requires state key %q to hold the conversation messages, but it is missing", cfg.messagesKey)
		}
		msgs, ok := raw.([]messages.Message)
		if !ok {
			return nil, fmt.Errorf("prebuilt: ToolNode requires state key %q to hold []messages.Message, got %T", cfg.messagesKey, raw)
		}

		calls := tools.PendingToolCalls(msgs)
		if len(calls) == 0 {
			return nil, nil
		}

		outcomes, err := node.InvokeToolCallsFull(ctx, calls, state)
		if err != nil {
			return nil, err
		}

		resultMessages := make([]messages.Message, len(outcomes))
		var merged *types.Command
		for i, outcome := range outcomes {
			resultMessages[i] = outcome.Message
			if outcome.Command != nil {
				merged = mergeCommand(merged, outcome.Command)
			}
		}

		if merged == nil {
			return map[string]any{cfg.messagesKey: resultMessages}, nil
		}
		if merged.Update == nil {
			merged.Update = map[string]any{}
		}
		merged.Update[cfg.messagesKey] = resultMessages
		return merged, nil
	}
}

// mergeCommand folds cmd into merged: Update maps merge (cmd's keys win on
// conflict) and Goto lists concatenate. Graph and Resume are taken from cmd
// when set (last-in-call-order wins on conflict). merged may be nil.
func mergeCommand(merged, cmd *types.Command) *types.Command {
	if merged == nil {
		merged = &types.Command{
			Graph:  cmd.Graph,
			Update: map[string]any{},
			Resume: cmd.Resume,
		}
	}
	for k, v := range cmd.Update {
		merged.Update[k] = v
	}
	if cmd.Graph != "" {
		merged.Graph = cmd.Graph
	}
	if cmd.Resume != nil {
		merged.Resume = cmd.Resume
	}
	merged.Goto = append(merged.Goto, cmd.Goto...)
	return merged
}
