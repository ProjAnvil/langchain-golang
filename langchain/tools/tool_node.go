// Package tools re-exports langchain_core's tool types for v1 (see tools.go)
// and implements a Go-idiomatic equivalent of Python's
// `langchain.tools.ToolNode` (backed by langgraph's `ToolNode` under the
// hood) in this file.
//
// Scope note: Python's `ToolNode` is a `langgraph` graph node that operates
// on arbitrary graph state, supports `Command`-based control flow returned by
// tools, `Send`-based distributed dispatch, and reflection-based
// `InjectedState`/`InjectedStore`/`ToolRuntime` argument injection. langgraph
// is not vendored in this Go port (see migration_plan/core-v1-migration-todo.md),
// so this ToolNode instead operates directly on a `[]messages.Message` slice:
// it finds the tool calls attached to the most recent AI message, executes
// them (concurrently, matching Python's default parallel execution), and
// returns the resulting tool messages. It does not support Send-based
// dispatch or reflection-based state/store injection; tools that need
// read-only access to conversation state can receive it explicitly via
// ToolCallRequest.State. Command-based control flow is surfaced (but not
// acted on) via InvokeToolCallsFull: a tool places a *types.Command in its
// Result.Artifact and the caller — e.g. langgraph/prebuilt.ToolNode — applies
// it to graph state.
package tools

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// ToolCallRequest is passed to a ToolCallWrapper, mirroring Python's
// `ToolCallRequest` dataclass (minus the `runtime` field, since this port has
// no langgraph runtime to attach).
type ToolCallRequest struct {
	// ToolCall is the tool call requested by the model.
	ToolCall messages.ToolCall
	// Tool is the registered Tool instance, or nil if ToolCall.Name is not
	// registered with the ToolNode. Wrappers may inspect this to
	// short-circuit unregistered calls before validation occurs.
	Tool Tool
	// State is optional read-only context (e.g. conversation state) threaded
	// through from the caller of InvokeToolCalls/AppendToolResults.
	State map[string]any
	// Store is the agent's cross-thread semantic store (Python's
	// `InjectedStore` / langgraph BaseStore), populated when the ToolNode was
	// built via WithToolNodeStore (i.e. the agent was configured with
	// WithAgentStore). nil when no store is configured. Go has no
	// annotation-based InjectedStore, so tools read it explicitly.
	Store store.Store
}

// ToolHandler executes a single tool call and returns the resulting message.
type ToolHandler func(context.Context, ToolCallRequest) (messages.Message, error)

// ToolCallWrapper lets middleware intercept execution of a single tool call,
// mirroring Python's `wrap_tool_call`. It receives the request and a `next`
// callable that performs (or continues) execution; `next` may be invoked
// zero, one, or multiple times (e.g. for retries), matching the documented
// Python contract.
type ToolCallWrapper func(context.Context, ToolCallRequest, ToolHandler) (messages.Message, error)

// HandleToolErrors converts a tool execution error into ToolMessage content.
// It returns (content, true) if the error should be converted into an error
// ToolMessage, or ("", false) if the error should instead propagate out of
// Invoke/InvokeToolCalls.
type HandleToolErrors func(error) (string, bool)

// DefaultHandleToolErrors mirrors Python's `handle_tool_errors=True` mode: it
// catches every error and formats it with a generic template. Python's actual
// default additionally distinguishes tool *invocation* errors (invalid
// arguments) from tool *execution* errors (re-raised); Go's core/tools does
// not expose that distinction uniformly, so this simpler catch-all default is
// used instead. Callers that need the stricter behavior can supply their own
// HandleToolErrors via WithHandleToolErrors.
func DefaultHandleToolErrors(err error) (string, bool) {
	return fmt.Sprintf("Error: %v\n Please fix your mistakes.", err), true
}

// ToolCallOutcome is the full result of executing a single tool call: the
// resulting message plus, when the tool signaled graph control flow by
// placing a *types.Command in its Result.Artifact, that command. It is
// returned by InvokeToolCallsFull; InvokeToolCalls discards the Command half.
type ToolCallOutcome struct {
	// Message is the ToolMessage produced for the call (including handled
	// error messages).
	Message messages.Message
	// Command is the graph command the tool returned via Result.Artifact, or
	// nil if the tool returned no command.
	Command *types.Command
}

// ToolNode executes tool calls requested by a model, mirroring the core
// execution behavior of Python's `langchain.tools.ToolNode`. See the package
// doc comment for what is intentionally out of scope.
type ToolNode struct {
	byName           map[string]Tool
	handleToolErrors HandleToolErrors
	wrap             ToolCallWrapper
	store            store.Store
}

// ToolNodeOption configures a ToolNode constructed by NewToolNode.
type ToolNodeOption func(*ToolNode)

// WithHandleToolErrors overrides the default tool error handling strategy.
func WithHandleToolErrors(handler HandleToolErrors) ToolNodeOption {
	return func(n *ToolNode) { n.handleToolErrors = handler }
}

// WithToolCallWrapper installs a ToolCallWrapper invoked around every tool
// call, mirroring Python's `wrap_tool_call` constructor argument.
func WithToolCallWrapper(wrap ToolCallWrapper) ToolNodeOption {
	return func(n *ToolNode) { n.wrap = wrap }
}

// WithToolNodeStore installs the agent's cross-thread semantic store
// (langgraph BaseStore), mirroring Python's `InjectedStore` plumbing (Go has
// no annotation-based injection, so the store is surfaced explicitly on each
// ToolCallRequest.Store). When non-nil, the ToolNode populates Store on every
// request it builds.
func WithToolNodeStore(s store.Store) ToolNodeOption {
	return func(n *ToolNode) { n.store = s }
}

// NewToolNode builds a ToolNode over toolList. Tool names must be unique and
// non-empty, and toolList must not be empty, mirroring the expectation that a
// ToolNode always has at least one tool to dispatch to.
func NewToolNode(toolList []Tool, opts ...ToolNodeOption) (*ToolNode, error) {
	if len(toolList) == 0 {
		return nil, fmt.Errorf("tool_node: at least one tool is required")
	}
	byName := make(map[string]Tool, len(toolList))
	for _, t := range toolList {
		if t == nil {
			return nil, fmt.Errorf("tool_node: tool must not be nil")
		}
		name := t.Name()
		if name == "" {
			return nil, fmt.Errorf("tool_node: tool name must not be empty")
		}
		if _, exists := byName[name]; exists {
			return nil, fmt.Errorf("tool_node: duplicate tool name %q", name)
		}
		byName[name] = t
	}

	n := &ToolNode{byName: byName, handleToolErrors: DefaultHandleToolErrors}
	for _, opt := range opts {
		opt(n)
	}
	if n.handleToolErrors == nil {
		n.handleToolErrors = DefaultHandleToolErrors
	}
	return n, nil
}

// ToolsByName returns a copy of the registered tools keyed by name.
func (n *ToolNode) ToolsByName() map[string]Tool {
	out := make(map[string]Tool, len(n.byName))
	for name, t := range n.byName {
		out[name] = t
	}
	return out
}

// PendingToolCalls returns the tool calls attached to the most recent AI
// message in msgs (searching from the end), or nil if no AI message with
// tool calls is found. This mirrors the state inspection performed by
// Python's `tools_condition` routing helper.
func PendingToolCalls(msgs []messages.Message) []messages.ToolCall {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == messages.RoleAI {
			return msgs[i].ToolCalls
		}
	}
	return nil
}

// HasPendingToolCalls reports whether the most recent AI message in msgs has
// tool calls to execute. It is the Go equivalent of Python's
// `tools_condition`, which routes to `"tools"` or `"__end__"`.
func HasPendingToolCalls(msgs []messages.Message) bool {
	return len(PendingToolCalls(msgs)) > 0
}

// Invoke executes the tool calls attached to the most recent AI message in
// msgs and returns the resulting tool messages, in call order. It returns an
// empty, non-nil slice if there are no pending tool calls. state is optional
// and is threaded through to every ToolCallRequest.State.
func (n *ToolNode) Invoke(ctx context.Context, msgs []messages.Message, state map[string]any) ([]messages.Message, error) {
	return n.InvokeToolCalls(ctx, PendingToolCalls(msgs), state)
}

// AppendToolResults executes the tool calls attached to the most recent AI
// message in msgs and returns msgs with the resulting tool messages appended,
// mirroring the typical `{"messages": [...]}` graph-state update Python's
// ToolNode produces for dict-shaped state. It returns msgs unchanged (same
// backing data, not copied) if there are no pending tool calls.
func (n *ToolNode) AppendToolResults(ctx context.Context, msgs []messages.Message) ([]messages.Message, error) {
	calls := PendingToolCalls(msgs)
	if len(calls) == 0 {
		return msgs, nil
	}
	results, err := n.InvokeToolCalls(ctx, calls, nil)
	if err != nil {
		return nil, err
	}
	out := make([]messages.Message, 0, len(msgs)+len(results))
	out = append(out, msgs...)
	out = append(out, results...)
	return out, nil
}

// InvokeToolCalls executes calls concurrently, matching Python ToolNode's
// default parallel execution, and returns one ToolMessage per call in the
// same order as calls. state is passed through to every ToolCallRequest.State
// and may be nil.
//
// If any call's HandleToolErrors declines to handle its error (returns
// handled=false), InvokeToolCalls returns that error; results for other,
// still-running calls are discarded, matching the "let it propagate" contract
// Python uses when `handle_tool_errors` does not cover the raised exception.
//
// Commands a tool signals via Result.Artifact (see InvokeToolCallsFull) are
// discarded here; use InvokeToolCallsFull to observe them.
func (n *ToolNode) InvokeToolCalls(ctx context.Context, calls []messages.ToolCall, state map[string]any) ([]messages.Message, error) {
	outcomes, err := n.InvokeToolCallsFull(ctx, calls, state)
	if err != nil {
		return nil, err
	}
	results := make([]messages.Message, len(outcomes))
	for i, outcome := range outcomes {
		results[i] = outcome.Message
	}
	return results, nil
}

// InvokeToolCallsFull is InvokeToolCalls with the full per-call outcome: in
// addition to the result message, it surfaces the *types.Command a tool
// placed in its Result.Artifact (the Go port's convention for a tool
// signaling graph control flow, used by langgraph/prebuilt.ToolNode). An
// Artifact of any other type is ignored, matching InvokeToolCalls.
//
// Execution, error handling, wrappers, and store plumbing are exactly those
// of InvokeToolCalls. Two wrapper interactions are worth noting: a
// ToolCallWrapper that short-circuits a call without invoking `next` yields
// no Command (the tool never ran), and a wrapper that invokes `next` more
// than once (e.g. retries) keeps the Command of the last invocation.
func (n *ToolNode) InvokeToolCallsFull(ctx context.Context, calls []messages.ToolCall, state map[string]any) ([]ToolCallOutcome, error) {
	outcomes := make([]ToolCallOutcome, len(calls))
	errs := make([]error, len(calls))

	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Add(1)
		go func(i int, call messages.ToolCall) {
			defer wg.Done()
			outcomes[i], errs[i] = n.runOne(ctx, call, state)
		}(i, call)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return outcomes, nil
}

func (n *ToolNode) runOne(ctx context.Context, call messages.ToolCall, state map[string]any) (ToolCallOutcome, error) {
	request := ToolCallRequest{ToolCall: call, Tool: n.byName[call.Name], State: state}
	if n.store != nil {
		request.Store = n.store
	}
	if n.wrap != nil {
		// The wrapper contract is message-only, so the Command is captured
		// from the innermost execute call(s) the wrapper delegates to.
		var cmd *types.Command
		next := func(ctx context.Context, req ToolCallRequest) (messages.Message, error) {
			outcome, err := n.execute(ctx, req)
			if err == nil {
				cmd = outcome.Command
			}
			return outcome.Message, err
		}
		msg, err := n.wrap(ctx, request, next)
		return ToolCallOutcome{Message: msg, Command: cmd}, err
	}
	return n.execute(ctx, request)
}

func (n *ToolNode) execute(ctx context.Context, req ToolCallRequest) (ToolCallOutcome, error) {
	call := req.ToolCall
	if req.Tool == nil {
		return ToolCallOutcome{Message: n.invalidToolMessage(call)}, nil
	}

	result, err := req.Tool.Invoke(ctx, call.Args)
	if err != nil {
		content, handled := n.handleToolErrors(err)
		if !handled {
			return ToolCallOutcome{}, err
		}
		return ToolCallOutcome{Message: errorToolMessage(call.ID, call.Name, content)}, nil
	}

	msg := messages.Tool(call.ID, result.Content)
	msg.Name = call.Name
	return ToolCallOutcome{Message: msg, Command: commandFromArtifact(result.Artifact)}, nil
}

// commandFromArtifact extracts the *types.Command a tool placed in its
// Result.Artifact to signal graph control flow, or nil if the artifact is
// absent or of any other type. Recognition goes through the
// messages.ToolOutput marker interface (Python's ToolOutputMixin,
// langchain_core/messages/tool.py:16): only ToolOutput-marked artifacts are
// considered direct tool outputs at all, and only *types.Command among them
// carries graph control flow in this port.
func commandFromArtifact(artifact any) *types.Command {
	out, ok := artifact.(messages.ToolOutput)
	if !ok {
		return nil
	}
	cmd, _ := out.(*types.Command)
	return cmd
}

func (n *ToolNode) invalidToolMessage(call messages.ToolCall) messages.Message {
	available := make([]string, 0, len(n.byName))
	for name := range n.byName {
		available = append(available, name)
	}
	sort.Strings(available)
	content := fmt.Sprintf("Error: %s is not a valid tool, try one of [%s].", call.Name, strings.Join(available, ", "))
	return errorToolMessage(call.ID, call.Name, content)
}

func errorToolMessage(toolCallID, toolName, content string) messages.Message {
	msg := messages.Tool(toolCallID, content)
	msg.Name = toolName
	msg.ResponseMetadata = map[string]any{"status": "error"}
	return msg
}
