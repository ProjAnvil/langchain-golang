package middleware

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	agenttools "github.com/projanvil/langchain-golang/langchain/tools"
)

// ToolErrorHandler is the Go shape of Python's OnError callable
// (langchain/agents/middleware/tool_error.py: `OnError = Callable[[Exception,
// ToolCallRequest], str | list[ContentBlock] | None]`). It is called for each
// error raised by tool execution. Return (content, true) to convert the
// error into an error ToolMessage — content is a string or a
// []messages.ContentBlock. Return (nil, false) to let the error propagate,
// halting the run.
type ToolErrorHandler func(err error, request ToolCallRequest) (content any, handled bool)

// ToolErrorMiddleware returns selected tool-execution errors to the model as
// error ToolMessages, the port of langchain's ToolErrorMiddleware (#38781).
//
// OnError is called for each error raised by tool execution (raw errors: the
// ToolNode applies its own error conversion OUTSIDE the wrap chain, mirroring
// Python's ToolNode._run_one, so wrap middleware observe errors as raised).
// Returning content converts the error into a ToolMessage with error status;
// returning not-handled lets the ORIGINAL error propagate and the run fails.
// Handling is opt-in — errors you do not return content for propagate
// unchanged, so arbitrary internal errors are never serialized to the model
// or end user unless you choose to surface them. Prefer content that names
// the error type over the raw error message, which may carry sensitive or
// internal detail.
//
// Control-flow signals never reach OnError and always propagate: in the Go
// port interrupts are panics (*types.GraphInterrupt, recovered by the graph
// executor), so they pass straight through this middleware, matching
// Python's `except GraphBubbleUp: raise` clause.
//
// This middleware does not retry. For retries, compose with
// ToolRetryMiddleware placed INNER — i.e. listed AFTER this middleware in
// the agent's middleware slice (the first-listed middleware wraps OUTERMOST;
// see CreateAgent's WrapToolCall composition) — and configured with
// WithToolRetryOnFailure("error") so exhausted retries' exceptions escape
// the retry layer and reach this middleware; with the retry layer's default
// on_failure ("continue") it converts failures itself and this middleware
// never sees them. Listing the retry middleware first would make it the
// outer layer: this middleware would convert the error first and the
// retries would never fire.
//
// Unlike Python, whose on_error sees only execution errors (argument-binding
// and validation errors are converted by ToolNode inside the wrap chain
// before the tool runs), Go's tools packages expose no uniform
// binding-vs-execution error marker, so OnError observes both (see
// DIVERGENCES.md). Python additionally declares
// `trace_policy = TracePolicy(process_inputs=omit_payload)` (#40098); the Go
// port has no trace-policy installation point for wrap-tool-call middleware
// (see DIVERGENCES.md, "AutoMode trace policy").
//
// Example:
//
//	mw, err := middleware.NewToolErrorMiddleware(func(err error, req middleware.ToolCallRequest) (any, bool) {
//		if errors.Is(err, errBadInput) {
//			return fmt.Sprintf("`%s` failed; fix the input and retry.", req.ToolCall.Name), true
//		}
//		return nil, false // propagate everything else
//	})
type ToolErrorMiddleware struct {
	// OnError converts a raw tool error into error-ToolMessage content, or
	// declines (not-handled) to let the error propagate. Required.
	OnError ToolErrorHandler
	// ToolFilter restricts handling to the named tools; nil applies to all
	// tools (Python's `tools=None`).
	ToolFilter map[string]bool
}

// ToolErrorOption configures a ToolErrorMiddleware built by
// NewToolErrorMiddleware.
type ToolErrorOption func(*ToolErrorMiddleware)

// NewToolErrorMiddleware builds a ToolErrorMiddleware around onError,
// mirroring Python's ToolErrorMiddleware.__init__ (tool_error.py): it errors
// on a missing handler ("ToolErrorMiddleware requires `on_error`") and
// optionally restricts handling to a set of tools.
func NewToolErrorMiddleware(onError ToolErrorHandler, opts ...ToolErrorOption) (*ToolErrorMiddleware, error) {
	if onError == nil {
		return nil, fmt.Errorf("ToolErrorMiddleware requires `on_error`")
	}
	m := &ToolErrorMiddleware{OnError: onError}
	for _, opt := range opts {
		opt(m)
	}
	return m, nil
}

// WithToolErrorTools restricts error handling to the named tools, mirroring
// Python's `tools: list[BaseTool | str] | None` name-string form.
func WithToolErrorTools(names ...string) ToolErrorOption {
	return func(m *ToolErrorMiddleware) {
		if m.ToolFilter == nil {
			m.ToolFilter = map[string]bool{}
		}
		for _, name := range names {
			m.ToolFilter[name] = true
		}
	}
}

// WithToolErrorToolInstances restricts error handling to the given tools,
// matched by name — Python's BaseTool-instance form of the `tools` argument.
func WithToolErrorToolInstances(instances ...coretools.Tool) ToolErrorOption {
	return func(m *ToolErrorMiddleware) {
		if m.ToolFilter == nil {
			m.ToolFilter = map[string]bool{}
		}
		for _, t := range instances {
			if t != nil {
				m.ToolFilter[t.Name()] = true
			}
		}
	}
}

// WrapToolCall intercepts tool execution and converts handled errors into
// error ToolMessages, mirroring Python's wrap_tool_call (tool_error.py):
// non-matching tools pass through untouched (their errors bypass this
// middleware entirely), handled errors become one error ToolMessage named
// for the tool, and declined errors propagate wrapped in
// tools.UnhandledToolError — the sentinel the ToolNode's runOne-level
// conversion unwraps to re-raise the ORIGINAL error unconverted, so the run
// fails like Python's None path. When an inner wrap layer already declined,
// OnError receives that decline as the sentinel-wrapped error (errors.Is/As
// chain through it to the original error); declining again nests sentinels,
// which runOne strips.
func (m *ToolErrorMiddleware) WrapToolCall(ctx context.Context, request ToolCallRequest, handler ToolHandler) (messages.Message, error) {
	toolName := request.ToolCall.Name
	if request.Tool != nil {
		toolName = request.Tool.Name()
	}
	if !m.shouldHandleTool(toolName) {
		return handler(ctx, request)
	}

	msg, err := handler(ctx, request)
	if err == nil {
		return msg, nil
	}
	content, handled := m.OnError(err, request)
	if !handled {
		// Python: `if content is None: raise` — the exception propagates.
		return messages.Message{}, &agenttools.UnhandledToolError{Err: err}
	}
	return toolErrorMessage(request.ToolCall.ID, toolName, content), nil
}

func (m *ToolErrorMiddleware) shouldHandleTool(toolName string) bool {
	if m.ToolFilter == nil {
		return true
	}
	return m.ToolFilter[toolName]
}

// toolErrorMessage builds the error ToolMessage for handled content: a
// string lands in Content, a []messages.ContentBlock in ContentBlocks (the
// Go carrier for Python's ToolMessage(status="error") is
// ResponseMetadata{"status": "error"}). Any other non-nil content is
// stringified.
func toolErrorMessage(toolCallID, toolName string, content any) messages.Message {
	msg := messages.Tool(toolCallID, "")
	msg.Name = toolName
	msg.ResponseMetadata = map[string]any{"status": "error"}
	switch c := content.(type) {
	case string:
		msg.Content = c
	case []messages.ContentBlock:
		msg.ContentBlocks = c
	case nil:
		// handled with nil content: an empty error message.
	default:
		msg.Content = fmt.Sprintf("%v", c)
	}
	return msg
}
