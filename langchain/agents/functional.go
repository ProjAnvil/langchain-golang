package agents

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// This file provides functional adapters that lift plain Go functions into the
// `*Hook` interfaces, the Go analog of Python's `before_model`/`after_model`/
// `wrap_model_call`/`wrap_tool_call`/`before_agent`/`after_agent` decorators.
// Middleware that is naturally a single function can be authored without
// declaring a named struct implementing the interface.

type (
	// BeforeModelFunc mirrors Python's `before_model` hook.
	BeforeModelFunc func(ctx context.Context, state map[string]any) (map[string]any, error)
	// BeforeModelCommandFunc mirrors a `before_model` returning a full Command.
	BeforeModelCommandFunc func(ctx context.Context, state map[string]any) (*middleware.Command, error)
	// AfterModelFunc mirrors Python's `after_model` hook.
	AfterModelFunc func(ctx context.Context, state map[string]any) (map[string]any, error)
	// WrapModelCallFunc mirrors Python's `wrap_model_call` hook.
	WrapModelCallFunc func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error)
	// WrapToolCallFunc mirrors Python's `wrap_tool_call` hook.
	WrapToolCallFunc func(ctx context.Context, request middleware.ToolCallRequest, handler middleware.ToolHandler) (messages.Message, error)
	// BeforeAgentFunc mirrors Python's `before_agent` hook.
	BeforeAgentFunc func(ctx context.Context, state map[string]any) (map[string]any, error)
	// AfterAgentFunc mirrors Python's `after_agent` hook.
	AfterAgentFunc func(ctx context.Context, state map[string]any) error
)

type beforeModelFuncAdapter struct{ fn BeforeModelFunc }

func (a beforeModelFuncAdapter) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return a.fn(ctx, state)
}

type beforeModelCommandFuncAdapter struct{ fn BeforeModelCommandFunc }

func (a beforeModelCommandFuncAdapter) BeforeModel(ctx context.Context, state map[string]any) (*middleware.Command, error) {
	return a.fn(ctx, state)
}

type afterModelFuncAdapter struct{ fn AfterModelFunc }

func (a afterModelFuncAdapter) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return a.fn(ctx, state)
}

type wrapModelCallFuncAdapter struct{ fn WrapModelCallFunc }

func (a wrapModelCallFuncAdapter) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	return a.fn(ctx, request, handler)
}

type wrapToolCallFuncAdapter struct{ fn WrapToolCallFunc }

func (a wrapToolCallFuncAdapter) WrapToolCall(ctx context.Context, request middleware.ToolCallRequest, handler middleware.ToolHandler) (messages.Message, error) {
	return a.fn(ctx, request, handler)
}

type beforeAgentFuncAdapter struct{ fn BeforeAgentFunc }

func (a beforeAgentFuncAdapter) BeforeAgent(ctx context.Context, state map[string]any) (map[string]any, error) {
	return a.fn(ctx, state)
}

type afterAgentFuncAdapter struct{ fn AfterAgentFunc }

func (a afterAgentFuncAdapter) AfterAgent(ctx context.Context, state map[string]any) error {
	return a.fn(ctx, state)
}

// FuncBeforeModel returns a BeforeModelHook backed by fn.
func FuncBeforeModel(fn BeforeModelFunc) BeforeModelHook { return beforeModelFuncAdapter{fn: fn} }

// FuncBeforeModelCommand returns a BeforeModelCommandHook backed by fn.
func FuncBeforeModelCommand(fn BeforeModelCommandFunc) BeforeModelCommandHook {
	return beforeModelCommandFuncAdapter{fn: fn}
}

// FuncAfterModel returns an AfterModelHook backed by fn.
func FuncAfterModel(fn AfterModelFunc) AfterModelHook { return afterModelFuncAdapter{fn: fn} }

// FuncWrapModelCall returns a WrapModelCallHook backed by fn.
func FuncWrapModelCall(fn WrapModelCallFunc) WrapModelCallHook { return wrapModelCallFuncAdapter{fn: fn} }

// FuncWrapToolCall returns a WrapToolCallHook backed by fn.
func FuncWrapToolCall(fn WrapToolCallFunc) WrapToolCallHook { return wrapToolCallFuncAdapter{fn: fn} }

// FuncBeforeAgent returns a BeforeAgentHook backed by fn.
func FuncBeforeAgent(fn BeforeAgentFunc) BeforeAgentHook { return beforeAgentFuncAdapter{fn: fn} }

// FuncAfterAgent returns an AfterAgentHook backed by fn.
func FuncAfterAgent(fn AfterAgentFunc) AfterAgentHook { return afterAgentFuncAdapter{fn: fn} }

// DynamicPromptFunc mirrors the function decorated with Python's
// `@dynamic_prompt` (middleware/types.py:1680): it computes the system prompt
// for one model call from the request. Return either a string (wrapped into a
// system message, mirroring Python's `str` branch) or a complete
// messages.Message (installed as-is, mirroring Python's `SystemMessage`
// branch). Returning (nil, nil) leaves the request unchanged (Go-only
// convenience; Python has no None return).
type DynamicPromptFunc func(ctx context.Context, request middleware.ModelRequest) (any, error)

type dynamicPromptAdapter struct{ fn DynamicPromptFunc }

func (a dynamicPromptAdapter) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	prompt, err := a.fn(ctx, request)
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	var next middleware.ModelRequest
	switch p := prompt.(type) {
	case nil:
		next = request
	case string:
		next, err = request.Override(middleware.WithSystemPrompt(p))
	case messages.Message:
		msg := p
		next, err = request.Override(middleware.WithSystemMessage(&msg))
	case *messages.Message:
		next, err = request.Override(middleware.WithSystemMessage(p))
	default:
		return middleware.ModelResponse{}, fmt.Errorf("agents: DynamicPromptFunc must return a string or messages.Message, got %T", prompt)
	}
	if err != nil {
		return middleware.ModelResponse{}, err
	}
	return handler(ctx, next)
}

// DynamicPrompt returns a WrapModelCallHook that installs fn's computed system
// prompt before delegating to the handler, mirroring Python's `@dynamic_prompt`
// decorator. When several DynamicPrompt middleware are chained, the one closest
// to the model (last in the CreateAgent middleware list) runs last and its
// prompt wins, matching Python's wrap_model_call composition order.
func DynamicPrompt(fn DynamicPromptFunc) WrapModelCallHook {
	return dynamicPromptAdapter{fn: fn}
}
