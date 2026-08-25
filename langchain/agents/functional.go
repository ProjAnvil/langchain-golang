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

// HookConfig declares static middleware hook metadata, mirroring Python's
// `@hook_config` decorator (middleware/types.py:867). Unlike Python, where
// can_jump_to drives conditional-edge construction (`factory._add_middleware_edge`,
// factory.py:1957), Go routes jumps dynamically via the types.Command returned
// by the model node (see popJumpTo, create_agent.go), so the declaration is
// metadata: CreateAgent validates the declared target names against Python's
// JumpTo literal ("model" | "tools" | "end") at build time, and tooling can
// introspect it via DeclaredCanJumpTo.
type HookConfig struct {
	// CanJumpTo lists the jump destinations the hook may request via
	// update["jump_to"]: "model", "tools", or "end".
	CanJumpTo []string
}

// CanJumpToHook is implemented by middleware that statically declares valid
// jump destinations for its hooks, mirroring Python's `__can_jump_to__`
// method metadata set by `@hook_config` / `@before_model(can_jump_to=...)`.
type CanJumpToHook interface {
	// CanJumpTo returns the declared jump targets for hookName
	// ("before_model"/"after_model"), or nil when the hook has no declaration.
	CanJumpTo(hookName string) []string
}

// DeclaredCanJumpTo mirrors Python's `factory._get_can_jump_to`
// (factory.py:491): it returns the declared jump targets for hookName on mw,
// or nil when mw does not implement CanJumpToHook (no false positives, like
// Python's overridden-method check).
func DeclaredCanJumpTo(mw any, hookName string) []string {
	if hook, ok := mw.(CanJumpToHook); ok {
		return hook.CanJumpTo(hookName)
	}
	return nil
}

type beforeModelConfiguredAdapter struct {
	fn        BeforeModelFunc
	canJumpTo []string
}

func (a beforeModelConfiguredAdapter) BeforeModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return a.fn(ctx, state)
}

func (a beforeModelConfiguredAdapter) CanJumpTo(hookName string) []string {
	if hookName != "before_model" {
		return nil
	}
	return append([]string(nil), a.canJumpTo...)
}

// FuncBeforeModelWithConfig mirrors `@before_model(can_jump_to=...)` /
// `@hook_config(can_jump_to=...)` on a before_model hook: a BeforeModelHook
// backed by fn that additionally declares its valid jump targets.
func FuncBeforeModelWithConfig(fn BeforeModelFunc, cfg HookConfig) BeforeModelHook {
	return beforeModelConfiguredAdapter{fn: fn, canJumpTo: cfg.CanJumpTo}
}

type afterModelConfiguredAdapter struct {
	fn        AfterModelFunc
	canJumpTo []string
}

func (a afterModelConfiguredAdapter) AfterModel(ctx context.Context, state map[string]any) (map[string]any, error) {
	return a.fn(ctx, state)
}

func (a afterModelConfiguredAdapter) CanJumpTo(hookName string) []string {
	if hookName != "after_model" {
		return nil
	}
	return append([]string(nil), a.canJumpTo...)
}

// FuncAfterModelWithConfig mirrors `@after_model(can_jump_to=...)`.
func FuncAfterModelWithConfig(fn AfterModelFunc, cfg HookConfig) AfterModelHook {
	return afterModelConfiguredAdapter{fn: fn, canJumpTo: cfg.CanJumpTo}
}
