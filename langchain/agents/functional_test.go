package agents

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

func TestFuncBeforeModelAdapter(t *testing.T) {
	var called bool
	hook := FuncBeforeModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		called = true
		state["k"] = "v"
		return state, nil
	})
	out, err := hook.BeforeModel(t.Context(), map[string]any{})
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if !called || out["k"] != "v" {
		t.Fatalf("before_model not invoked: called=%v out=%#v", called, out)
	}
}

func TestFuncAfterModelAdapter(t *testing.T) {
	hook := FuncAfterModel(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		return map[string]any{"seen": true}, nil
	})
	out, err := hook.AfterModel(t.Context(), map[string]any{})
	if err != nil || out["seen"] != true {
		t.Fatalf("after_model: err=%v out=%#v", err, out)
	}
}

func TestFuncWrapModelCallAdapter(t *testing.T) {
	hook := FuncWrapModelCall(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
		return handler(ctx, request)
	})
	resp, err := hook.WrapModelCall(t.Context(), middleware.ModelRequest{}, func(ctx context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
		return middleware.ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil || len(resp.Result) != 1 {
		t.Fatalf("wrap_model_call: err=%v resp=%#v", err, resp)
	}
}

func TestFuncWrapToolCallAdapter(t *testing.T) {
	hook := FuncWrapToolCall(func(ctx context.Context, request middleware.ToolCallRequest, handler middleware.ToolHandler) (messages.Message, error) {
		return handler(ctx, request)
	})
	msg, err := hook.WrapToolCall(t.Context(), middleware.ToolCallRequest{}, func(ctx context.Context, r middleware.ToolCallRequest) (messages.Message, error) {
		return messages.Tool("id", "result"), nil
	})
	if err != nil || msg.Content != "result" {
		t.Fatalf("wrap_tool_call: err=%v msg=%#v", err, msg)
	}
}

func TestFuncBeforeAndAfterAgentAdapters(t *testing.T) {
	before := FuncBeforeAgent(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		state["before"] = true
		return state, nil
	})
	after := FuncAfterAgent(func(ctx context.Context, state map[string]any) error {
		return nil
	})
	out, err := before.BeforeAgent(t.Context(), map[string]any{})
	if err != nil || out["before"] != true {
		t.Fatalf("before_agent: err=%v out=%#v", err, out)
	}
	if err := after.AfterAgent(t.Context(), map[string]any{}); err != nil {
		t.Fatalf("after_agent: %v", err)
	}
}

func TestFuncBeforeModelCommandAdapter(t *testing.T) {
	var called bool
	hook := FuncBeforeModelCommand(func(ctx context.Context, state map[string]any) (*middleware.Command, error) {
		called = true
		return &middleware.Command{Update: map[string]any{"k": "v"}, Goto: "end"}, nil
	})
	cmd, err := hook.BeforeModel(t.Context(), map[string]any{})
	if err != nil {
		t.Fatalf("BeforeModel: %v", err)
	}
	if !called || cmd == nil || cmd.Goto != "end" || cmd.Update["k"] != "v" {
		t.Fatalf("before_model command not invoked: called=%v cmd=%#v", called, cmd)
	}
}
