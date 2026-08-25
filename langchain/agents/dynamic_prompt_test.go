package agents

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// echoSystemPromptHandler returns the request's effective system prompt text
// as the AI response, like the mock_handler closures in the Python tests.
func echoSystemPromptHandler(_ context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
	return middleware.ModelResponse{Result: []messages.Message{messages.AI(r.SystemPromptText())}}, nil
}

// TestDynamicPromptStringPrompt mirrors test_dynamic_prompt_decorator
// (tests/unit_tests/agents/middleware/core/test_decorators.py:630).
func TestDynamicPromptStringPrompt(t *testing.T) {
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return "Dynamic test prompt", nil
	})
	req, err := middleware.NewModelRequest(middleware.ModelRequest{
		SystemPrompt: "Original",
		Messages:     []messages.Message{messages.Human("Hello")},
	})
	if err != nil {
		t.Fatalf("NewModelRequest: %v", err)
	}
	resp, err := hook.WrapModelCall(context.Background(), req, echoSystemPromptHandler)
	if err != nil {
		t.Fatalf("WrapModelCall: %v", err)
	}
	if len(resp.Result) != 1 || resp.Result[0].Content != "Dynamic test prompt" {
		t.Fatalf("response mismatch: %#v", resp.Result)
	}
}

// TestDynamicPromptUsesState mirrors test_dynamic_prompt_uses_state
// (test_decorators.py:662).
func TestDynamicPromptUsesState(t *testing.T) {
	hook := DynamicPrompt(func(_ context.Context, request middleware.ModelRequest) (any, error) {
		msgs, _ := request.State["messages"].([]messages.Message)
		return fmt.Sprintf("Prompt with %d messages", len(msgs)), nil
	})
	req, err := middleware.NewModelRequest(middleware.ModelRequest{
		SystemPrompt: "Original",
		Messages:     []messages.Message{messages.Human("Hello")},
		State: map[string]any{"messages": []messages.Message{
			messages.Human("Hello"), messages.Human("World"),
		}},
	})
	if err != nil {
		t.Fatalf("NewModelRequest: %v", err)
	}
	resp, err := hook.WrapModelCall(context.Background(), req, echoSystemPromptHandler)
	if err != nil {
		t.Fatalf("WrapModelCall: %v", err)
	}
	if len(resp.Result) != 1 || resp.Result[0].Content != "Prompt with 2 messages" {
		t.Fatalf("response mismatch: %#v", resp.Result)
	}
}

// TestDynamicPromptSystemMessage covers Python's `SystemMessage` return branch
// (`types.py:1779-1782`): a complete message is installed as-is.
func TestDynamicPromptSystemMessage(t *testing.T) {
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return messages.System("full system message"), nil
	})
	req, err := middleware.NewModelRequest(middleware.ModelRequest{
		SystemPrompt: "Original",
		Messages:     []messages.Message{messages.Human("Hello")},
	})
	if err != nil {
		t.Fatalf("NewModelRequest: %v", err)
	}
	resp, err := hook.WrapModelCall(context.Background(), req, func(_ context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
		if r.SystemMessage == nil || r.SystemMessage.Content != "full system message" {
			t.Fatalf("system message mismatch: %#v", r.SystemMessage)
		}
		return middleware.ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil || resp.Result[0].Content != "ok" {
		t.Fatalf("WrapModelCall: err=%v resp=%#v", err, resp)
	}
}

// TestDynamicPromptSystemMessagePointer covers the *messages.Message return
// branch: a complete message pointer is installed as-is.
func TestDynamicPromptSystemMessagePointer(t *testing.T) {
	msg := messages.System("pointer system message")
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return &msg, nil
	})
	req, err := middleware.NewModelRequest(middleware.ModelRequest{
		SystemPrompt: "Original",
		Messages:     []messages.Message{messages.Human("Hello")},
	})
	if err != nil {
		t.Fatalf("NewModelRequest: %v", err)
	}
	resp, err := hook.WrapModelCall(context.Background(), req, func(_ context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
		if r.SystemMessage == nil || r.SystemMessage.Content != "pointer system message" {
			t.Fatalf("system message mismatch: %#v", r.SystemMessage)
		}
		return middleware.ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil || resp.Result[0].Content != "ok" {
		t.Fatalf("WrapModelCall: err=%v resp=%#v", err, resp)
	}
}

// TestDynamicPromptNilPromptLeavesRequest covers the Go-only nil branch: no
// override is applied and the original system prompt survives.
func TestDynamicPromptNilPromptLeavesRequest(t *testing.T) {
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return nil, nil
	})
	req, err := middleware.NewModelRequest(middleware.ModelRequest{
		SystemPrompt: "Original",
		Messages:     []messages.Message{messages.Human("Hello")},
	})
	if err != nil {
		t.Fatalf("NewModelRequest: %v", err)
	}
	resp, err := hook.WrapModelCall(context.Background(), req, echoSystemPromptHandler)
	if err != nil || resp.Result[0].Content != "Original" {
		t.Fatalf("WrapModelCall: err=%v resp=%#v", err, resp)
	}
}

// TestDynamicPromptInvalidReturn pins the unsupported-return-type error.
func TestDynamicPromptInvalidReturn(t *testing.T) {
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return 42, nil
	})
	handlerCalled := false
	_, err := hook.WrapModelCall(context.Background(), middleware.ModelRequest{}, func(_ context.Context, r middleware.ModelRequest) (middleware.ModelResponse, error) {
		handlerCalled = true
		return middleware.ModelResponse{}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "must return a string or messages.Message") {
		t.Fatalf("expected return-type error, got %v", err)
	}
	if handlerCalled {
		t.Fatal("handler must not be called on invalid prompt return")
	}
}

// TestDynamicPromptFuncErrorPropagates covers the fn-error branch.
func TestDynamicPromptFuncErrorPropagates(t *testing.T) {
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return nil, fmt.Errorf("prompt boom")
	})
	_, err := hook.WrapModelCall(context.Background(), middleware.ModelRequest{}, echoSystemPromptHandler)
	if err == nil || !strings.Contains(err.Error(), "prompt boom") {
		t.Fatalf("expected fn error, got %v", err)
	}
}

// TestDynamicPromptIntegration mirrors test_dynamic_prompt_integration
// (test_decorators.py:690).
func TestDynamicPromptIntegration(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	calls := 0
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		calls++
		return "you are a helpful assistant.", nil
	})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Hello")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if calls != 1 {
		t.Fatalf("prompt func calls = %d, want 1", calls)
	}
	if len(out) != 2 || out[1].Content != "done" {
		t.Fatalf("unexpected result: %#v", out)
	}
	if len(model.invocations) != 1 {
		t.Fatalf("expected 1 invocation, got %d", len(model.invocations))
	}
	invoked := model.invocations[0]
	if len(invoked) != 2 || invoked[0].Role != messages.RoleSystem || invoked[0].Content != "you are a helpful assistant." {
		t.Fatalf("expected leading dynamic system message, got %#v", invoked)
	}
}

// TestDynamicPromptOverwritesSystemPrompt mirrors
// test_dynamic_prompt_overwrites_system_prompt (test_decorators.py:740).
func TestDynamicPromptOverwritesSystemPrompt(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	hook := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return "Overridden prompt.", nil
	})
	agent, err := CreateAgent(model, nil,
		WithAgentSystemPrompt("Original static prompt"),
		WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Hello")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	invoked := model.invocations[0]
	if len(invoked) != 2 || invoked[0].Role != messages.RoleSystem || invoked[0].Content != "Overridden prompt." {
		t.Fatalf("expected overridden system message, got %#v", invoked)
	}
}

// TestDynamicPromptLastInChainWins mirrors
// test_dynamic_prompt_multiple_in_sequence (test_decorators.py:758): the
// middleware closest to the model (last in the list) applies last and wins.
func TestDynamicPromptLastInChainWins(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("done")}}
	first := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return "First prompt.", nil
	})
	second := DynamicPrompt(func(_ context.Context, _ middleware.ModelRequest) (any, error) {
		return "Second prompt.", nil
	})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(first, second))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if _, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Hello")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	invoked := model.invocations[0]
	if len(invoked) != 2 || invoked[0].Content != "Second prompt." {
		t.Fatalf("expected last-chain prompt to win, got %#v", invoked)
	}
}
