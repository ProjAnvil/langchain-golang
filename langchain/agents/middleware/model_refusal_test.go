package middleware_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/langchain/agents"
	middleware "github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// Ports langchain tests/unit_tests/agents/middleware/implementations/
// test_model_refusal.py（164 行，#41582 + #41689）。

func refusedHandler(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "Cannot answer."}}
	return middleware.ModelResponse{Result: []messages.Message{msg}}, nil
}

func cleanHandler(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
	return middleware.ModelResponse{Result: []messages.Message{messages.AI("fine")}}, nil
}

func TestModelRefusalMiddlewareRaises(t *testing.T) {
	mw := middleware.NewModelRefusalMiddleware()
	_, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, refusedHandler)
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError, got %v", err)
	}
}

func TestModelRefusalMiddlewarePassesClean(t *testing.T) {
	mw := middleware.NewModelRefusalMiddleware()
	resp, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, cleanHandler)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Result) != 1 {
		t.Fatalf("response lost: %+v", resp)
	}
}

// refusal 藏在 StructuredOutputValidationError 的 AI 消息里也要抛出（上游
// wrap_model_call 先查 exc.ai_message 再 re-raise 的分支）。
func TestModelRefusalMiddlewareChecksValidationErrorMessage(t *testing.T) {
	refused := messages.AI("")
	refused.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: "no"}}
	handler := func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		return middleware.ModelResponse{}, agents.NewStructuredOutputValidationError("out", errors.New("parse fail"), refused)
	}
	mw := middleware.NewModelRefusalMiddleware()
	_, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, handler)
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("want ModelRefusalError from validation error's message, got %v", err)
	}
}

// 校验错误里的消息不含 refusal 时，原错误原样传播。
func TestModelRefusalMiddlewareReraisesUnrelatedValidationError(t *testing.T) {
	clean := messages.AI("not a refusal")
	handler := func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		return middleware.ModelResponse{}, agents.NewStructuredOutputValidationError("out", errors.New("parse fail"), clean)
	}
	mw := middleware.NewModelRefusalMiddleware()
	_, err := mw.WrapModelCall(context.Background(), middleware.ModelRequest{}, handler)
	var ve *agents.StructuredOutputValidationError
	if !errors.As(err, &ve) {
		t.Fatalf("unrelated validation error must propagate, got %v", err)
	}
}

// #41689：trace policy omit inputs。
func TestModelRefusalMiddlewareTracePolicyOmitsInputs(t *testing.T) {
	mw := middleware.NewModelRefusalMiddleware()
	policy := mw.TracePolicy()
	if policy.ProcessInputs == nil {
		t.Fatal("ProcessInputs must be installed (#41689)")
	}
	scrubbed := policy.ProcessInputs(map[string]any{"messages": []string{"secret"}})
	m, ok := scrubbed.(map[string]any)
	if !ok || len(m) == 0 {
		t.Fatalf("scrubbed inputs must be an omitted marker, got %v", scrubbed)
	}
	if m["messages"] != nil {
		t.Fatalf("inputs must be omitted, got %v", m)
	}
}

// #41689 端到端：fake tracer 断言 wrap_model_call 层内 inputs 被 omit。
// 复用 trace_policy_test.go 的 invokeTracePolicyAgent / tracePolicyChatStart
// fixture（同 package middleware_test）：模型正常应答 "done"，断言 tracer
// 收到的 chat_model_start 输入是 omitted marker 而非原始消息。
func TestModelRefusalMiddlewareTracePolicyOmitsInputsEndToEnd(t *testing.T) {
	model := &tracePolicyModel{response: messages.AI("done")}
	start := tracePolicyChatStart(t, invokeTracePolicyAgent(t, model, middleware.NewModelRefusalMiddleware()))
	payload, ok := start.Input.(map[string]any)
	if !ok {
		t.Fatalf("omitted Input = %#v, want the omitted marker map", start.Input)
	}
	if got := payload["omitted"]; got != true {
		t.Fatalf("omitted Input = %#v, want map[omitted:true]", payload)
	}
	if payload["messages"] != nil {
		t.Fatalf("inputs must be omitted, got %#v", payload)
	}
}
