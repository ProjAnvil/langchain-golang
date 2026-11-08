package middleware_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
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

// --- 上游中间件联动用例（test_model_refusal.py, #41582）---

// refusalMessage 构造一条只带 refusal 块的 AI 消息。
func refusalMessage(text string) messages.Message {
	msg := messages.AI("")
	msg.ContentBlocks = []messages.ContentBlock{messages.RefusalBlock{Refusal: text}}
	return msg
}

// refusalCountingModel 恒返回 refusal 消息并记录 Invoke 次数。不用
// language.FakeChatModel：它的 BindTools 返回副本，计数无法共享给断言。
type refusalCountingModel struct {
	calls atomic.Int32
	msg   messages.Message
}

func (m *refusalCountingModel) Invoke(_ context.Context, _ []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	m.calls.Add(1)
	return m.msg, nil
}

func (m *refusalCountingModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *refusalCountingModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	m.calls.Add(1)
	return runnables.NewSliceStream([]messages.Message{m.msg}), nil
}

func (m *refusalCountingModel) InputSchema() schema.Schema  { return schema.Object(map[string]schema.Schema{}) }
func (m *refusalCountingModel) OutputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }
func (m *refusalCountingModel) BindTools([]coretools.Tool) (language.ChatModel, error) {
	return m, nil
}
func (m *refusalCountingModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{}
}

// 对应上游 test_fallback_can_handle_refusal：fallback 中间件在外层，能把
// refusal 错误落到 fallback 模型并拿到成功应答。
func TestFallbackCanHandleRefusal(t *testing.T) {
	primary := language.NewFakeChatModel(language.WithResponses(refusalMessage("blocked")))
	fallback := language.NewFakeChatModel(language.WithResponses(messages.AI("fallback answer")))
	agent, err := agents.CreateAgent(primary, nil, agents.WithAgentMiddleware(
		middleware.NewModelFallbackMiddleware(fallback),
		middleware.NewModelRefusalMiddleware(),
	))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	out, err := agent.Invoke(t.Context(), []messages.Message{messages.Human("request")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(out) == 0 || out[len(out)-1].Content != "fallback answer" {
		t.Fatalf("agent output = %#v, want the fallback model's answer", out)
	}
}

// 对应上游 test_retry_middleware_does_not_retry_refusal：retry 中间件在外层、
// max_retries=2、默认 on_failure=continue，模型恒 refusal。refusal 不可重试
// 且必须立即传播（Python: retry_on 拒绝时绕过 on_failure 直接 raise），所以
// 模型只被调用一次，调用方拿到 ModelRefusalError 而不是 continue 消息。
func TestRetryMiddlewareDoesNotRetryRefusal(t *testing.T) {
	model := &refusalCountingModel{msg: refusalMessage("blocked")}
	retry, err := middleware.NewModelRetryMiddleware(
		middleware.WithModelRetryMaxRetries(2),
		middleware.WithModelRetryBackoff(0, 0, 0, false),
		middleware.WithModelRetrySleep(func(time.Duration) {}),
	)
	if err != nil {
		t.Fatalf("NewModelRetryMiddleware: %v", err)
	}
	agent, err := agents.CreateAgent(model, nil, agents.WithAgentMiddleware(
		retry,
		middleware.NewModelRefusalMiddleware(),
	))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	_, err = agent.Invoke(t.Context(), []messages.Message{messages.Human("request")})
	var mre *lcerrors.ModelRefusalError
	if !errors.As(err, &mre) {
		t.Fatalf("Invoke error = %v, want ModelRefusalError", err)
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("model invoked %d times, want 1 (refusal must not be retried)", got)
	}
}

// 对应上游 test_stream_surfaces_refusal：流式路径下 refusal 块先以
// content-block-finish delta 透传给消费者，聚合完成后 ModelRefusalMiddleware
// 抛出，终止事件携带 ModelRefusalError。
func TestStreamSurfacesRefusal(t *testing.T) {
	model := language.NewFakeChatModel(language.WithStreamChunks(refusalMessage("blocked")))
	agent, err := agents.CreateAgent(model, nil, agents.WithAgentMiddleware(
		middleware.NewModelRefusalMiddleware(),
	))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	stream, err := agent.StreamEvents(t.Context(), []messages.Message{messages.Human("request")})
	if err != nil {
		t.Fatalf("StreamEvents: %v", err)
	}
	defer stream.Close()

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	sawRefusalDelta := false
	var endErr error
	for {
		ev, ok, err := stream.Next(ctx)
		if err != nil {
			t.Fatalf("stream Next: %v", err)
		}
		if !ok {
			break
		}
		// 消费者在错误抛出前就能看到 refusal 块（bridge 的
		// content-block-finish 事件以 model_delta 形式透传）。
		if ev.Type == agents.StreamModelDelta && ev.Delta != nil && ev.Delta.Content != nil {
			if messages.BlockToMap(ev.Delta.Content)["type"] == "refusal" {
				sawRefusalDelta = true
			}
		}
		if ev.Type == agents.StreamEnd {
			endErr = ev.Err
		}
	}
	if !sawRefusalDelta {
		t.Fatal("consumer never saw the refusal block delta")
	}
	var mre *lcerrors.ModelRefusalError
	if !errors.As(endErr, &mre) {
		t.Fatalf("terminal error = %v, want ModelRefusalError", endErr)
	}
}
