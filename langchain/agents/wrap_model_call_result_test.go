package agents

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// alwaysFailModel is a language.ChatModel whose Invoke always errors, used to
// exercise the error-to-AIMessage fallback path (mirrors AlwaysFailModel in
// the Python tests).
type alwaysFailModel struct{}

func (m *alwaysFailModel) Invoke(context.Context, []messages.Message, ...runnables.Option) (messages.Message, error) {
	return messages.Message{}, fmt.Errorf("model error")
}

func (m *alwaysFailModel) Batch(_ context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		msg, err := m.Invoke(context.Background(), in, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = msg
	}
	return out, nil
}

func (m *alwaysFailModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, fmt.Errorf("model error")
}

func (m *alwaysFailModel) InputSchema() schema.Schema  { return schema.Object(map[string]schema.Schema{}) }
func (m *alwaysFailModel) OutputSchema() schema.Schema { return schema.Object(map[string]schema.Schema{}) }

func (m *alwaysFailModel) BindTools([]coretools.Tool) (language.ChatModel, error) { return m, nil }

func (m *alwaysFailModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{ToolCalling: true}
}

// TestWrapModelCallResultUppercaseShortForm mirrors test_uppercase_response
// (tests/unit_tests/agents/middleware/core/test_wrap_model_call.py:299): the
// middleware returns a bare AIMessage instead of a ModelResponse.
func TestWrapModelCallResultUppercaseShortForm(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello world")}}
	hook := FuncWrapModelCallResult(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
		resp, err := handler(ctx, request)
		if err != nil {
			return nil, err
		}
		return messages.AI(strings.ToUpper(resp.Result[0].Content)), nil
	})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Test")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 2 || out[1].Content != "HELLO WORLD" {
		t.Fatalf("unexpected result: %#v", out)
	}
}

// TestWrapModelCallResultPrefixShortForm mirrors test_prefix_response
// (test_wrap_model_call.py:321).
func TestWrapModelCallResultPrefixShortForm(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("Response")}}
	hook := FuncWrapModelCallResult(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
		resp, err := handler(ctx, request)
		if err != nil {
			return nil, err
		}
		return messages.AI("[BOT]: " + resp.Result[0].Content), nil
	})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Test")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 2 || out[1].Content != "[BOT]: Response" {
		t.Fatalf("unexpected result: %#v", out)
	}
}

// TestWrapModelCallResultMultiStage mirrors test_multi_stage_transformation
// (test_wrap_model_call.py:346).
func TestWrapModelCallResultMultiStage(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("hello")}}
	hook := FuncWrapModelCallResult(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
		resp, err := handler(ctx, request)
		if err != nil {
			return nil, err
		}
		content := strings.ToUpper(resp.Result[0].Content)
		return messages.AI("[START] " + content + " [END]"), nil
	})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Test")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 2 || out[1].Content != "[START] HELLO [END]" {
		t.Fatalf("unexpected result: %#v", out)
	}
}

// TestWrapModelCallResultErrorToShortForm mirrors test_convert_error_to_response
// (test_wrap_model_call.py:377): a handler error is converted to an AIMessage
// fallback, and the run succeeds.
func TestWrapModelCallResultErrorToShortForm(t *testing.T) {
	hook := FuncWrapModelCallResult(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
		resp, err := handler(ctx, request)
		if err != nil {
			return messages.AI(fmt.Sprintf("Error occurred: %v. Using fallback response.", err)), nil
		}
		return resp, nil
	})
	agent, err := CreateAgent(&alwaysFailModel{}, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Test")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 2 || !strings.Contains(out[1].Content, "Error occurred") ||
		!strings.Contains(out[1].Content, "fallback response") {
		t.Fatalf("unexpected result: %#v", out)
	}
}

// TestWrapModelCallResultInvalidType pins the normalization error surfacing
// through Invoke when a hook returns an unsupported type.
func TestWrapModelCallResultInvalidType(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("ok")}}
	hook := FuncWrapModelCallResult(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
		return 42, nil
	})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	_, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("Test")})
	if err == nil || !strings.Contains(err.Error(), "unsupported ModelCallResult type") {
		t.Fatalf("expected normalization error, got %v", err)
	}
}

// bothHooksMiddleware implements both WrapModelCallHook and
// WrapModelCallResultHook; the Result variant takes precedence (documented on
// WrapModelCallResultHook).
type bothHooksMiddleware struct{}

func (bothHooksMiddleware) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	return middleware.ModelResponse{Result: []messages.Message{messages.AI("plain")}}, nil
}

func (bothHooksMiddleware) WrapModelCallResult(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
	return messages.AI("result-variant"), nil
}

// TestWrapModelCallResultPrecedenceOverPlainHook pins the both-implemented
// precedence rule.
func TestWrapModelCallResultPrecedenceOverPlainHook(t *testing.T) {
	agent, err := CreateAgent(&sequenceModel{responses: []messages.Message{messages.AI("model")}}, nil,
		WithAgentMiddleware(bothHooksMiddleware{}))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("Test")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(out) != 2 || out[1].Content != "result-variant" {
		t.Fatalf("unexpected result: %#v", out)
	}
}

// TestFuncWrapModelCallResultAdapter mirrors the Func* adapter tests in
// functional_test.go: the adapter delegates and passes the union through.
func TestFuncWrapModelCallResultAdapter(t *testing.T) {
	hook := FuncWrapModelCallResult(func(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelCallResult, error) {
		return handler(ctx, request)
	})
	result, err := hook.WrapModelCallResult(context.Background(), middleware.ModelRequest{}, func(_ context.Context, _ middleware.ModelRequest) (middleware.ModelResponse, error) {
		return middleware.ModelResponse{Result: []messages.Message{messages.AI("ok")}}, nil
	})
	if err != nil {
		t.Fatalf("WrapModelCallResult: %v", err)
	}
	resp, err := middleware.NormalizeModelCallResult(result)
	if err != nil || len(resp.Result) != 1 || resp.Result[0].Content != "ok" {
		t.Fatalf("normalized response: err=%v resp=%#v", err, resp)
	}
}
