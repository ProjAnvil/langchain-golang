package agents

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// TestFuncBeforeModelWithConfigDeclaresJumpTargets mirrors
// test_can_jump_to_with_before_model_decorator (test_decorators.py:214).
func TestFuncBeforeModelWithConfigDeclaresJumpTargets(t *testing.T) {
	hook := FuncBeforeModelWithConfig(func(_ context.Context, _ map[string]any) (map[string]any, error) {
		return nil, nil
	}, HookConfig{CanJumpTo: []string{"end"}})
	if got := DeclaredCanJumpTo(hook, "before_model"); !reflect.DeepEqual(got, []string{"end"}) {
		t.Fatalf("before_model can_jump_to = %#v, want [end]", got)
	}
	if got := DeclaredCanJumpTo(hook, "after_model"); got != nil {
		t.Fatalf("after_model can_jump_to = %#v, want nil", got)
	}
	// The wrapped function still runs.
	out, err := hook.BeforeModel(context.Background(), map[string]any{})
	if err != nil || out != nil {
		t.Fatalf("BeforeModel: err=%v out=%#v", err, out)
	}
}

// TestFuncAfterModelWithConfigDeclaresJumpTargets mirrors
// test_can_jump_to_with_after_model_decorator (test_decorators.py:230).
func TestFuncAfterModelWithConfigDeclaresJumpTargets(t *testing.T) {
	hook := FuncAfterModelWithConfig(func(_ context.Context, _ map[string]any) (map[string]any, error) {
		return nil, nil
	}, HookConfig{CanJumpTo: []string{"model", "end"}})
	if got := DeclaredCanJumpTo(hook, "after_model"); !reflect.DeepEqual(got, []string{"model", "end"}) {
		t.Fatalf("after_model can_jump_to = %#v, want [model end]", got)
	}
	if got := DeclaredCanJumpTo(hook, "before_model"); got != nil {
		t.Fatalf("before_model can_jump_to = %#v, want nil", got)
	}
	out, err := hook.AfterModel(context.Background(), map[string]any{})
	if err != nil || out != nil {
		t.Fatalf("AfterModel: err=%v out=%#v", err, out)
	}
}

// jumpConfigMiddleware mirrors the class-based declaration in
// test_hook_config_decorator_on_class_method (test_decorators.py:189).
type jumpConfigMiddleware struct{}

func (jumpConfigMiddleware) BeforeModel(_ context.Context, _ map[string]any) (map[string]any, error) {
	return nil, nil
}

func (jumpConfigMiddleware) AfterModel(_ context.Context, _ map[string]any) (map[string]any, error) {
	return map[string]any{"jump_to": "tools"}, nil
}

func (jumpConfigMiddleware) CanJumpTo(hookName string) []string {
	switch hookName {
	case "before_model":
		return []string{"end", "model"}
	case "after_model":
		return []string{"tools"}
	default:
		return nil
	}
}

// TestHookConfigOnStructMiddleware mirrors
// test_hook_config_decorator_on_class_method (test_decorators.py:189).
func TestHookConfigOnStructMiddleware(t *testing.T) {
	var mw any = jumpConfigMiddleware{}
	if got := DeclaredCanJumpTo(mw, "before_model"); !reflect.DeepEqual(got, []string{"end", "model"}) {
		t.Fatalf("before_model can_jump_to = %#v, want [end model]", got)
	}
	if got := DeclaredCanJumpTo(mw, "after_model"); !reflect.DeepEqual(got, []string{"tools"}) {
		t.Fatalf("after_model can_jump_to = %#v, want [tools]", got)
	}
	if got := DeclaredCanJumpTo(mw, "wrap_model_call"); got != nil {
		t.Fatalf("unknown hook can_jump_to = %#v, want nil", got)
	}
}

// TestDeclaredCanJumpToNoFalsePositives mirrors
// test_get_can_jump_to_no_false_positives (test_decorators.py:511): middleware
// without a declaration returns nil.
func TestDeclaredCanJumpToNoFalsePositives(t *testing.T) {
	hook := FuncBeforeModel(func(_ context.Context, _ map[string]any) (map[string]any, error) {
		return nil, nil
	})
	if got := DeclaredCanJumpTo(hook, "before_model"); got != nil {
		t.Fatalf("undeclared hook can_jump_to = %#v, want nil", got)
	}
	if got := DeclaredCanJumpTo(struct{}{}, "before_model"); got != nil {
		t.Fatalf("non-middleware can_jump_to = %#v, want nil", got)
	}
}

// TestCanJumpToIntegration mirrors test_can_jump_to_integration
// (test_decorators.py:249): a before_model hook returning {"jump_to": "end"}
// short-circuits the run before the model call.
func TestCanJumpToIntegration(t *testing.T) {
	model := &sequenceModel{responses: []messages.Message{messages.AI("reply")}}
	calls := 0
	hook := FuncBeforeModelWithConfig(func(_ context.Context, state map[string]any) (map[string]any, error) {
		calls++
		msgs, _ := state["messages"].([]messages.Message)
		if len(msgs) > 0 && msgs[0].Content == "exit" {
			return map[string]any{"jump_to": "end"}, nil
		}
		return nil, nil
	}, HookConfig{CanJumpTo: []string{"end"}})
	agent, err := CreateAgent(model, nil, WithAgentMiddleware(hook))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}

	// Early exit: the model never runs, only the human message remains.
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("exit")})
	if err != nil {
		t.Fatalf("invoke (exit): %v", err)
	}
	if calls != 1 || len(out) != 1 {
		t.Fatalf("early-exit mismatch: calls=%d out=%#v", calls, out)
	}

	// Normal path: the model runs and appends its reply.
	out, err = agent.Invoke(context.Background(), []messages.Message{messages.Human("hello")})
	if err != nil {
		t.Fatalf("invoke (hello): %v", err)
	}
	if calls != 2 || len(out) != 2 || out[1].Content != "reply" {
		t.Fatalf("normal-path mismatch: calls=%d out=%#v", calls, out)
	}
}

// TestCreateAgentRejectsInvalidDeclaredJumpTarget enforces Python's JumpTo
// literal ("tools" | "model" | "end") on declared targets at build time.
func TestCreateAgentRejectsInvalidDeclaredJumpTarget(t *testing.T) {
	hook := FuncBeforeModelWithConfig(func(_ context.Context, _ map[string]any) (map[string]any, error) {
		return nil, nil
	}, HookConfig{CanJumpTo: []string{"sideways"}})
	_, err := CreateAgent(&sequenceModel{responses: []messages.Message{messages.AI("x")}}, nil, WithAgentMiddleware(hook))
	if err == nil || !strings.Contains(err.Error(), `invalid can_jump_to target "sideways"`) {
		t.Fatalf("expected invalid can_jump_to error, got %v", err)
	}
}

// TestCreateAgentAcceptsValidDeclaredJumpTargets covers the valid-target
// branches of validateDeclaredJumpTargets for both hook names.
func TestCreateAgentAcceptsValidDeclaredJumpTargets(t *testing.T) {
	before := FuncBeforeModelWithConfig(func(_ context.Context, _ map[string]any) (map[string]any, error) {
		return nil, nil
	}, HookConfig{CanJumpTo: []string{"model", "tools", "end"}})
	after := FuncAfterModelWithConfig(func(_ context.Context, _ map[string]any) (map[string]any, error) {
		return nil, nil
	}, HookConfig{CanJumpTo: []string{"model", "end"}})
	agent, err := CreateAgent(&sequenceModel{responses: []messages.Message{messages.AI("ok")}}, nil,
		WithAgentMiddleware(before, after))
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	out, err := agent.Invoke(context.Background(), []messages.Message{messages.Human("hi")})
	if err != nil || len(out) != 2 || out[1].Content != "ok" {
		t.Fatalf("invoke: err=%v out=%#v", err, out)
	}
}
