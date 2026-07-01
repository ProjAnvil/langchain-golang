package runnables

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestPassthrough(t *testing.T) {
	called := false
	runnable := NewPassthrough[string](schema.String(""))
	runnable.OnInvoke = func(_ context.Context, input string, _ ...Option) error {
		called = input == "hello"
		return nil
	}

	got, err := runnable.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "hello" || !called {
		t.Fatalf("got=%q called=%v", got, called)
	}
}

func TestAssign(t *testing.T) {
	assign := NewAssign(map[string]Runnable[map[string]any, any]{
		"total": NewFunc(func(_ context.Context, input map[string]any, _ ...Option) (any, error) {
			return input["a"].(int) + input["b"].(int), nil
		}, schema.Schema{"type": "object"}, schema.Integer("")),
	})

	input := map[string]any{"a": 2, "b": 3}
	got, err := assign.Invoke(context.Background(), input)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got["total"] != 5 {
		t.Fatalf("total: %#v", got)
	}
	if _, ok := input["total"]; ok {
		t.Fatalf("assign mutated input: %#v", input)
	}
}

func TestAssignPropagatesChildConfig(t *testing.T) {
	seen := []Config{}
	assign := NewAssign(map[string]Runnable[map[string]any, any]{
		"total": configCaptureRunnable[map[string]any, any]{output: 5, seen: &seen},
	})

	got, err := assign.Invoke(
		context.Background(),
		map[string]any{"a": 2, "b": 3},
		WithRunID("root"),
		WithTags("parent"),
		WithMetadata("trace", "yes"),
		WithConfigurable("mode", "fast"),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got["total"] != 5 {
		t.Fatalf("got %#v", got)
	}
	assertChildConfig(t, seen[0], "assign:key:total")
}

func TestBranch(t *testing.T) {
	isEven := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input%2 == 0, nil
	}, schema.Integer(""), schema.Boolean(""))
	even := NewFunc(func(_ context.Context, input int, _ ...Option) (string, error) {
		return "even", nil
	}, schema.Integer(""), schema.String(""))
	odd := NewFunc(func(_ context.Context, input int, _ ...Option) (string, error) {
		return "odd", nil
	}, schema.Integer(""), schema.String(""))

	branch, err := NewBranch([]BranchCase[int, string]{{Condition: isEven, Runnable: even}}, odd)
	if err != nil {
		t.Fatalf("new branch: %v", err)
	}
	got, err := branch.Invoke(context.Background(), 4)
	if err != nil {
		t.Fatalf("invoke even: %v", err)
	}
	if got != "even" {
		t.Fatalf("even got %q", got)
	}
	got, err = branch.Invoke(context.Background(), 3)
	if err != nil {
		t.Fatalf("invoke odd: %v", err)
	}
	if got != "odd" {
		t.Fatalf("odd got %q", got)
	}
}

func TestBranchPropagatesConditionAndBranchConfig(t *testing.T) {
	seen := []Config{}
	condition := configCaptureRunnable[int, bool]{output: true, seen: &seen}
	selected := configCaptureRunnable[int, string]{output: "selected", seen: &seen}
	def := configCaptureRunnable[int, string]{output: "default", seen: &seen}
	branch, err := NewBranch([]BranchCase[int, string]{{Condition: condition, Runnable: selected}}, def)
	if err != nil {
		t.Fatalf("new branch: %v", err)
	}

	got, err := branch.Invoke(
		context.Background(),
		1,
		WithRunID("root"),
		WithTags("parent"),
		WithMetadata("trace", "yes"),
		WithConfigurable("mode", "fast"),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "selected" {
		t.Fatalf("got %q", got)
	}
	assertChildConfig(t, seen[0], "condition:1")
	assertChildConfig(t, seen[1], "branch:1")
}

func TestWithFallbacks(t *testing.T) {
	primaryErr := errors.New("primary failed")
	primary := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return "", primaryErr
	}, schema.String(""), schema.String(""))
	fallback := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "-fallback", nil
	}, schema.String(""), schema.String(""))

	runnable, err := NewWithFallbacks[string, string](primary, fallback)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), "ok")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "ok-fallback" {
		t.Fatalf("got %q", got)
	}
}

func TestWithFallbacksPropagatesAttemptConfig(t *testing.T) {
	seen := []Config{}
	primaryErr := errors.New("primary failed")
	primary := NewFunc(func(_ context.Context, _ string, opts ...Option) (string, error) {
		seen = append(seen, NewConfig(opts...))
		return "", primaryErr
	}, schema.String(""), schema.String(""))
	fallback := configCaptureRunnable[string, string]{output: "ok", seen: &seen}

	runnable, err := NewWithFallbacks[string, string](primary, fallback)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	got, err := runnable.Invoke(
		context.Background(),
		"input",
		WithRunID("root"),
		WithTags("parent"),
		WithMetadata("trace", "yes"),
		WithConfigurable("mode", "fast"),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "ok" {
		t.Fatalf("got %q", got)
	}
	assertChildConfig(t, seen[0], "fallback:primary")
	assertChildConfig(t, seen[1], "fallback:1")
}

func TestWithFallbacksReturnsFirstError(t *testing.T) {
	primaryErr := errors.New("primary failed")
	primary := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return "", primaryErr
	}, schema.String(""), schema.String(""))

	runnable, err := NewWithFallbacks[string, string](primary)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	_, err = runnable.Invoke(context.Background(), "ok")
	if !errors.Is(err, primaryErr) {
		t.Fatalf("err: got %v want %v", err, primaryErr)
	}
}
