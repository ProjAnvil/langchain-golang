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

func TestPassthroughOnInvokeError(t *testing.T) {
	runnable := NewPassthrough[string](schema.String(""))
	runnable.OnInvoke = func(context.Context, string, ...Option) error {
		return errTestSentinel
	}

	if _, err := runnable.Invoke(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
	if _, err := runnable.Stream(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
}

func TestPassthroughBatch(t *testing.T) {
	runnable := NewPassthrough[string](schema.String(""))
	got, err := runnable.Batch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("got %#v", got)
	}

	calls := 0
	runnable.OnInvoke = func(_ context.Context, input string, _ ...Option) error {
		calls++
		if input == "bad" {
			return errTestSentinel
		}
		return nil
	}
	got, err = runnable.Batch(context.Background(), []string{"ok", "bad"})
	if err == nil {
		t.Fatal("expected joined hook error")
	}
	if calls != 2 {
		t.Fatalf("hook calls: %d", calls)
	}
	if len(got) != 2 || got[0] != "ok" || got[1] != "bad" {
		t.Fatalf("batch must still return the inputs: %#v", got)
	}
}

func TestPassthroughStreamAndSchemas(t *testing.T) {
	runnable := NewPassthrough[int](schema.Integer("number"))
	stream, err := runnable.Stream(context.Background(), 7)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	got, ok, err := stream.Next(context.Background())
	if err != nil || !ok || got != 7 {
		t.Fatalf("next got=%d ok=%v err=%v", got, ok, err)
	}

	if runnable.InputSchema()["description"] != "number" {
		t.Fatalf("input schema: %#v", runnable.InputSchema())
	}
	if runnable.OutputSchema()["description"] != "number" {
		t.Fatalf("output schema: %#v", runnable.OutputSchema())
	}
	if _, ok := configurableSchema(runnable.ConfigSchema()); ok {
		t.Fatalf("expected empty config schema, got %#v", runnable.ConfigSchema())
	}
}

func TestAssignBatchStreamAndSchemas(t *testing.T) {
	assign := NewAssign(map[string]Runnable[map[string]any, any]{
		"double": NewFunc(func(_ context.Context, input map[string]any, _ ...Option) (any, error) {
			return input["n"].(int) * 2, nil
		}, schema.Schema{"type": "object"}, schema.Integer("")),
	})

	got, err := assign.Batch(context.Background(), []map[string]any{{"n": 2}, {"n": 3}})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if got[0]["double"] != 4 || got[1]["double"] != 6 {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := assign.Stream(context.Background(), map[string]any{"n": 5})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk["double"] != 10 {
		t.Fatalf("chunk=%#v ok=%v err=%v", chunk, ok, err)
	}

	if assign.InputSchema()["type"] != "object" || assign.OutputSchema()["type"] != "object" {
		t.Fatalf("schemas: in=%#v out=%#v", assign.InputSchema(), assign.OutputSchema())
	}
	if assign.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", assign.ConfigSchema())
	}
}

func TestAssignStepError(t *testing.T) {
	assign := NewAssign(map[string]Runnable[map[string]any, any]{
		"fail": NewFunc(func(context.Context, map[string]any, ...Option) (any, error) {
			return nil, errTestSentinel
		}, schema.Schema{}, schema.Schema{}),
	})
	if _, err := assign.Invoke(context.Background(), map[string]any{}); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
	if _, err := assign.Stream(context.Background(), map[string]any{}); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
	if _, err := assign.Batch(context.Background(), []map[string]any{{}}); err == nil {
		t.Fatal("expected batch error")
	}
}

func TestNewBranchErrors(t *testing.T) {
	def := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input, nil
	}, schema.Integer(""), schema.Integer(""))

	if _, err := NewBranch[int, int](nil, def); err == nil {
		t.Fatal("expected error for empty cases")
	}
	cond := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return true, nil
	}, schema.Integer(""), schema.Boolean(""))
	if _, err := NewBranch[int, int]([]BranchCase[int, int]{{Condition: cond, Runnable: def}}, nil); err == nil {
		t.Fatal("expected error for nil default")
	}
}

func TestBranchBatchStreamAndSchemas(t *testing.T) {
	isPositive := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input > 0, nil
	}, schema.Integer(""), schema.Boolean(""))
	positive := NewFunc(func(_ context.Context, input int, _ ...Option) (string, error) {
		return "positive", nil
	}, schema.Integer(""), schema.String(""))
	negative := NewFunc(func(_ context.Context, input int, _ ...Option) (string, error) {
		return "negative", nil
	}, schema.Integer(""), schema.String(""))

	branch, err := NewBranch([]BranchCase[int, string]{{Condition: isPositive, Runnable: positive}}, negative)
	if err != nil {
		t.Fatalf("new branch: %v", err)
	}

	got, err := branch.Batch(context.Background(), []int{1, -1})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if got[0] != "positive" || got[1] != "negative" {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := branch.Stream(context.Background(), 5)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk != "positive" {
		t.Fatalf("chunk=%q ok=%v err=%v", chunk, ok, err)
	}

	stream, err = branch.Stream(context.Background(), -5)
	if err != nil {
		t.Fatalf("default stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err = stream.Next(context.Background())
	if err != nil || !ok || chunk != "negative" {
		t.Fatalf("default chunk=%q ok=%v err=%v", chunk, ok, err)
	}

	if branch.InputSchema()["type"] != "integer" || branch.OutputSchema()["type"] != "string" {
		t.Fatalf("schemas: in=%#v out=%#v", branch.InputSchema(), branch.OutputSchema())
	}
	if branch.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", branch.ConfigSchema())
	}
}

func TestBranchConditionError(t *testing.T) {
	cond := NewFunc(func(context.Context, int, ...Option) (bool, error) {
		return false, errTestSentinel
	}, schema.Integer(""), schema.Boolean(""))
	body := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input, nil
	}, schema.Integer(""), schema.Integer(""))
	branch, err := NewBranch([]BranchCase[int, int]{{Condition: cond, Runnable: body}}, body)
	if err != nil {
		t.Fatalf("new branch: %v", err)
	}

	if _, err := branch.Invoke(context.Background(), 1); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
	if _, err := branch.Stream(context.Background(), 1); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
}

func TestNewWithFallbacksNilRunnable(t *testing.T) {
	if _, err := NewWithFallbacks[string, string](nil); err == nil {
		t.Fatal("expected error for nil primary runnable")
	}
}

func TestWithFallbacksBatchStreamAndSchemas(t *testing.T) {
	primary := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		if input == "bad" {
			return "", errTestSentinel
		}
		return input + "-primary", nil
	}, schema.String("primary in"), schema.String("primary out"))
	fallback := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "-fallback", nil
	}, schema.String(""), schema.String(""))

	runnable, err := NewWithFallbacks[string, string](primary, fallback)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}

	got, err := runnable.Batch(context.Background(), []string{"good", "bad"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if got[0] != "good-primary" || got[1] != "bad-fallback" {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := runnable.Stream(context.Background(), "bad")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk != "bad-fallback" {
		t.Fatalf("chunk=%q ok=%v err=%v", chunk, ok, err)
	}

	if runnable.InputSchema()["description"] != "primary in" {
		t.Fatalf("input schema: %#v", runnable.InputSchema())
	}
	if runnable.OutputSchema()["description"] != "primary out" {
		t.Fatalf("output schema: %#v", runnable.OutputSchema())
	}
	if runnable.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", runnable.ConfigSchema())
	}
}

func TestWithFallbacksStreamReturnsFirstError(t *testing.T) {
	failing := NewFunc(func(context.Context, string, ...Option) (string, error) {
		return "", errTestSentinel
	}, schema.String(""), schema.String(""))
	runnable, err := NewWithFallbacks[string, string](failing, failing)
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	if _, err := runnable.Stream(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
}

func TestAssignNilMapInput(t *testing.T) {
	assign := NewAssign(map[string]Runnable[map[string]any, any]{
		"x": NewFunc(func(_ context.Context, input map[string]any, _ ...Option) (any, error) {
			if len(input) != 0 {
				t.Fatalf("expected empty input map, got %#v", input)
			}
			return 1, nil
		}, schema.Schema{"type": "object"}, schema.Integer("")),
	})
	got, err := assign.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got["x"] != 1 {
		t.Fatalf("got %#v", got)
	}
}
