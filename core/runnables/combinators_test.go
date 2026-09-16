package runnables

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestBindCallTimeConfigWins(t *testing.T) {
	seen := []Config{}
	inner := NewFunc(func(_ context.Context, _ string, opts ...Option) (string, error) {
		seen = append(seen, NewConfig(opts...))
		return "ok", nil
	}, schema.String(""), schema.String(""))

	bound, err := Bind[string, string](inner,
		WithMaxConcurrency(5),
		WithMetadata("trace", "bound"),
		WithConfigurable("mode", "bound"),
		WithTags("bound"),
	)
	if err != nil {
		t.Fatalf("bind: %v", err)
	}

	_, err = bound.Invoke(t.Context(), "x",
		WithMaxConcurrency(1),
		WithMetadata("trace", "call"),
		WithConfigurable("mode", "call"),
		WithTags("call"),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(seen) != 1 {
		t.Fatalf("inner invocations: %d", len(seen))
	}
	cfg := seen[0]
	if cfg.MaxConcurrency != 1 {
		t.Fatalf("call-time MaxConcurrency must win: got %d want 1", cfg.MaxConcurrency)
	}
	if cfg.Metadata["trace"] != "call" {
		t.Fatalf("call-time metadata must win: %#v", cfg.Metadata)
	}
	if cfg.Configurable["mode"] != "call" {
		t.Fatalf("call-time configurable must win: %#v", cfg.Configurable)
	}
	if !reflect.DeepEqual(cfg.Tags, []string{"bound", "call"}) {
		t.Fatalf("tags must accumulate: %#v", cfg.Tags)
	}

	// Without call-site overrides the bound settings still apply.
	_, err = bound.Invoke(t.Context(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	cfg = seen[1]
	if cfg.MaxConcurrency != 5 || cfg.Metadata["trace"] != "bound" || cfg.Configurable["mode"] != "bound" {
		t.Fatalf("bound settings must apply when the call site is silent: %#v", cfg)
	}
	if !reflect.DeepEqual(cfg.Tags, []string{"bound"}) {
		t.Fatalf("tags: %#v", cfg.Tags)
	}
}

func TestBindBatchAndStreamDelegate(t *testing.T) {
	inner := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		return n * 2, nil
	}, schema.Integer(""), schema.Integer(""))

	bound, err := Bind[int, int](inner)
	if err != nil {
		t.Fatalf("bind: %v", err)
	}

	got, err := bound.Batch(t.Context(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if !reflect.DeepEqual(got, []int{2, 4, 6}) {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := bound.Stream(t.Context(), 21)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(t.Context())
	if err != nil || !ok || chunk != 42 {
		t.Fatalf("chunk=%d ok=%v err=%v", chunk, ok, err)
	}

	if bound.InputSchema()["type"] != "integer" || bound.OutputSchema()["type"] != "integer" {
		t.Fatalf("schemas: in=%#v out=%#v", bound.InputSchema(), bound.OutputSchema())
	}
}

func TestBindChainedBindMerges(t *testing.T) {
	seen := []Config{}
	inner := NewFunc(func(_ context.Context, _ string, opts ...Option) (string, error) {
		seen = append(seen, NewConfig(opts...))
		return "ok", nil
	}, schema.String(""), schema.String(""))

	bound, err := Bind[string, string](inner, WithMaxConcurrency(2))
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	chained := bound.Bind(WithMaxConcurrency(7))

	if _, err := chained.Invoke(t.Context(), "x"); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if seen[0].MaxConcurrency != 7 {
		t.Fatalf("later bind must win: got %d want 7", seen[0].MaxConcurrency)
	}
	// The original binding is unchanged.
	if _, err := bound.Invoke(t.Context(), "x"); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if seen[1].MaxConcurrency != 2 {
		t.Fatalf("original bind must be intact: got %d want 2", seen[1].MaxConcurrency)
	}
}

func TestBindNilRunnable(t *testing.T) {
	if _, err := Bind[string, string](nil); err == nil {
		t.Fatal("expected error for nil runnable")
	}
}

func TestPickSubset(t *testing.T) {
	inner := NewFunc(func(_ context.Context, _ string, _ ...Option) (map[string]any, error) {
		return map[string]any{"a": 1, "b": 2, "c": 3}, nil
	}, schema.String(""), schema.Schema{"type": "object"})

	pick, err := NewPick[string, map[string]any](inner, "a", "c")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	got, err := pick.Invoke(t.Context(), "in")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"a": 1, "c": 3}) {
		t.Fatalf("got %#v", got)
	}
}

func TestPickSingleKeyReturnsValue(t *testing.T) {
	inner := NewFunc(func(_ context.Context, _ string, _ ...Option) (map[string]any, error) {
		return map[string]any{"a": 1, "answer": "42"}, nil
	}, schema.String(""), schema.Schema{"type": "object"})

	pick, err := NewPick[string, any](inner, "answer")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	got, err := pick.Invoke(t.Context(), "in")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "42" {
		t.Fatalf("got %#v", got)
	}
}

func TestPickNoKeysIsIdentity(t *testing.T) {
	inner := NewFunc(func(_ context.Context, _ string, _ ...Option) (map[string]any, error) {
		return map[string]any{"a": 1}, nil
	}, schema.String(""), schema.Schema{"type": "object"})

	pick, err := NewPick[string, map[string]any](inner)
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	got, err := pick.Invoke(t.Context(), "in")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"a": 1}) {
		t.Fatalf("got %#v", got)
	}
}

func TestPickMissingKeyErrors(t *testing.T) {
	inner := NewFunc(func(_ context.Context, _ string, _ ...Option) (map[string]any, error) {
		return map[string]any{"a": 1}, nil
	}, schema.String(""), schema.Schema{"type": "object"})

	single, err := NewPick[string, any](inner, "missing")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	if _, err := single.Invoke(t.Context(), "in"); err == nil {
		t.Fatal("expected error for missing single key")
	}

	multi, err := NewPick[string, map[string]any](inner, "a", "missing")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}
	if _, err := multi.Batch(t.Context(), []string{"in", "in"}); err == nil {
		t.Fatal("expected error for missing key in batch")
	}
}

func TestPickBatchStreamAndSchemas(t *testing.T) {
	inner := NewFunc(func(_ context.Context, _ string, _ ...Option) (map[string]any, error) {
		return map[string]any{"a": 1, "b": 2}, nil
	}, schema.String("pick in"), schema.Object(map[string]schema.Schema{
		"a": schema.Integer("a"),
		"b": schema.Integer("b"),
	}))

	pick, err := NewPick[string, map[string]any](inner, "b", "a")
	if err != nil {
		t.Fatalf("new pick: %v", err)
	}

	got, err := pick.Batch(t.Context(), []string{"x", "y"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if !reflect.DeepEqual(got, []map[string]any{{"a": 1, "b": 2}, {"a": 1, "b": 2}}) {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := pick.Stream(t.Context(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(t.Context())
	if err != nil || !ok || !reflect.DeepEqual(chunk, map[string]any{"a": 1, "b": 2}) {
		t.Fatalf("chunk=%#v ok=%v err=%v", chunk, ok, err)
	}

	if pick.InputSchema()["description"] != "pick in" {
		t.Fatalf("input schema: %#v", pick.InputSchema())
	}
	out := pick.OutputSchema()
	props := schemaProperties(out)
	if _, ok := props["b"]; !ok {
		t.Fatalf("output schema should project picked keys: %#v", out)
	}
}

func TestPickNilRunnable(t *testing.T) {
	if _, err := NewPick[string, map[string]any](nil, "a"); err == nil {
		t.Fatal("expected error for nil runnable")
	}
}

func TestEachInvokesEveryElementInOrder(t *testing.T) {
	double := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		return n * 2, nil
	}, schema.Integer(""), schema.Integer(""))

	each, err := NewEach[int, int](double)
	if err != nil {
		t.Fatalf("new each: %v", err)
	}

	got, err := each.Invoke(t.Context(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !reflect.DeepEqual(got, []int{2, 4, 6}) {
		t.Fatalf("got %#v", got)
	}
}

func TestEachRunsElementsConcurrently(t *testing.T) {
	var inFlight atomic.Int32
	release := make(chan struct{})
	blocked := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		inFlight.Add(1)
		<-release
		inFlight.Add(-1)
		return n, nil
	}, schema.Integer(""), schema.Integer(""))

	each, err := NewEach[int, int](blocked)
	if err != nil {
		t.Fatalf("new each: %v", err)
	}

	type result struct {
		out []int
		err error
	}
	done := make(chan result, 1)
	go func() {
		out, err := each.Invoke(t.Context(), []int{1, 2, 3, 4}, WithMaxConcurrency(4))
		done <- result{out, err}
	}()

	// Sequential execution would deadlock on the barrier below.
	deadline := time.After(5 * time.Second)
	for {
		if inFlight.Load() == 4 {
			break
		}
		select {
		case <-deadline:
			close(release)
			t.Fatalf("elements did not run concurrently: inFlight=%d", inFlight.Load())
		case <-time.After(time.Millisecond):
		}
	}
	close(release)

	select {
	case res := <-done:
		if res.err != nil {
			t.Fatalf("invoke: %v", res.err)
		}
		if !reflect.DeepEqual(res.out, []int{1, 2, 3, 4}) {
			t.Fatalf("got %#v", res.out)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("invoke did not finish after release")
	}
}

func TestEachHonorsMaxConcurrency(t *testing.T) {
	var inFlight, peak atomic.Int32
	var mu sync.Mutex
	slow := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		cur := inFlight.Add(1)
		mu.Lock()
		if cur > peak.Load() {
			peak.Store(cur)
		}
		mu.Unlock()
		time.Sleep(2 * time.Millisecond)
		inFlight.Add(-1)
		return n, nil
	}, schema.Integer(""), schema.Integer(""))

	each, err := NewEach[int, int](slow)
	if err != nil {
		t.Fatalf("new each: %v", err)
	}
	inputs := make([]int, 24)
	for i := range inputs {
		inputs[i] = i
	}
	got, err := each.Invoke(t.Context(), inputs, WithMaxConcurrency(3))
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if len(got) != len(inputs) {
		t.Fatalf("len(got)=%d want %d", len(got), len(inputs))
	}
	if peak.Load() > 3 {
		t.Fatalf("peak concurrency = %d, want <= 3", peak.Load())
	}
}

func TestEachPropagatesElementError(t *testing.T) {
	calls := atomic.Int32{}
	flaky := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		calls.Add(1)
		if n == 1 {
			return 0, errTestSentinel
		}
		return n, nil
	}, schema.Integer(""), schema.Integer(""))

	each, err := NewEach[int, int](flaky)
	if err != nil {
		t.Fatalf("new each: %v", err)
	}
	_, err = each.Invoke(t.Context(), []int{0, 1, 2})
	if err == nil || !errors.Is(err, errTestSentinel) {
		t.Fatalf("err: got %v want %v inside", err, errTestSentinel)
	}
	if calls.Load() != 3 {
		t.Fatalf("calls = %d, want 3 (other elements still run)", calls.Load())
	}
}

func TestEachBatchStreamAndSchemas(t *testing.T) {
	double := NewFunc(func(_ context.Context, n int, _ ...Option) (int, error) {
		return n * 2, nil
	}, schema.Integer("each in"), schema.Integer("each out"))

	each, err := NewEach[int, int](double)
	if err != nil {
		t.Fatalf("new each: %v", err)
	}

	got, err := each.Batch(t.Context(), [][]int{{1, 2}, {3}})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if !reflect.DeepEqual(got, [][]int{{2, 4}, {6}}) {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := each.Stream(t.Context(), []int{5, 6})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(t.Context())
	if err != nil || !ok || !reflect.DeepEqual(chunk, []int{10, 12}) {
		t.Fatalf("chunk=%#v ok=%v err=%v", chunk, ok, err)
	}

	if each.InputSchema()["type"] != "array" || each.OutputSchema()["type"] != "array" {
		t.Fatalf("schemas: in=%#v out=%#v", each.InputSchema(), each.OutputSchema())
	}
}

func TestEachNilRunnable(t *testing.T) {
	if _, err := NewEach[int, int](nil); err == nil {
		t.Fatal("expected error for nil runnable")
	}
}
