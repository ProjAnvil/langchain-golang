package runnables

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestRetry(t *testing.T) {
	attempts := 0
	runnable := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		attempts++
		if attempts < 3 {
			return "", errors.New("temporary")
		}
		return input + "-ok", nil
	}, schema.String(""), schema.String(""))

	retrying, err := NewRetry[string, string](runnable, 3)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	got, err := retrying.Invoke(context.Background(), "value")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "value-ok" || attempts != 3 {
		t.Fatalf("got=%q attempts=%d", got, attempts)
	}
}

func TestRouter(t *testing.T) {
	add := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input + 1, nil
	}, schema.Integer(""), schema.Integer(""))
	double := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input * 2, nil
	}, schema.Integer(""), schema.Integer(""))

	router := NewRouter(map[string]Runnable[int, int]{
		"add":    add,
		"double": double,
	})
	got, err := router.Invoke(context.Background(), RouterInput[int]{Key: "double", Input: 4})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != 8 {
		t.Fatalf("got %d", got)
	}
	_, err = router.Invoke(context.Background(), RouterInput[int]{Key: "missing", Input: 4})
	if err == nil {
		t.Fatal("expected missing route error")
	}
}

func TestConfigurableAlternatives(t *testing.T) {
	add := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input + 1, nil
	}, schema.Integer(""), schema.Integer(""))
	double := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input * 2, nil
	}, schema.Integer(""), schema.Integer(""))

	runnable, err := NewConfigurableAlternatives[int, int](
		"mode",
		"add",
		add,
		map[string]Runnable[int, int]{"double": double},
	)
	if err != nil {
		t.Fatalf("new alternatives: %v", err)
	}

	got, err := runnable.Invoke(context.Background(), 4)
	if err != nil {
		t.Fatalf("default invoke: %v", err)
	}
	if got != 5 {
		t.Fatalf("default got %d", got)
	}

	got, err = runnable.Invoke(context.Background(), 4, WithConfigurable("mode", "double"))
	if err != nil {
		t.Fatalf("configured invoke: %v", err)
	}
	if got != 8 {
		t.Fatalf("configured got %d", got)
	}

	batch, err := runnable.Batch(context.Background(), []int{2, 3}, WithConfigurable("mode", "double"))
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if len(batch) != 2 || batch[0] != 4 || batch[1] != 6 {
		t.Fatalf("batch got %#v", batch)
	}
}

func TestConfigurableAlternativesStreamAndUnknownKey(t *testing.T) {
	defaultRunnable := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "-default", nil
	}, schema.String(""), schema.String(""))
	streamRunnable := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "-stream", nil
	}, schema.String(""), schema.String(""))

	runnable, err := NewConfigurableAlternatives[string, string](
		"model",
		"default",
		defaultRunnable,
		map[string]Runnable[string, string]{"stream": streamRunnable},
	)
	if err != nil {
		t.Fatalf("new alternatives: %v", err)
	}

	stream, err := runnable.Stream(context.Background(), "x", WithConfigurable("model", "stream"))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	got, ok, err := stream.Next(context.Background())
	if err != nil || !ok || got != "x-stream" {
		t.Fatalf("next got=%q ok=%v err=%v", got, ok, err)
	}

	_, err = runnable.Invoke(context.Background(), "x", WithConfigurable("model", "missing"))
	if err == nil {
		t.Fatal("expected unknown alternative error")
	}
}

func TestSequence(t *testing.T) {
	toLen := NewFunc(func(_ context.Context, input string, _ ...Option) (int, error) {
		return len(input), nil
	}, schema.String(""), schema.Integer(""))
	isEven := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input%2 == 0, nil
	}, schema.Integer(""), schema.Boolean(""))

	seq, err := NewSequence[string, int, bool](toLen, isEven)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	got, err := seq.Invoke(context.Background(), "four")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if !got {
		t.Fatalf("got false")
	}
}

func TestSequencePropagatesChildConfig(t *testing.T) {
	seen := []Config{}
	first := configCaptureRunnable[string, int]{output: 3, seen: &seen}
	second := configCaptureRunnable[int, string]{output: "done", seen: &seen}
	seq, err := NewSequence[string, int, string](first, second)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}

	got, err := seq.Invoke(
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
	if got != "done" {
		t.Fatalf("got %q", got)
	}
	assertChildConfig(t, seen[0], "seq:step:1")
	assertChildConfig(t, seen[1], "seq:step:2")
}

func TestSequenceStreamFlattensFirstAndSecondStreams(t *testing.T) {
	first := intStreamingRunnable{
		invoke: 1,
		stream: []int{2, 3},
	}
	seq, err := NewSequence[string, int, int](first, secondWithStream{})
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	stream, err := seq.Stream(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	got := collectStreamValues(t, stream)
	want := []int{2, 20, 3, 30}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream got %#v want %#v", got, want)
	}
}

func TestParallel(t *testing.T) {
	length := NewFunc(func(_ context.Context, input string, _ ...Option) (any, error) {
		return len(input), nil
	}, schema.String(""), schema.Integer(""))
	upper := NewFunc(func(_ context.Context, input string, _ ...Option) (any, error) {
		return input + "!", nil
	}, schema.String(""), schema.String(""))

	parallel := NewParallel(map[string]Runnable[string, any]{
		"length": length,
		"text":   upper,
	})
	got, err := parallel.Invoke(context.Background(), "go")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got["length"] != 2 || got["text"] != "go!" {
		t.Fatalf("got %#v", got)
	}
}

func TestParallelPropagatesChildConfigByKey(t *testing.T) {
	seen := map[string]Config{}
	parallel := NewParallel(map[string]Runnable[string, any]{
		"a": configCaptureRunnable[string, any]{output: 1, byName: seen},
		"b": configCaptureRunnable[string, any]{output: 2, byName: seen},
	})

	_, err := parallel.Invoke(
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
	assertChildConfig(t, seen["map:key:a"], "map:key:a")
	assertChildConfig(t, seen["map:key:b"], "map:key:b")
}

func TestRouterPropagatesRouteConfig(t *testing.T) {
	seen := []Config{}
	router := NewRouter(map[string]Runnable[string, string]{
		"chosen": configCaptureRunnable[string, string]{output: "ok", seen: &seen},
	})

	got, err := router.Invoke(
		context.Background(),
		RouterInput[string]{Key: "chosen", Input: "input"},
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
	assertChildConfig(t, seen[0], "route:chosen")
}

func TestConfigurableAlternativesPropagatesSelectedConfig(t *testing.T) {
	seen := []Config{}
	def := configCaptureRunnable[string, string]{output: "default", seen: &seen}
	alt := configCaptureRunnable[string, string]{output: "alt", seen: &seen}
	runnable, err := NewConfigurableAlternatives[string, string](
		"model",
		"default",
		def,
		map[string]Runnable[string, string]{"alt": alt},
	)
	if err != nil {
		t.Fatalf("new alternatives: %v", err)
	}

	got, err := runnable.Invoke(
		context.Background(),
		"input",
		WithRunID("root"),
		WithTags("parent"),
		WithMetadata("trace", "yes"),
		WithConfigurable("model", "alt"),
	)
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got != "alt" {
		t.Fatalf("got %q", got)
	}
	assertChildConfig(t, seen[0], "configurable:alt")
}

func TestParallelStreamEmitsKeyedChunks(t *testing.T) {
	parallel := NewParallel(map[string]Runnable[string, any]{
		"a": anyStreamingRunnable{values: []any{1, 2}},
		"b": anyStreamingRunnable{values: []any{10}},
	})

	stream, err := parallel.Stream(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	got := collectStreamValues(t, stream)
	want := []map[string]any{
		{"a": 1},
		{"b": 10},
		{"a": 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stream got %#v want %#v", got, want)
	}
}

func TestRunnableGraphSequence(t *testing.T) {
	toLen := NewFunc(func(_ context.Context, input string, _ ...Option) (int, error) {
		return len(input), nil
	}, schema.String(""), schema.Integer(""))
	isEven := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input%2 == 0, nil
	}, schema.Integer(""), schema.Boolean(""))

	seq, err := NewSequence[string, int, bool](toLen, isEven)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	graph := GetGraph(seq)
	if len(graph.Nodes) != 2 {
		t.Fatalf("nodes: %#v", graph.Nodes)
	}
	if len(graph.Edges) != 1 || graph.Edges[0].Label != "then" {
		t.Fatalf("edges: %#v", graph.Edges)
	}
	if graph.Edges[0].Source != "first.first" || graph.Edges[0].Target != "second.second" {
		t.Fatalf("sequence edge: %#v", graph.Edges[0])
	}
}

func TestRunnableGraphParallelAndConfigurable(t *testing.T) {
	length := NewFunc(func(_ context.Context, input string, _ ...Option) (any, error) {
		return len(input), nil
	}, schema.String(""), schema.Integer(""))
	text := NewFunc(func(_ context.Context, input string, _ ...Option) (any, error) {
		return input + "!", nil
	}, schema.String(""), schema.String(""))
	parallel := NewParallel(map[string]Runnable[string, any]{
		"length": length,
		"text":   text,
	})

	graph := GetGraph(parallel)
	if len(graph.Nodes) != 3 {
		t.Fatalf("parallel nodes: %#v", graph.Nodes)
	}
	labels := []string{graph.Edges[0].Label, graph.Edges[1].Label}
	if !reflect.DeepEqual(labels, []string{"length", "text"}) {
		t.Fatalf("parallel edge labels: %#v", graph.Edges)
	}

	def := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))
	alt := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "!", nil
	}, schema.String(""), schema.String(""))
	configurable, err := NewConfigurableAlternatives[string, string]("model", "default", def, map[string]Runnable[string, string]{"alt": alt})
	if err != nil {
		t.Fatalf("new configurable: %v", err)
	}
	graph = GetGraph(configurable)
	if graph.Nodes[0].Metadata["field"] != "model" || graph.Nodes[0].Metadata["default_key"] != "default" {
		t.Fatalf("metadata: %#v", graph.Nodes[0].Metadata)
	}
	if len(graph.Edges) != 2 || graph.Edges[0].Label != "default" || graph.Edges[1].Label != "alt" {
		t.Fatalf("configurable edges: %#v", graph.Edges)
	}
}

func TestRunnableGraphLeaf(t *testing.T) {
	runnable := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))

	graph := GetGraph(runnable)
	if len(graph.Nodes) != 1 || len(graph.Edges) != 0 {
		t.Fatalf("graph: %#v", graph)
	}
	if graph.Nodes[0].Type != "Func[string,string]" {
		t.Fatalf("node type: %#v", graph.Nodes[0])
	}
}

func TestRunnableGraphExports(t *testing.T) {
	toLen := NewFunc(func(_ context.Context, input string, _ ...Option) (int, error) {
		return len(input), nil
	}, schema.String(""), schema.Integer(""))
	isEven := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input%2 == 0, nil
	}, schema.Integer(""), schema.Boolean(""))
	seq, err := NewSequence[string, int, bool](toLen, isEven)
	if err != nil {
		t.Fatal(err)
	}
	graph := GetGraph(seq)
	ascii := graph.DrawASCII()
	for _, want := range []string{"| first |", "| second |", "*"} {
		if !strings.Contains(ascii, want) {
			t.Fatalf("ASCII %q missing %q", ascii, want)
		}
	}
	mermaid := graph.DrawMermaid()
	for _, want := range []string{"graph TD;", `first\2efirst`, "-- &nbsp;then&nbsp; -->"} {
		if !strings.Contains(mermaid, want) {
			t.Fatalf("Mermaid %q missing %q", mermaid, want)
		}
	}
	data, err := graph.MarshalJSONStable()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"Nodes"`) || !strings.Contains(string(data), `"Edges"`) {
		t.Fatalf("JSON = %s", data)
	}
	if _, err := graph.DrawPNG(); err == nil {
		t.Fatal("expected PNG unsupported error")
	}
}

func collectStreamValues[T any](t *testing.T, stream Stream[T]) []T {
	t.Helper()
	out := []T{}
	for {
		value, ok, err := stream.Next(context.Background())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			return out
		}
		out = append(out, value)
	}
}

type intStreamingRunnable struct {
	invoke int
	stream []int
}

func (r intStreamingRunnable) Invoke(context.Context, string, ...Option) (int, error) {
	return r.invoke, nil
}

func (r intStreamingRunnable) Batch(_ context.Context, inputs []string, _ ...Option) ([]int, error) {
	out := make([]int, len(inputs))
	for i := range inputs {
		out[i] = r.invoke
	}
	return out, nil
}

func (r intStreamingRunnable) Stream(context.Context, string, ...Option) (Stream[int], error) {
	return NewSliceStream(r.stream), nil
}

func (r intStreamingRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (r intStreamingRunnable) OutputSchema() schema.Schema { return schema.Integer("") }

type secondWithStream struct{}

func (r secondWithStream) Invoke(context.Context, int, ...Option) (int, error) { return 0, nil }

func (r secondWithStream) Batch(_ context.Context, inputs []int, _ ...Option) ([]int, error) {
	return make([]int, len(inputs)), nil
}

func (r secondWithStream) Stream(_ context.Context, input int, _ ...Option) (Stream[int], error) {
	return NewSliceStream([]int{input, input * 10}), nil
}

func (r secondWithStream) InputSchema() schema.Schema  { return schema.Integer("") }
func (r secondWithStream) OutputSchema() schema.Schema { return schema.Integer("") }

type anyStreamingRunnable struct {
	values []any
}

func (r anyStreamingRunnable) Invoke(context.Context, string, ...Option) (any, error) {
	if len(r.values) == 0 {
		return nil, nil
	}
	return r.values[0], nil
}

func (r anyStreamingRunnable) Batch(_ context.Context, inputs []string, _ ...Option) ([]any, error) {
	out := make([]any, len(inputs))
	for i := range inputs {
		if len(r.values) > 0 {
			out[i] = r.values[0]
		}
	}
	return out, nil
}

func (r anyStreamingRunnable) Stream(context.Context, string, ...Option) (Stream[any], error) {
	return NewSliceStream(r.values), nil
}

func (r anyStreamingRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (r anyStreamingRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

type configCaptureRunnable[I any, O any] struct {
	output O
	seen   *[]Config
	byName map[string]Config
}

func (r configCaptureRunnable[I, O]) Invoke(_ context.Context, _ I, opts ...Option) (O, error) {
	cfg := NewConfig(opts...)
	if r.seen != nil {
		*r.seen = append(*r.seen, cfg)
	}
	if r.byName != nil {
		r.byName[cfg.Name] = cfg
	}
	return r.output, nil
}

func (r configCaptureRunnable[I, O]) Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error) {
	out := make([]O, len(inputs))
	for i, input := range inputs {
		value, err := r.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = value
	}
	return out, nil
}

func (r configCaptureRunnable[I, O]) Stream(ctx context.Context, input I, opts ...Option) (Stream[O], error) {
	value, err := r.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	return NewSliceStream([]O{value}), nil
}

func (r configCaptureRunnable[I, O]) InputSchema() schema.Schema  { return schema.Schema{} }
func (r configCaptureRunnable[I, O]) OutputSchema() schema.Schema { return schema.Schema{} }

func assertChildConfig(t *testing.T, cfg Config, name string) {
	t.Helper()
	if cfg.Name != name {
		t.Fatalf("name: got %q want %q", cfg.Name, name)
	}
	if cfg.RunID != "" || cfg.ParentID != "root" {
		t.Fatalf("run IDs: got run=%q parent=%q", cfg.RunID, cfg.ParentID)
	}
	if !reflect.DeepEqual(cfg.Tags, []string{"parent"}) {
		t.Fatalf("tags: %#v", cfg.Tags)
	}
	if cfg.Metadata["trace"] != "yes" {
		t.Fatalf("metadata: %#v", cfg.Metadata)
	}
	if cfg.Configurable["mode"] != "fast" && cfg.Configurable["model"] != "alt" {
		t.Fatalf("configurable: %#v", cfg.Configurable)
	}
}

func TestNewRetryErrorsAndDefaults(t *testing.T) {
	if _, err := NewRetry[string, string](nil, 3); err == nil {
		t.Fatal("expected error for nil runnable")
	}

	attempts := 0
	alwaysFail := NewFunc(func(context.Context, string, ...Option) (string, error) {
		attempts++
		return "", errTestSentinel
	}, schema.String(""), schema.String(""))
	// maxAttempts <= 0 falls back to three attempts.
	retrying, err := NewRetry[string, string](alwaysFail, 0)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	if _, err := retrying.Invoke(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
	if attempts != 3 {
		t.Fatalf("attempts: got %d want 3", attempts)
	}
}

func TestRetryBatchAndSchemas(t *testing.T) {
	base := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		if input < 0 {
			return 0, errTestSentinel
		}
		return input * 2, nil
	}, schema.Integer("in"), schema.Integer("out"))
	retrying, err := NewRetry[int, int](base, 2)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}

	got, err := retrying.Batch(context.Background(), []int{2, -1})
	if err == nil {
		t.Fatal("expected joined batch error")
	}
	if got[0] != 4 {
		t.Fatalf("batch got %#v", got)
	}

	if retrying.InputSchema()["description"] != "in" {
		t.Fatalf("input schema: %#v", retrying.InputSchema())
	}
	if retrying.OutputSchema()["description"] != "out" {
		t.Fatalf("output schema: %#v", retrying.OutputSchema())
	}
	if retrying.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", retrying.ConfigSchema())
	}
}

// flakyStreamRunnable fails Stream construction a fixed number of times before
// delegating to a one-value stream, to exercise Retry.Stream retries.
type flakyStreamRunnable struct {
	failures int
	calls    int
}

func (r *flakyStreamRunnable) Invoke(_ context.Context, input string, _ ...Option) (string, error) {
	return input, nil
}

func (r *flakyStreamRunnable) Batch(_ context.Context, inputs []string, _ ...Option) ([]string, error) {
	return inputs, nil
}

func (r *flakyStreamRunnable) Stream(_ context.Context, input string, _ ...Option) (Stream[string], error) {
	r.calls++
	if r.calls <= r.failures {
		return nil, errTestSentinel
	}
	return NewSliceStream([]string{input}), nil
}

func (r *flakyStreamRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (r *flakyStreamRunnable) OutputSchema() schema.Schema { return schema.String("") }

func TestRetryStream(t *testing.T) {
	base := &flakyStreamRunnable{failures: 2}
	retrying, err := NewRetry[string, string](base, 3)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	stream, err := retrying.Stream(context.Background(), "chunk")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	got, ok, err := stream.Next(context.Background())
	if err != nil || !ok || got != "chunk" {
		t.Fatalf("next got=%q ok=%v err=%v", got, ok, err)
	}
	if base.calls != 3 {
		t.Fatalf("stream attempts: got %d want 3", base.calls)
	}

	exhausted := &flakyStreamRunnable{failures: 5}
	retrying, err = NewRetry[string, string](exhausted, 2)
	if err != nil {
		t.Fatalf("new retry: %v", err)
	}
	if _, err := retrying.Stream(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
	if exhausted.calls != 2 {
		t.Fatalf("stream attempts: got %d want 2", exhausted.calls)
	}
}

func TestRouterBatchStreamAndSchemas(t *testing.T) {
	add := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input + 1, nil
	}, schema.Integer(""), schema.Integer("routed"))
	router := NewRouter(map[string]Runnable[int, int]{"add": add})

	got, err := router.Batch(context.Background(), []RouterInput[int]{
		{Key: "add", Input: 1},
		{Key: "missing", Input: 2},
	})
	if err == nil {
		t.Fatal("expected joined batch error for missing route")
	}
	if got[0] != 2 {
		t.Fatalf("batch got %#v", got)
	}

	stream, err := router.Stream(context.Background(), RouterInput[int]{Key: "add", Input: 4})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk != 5 {
		t.Fatalf("chunk=%d ok=%v err=%v", chunk, ok, err)
	}

	if _, err := router.Stream(context.Background(), RouterInput[int]{Key: "missing", Input: 1}); err == nil {
		t.Fatal("expected missing route stream error")
	}

	input := router.InputSchema()
	props := schemaProperties(input)
	if _, ok := props["key"]; !ok {
		t.Fatalf("input schema missing key property: %#v", input)
	}
	if _, ok := props["input"]; !ok {
		t.Fatalf("input schema missing input property: %#v", input)
	}
	if router.OutputSchema()["description"] != "routed" {
		t.Fatalf("output schema: %#v", router.OutputSchema())
	}
	if router.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", router.ConfigSchema())
	}

	empty := NewRouter[int, int](map[string]Runnable[int, int]{})
	if len(empty.OutputSchema()) != 0 {
		t.Fatalf("empty router output schema: %#v", empty.OutputSchema())
	}
}

func TestNewConfigurableAlternativesErrors(t *testing.T) {
	base := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input, nil
	}, schema.String(""), schema.String(""))

	if _, err := NewConfigurableAlternatives[string, string]("", "default", base, nil); err == nil {
		t.Fatal("expected error for empty field")
	}
	if _, err := NewConfigurableAlternatives[string, string]("mode", "default", nil, nil); err == nil {
		t.Fatal("expected error for nil default runnable")
	}
	if _, err := NewConfigurableAlternatives[string, string]("mode", "default", base, map[string]Runnable[string, string]{"": base}); err == nil {
		t.Fatal("expected error for empty alternative key")
	}
	if _, err := NewConfigurableAlternatives[string, string]("mode", "default", base, map[string]Runnable[string, string]{"alt": nil}); err == nil {
		t.Fatal("expected error for nil alternative runnable")
	}
}

func TestConfigurableAlternativesSelectionEdgeCases(t *testing.T) {
	def := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "-default", nil
	}, schema.String("in"), schema.String("out"))
	alt := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "-alt", nil
	}, schema.String(""), schema.String(""))

	runnable, err := NewConfigurableAlternatives[string, string](
		"mode",
		"",
		def,
		map[string]Runnable[string, string]{"alt": alt},
	)
	if err != nil {
		t.Fatalf("new alternatives: %v", err)
	}

	// A nil configurable value falls back to the default runnable.
	got, err := runnable.Invoke(context.Background(), "x", WithConfigurable("mode", nil))
	if err != nil {
		t.Fatalf("nil value invoke: %v", err)
	}
	if got != "x-default" {
		t.Fatalf("nil value got %q", got)
	}

	// A non-string configurable value is stringified before lookup.
	_, err = runnable.Invoke(context.Background(), "x", WithConfigurable("mode", 42))
	if err == nil || !strings.Contains(err.Error(), `"42"`) {
		t.Fatalf("expected unknown alternative error for numeric key, got %v", err)
	}

	// Schemas come from the default runnable.
	if runnable.InputSchema()["description"] != "in" || runnable.OutputSchema()["description"] != "out" {
		t.Fatalf("schemas: in=%#v out=%#v", runnable.InputSchema(), runnable.OutputSchema())
	}

	// Unknown key with no choices and empty default key renders "[]" for the
	// available list (covers stringsList's empty branch).
	strict, err := NewConfigurableAlternatives[string, string]("mode", "", def, nil)
	if err != nil {
		t.Fatalf("new strict alternatives: %v", err)
	}
	_, err = strict.Invoke(context.Background(), "x", WithConfigurable("mode", "nope"))
	if err == nil || !strings.Contains(err.Error(), "available: []") {
		t.Fatalf("expected empty available list error, got %v", err)
	}

	// A zero-value wrapper (nil Default) returns empty schemas.
	var zero ConfigurableAlternatives[string, string]
	if len(zero.InputSchema()) != 0 || len(zero.OutputSchema()) != 0 {
		t.Fatalf("zero schemas: in=%#v out=%#v", zero.InputSchema(), zero.OutputSchema())
	}
}

func TestSequenceBatchAndErrors(t *testing.T) {
	toLen := NewFunc(func(_ context.Context, input string, _ ...Option) (int, error) {
		if input == "bad" {
			return 0, errTestSentinel
		}
		return len(input), nil
	}, schema.String("text"), schema.Integer(""))
	double := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input * 2, nil
	}, schema.Integer(""), schema.Integer("num"))

	seq, err := NewSequence[string, int, int](toLen, double)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}

	got, err := seq.Batch(context.Background(), []string{"ab", "abc"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if got[0] != 4 || got[1] != 6 {
		t.Fatalf("batch got %#v", got)
	}

	if _, err := seq.Batch(context.Background(), []string{"bad"}); err == nil {
		t.Fatal("expected batch error from first step")
	}
	if _, err := seq.Invoke(context.Background(), "bad"); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}

	if seq.InputSchema()["description"] != "text" {
		t.Fatalf("input schema: %#v", seq.InputSchema())
	}
	if seq.OutputSchema()["description"] != "num" {
		t.Fatalf("output schema: %#v", seq.OutputSchema())
	}
	if seq.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", seq.ConfigSchema())
	}

	if _, err := NewSequence[string, int, int](nil, double); err == nil {
		t.Fatal("expected error for nil first runnable")
	}
	if _, err := NewSequence[string, int, int](toLen, nil); err == nil {
		t.Fatal("expected error for nil second runnable")
	}
}

func TestParallelBatchAndSchemas(t *testing.T) {
	length := NewFunc(func(_ context.Context, input string, _ ...Option) (any, error) {
		return len(input), nil
	}, schema.String("text"), schema.Integer(""))
	parallel := NewParallel(map[string]Runnable[string, any]{"length": length})

	got, err := parallel.Batch(context.Background(), []string{"a", "bb"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	if got[0]["length"] != 1 || got[1]["length"] != 2 {
		t.Fatalf("batch got %#v", got)
	}

	if parallel.InputSchema()["description"] != "text" {
		t.Fatalf("input schema: %#v", parallel.InputSchema())
	}
	out := parallel.OutputSchema()
	if out["type"] != "object" {
		t.Fatalf("output schema: %#v", out)
	}
	if _, ok := schemaProperties(out)["length"]; !ok {
		t.Fatalf("output schema missing length property: %#v", out)
	}
	if parallel.ConfigSchema()["type"] != "object" {
		t.Fatalf("config schema: %#v", parallel.ConfigSchema())
	}

	empty := NewParallel[string](map[string]Runnable[string, any]{})
	if len(empty.InputSchema()) != 0 {
		t.Fatalf("empty parallel input schema: %#v", empty.InputSchema())
	}
}

// failingStream wraps a stream whose Next fails, to exercise Parallel.Stream
// error propagation.
type failingStream[T any] struct {
	err error
}

func (s failingStream[T]) Next(context.Context) (T, bool, error) {
	var zero T
	return zero, false, s.err
}

func (s failingStream[T]) Close() error { return nil }

type failingStreamRunnable struct{}

func (r failingStreamRunnable) Invoke(context.Context, string, ...Option) (any, error) {
	return nil, nil
}

func (r failingStreamRunnable) Batch(context.Context, []string, ...Option) ([]any, error) {
	return nil, nil
}

func (r failingStreamRunnable) Stream(context.Context, string, ...Option) (Stream[any], error) {
	return failingStream[any]{err: errTestSentinel}, nil
}

func (r failingStreamRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (r failingStreamRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

func TestParallelStreamStepError(t *testing.T) {
	parallel := NewParallel(map[string]Runnable[string, any]{
		"fail": failingStreamRunnable{},
	})
	stream, err := parallel.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); err != errTestSentinel {
		t.Fatalf("next err: got %v want %v", err, errTestSentinel)
	}
}

// closeErrStream returns values then fails on Close, to exercise the
// closeParallelStreams error join in Parallel.Stream construction cleanup.
type closeErrStream struct {
	values []any
	index  int
}

func (s *closeErrStream) Next(context.Context) (any, bool, error) {
	if s.index >= len(s.values) {
		return nil, false, nil
	}
	value := s.values[s.index]
	s.index++
	return value, true, nil
}

func (s *closeErrStream) Close() error { return errTestSentinel }

type closeErrStreamRunnable struct {
	failStream bool
}

func (r closeErrStreamRunnable) Invoke(context.Context, string, ...Option) (any, error) {
	return nil, nil
}

func (r closeErrStreamRunnable) Batch(context.Context, []string, ...Option) ([]any, error) {
	return nil, nil
}

func (r closeErrStreamRunnable) Stream(context.Context, string, ...Option) (Stream[any], error) {
	if r.failStream {
		return nil, errTestSentinel
	}
	return &closeErrStream{values: []any{1}}, nil
}

func (r closeErrStreamRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (r closeErrStreamRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

func TestParallelStreamConstructionError(t *testing.T) {
	// Step "b" fails Stream construction; the already-opened "a" stream must be
	// closed during cleanup (its Close error is swallowed in favor of the
	// construction error).
	parallel := NewParallel(map[string]Runnable[string, any]{
		"a": closeErrStreamRunnable{},
		"b": closeErrStreamRunnable{failStream: true},
	})
	_, err := parallel.Stream(context.Background(), "x")
	if err == nil || !strings.Contains(err.Error(), "b:") {
		t.Fatalf("stream err: %v", err)
	}
}

func TestParallelStreamCloseErr(t *testing.T) {
	// When a step's stream exhausts, its Close error propagates to the
	// consumer; closing the parallel stream itself joins remaining Close
	// errors.
	parallel := NewParallel(map[string]Runnable[string, any]{
		"a": closeErrStreamRunnable{},
	})
	stream, err := parallel.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	value, ok, err := stream.Next(context.Background())
	if err != nil || !ok || value["a"] != 1 {
		t.Fatalf("first chunk=%#v ok=%v err=%v", value, ok, err)
	}
	if _, _, err := stream.Next(context.Background()); err != errTestSentinel {
		t.Fatalf("expected close error on exhaustion, got %v", err)
	}
	// The failed-close item stays in the stream's item list, so Close joins
	// the same close error again (documents current behavior).
	if err := stream.Close(); !errors.Is(err, errTestSentinel) {
		t.Fatalf("close after drain: %v", err)
	}
}

func TestConfigurableAlternativesBatchAndStreamErrors(t *testing.T) {
	def := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input + 1, nil
	}, schema.Integer(""), schema.Integer(""))
	alt := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input * 2, nil
	}, schema.Integer(""), schema.Integer(""))
	runnable, err := NewConfigurableAlternatives[int, int](
		"mode",
		"default",
		def,
		map[string]Runnable[int, int]{"alt": alt},
	)
	if err != nil {
		t.Fatalf("new alternatives: %v", err)
	}

	if _, err := runnable.Batch(context.Background(), []int{1}, WithConfigurable("mode", "nope")); err == nil {
		t.Fatal("expected batch error for unknown alternative")
	}
	if _, err := runnable.Stream(context.Background(), 1, WithConfigurable("mode", "nope")); err == nil {
		t.Fatal("expected stream error for unknown alternative")
	}

	// An explicit default key selects the default runnable for the batch.
	got, err := runnable.Batch(context.Background(), []int{1, 2}, WithConfigurable("mode", "default"))
	if err != nil {
		t.Fatalf("default batch: %v", err)
	}
	if got[0] != 2 || got[1] != 3 {
		t.Fatalf("default batch got %#v", got)
	}

	// Streaming from the default alternative works too.
	stream, err := runnable.Stream(context.Background(), 4)
	if err != nil {
		t.Fatalf("default stream: %v", err)
	}
	defer stream.Close()
	chunk, ok, err := stream.Next(context.Background())
	if err != nil || !ok || chunk != 5 {
		t.Fatalf("chunk=%d ok=%v err=%v", chunk, ok, err)
	}
}

func TestParallelInvokeStepError(t *testing.T) {
	parallel := NewParallel(map[string]Runnable[string, any]{
		"fail": NewFunc(func(context.Context, string, ...Option) (any, error) {
			return nil, errTestSentinel
		}, schema.String(""), schema.Schema{}),
		"ok": NewFunc(func(_ context.Context, input string, _ ...Option) (any, error) {
			return len(input), nil
		}, schema.String(""), schema.Integer("")),
	})

	got, err := parallel.Invoke(context.Background(), "abc")
	if err == nil || !strings.Contains(err.Error(), "fail:") {
		t.Fatalf("expected keyed step error, got %v", err)
	}
	if got["ok"] != 3 {
		t.Fatalf("successful steps must still be collected: %#v", got)
	}
}

func TestSequenceStreamFirstStepError(t *testing.T) {
	// A first step whose Stream constructor fails propagates the error.
	seq, err := NewSequence[string, any, any](sequenceStreamConstructErrRunnable{}, streamOnlyRunnable{})
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	if _, err := seq.Stream(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
}

// sequenceStreamConstructErrRunnable fails Stream construction.
type sequenceStreamConstructErrRunnable struct{}

func (sequenceStreamConstructErrRunnable) Invoke(context.Context, string, ...Option) (any, error) {
	return nil, nil
}

func (sequenceStreamConstructErrRunnable) Batch(context.Context, []string, ...Option) ([]any, error) {
	return nil, nil
}

func (sequenceStreamConstructErrRunnable) Stream(context.Context, string, ...Option) (Stream[any], error) {
	return nil, errTestSentinel
}

func (sequenceStreamConstructErrRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (sequenceStreamConstructErrRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

// anyStreamConstructErrRunnable is the Runnable[any, any] counterpart used as
// a second sequence stage.
type anyStreamConstructErrRunnable struct{}

func (anyStreamConstructErrRunnable) Invoke(context.Context, any, ...Option) (any, error) {
	return nil, nil
}

func (anyStreamConstructErrRunnable) Batch(context.Context, []any, ...Option) ([]any, error) {
	return nil, nil
}

func (anyStreamConstructErrRunnable) Stream(context.Context, any, ...Option) (Stream[any], error) {
	return nil, errTestSentinel
}

func (anyStreamConstructErrRunnable) InputSchema() schema.Schema  { return schema.Schema{} }
func (anyStreamConstructErrRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

// anyCloseErrStreamRunnable streams one value then fails on Close.
type anyCloseErrStreamRunnable struct{}

func (anyCloseErrStreamRunnable) Invoke(context.Context, any, ...Option) (any, error) {
	return nil, nil
}

func (anyCloseErrStreamRunnable) Batch(context.Context, []any, ...Option) ([]any, error) {
	return nil, nil
}

func (anyCloseErrStreamRunnable) Stream(context.Context, any, ...Option) (Stream[any], error) {
	return &closeErrStream{values: []any{1}}, nil
}

func (anyCloseErrStreamRunnable) InputSchema() schema.Schema  { return schema.Schema{} }
func (anyCloseErrStreamRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

func TestSequenceStreamNextErrors(t *testing.T) {
	ctx := context.Background()

	// The first stage's stream fails mid-iteration.
	seq, err := NewSequence[string, any, any](
		firstNextErrRunnable{},
		streamOnlyRunnable{stream: []any{"unused"}},
	)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	stream, err := seq.Stream(ctx, "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(ctx); err != errTestSentinel {
		t.Fatalf("first-stage next err: got %v want %v", err, errTestSentinel)
	}

	// The second stage fails Stream construction for a mid value.
	seq, err = NewSequence[string, any, any](
		anyStreamingRunnable{values: []any{"x"}},
		anyStreamConstructErrRunnable{},
	)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	stream, err = seq.Stream(ctx, "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(ctx); err != errTestSentinel {
		t.Fatalf("second-stage construct err: got %v want %v", err, errTestSentinel)
	}

	// The second stage's stream fails on Close when exhausted.
	seq, err = NewSequence[string, any, any](
		anyStreamingRunnable{values: []any{"x"}},
		anyCloseErrStreamRunnable{},
	)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	stream, err = seq.Stream(ctx, "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	value, ok, err := stream.Next(ctx)
	if err != nil || !ok || value != 1 {
		t.Fatalf("chunk=%#v ok=%v err=%v", value, ok, err)
	}
	if _, _, err := stream.Next(ctx); err != errTestSentinel {
		t.Fatalf("close-on-exhaust err: got %v want %v", err, errTestSentinel)
	}
}

// firstNextErrRunnable returns a stream that fails on the first Next.
type firstNextErrRunnable struct{}

func (firstNextErrRunnable) Invoke(context.Context, string, ...Option) (any, error) {
	return nil, nil
}

func (firstNextErrRunnable) Batch(context.Context, []string, ...Option) ([]any, error) {
	return nil, nil
}

func (firstNextErrRunnable) Stream(context.Context, string, ...Option) (Stream[any], error) {
	return failingStream[any]{err: errTestSentinel}, nil
}

func (firstNextErrRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (firstNextErrRunnable) OutputSchema() schema.Schema { return schema.Schema{} }

func TestSequenceStreamCloseWithOpenStage(t *testing.T) {
	// Closing a partially-drained sequence stream closes both the open second
	// stage and the first stage, joining their Close errors.
	seq, err := NewSequence[string, any, any](
		anyStreamingRunnable{values: []any{"x"}},
		anyCloseErrStreamRunnable{},
	)
	if err != nil {
		t.Fatalf("new sequence: %v", err)
	}
	stream, err := seq.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if _, ok, err := stream.Next(context.Background()); err != nil || !ok {
		t.Fatalf("first next: ok=%v err=%v", ok, err)
	}
	if err := stream.Close(); !errors.Is(err, errTestSentinel) {
		t.Fatalf("close: got %v want %v", err, errTestSentinel)
	}
}
