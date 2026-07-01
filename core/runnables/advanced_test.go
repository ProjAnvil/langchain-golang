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
	for _, want := range []string{"graph:", "[first.first]", "first.first --then--> second.second"} {
		if !strings.Contains(ascii, want) {
			t.Fatalf("ASCII %q missing %q", ascii, want)
		}
	}
	mermaid := graph.DrawMermaid()
	for _, want := range []string{"graph TD;", "first_first", "-->|then|"} {
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
