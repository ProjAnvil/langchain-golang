package runnables

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

// pipeStrFn makes a Runnable[string,string] that appends tag to its input,
// for pipe composition tests.
func pipeStrFn(tag string) Runnable[string, string] {
	return NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return input + "+" + tag, nil
	}, schema.String(""), schema.String(""))
}

func TestPipe_InvokeTwoSteps(t *testing.T) {
	chain := Pipe(pipeStrFn("a"), pipeStrFn("b"))
	got, err := chain.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if want := "x+a+b"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPipe3_InvokeThreeSteps(t *testing.T) {
	chain := Pipe3(pipeStrFn("a"), pipeStrFn("b"), pipeStrFn("c"))
	got, err := chain.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if want := "x+a+b+c"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPipe_SatisfiesRunnable(t *testing.T) {
	// Compile-time assertion: SeqN[I,O] implements Runnable[I,O], so a
	// Pipe result composes with every existing combinator (NewWithFallbacks,
	// NewRetry, NewBranch, ...).
	var _ Runnable[string, string] = SeqN[string, string]{}
}

func TestPipe_Batch(t *testing.T) {
	chain := Pipe(pipeStrFn("a"), pipeStrFn("b"))
	got, err := chain.Batch(context.Background(), []string{"x", "y"})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}
	want := []string{"x+a+b", "y+a+b"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// streamFn is a Runnable[string,string] whose Stream behavior is controlled by
// a closure, so pipe-streaming tests can assert chunk order across stages.
type streamFn struct {
	invokeFn func(string) (string, error)
	streamFn func(string) ([]string, error)
}

func (s streamFn) Invoke(_ context.Context, input string, _ ...Option) (string, error) {
	return s.invokeFn(input)
}

func (s streamFn) Batch(ctx context.Context, inputs []string, _ ...Option) ([]string, error) {
	out := make([]string, len(inputs))
	for i, in := range inputs {
		o, err := s.invokeFn(in)
		if err != nil {
			return nil, err
		}
		out[i] = o
	}
	return out, nil
}

func (s streamFn) Stream(_ context.Context, input string, _ ...Option) (Stream[string], error) {
	chunks, err := s.streamFn(input)
	if err != nil {
		return nil, err
	}
	return NewSliceStream(chunks), nil
}

func (s streamFn) InputSchema() schema.Schema  { return schema.String("") }
func (s streamFn) OutputSchema() schema.Schema { return schema.String("") }

func TestPipe_Stream(t *testing.T) {
	// src streams two chunks "a","b" (input-independent).
	// transform expands each received chunk into "<chunk>1","<chunk>2".
	// Flattened order must be a1,a2,b1,b2.
	src := streamFn{
		invokeFn: func(s string) (string, error) { return "a" + "b", nil },
		streamFn: func(_ string) ([]string, error) { return []string{"a", "b"}, nil },
	}
	transform := streamFn{
		invokeFn: func(s string) (string, error) { return s + "1" + s + "2", nil },
		streamFn: func(s string) ([]string, error) { return []string{s + "1", s + "2"}, nil },
	}

	stream, err := Pipe(src, transform).Stream(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	got, err := drainStringStream(stream)
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	want := []string{"a1", "a2", "b1", "b2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func drainStringStream(s Stream[string]) ([]string, error) {
	var out []string
	for {
		v, ok, err := s.Next(context.Background())
		if err != nil {
			return nil, err
		}
		if !ok {
			return out, s.Close()
		}
		out = append(out, v)
	}
}

// configSchemaFn extends streamFn with an explicit ConfigSchema, so tests can
// assert that Pipe preserves each step's config schema through the type-erased
// SeqN.
type configSchemaFn struct {
	streamFn
	cfg schema.Schema
}

func (c configSchemaFn) ConfigSchema() schema.Schema { return c.cfg }

func TestPipe_NilPanics(t *testing.T) {
	var nilStr Runnable[string, string]
	for _, tc := range []struct {
		name string
		call func()
	}{
		{"Pipe(nil, b)", func() { Pipe(nilStr, pipeStrFn("b")) }},
		{"Pipe(a, nil)", func() { Pipe(pipeStrFn("a"), nilStr) }},
		{"Pipe3(nil, b, c)", func() { Pipe3(nilStr, pipeStrFn("b"), pipeStrFn("c")) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatalf("%s did not panic", tc.name)
				}
			}()
			tc.call()
		})
	}
}

func TestPipe_NestedEqualsPipe3(t *testing.T) {
	a, b, c := pipeStrFn("a"), pipeStrFn("b"), pipeStrFn("c")
	nested := Pipe(Pipe(a, b), c) // Pipe(a,b) is SeqN[string,string], feeds outer Pipe
	flat := Pipe3(a, b, c)
	ctx := context.Background()

	gotNested, err := nested.Invoke(ctx, "x")
	if err != nil {
		t.Fatalf("nested invoke: %v", err)
	}
	gotFlat, err := flat.Invoke(ctx, "x")
	if err != nil {
		t.Fatalf("flat invoke: %v", err)
	}
	want := "x+a+b+c"
	if gotNested != want || gotFlat != want {
		t.Fatalf("nested=%q flat=%q want=%q", gotNested, gotFlat, want)
	}
}

func TestPipe_ComposesWithExisting(t *testing.T) {
	// A Pipe result must be consumable by existing combinators (NewWithFallbacks)
	// without adapter, proving SeqN[I,O] satisfies Runnable[I,O].
	fail := NewFunc(func(_ context.Context, _ string, _ ...Option) (string, error) {
		return "", errors.New("boom")
	}, schema.String(""), schema.String(""))
	primary := Pipe(fail, pipeStrFn("b")) // always fails at first step

	fb, err := NewWithFallbacks(primary, pipeStrFn("fb"))
	if err != nil {
		t.Fatalf("new fallbacks: %v", err)
	}
	got, err := fb.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if want := "x+fb"; got != want {
		t.Fatalf("got %q, want %q (fallback should have run)", got, want)
	}
}

func TestPipe_ConfigSchemaMergedFromSteps(t *testing.T) {
	// Each step declares its own configurable field; SeqN.ConfigSchema must
	// surface both, proving the type-erasing adapter forwards ConfigSchema.
	stepWithCfg := func(tag, key string) configSchemaFn {
		return configSchemaFn{
			streamFn: streamFn{
				invokeFn: func(s string) (string, error) { return s + "+" + tag, nil },
				streamFn: func(s string) ([]string, error) { return []string{s + "+" + tag}, nil },
			},
			cfg: configurableConfigSchema(map[string]schema.Schema{key: schema.String(tag)}, key),
		}
	}
	chain := Pipe(stepWithCfg("a", "keyA"), stepWithCfg("b", "keyB"))

	got := chain.ConfigSchema()
	configurable, ok := configurableSchema(got)
	if !ok {
		t.Fatalf("no configurable object in schema: %v", got)
	}
	props := schemaProperties(configurable)
	if _, ok := props["keyA"]; !ok {
		t.Errorf("missing keyA in configurable schema; got %v", props)
	}
	if _, ok := props["keyB"]; !ok {
		t.Errorf("missing keyB in configurable schema; got %v", props)
	}
}

func TestPipe4_Invoke(t *testing.T) {
	chain := Pipe4(pipeStrFn("a"), pipeStrFn("b"), pipeStrFn("c"), pipeStrFn("d"))
	got, err := chain.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if want := "x+a+b+c+d"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPipe5_Invoke(t *testing.T) {
	chain := Pipe5(pipeStrFn("a"), pipeStrFn("b"), pipeStrFn("c"), pipeStrFn("d"), pipeStrFn("e"))
	got, err := chain.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if want := "x+a+b+c+d+e"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPipe6_Invoke(t *testing.T) {
	chain := Pipe6(pipeStrFn("a"), pipeStrFn("b"), pipeStrFn("c"), pipeStrFn("d"), pipeStrFn("e"), pipeStrFn("f"))
	got, err := chain.Invoke(context.Background(), "x")
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if want := "x+a+b+c+d+e+f"; got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestPipe3_Stream(t *testing.T) {
	// Three stages: src emits ["a","b"]; mid expands each to [<s>+"1"]; tail
	// expands each to [<s>+"!"]. Flattened: a1!, b1!.
	src := streamFn{
		invokeFn: func(s string) (string, error) { return s, nil },
		streamFn: func(_ string) ([]string, error) { return []string{"a", "b"}, nil },
	}
	mid := streamFn{
		invokeFn: func(s string) (string, error) { return s + "1", nil },
		streamFn: func(s string) ([]string, error) { return []string{s + "1"}, nil },
	}
	tail := streamFn{
		invokeFn: func(s string) (string, error) { return s + "!", nil },
		streamFn: func(s string) ([]string, error) { return []string{s + "!"}, nil },
	}

	stream, err := Pipe3(src, mid, tail).Stream(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	got, err := drainStringStream(stream)
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	want := []string{"a1!", "b1!"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
