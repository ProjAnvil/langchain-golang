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

func TestPipe456_NilPanics(t *testing.T) {
	var nilStr Runnable[string, string]
	a, b, c := pipeStrFn("a"), pipeStrFn("b"), pipeStrFn("c")
	d, e, f := pipeStrFn("d"), pipeStrFn("e"), pipeStrFn("f")
	for _, tc := range []struct {
		name string
		call func()
	}{
		{"Pipe4(nil,...)", func() { Pipe4(nilStr, b, c, d) }},
		{"Pipe4(...,nil)", func() { Pipe4(a, b, c, nilStr) }},
		{"Pipe5(nil,...)", func() { Pipe5(nilStr, b, c, d, e) }},
		{"Pipe5(...,nil)", func() { Pipe5(a, b, c, d, nilStr) }},
		{"Pipe6(nil,...)", func() { Pipe6(nilStr, b, c, d, e, f) }},
		{"Pipe6(...,nil)", func() { Pipe6(a, b, c, d, e, nilStr) }},
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

func TestPipe_Schemas(t *testing.T) {
	toLen := NewFunc(func(_ context.Context, input string, _ ...Option) (int, error) {
		return len(input), nil
	}, schema.String("text"), schema.Integer("length"))
	isEven := NewFunc(func(_ context.Context, input int, _ ...Option) (bool, error) {
		return input%2 == 0, nil
	}, schema.Integer(""), schema.Boolean("even"))
	chain := Pipe(toLen, isEven)
	if chain.InputSchema()["description"] != "text" {
		t.Fatalf("input schema: %#v", chain.InputSchema())
	}
	if chain.OutputSchema()["description"] != "even" {
		t.Fatalf("output schema: %#v", chain.OutputSchema())
	}
}

func TestSeqN_StreamEmptySequence(t *testing.T) {
	var chain SeqN[string, string]
	if _, err := chain.Stream(context.Background(), "x"); err == nil {
		t.Fatal("expected error for empty sequence")
	}
}

func TestSeqN_InvokeStepError(t *testing.T) {
	fail := NewFunc(func(context.Context, string, ...Option) (string, error) {
		return "", errTestSentinel
	}, schema.String(""), schema.String(""))
	chain := Pipe(pipeStrFn("a"), fail)
	if _, err := chain.Invoke(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("invoke err: got %v want %v", err, errTestSentinel)
	}
}

func TestSeqN_BatchStepError(t *testing.T) {
	fail := NewFunc(func(context.Context, string, ...Option) (string, error) {
		return "", errTestSentinel
	}, schema.String(""), schema.String(""))
	chain := Pipe(pipeStrFn("a"), fail)
	if _, err := chain.Batch(context.Background(), []string{"x"}); err == nil {
		t.Fatal("expected batch error from failing step")
	}
}

func TestErasedRunnableTypeMismatch(t *testing.T) {
	// erase is exercised directly here with deliberately mismatched runtime
	// types; Pipe* call sites prevent this at compile time.
	intToInt := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		return input, nil
	}, schema.Integer("in"), schema.Integer("out"))
	erased := erase[int, int](intToInt)

	if _, err := erased.Invoke(context.Background(), "not-an-int"); err == nil {
		t.Fatal("expected invoke type mismatch error")
	}
	if _, err := erased.Batch(context.Background(), []any{1, "nope"}); err == nil {
		t.Fatal("expected batch type mismatch error")
	}
	if _, err := erased.Stream(context.Background(), 1.5); err == nil {
		t.Fatal("expected stream type mismatch error")
	}

	if erased.InputSchema()["description"] != "in" {
		t.Fatalf("input schema: %#v", erased.InputSchema())
	}
	if erased.OutputSchema()["description"] != "out" {
		t.Fatalf("output schema: %#v", erased.OutputSchema())
	}
}

func TestSeqN_StreamTailTypeMismatch(t *testing.T) {
	// A step whose stream emits a value of the wrong type fails at the tail
	// stream's type assertion.
	src := streamFn{
		invokeFn: func(s string) (string, error) { return s, nil },
		streamFn: func(_ string) ([]string, error) { return []string{"a"}, nil },
	}
	chain := Pipe(src, pipeStrFn("b"))
	stream, err := chain.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	// Rebind the tail to an impossible output type via a manual tail stream.
	tail := &seqNTailStream[int]{inner: mustAnyStream(t, src, "x")}
	if _, _, err := tail.Next(context.Background()); err == nil {
		t.Fatal("expected tail type mismatch error")
	}
}

func mustAnyStream(t *testing.T, r streamFn, input string) Stream[any] {
	t.Helper()
	stream, err := r.Stream(context.Background(), input)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	return &erasedStream[string]{inner: stream}
}

func TestSeqN_StreamFirstStepError(t *testing.T) {
	fail := streamFn{
		invokeFn: func(s string) (string, error) { return "", errTestSentinel },
		streamFn: func(_ string) ([]string, error) { return nil, errTestSentinel },
	}
	chain := Pipe(fail, pipeStrFn("b"))
	if _, err := chain.Stream(context.Background(), "x"); err != errTestSentinel {
		t.Fatalf("stream err: got %v want %v", err, errTestSentinel)
	}
}

func TestPipe_StreamCloseMidDrain(t *testing.T) {
	src := streamFn{
		invokeFn: func(s string) (string, error) { return s, nil },
		streamFn: func(_ string) ([]string, error) { return []string{"a", "b"}, nil },
	}
	transform := streamFn{
		invokeFn: func(s string) (string, error) { return s, nil },
		streamFn: func(s string) ([]string, error) { return []string{s + "1", s + "2"}, nil },
	}
	stream, err := Pipe(src, transform).Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	got, ok, err := stream.Next(context.Background())
	if err != nil || !ok || got != "a1" {
		t.Fatalf("first chunk=%q ok=%v err=%v", got, ok, err)
	}
	// Closing with an open inner stage must succeed and release both stages.
	if err := stream.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// errNextStringStream fails on the first Next call, exercising the erased
// stream's error forwarding.
type errNextStringStream struct{}

func (errNextStringStream) Next(context.Context) (string, bool, error) {
	return "", false, errTestSentinel
}

func (errNextStringStream) Close() error { return nil }

type errNextStringRunnable struct{}

func (errNextStringRunnable) Invoke(_ context.Context, input string, _ ...Option) (string, error) {
	return input, nil
}

func (errNextStringRunnable) Batch(_ context.Context, inputs []string, _ ...Option) ([]string, error) {
	return inputs, nil
}

func (errNextStringRunnable) Stream(context.Context, string, ...Option) (Stream[string], error) {
	return errNextStringStream{}, nil
}

func (errNextStringRunnable) InputSchema() schema.Schema  { return schema.String("") }
func (errNextStringRunnable) OutputSchema() schema.Schema { return schema.String("") }

func TestPipe_StreamFirstStageNextError(t *testing.T) {
	chain := Pipe(errNextStringRunnable{}, pipeStrFn("b"))
	stream, err := chain.Stream(context.Background(), "x")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	if _, _, err := stream.Next(context.Background()); err != errTestSentinel {
		t.Fatalf("next err: got %v want %v", err, errTestSentinel)
	}
}
