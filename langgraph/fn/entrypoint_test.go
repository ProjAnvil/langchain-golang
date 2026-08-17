package fn

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// Mirrors test_pregel.py:6307 test_entrypoint_without_checkpointer: with no
// checkpointer, previous is never injected (Python: previous is always None).
func TestEntrypointWithoutCheckpointer(t *testing.T) {
	var hasPrevs []bool
	var prevs []map[string]any
	e := NewEntrypoint[map[string]any, map[string]any, map[string]any](EntrypointOpts{},
		func(_ runtime.Runtime, in, prev map[string]any, hasPrev bool) (map[string]any, error) {
			hasPrevs = append(hasPrevs, hasPrev)
			prevs = append(prevs, prev)
			return map[string]any{"previous": nil, "current": in}, nil
		})

	opts := graph.Options{ThreadID: "1"}
	want := map[string]any{"previous": nil, "current": map[string]any{"a": "1"}}
	for i := 0; i < 2; i++ {
		out, err := e.Invoke(context.Background(), map[string]any{"a": "1"}, opts)
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		if !reflect.DeepEqual(out, want) {
			t.Fatalf("Invoke() = %v, want %v", out, want)
		}
	}
	if !reflect.DeepEqual(hasPrevs, []bool{false, false}) {
		t.Fatalf("hasPrev across calls = %v, want [false false]", hasPrevs)
	}
	for i, p := range prevs {
		if p != nil {
			t.Fatalf("prev[%d] = %v, want nil (zero value)", i, p)
		}
	}
}

// Mirrors test_pregel.py:6329 test_entrypoint_stateful: previous threads the
// prior invocation's save value through the checkpointer.
func TestEntrypointStateful(t *testing.T) {
	e := NewEntrypoint[map[string]any, map[string]any, map[string]any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(_ runtime.Runtime, in, prev map[string]any, hasPrev bool) (map[string]any, error) {
			var p any
			if hasPrev {
				p = prev
			}
			return map[string]any{"previous": p, "current": in}, nil
		})

	opts := graph.Options{ThreadID: "1"}
	var outs []map[string]any
	for _, a := range []string{"1", "2", "3"} {
		out, err := e.Invoke(context.Background(), map[string]any{"a": a}, opts)
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		outs = append(outs, out)
	}
	want := []map[string]any{
		{"current": map[string]any{"a": "1"}, "previous": nil},
		{"current": map[string]any{"a": "2"},
			"previous": map[string]any{"current": map[string]any{"a": "1"}, "previous": nil}},
		{"current": map[string]any{"a": "3"},
			"previous": map[string]any{"current": map[string]any{"a": "2"},
				"previous": map[string]any{"current": map[string]any{"a": "1"}, "previous": nil}}},
	}
	if !reflect.DeepEqual(outs, want) {
		t.Fatalf("Invoke() results = %v, want %v", outs, want)
	}
}

// Mirrors test_pregel.py:6785 test_entrypoint_with_return_and_save:
// entrypoint.final(value=, save=) decouples the returned value from the save
// value threaded through previous.
func TestEntrypointFinalValueSave(t *testing.T) {
	var hasPrevs []bool
	var prevs [][]string
	e := NewEntrypointFinal[string, int, []string](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(_ runtime.Runtime, in string, prev []string, hasPrev bool) (Final[int, []string], error) {
			hasPrevs = append(hasPrevs, hasPrev)
			prevs = append(prevs, append([]string(nil), prev...))
			save := append(append([]string(nil), prev...), in)
			return Final[int, []string]{Value: len(prev), Save: save}, nil
		})

	opts := graph.Options{ThreadID: "1"}
	var outs []int
	for _, msg := range []string{"hello", "goodbye", "definitely"} {
		out, err := e.Invoke(context.Background(), msg, opts)
		if err != nil {
			t.Fatalf("Invoke() error = %v", err)
		}
		outs = append(outs, out)
	}
	if !reflect.DeepEqual(outs, []int{0, 1, 2}) {
		t.Fatalf("Invoke() results = %v, want [0 1 2]", outs)
	}
	if !reflect.DeepEqual(hasPrevs, []bool{false, true, true}) {
		t.Fatalf("hasPrev across calls = %v, want [false true true]", hasPrevs)
	}
	wantPrevs := [][]string{nil, {"hello"}, {"hello", "goodbye"}}
	if !reflect.DeepEqual(prevs, wantPrevs) {
		t.Fatalf("prev across calls = %v, want %v", prevs, wantPrevs)
	}
}

// Mirrors the stream part of test_pregel.py:6329 (Python asserts
// items == [{"foo": {...}}]; the Go node name is the fixed "entrypoint").
func TestEntrypointStream(t *testing.T) {
	e := NewEntrypoint[string, string, string](EntrypointOpts{},
		func(_ runtime.Runtime, in, _ string, _ bool) (string, error) {
			return "done", nil
		})

	var chunks []graph.StreamChunk
	for chunk, err := range e.Stream(context.Background(), "in", graph.Options{ThreadID: "1"}) {
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1: %+v", len(chunks), chunks)
	}
	c := chunks[0]
	if c.Mode != graph.StreamUpdates {
		t.Fatalf("chunk mode = %q, want %q", c.Mode, graph.StreamUpdates)
	}
	if !reflect.DeepEqual(c.Payload, map[string]any{"entrypoint": "done"}) {
		t.Fatalf("chunk payload = %+v, want %+v", c.Payload, map[string]any{"entrypoint": "done"})
	}
	payload := fmt.Sprintf("%+v", c.Payload)
	for _, reserved := range []string{"__start__", "__end__", "__previous__"} {
		if strings.Contains(payload, reserved) {
			t.Fatalf("reserved key %q leaked into stream chunk payload %s", reserved, payload)
		}
	}
}

// Mirrors the skeleton of test_pregel.py:4985 test_interrupt_functional
// (task-free variant): an interrupt inside the entrypoint pauses the run and
// a resumed Invoke feeds the resume value to the pending Interrupt call.
func TestEntrypointInterruptResume(t *testing.T) {
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(ctx runtime.Runtime, _ any, _ any, _ bool) (string, error) {
			v := graph.Interrupt(ctx, "Provide value")
			return fmt.Sprintf("got %v", v), nil
		})

	ctx := context.Background()
	_, err := e.Invoke(ctx, map[string]any{"a": ""}, graph.Options{ThreadID: "1"})
	var ierr *InterruptError
	if !errors.As(err, &ierr) {
		t.Fatalf("Invoke() error = %v (%T), want *InterruptError", err, err)
	}
	if len(ierr.Interrupts) != 1 || ierr.Interrupts[0].Value != "Provide value" {
		t.Fatalf("InterruptError.Interrupts = %+v, want one interrupt with value %q", ierr.Interrupts, "Provide value")
	}

	out, err := e.Invoke(ctx, nil, graph.Options{ThreadID: "1", Resume: "bar"})
	if err != nil {
		t.Fatalf("resumed Invoke() error = %v", err)
	}
	if out != "got bar" {
		t.Fatalf("resumed Invoke() = %q, want %q", out, "got bar")
	}
}

// An entrypoint function error propagates as a plain error, not an
// InterruptError.
func TestEntrypointErrorPropagation(t *testing.T) {
	boom := errors.New("boom")
	e := NewEntrypoint[string, string, string](EntrypointOpts{},
		func(_ runtime.Runtime, _, _ string, _ bool) (string, error) {
			return "", boom
		})

	_, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if !errors.Is(err, boom) {
		t.Fatalf("Invoke() error = %v, want boom", err)
	}
	var ierr *InterruptError
	if errors.As(err, &ierr) {
		t.Fatalf("Invoke() error = %v, must not be *InterruptError", err)
	}
}

// Construction with a nil function is a programmer error and panics, for
// both the plain and the Final form.
func TestNewEntrypointNilFuncPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "fn: entrypoint function must be non-nil" {
			t.Fatalf("recover = %v, want the nil-function panic", r)
		}
	}()
	NewEntrypoint[string, string, string](EntrypointOpts{}, nil)
}

func TestNewEntrypointFinalNilFuncPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != "fn: entrypoint function must be non-nil" {
			t.Fatalf("recover = %v, want the nil-function panic", r)
		}
	}()
	NewEntrypointFinal[string, int, string](EntrypointOpts{}, nil)
}

// A state missing the __start__ channel or carrying an input of the wrong
// type is a serde-contract violation and surfaces as a descriptive error,
// never a silent zero-value downgrade. Reached by driving the compiled
// graph directly (Invoke always supplies a well-typed __start__).
func TestEntrypointInputContractViolations(t *testing.T) {
	e := NewEntrypoint[string, string, string](EntrypointOpts{},
		func(_ runtime.Runtime, in, _ string, _ bool) (string, error) { return in, nil })

	_, err := e.graph.InvokeWithOptions(context.Background(), map[string]any{}, graph.Options{})
	if err == nil || !strings.Contains(err.Error(), `missing the "__start__" channel`) {
		t.Fatalf("InvokeWithOptions() error = %v, want a missing-channel error", err)
	}

	_, err = e.graph.InvokeWithOptions(context.Background(), map[string]any{channelStart: 42}, graph.Options{})
	if err == nil || !strings.Contains(err.Error(), "entrypoint input has type int") {
		t.Fatalf("InvokeWithOptions() error = %v, want an input-type error", err)
	}

	// The Final form's node wrapper surfaces the same contract violation.
	ef := NewEntrypointFinal[string, int, string](EntrypointOpts{},
		func(_ runtime.Runtime, in string, _ string, _ bool) (Final[int, string], error) {
			return Final[int, string]{Value: len(in), Save: in}, nil
		})
	_, err = ef.graph.InvokeWithOptions(context.Background(), map[string]any{channelStart: 42}, graph.Options{})
	if err == nil || !strings.Contains(err.Error(), "entrypoint input has type int") {
		t.Fatalf("InvokeWithOptions() error = %v, want an input-type error", err)
	}
}

// The plain form saves the returned value itself; when that value is not
// assignable to the declared save type S, the mismatch surfaces as a
// descriptive error on the NEXT invocation's previous assertion (documented
// NewEntrypoint behavior) — not silently.
func TestEntrypointPreviousTypeMismatch(t *testing.T) {
	e := NewEntrypoint[string, string, int](
		EntrypointOpts{Checkpointer: checkpoint.NewMemorySaver()},
		func(_ runtime.Runtime, in string, _ int, _ bool) (string, error) { return "out:" + in, nil })

	opts := graph.Options{ThreadID: "1"}
	if _, err := e.Invoke(context.Background(), "a", opts); err != nil {
		t.Fatalf("first Invoke() error = %v, want nil", err)
	}
	_, err := e.Invoke(context.Background(), "b", opts)
	if err == nil || !strings.Contains(err.Error(), "previous value has type string, want the declared save type") {
		t.Fatalf("second Invoke() error = %v, want a previous-type mismatch error", err)
	}
}

// A Final entrypoint function's error propagates as a plain error.
func TestEntrypointFinalErrorPropagation(t *testing.T) {
	boom := errors.New("boom")
	e := NewEntrypointFinal[string, int, string](EntrypointOpts{},
		func(_ runtime.Runtime, _ string, _ string, _ bool) (Final[int, string], error) {
			return Final[int, string]{}, boom
		})

	_, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if !errors.Is(err, boom) {
		t.Fatalf("Invoke() error = %v, want boom", err)
	}
}

// The __end__ channel value must match the declared output type O; a
// mismatch (here: a hand-built graph whose node writes a string into
// __end__ while O is int) fails Invoke with a descriptive error.
func TestEntrypointOutputTypeMismatch(t *testing.T) {
	g := graph.NewStateGraph().
		AddChannel(channelStart, channels.NewEphemeral(true)).
		AddChannel(channelEnd, channels.NewLastValue()).
		AddChannel(channelPrevious, channels.NewLastValue()).
		AddNode(entrypointNode, func(_ runtime.Runtime, _ map[string]any) (any, error) {
			return map[string]any{channelEnd: "not-an-int"}, nil
		}).
		SetEntryPoint(entrypointNode).
		AddEdge(entrypointNode, types.END)
	cg, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v, want nil", err)
	}
	e := &Entrypoint[string, int, any]{graph: cg}

	_, err = e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if err == nil || !strings.Contains(err.Error(), "entrypoint returned a value of type string, want the declared output type") {
		t.Fatalf("Invoke() error = %v, want an output-type mismatch error", err)
	}
}

func TestInterruptErrorMessage(t *testing.T) {
	err := &InterruptError{Interrupts: []types.Interrupt{{Value: "a"}, {Value: "b"}}}
	if got := err.Error(); got != "fn: entrypoint interrupted (2 pending)" {
		t.Fatalf("Error() = %q, want %q", got, "fn: entrypoint interrupted (2 pending)")
	}
}

// alwaysFailGetTupleSaver fails every GetTuple, so prepare's checkpoint load
// fails before the run starts.
type alwaysFailGetTupleSaver struct {
	checkpoint.Saver
	err error
}

func (s *alwaysFailGetTupleSaver) GetTuple(context.Context, checkpoint.Config) (*checkpoint.Tuple, error) {
	return nil, s.err
}

// A checkpointer failure during prepare aborts Invoke before the run starts.
func TestEntrypointPrepareCheckpointLoadError(t *testing.T) {
	boom := errors.New("saver down")
	e := NewEntrypoint[string, string, string](
		EntrypointOpts{Checkpointer: &alwaysFailGetTupleSaver{Saver: checkpoint.NewMemorySaver(), err: boom}},
		func(_ runtime.Runtime, in, _ string, _ bool) (string, error) { return in, nil })

	_, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "fn: entrypoint checkpoint load") {
		t.Fatalf("Invoke() error = %v, want the wrapped checkpoint-load error", err)
	}
}

// The same prepare failure surfaces as a yielded error from Stream.
func TestEntrypointStreamPrepareError(t *testing.T) {
	boom := errors.New("saver down")
	e := NewEntrypoint[string, string, string](
		EntrypointOpts{Checkpointer: &alwaysFailGetTupleSaver{Saver: checkpoint.NewMemorySaver(), err: boom}},
		func(_ runtime.Runtime, in, _ string, _ bool) (string, error) { return in, nil })

	var gotErr error
	chunks := 0
	for _, err := range e.Stream(context.Background(), "in", graph.Options{ThreadID: "1"}) {
		chunks++
		gotErr = err
	}
	if chunks != 1 || !errors.Is(gotErr, boom) {
		t.Fatalf("Stream() yielded %d chunks, error = %v; want one chunk with the load error", chunks, gotErr)
	}
}

// toggleGetTupleSaver delegates until armed, then fails (failErr non-nil) or
// reports no checkpoint (nilTuple). Arming from inside the entrypoint
// function places the failure after prepare's load and the graph's own
// loads, squarely on persistResults' GetTuple.
type toggleGetTupleSaver struct {
	checkpoint.Saver
	armed    atomic.Bool
	failErr  error
	nilTuple bool
}

func (s *toggleGetTupleSaver) GetTuple(ctx context.Context, cfg checkpoint.Config) (*checkpoint.Tuple, error) {
	if s.armed.Load() {
		if s.failErr != nil {
			return nil, s.failErr
		}
		if s.nilTuple {
			return nil, nil
		}
	}
	return s.Saver.GetTuple(ctx, cfg)
}

// A checkpointer failure inside persistResults (after a successful run)
// fails Invoke with a wrapped "persisting task results" error.
func TestEntrypointPersistResultsLoadError(t *testing.T) {
	boom := errors.New("saver down")
	saver := &toggleGetTupleSaver{Saver: checkpoint.NewMemorySaver(), failErr: boom}
	task := NewTask[any, string]("t", func(_ runtime.Runtime, _ any) (string, error) { return "v", nil }, TaskOpts{})
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			v, err := task.Call(ctx, nil).Get(ctx)
			if err != nil {
				return "", err
			}
			saver.armed.Store(true) // persistResults' GetTuple fails
			return v, nil
		})

	_, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "fn: persisting task results") {
		t.Fatalf("Invoke() error = %v, want the wrapped persist error", err)
	}
}

// When the run committed no checkpoint persistResults can find (nil tuple),
// persisting is skipped and the run's result still reaches the caller.
func TestEntrypointPersistResultsNoCheckpoint(t *testing.T) {
	saver := &toggleGetTupleSaver{Saver: checkpoint.NewMemorySaver(), nilTuple: true}
	task := NewTask[any, string]("t", func(_ runtime.Runtime, _ any) (string, error) { return "v", nil }, TaskOpts{})
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			v, err := task.Call(ctx, nil).Get(ctx)
			if err != nil {
				return "", err
			}
			saver.armed.Store(true) // persistResults' GetTuple reports no checkpoint
			return v, nil
		})

	out, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if err != nil || out != "v" {
		t.Fatalf("Invoke() = %q, %v; want %q, nil", out, err, "v")
	}
}

// putWritesPathSaver fails PutWrites for nested (non-empty) task paths,
// which only fn's persistResults issues — the graph's own PutWrites always
// use the root path.
type putWritesPathSaver struct {
	checkpoint.Saver
	err error
}

func (s *putWritesPathSaver) PutWrites(ctx context.Context, cfg checkpoint.Config, writes []checkpoint.Write, taskID, taskPath string) error {
	if taskPath != "" {
		return s.err
	}
	return s.Saver.PutWrites(ctx, cfg, writes, taskID, taskPath)
}

// A PutWrites failure while persisting a (nested) task result fails Invoke
// with a wrapped "persisting task results" error.
func TestEntrypointPersistResultsWriteError(t *testing.T) {
	boom := errors.New("writes rejected")
	saver := &putWritesPathSaver{Saver: checkpoint.NewMemorySaver(), err: boom}
	inner := NewTask[any, string]("inner", func(_ runtime.Runtime, _ any) (string, error) { return "i", nil }, TaskOpts{})
	outer := NewTask[any, string]("outer", func(ctx runtime.Runtime, _ any) (string, error) {
		return inner.Call(ctx, nil).Get(ctx)
	}, TaskOpts{})
	e := NewEntrypoint[any, string, any](
		EntrypointOpts{Checkpointer: saver},
		func(ctx runtime.Runtime, _, _ any, _ bool) (string, error) {
			return outer.Call(ctx, nil).Get(ctx)
		})

	_, err := e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "fn: persisting task results") {
		t.Fatalf("Invoke() error = %v, want the wrapped persist error", err)
	}
}

// A run failure (the entrypoint function returns an error) surfaces from
// Stream as a yielded error.
func TestEntrypointStreamRunError(t *testing.T) {
	boom := errors.New("boom")
	e := NewEntrypoint[string, string, string](EntrypointOpts{},
		func(_ runtime.Runtime, _, _ string, _ bool) (string, error) { return "", boom })

	var gotErr error
	for _, err := range e.Stream(context.Background(), "in", graph.Options{ThreadID: "1"}) {
		gotErr = err
	}
	if !errors.Is(gotErr, boom) {
		t.Fatalf("Stream() error = %v, want boom", gotErr)
	}
}

// Breaking out of the stream early cancels the run; the teardown still runs
// and the iterator terminates cleanly.
func TestEntrypointStreamEarlyBreak(t *testing.T) {
	e := NewEntrypoint[string, string, string](EntrypointOpts{},
		func(_ runtime.Runtime, in, _ string, _ bool) (string, error) { return "done:" + in, nil })

	chunks := 0
	for range e.Stream(context.Background(), "in", graph.Options{ThreadID: "1"}) {
		chunks++
		break
	}
	if chunks != 1 {
		t.Fatalf("consumed %d chunks before break, want 1", chunks)
	}
}

// panicPutSaver panics in Put, simulating a saver failure unwinding through
// the run in the caller's goroutine (sync durability).
type panicPutSaver struct {
	checkpoint.Saver
}

func (s *panicPutSaver) Put(context.Context, checkpoint.Config, checkpoint.Checkpoint, checkpoint.Metadata, map[string]int64) (checkpoint.Config, error) {
	panic("put boom")
}

// A panic unwinding through the graph run still gets the pinned teardown
// (cancel -> seal -> best-effort persist) and the ORIGINAL panic value
// continues to propagate out of Invoke.
func TestEntrypointInvokePanicTeardown(t *testing.T) {
	e := NewEntrypoint[string, string, string](
		EntrypointOpts{Checkpointer: &panicPutSaver{Saver: checkpoint.NewMemorySaver()}},
		func(_ runtime.Runtime, in, _ string, _ bool) (string, error) { return in, nil })

	defer func() {
		if r := recover(); r != "put boom" {
			t.Fatalf("recover = %v, want the original saver panic to propagate", r)
		}
	}()
	_, _ = e.Invoke(context.Background(), "in", graph.Options{ThreadID: "1"})
}
