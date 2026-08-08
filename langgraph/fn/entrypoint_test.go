package fn

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/langgraph/checkpoint"
	"github.com/projanvil/langchain-golang/langgraph/graph"
)

// Mirrors test_pregel.py:6307 test_entrypoint_without_checkpointer: with no
// checkpointer, previous is never injected (Python: previous is always None).
func TestEntrypointWithoutCheckpointer(t *testing.T) {
	var hasPrevs []bool
	var prevs []map[string]any
	e := NewEntrypoint[map[string]any, map[string]any, map[string]any](EntrypointOpts{},
		func(_ context.Context, in, prev map[string]any, hasPrev bool) (map[string]any, error) {
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
		func(_ context.Context, in, prev map[string]any, hasPrev bool) (map[string]any, error) {
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
		func(_ context.Context, in string, prev []string, hasPrev bool) (Final[int, []string], error) {
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
		func(_ context.Context, in, _ string, _ bool) (string, error) {
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
		func(ctx context.Context, _ any, _ any, _ bool) (string, error) {
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
		func(_ context.Context, _, _ string, _ bool) (string, error) {
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
