package tracers

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestListenerTracerOnEndAndEvents(t *testing.T) {
	var ended []Event
	listener := NewListenerTracer(nil, func(event Event) { ended = append(ended, event) }, nil)

	// Nil hooks must be safe no-ops.
	listener.OnStart(Event{Name: "root", RunID: "1"})
	listener.OnError(Event{Name: "root", RunID: "1", Error: "boom"})

	listener.OnEnd(Event{Name: "root", RunID: "1"})
	if len(ended) != 1 || ended[0].Name != "root" {
		t.Fatalf("ended events: %#v", ended)
	}
	if events := listener.Events(); events != nil {
		t.Fatalf("listener Events() should be nil, got %#v", events)
	}
}

func TestListenerTracerNilHooksAreNoOps(t *testing.T) {
	listener := ListenerTracer{}
	listener.OnStart(Event{Name: "root"})
	listener.OnEnd(Event{Name: "root"})
	listener.OnError(Event{Name: "root", Error: "boom"})
	if listener.Events() != nil {
		t.Fatal("expected nil events from empty listener")
	}
}

func TestListenerTracerRootOnlySkipsChildEndAndError(t *testing.T) {
	var calls int
	hook := func(Event) { calls++ }
	listener := NewRootListenerTracer(hook, hook, hook)
	listener.OnEnd(Event{Name: "child", RunID: "2", ParentID: "1"})
	listener.OnError(Event{Name: "child", RunID: "2", ParentID: "1", Error: "boom"})
	listener.OnEnd(Event{Name: "root", RunID: "1"})
	listener.OnError(Event{Name: "root", RunID: "1", Error: "boom"})
	if calls != 2 {
		t.Fatalf("expected 2 root calls, got %d", calls)
	}
}

func TestMemoryTracerFilterNilPredicate(t *testing.T) {
	tracer := NewMemoryTracer()
	tracer.OnStart(Event{Name: "root", RunID: "1"})
	tracer.OnEnd(Event{Name: "root", RunID: "1"})
	all := tracer.Filter(nil)
	if len(all) != 2 {
		t.Fatalf("nil predicate should return all events, got %#v", all)
	}
}

func TestMemoryTracerReplayNilTarget(t *testing.T) {
	tracer := NewMemoryTracer()
	tracer.OnStart(Event{Name: "root", RunID: "1"})
	tracer.Replay(nil) // must not panic
	if len(tracer.Events()) != 1 {
		t.Fatal("replay to nil target should not mutate source")
	}
}

func TestMemoryTracerSubscribeNegativeBufferAndCancel(t *testing.T) {
	tracer := NewMemoryTracer()
	ch, cancel := tracer.Subscribe(-1)
	// Negative buffer is clamped to 0 (unbuffered), so non-blocking delivery
	// drops events when no receiver is ready; the tracer must not block.
	tracer.OnStart(Event{Name: "root", RunID: "1"})
	select {
	case event := <-ch:
		t.Fatalf("expected dropped event on unbuffered subscriber, got %#v", event)
	default:
	}

	cancel()
	// After cancel the channel is closed and further events are not delivered.
	if _, ok := <-ch; ok {
		t.Fatal("expected closed channel after cancel")
	}
	tracer.OnEnd(Event{Name: "root", RunID: "1"})
	if len(tracer.Events()) != 2 {
		t.Fatalf("tracer should still record events, got %#v", tracer.Events())
	}

	// Cancelling twice must be safe.
	cancel()
}

func TestMemoryTracerSubscriberBackpressure(t *testing.T) {
	tracer := NewMemoryTracer()
	ch, cancel := tracer.Subscribe(1)
	defer cancel()
	// Fill the buffer; subsequent events must be dropped, not block.
	tracer.OnStart(Event{Name: "first"})
	tracer.OnStart(Event{Name: "second"})
	event := <-ch
	if event.Name != "first" {
		t.Fatalf("expected first event, got %#v", event)
	}
	select {
	case extra := <-ch:
		t.Fatalf("expected dropped event, got %#v", extra)
	case <-time.After(50 * time.Millisecond):
	}
	if len(tracer.Events()) != 2 {
		t.Fatal("dropped subscriber events must still be recorded")
	}
}

func TestStdOutTracerDefaultsAndEnd(t *testing.T) {
	tracer := NewStdOutTracer(nil)
	if tracer.Writer != os.Stdout {
		t.Fatalf("nil writer should default to os.Stdout, got %#v", tracer.Writer)
	}
	if events := tracer.Events(); events != nil {
		t.Fatalf("stdout tracer Events() should be nil, got %#v", events)
	}

	var buffer bytes.Buffer
	tracer = NewStdOutTracer(&buffer)
	tracer.OnEnd(Event{Name: "chain", RunID: "1", ParentID: "0"})
	got := buffer.String()
	for _, want := range []string{"[trace/end]", "chain", "run_id=1", "parent_id=0"} {
		if !strings.Contains(got, want) {
			t.Fatalf("stdout %q missing %q", got, want)
		}
	}
}

func TestStdOutTracerZeroValueWritesToStdout(t *testing.T) {
	// A zero-value StdOutTracer falls back to os.Stdout.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	original := os.Stdout
	os.Stdout = w
	defer func() { os.Stdout = original }()

	tracer := StdOutTracer{}
	tracer.OnStart(Event{Name: "root", RunID: "1"})

	os.Stdout = original
	_ = w.Close()
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	if !strings.Contains(string(out), "[trace/start] root run_id=1") {
		t.Fatalf("unexpected stdout output: %q", out)
	}
}

func TestRunCollectorOnError(t *testing.T) {
	collector := NewRunCollector()
	collector.OnError(Event{Name: "root", RunID: "1", Error: "boom"})
	events := collector.Events()
	if len(events) != 1 || events[0].EventType != "error" || events[0].Error != "boom" {
		t.Fatalf("events: %#v", events)
	}
	if runs := collector.Runs(); len(runs) != 0 {
		t.Fatalf("error events are not runs: %#v", runs)
	}
}

func TestMemoryTracerSetsZeroTime(t *testing.T) {
	tracer := NewMemoryTracer()
	tracer.OnStart(Event{Name: "root"})
	if tracer.Events()[0].Time.IsZero() {
		t.Fatal("expected tracer to stamp zero event times")
	}
	stamped := time.Now()
	tracer.OnEnd(Event{Name: "root", Time: stamped})
	if !tracer.Events()[1].Time.Equal(stamped) {
		t.Fatal("expected tracer to preserve explicit event times")
	}
}
