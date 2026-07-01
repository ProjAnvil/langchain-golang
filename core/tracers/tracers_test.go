package tracers

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"
)

func TestMemoryTracerOrdersAndCopiesEvents(t *testing.T) {
	tracer := NewMemoryTracer()
	meta := map[string]any{"k": "v"}
	tracer.OnStart(Event{Name: "root", RunID: "1", Metadata: meta})
	tracer.OnEnd(Event{Name: "root", RunID: "1"})
	events := tracer.Events()
	if len(events) != 2 || events[0].EventType != "start" || events[1].EventType != "end" {
		t.Fatalf("unexpected events: %#v", events)
	}
	events[0].Metadata["k"] = "changed"
	if tracer.Events()[0].Metadata["k"] != "v" {
		t.Fatal("metadata was not copied")
	}
}

func TestMemoryTracerFilterReplayAndSubscribe(t *testing.T) {
	tracer := NewMemoryTracer()
	ch, cancel := tracer.Subscribe(1)
	defer cancel()
	tracer.OnStart(Event{Name: "root", RunID: "1"})
	select {
	case event := <-ch:
		if event.Name != "root" || event.EventType != "start" {
			t.Fatalf("unexpected subscribed event: %#v", event)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for subscribed event")
	}
	tracer.OnError(Event{Name: "root", RunID: "1", Error: "boom"})
	errors := tracer.Filter(func(event Event) bool { return event.EventType == "error" })
	if len(errors) != 1 || errors[0].Error != "boom" {
		t.Fatalf("filtered errors: %#v", errors)
	}
	target := NewMemoryTracer()
	tracer.Replay(target)
	if len(target.Events()) != 2 || target.Events()[1].EventType != "error" {
		t.Fatalf("replayed events: %#v", target.Events())
	}
}

func TestContextTracerRoundTrip(t *testing.T) {
	tracer := NewMemoryTracer()
	ctx := ContextWithTracer(context.Background(), tracer)
	got, ok := TracerFromContext(ctx)
	if !ok || got != tracer {
		t.Fatalf("tracer from context: %#v ok=%v", got, ok)
	}
	if _, ok := TracerFromContext(context.Background()); ok {
		t.Fatal("unexpected tracer in empty context")
	}
}

func TestListenerTracerAndRootFiltering(t *testing.T) {
	events := []Event{}
	listener := NewRootListenerTracer(
		func(event Event) { events = append(events, event) },
		func(event Event) { events = append(events, event) },
		func(event Event) { events = append(events, event) },
	)
	meta := map[string]any{"key": "value"}
	listener.OnStart(Event{Name: "root", RunID: "root", Metadata: meta})
	listener.OnStart(Event{Name: "child", RunID: "child", ParentID: "root"})
	listener.OnError(Event{Name: "root", RunID: "root", Metadata: meta, Error: "boom"})
	meta["key"] = "mutated"

	if len(events) != 2 {
		t.Fatalf("events: %#v", events)
	}
	if events[0].Name != "root" || events[1].Error != "boom" {
		t.Fatalf("events: %#v", events)
	}
	if events[0].Metadata["key"] != "value" {
		t.Fatalf("event metadata was not copied: %#v", events[0].Metadata)
	}
}

func TestStdOutTracer(t *testing.T) {
	var buffer bytes.Buffer
	tracer := NewStdOutTracer(&buffer)
	tracer.OnStart(Event{Name: "tool", RunID: "child", ParentID: "root"})
	tracer.OnError(Event{Name: "tool", RunID: "child", ParentID: "root", Error: "failed"})
	got := buffer.String()
	for _, want := range []string{"[trace/start]", "tool", "run_id=child", "parent_id=root", "error=failed"} {
		if !strings.Contains(got, want) {
			t.Fatalf("stdout %q missing %q", got, want)
		}
	}
}

func TestRunCollector(t *testing.T) {
	collector := NewRunCollector()
	collector.OnStart(Event{Name: "root", RunID: "1"})
	collector.OnStart(Event{Name: "child", RunID: "2", ParentID: "1"})
	collector.OnEnd(Event{Name: "child", RunID: "2", ParentID: "1"})
	runs := collector.Runs()
	if len(runs) != 2 || runs[1].ParentID != "1" {
		t.Fatalf("runs: %#v", runs)
	}
	events := collector.Events()
	events[0].Metadata = map[string]any{"mutated": true}
	if len(collector.Events()[0].Metadata) != 0 {
		t.Fatal("collector events were not copied")
	}
}
