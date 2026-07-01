package tracers

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// Event is a normalized tracer event.
type Event struct {
	Name      string         `json:"name"`
	RunID     string         `json:"run_id,omitempty"`
	ParentID  string         `json:"parent_id,omitempty"`
	Time      time.Time      `json:"time"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	Error     string         `json:"error,omitempty"`
	EventType string         `json:"event_type,omitempty"`
}

// Tracer receives run lifecycle events.
type Tracer interface {
	OnStart(Event)
	OnEnd(Event)
	OnError(Event)
	Events() []Event
}

type contextKey struct{}

// ContextWithTracer returns a context carrying tracer.
func ContextWithTracer(ctx context.Context, tracer Tracer) context.Context {
	return context.WithValue(ctx, contextKey{}, tracer)
}

// TracerFromContext returns the tracer stored in ctx, if any.
func TracerFromContext(ctx context.Context) (Tracer, bool) {
	tracer, ok := ctx.Value(contextKey{}).(Tracer)
	return tracer, ok && tracer != nil
}

// ListenerTracer dispatches lifecycle events to optional hooks.
type ListenerTracer struct {
	OnStartFunc func(Event)
	OnEndFunc   func(Event)
	OnErrorFunc func(Event)
	RootOnly    bool
}

// NewListenerTracer creates a hook-backed tracer.
func NewListenerTracer(onStart func(Event), onEnd func(Event), onError func(Event)) ListenerTracer {
	return ListenerTracer{
		OnStartFunc: onStart,
		OnEndFunc:   onEnd,
		OnErrorFunc: onError,
	}
}

// NewRootListenerTracer creates a hook-backed tracer that only receives root
// run events.
func NewRootListenerTracer(onStart func(Event), onEnd func(Event), onError func(Event)) ListenerTracer {
	tracer := NewListenerTracer(onStart, onEnd, onError)
	tracer.RootOnly = true
	return tracer
}

func (t ListenerTracer) OnStart(event Event) {
	if t.skip(event) || t.OnStartFunc == nil {
		return
	}
	t.OnStartFunc(cloneEvent(event))
}

func (t ListenerTracer) OnEnd(event Event) {
	if t.skip(event) || t.OnEndFunc == nil {
		return
	}
	t.OnEndFunc(cloneEvent(event))
}

func (t ListenerTracer) OnError(event Event) {
	if t.skip(event) || t.OnErrorFunc == nil {
		return
	}
	t.OnErrorFunc(cloneEvent(event))
}

func (t ListenerTracer) Events() []Event { return nil }

func (t ListenerTracer) skip(event Event) bool {
	return t.RootOnly && event.ParentID != ""
}

// MemoryTracer records events in memory for tests and local inspection.
type MemoryTracer struct {
	mu          sync.Mutex
	events      []Event
	subscribers []chan Event
}

func NewMemoryTracer() *MemoryTracer {
	return &MemoryTracer{}
}

func (t *MemoryTracer) OnStart(event Event) { t.append("start", event) }
func (t *MemoryTracer) OnEnd(event Event)   { t.append("end", event) }
func (t *MemoryTracer) OnError(event Event) { t.append("error", event) }

func (t *MemoryTracer) Events() []Event {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([]Event, len(t.events))
	for i, event := range t.events {
		event.Metadata = cloneMap(event.Metadata)
		out[i] = event
	}
	return out
}

// Filter returns recorded events matching predicate. A nil predicate returns
// all events.
func (t *MemoryTracer) Filter(predicate func(Event) bool) []Event {
	events := t.Events()
	if predicate == nil {
		return events
	}
	out := []Event{}
	for _, event := range events {
		if predicate(event) {
			out = append(out, event)
		}
	}
	return out
}

// Replay sends recorded events to another tracer in original order.
func (t *MemoryTracer) Replay(target Tracer) {
	if target == nil {
		return
	}
	for _, event := range t.Events() {
		switch event.EventType {
		case "start":
			target.OnStart(event)
		case "end":
			target.OnEnd(event)
		case "error":
			target.OnError(event)
		}
	}
}

// Subscribe returns a channel that receives future events and a cancel
// function. The channel is buffered so tracing does not block normal execution.
func (t *MemoryTracer) Subscribe(buffer int) (<-chan Event, func()) {
	if buffer < 0 {
		buffer = 0
	}
	ch := make(chan Event, buffer)
	t.mu.Lock()
	t.subscribers = append(t.subscribers, ch)
	t.mu.Unlock()
	cancel := func() {
		t.mu.Lock()
		defer t.mu.Unlock()
		for i, subscriber := range t.subscribers {
			if subscriber == ch {
				t.subscribers = append(t.subscribers[:i], t.subscribers[i+1:]...)
				close(ch)
				return
			}
		}
	}
	return ch, cancel
}

func (t *MemoryTracer) append(kind string, event Event) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if event.Time.IsZero() {
		event.Time = time.Now()
	}
	event.EventType = kind
	event.Metadata = cloneMap(event.Metadata)
	t.events = append(t.events, event)
	for _, subscriber := range t.subscribers {
		select {
		case subscriber <- cloneEvent(event):
		default:
		}
	}
}

// StdOutTracer writes trace lifecycle events to a writer.
type StdOutTracer struct {
	Writer io.Writer
}

func NewStdOutTracer(writer io.Writer) StdOutTracer {
	if writer == nil {
		writer = os.Stdout
	}
	return StdOutTracer{Writer: writer}
}

func (t StdOutTracer) OnStart(event Event) { t.write("start", event) }
func (t StdOutTracer) OnEnd(event Event)   { t.write("end", event) }
func (t StdOutTracer) OnError(event Event) { t.write("error", event) }
func (t StdOutTracer) Events() []Event     { return nil }

func (t StdOutTracer) write(kind string, event Event) {
	writer := t.Writer
	if writer == nil {
		writer = os.Stdout
	}
	if event.Time.IsZero() {
		event.Time = time.Now()
	}
	if kind == "error" {
		_, _ = fmt.Fprintf(writer, "[trace/%s] %s run_id=%s parent_id=%s error=%s\n", kind, event.Name, event.RunID, event.ParentID, event.Error)
		return
	}
	_, _ = fmt.Fprintf(writer, "[trace/%s] %s run_id=%s parent_id=%s\n", kind, event.Name, event.RunID, event.ParentID)
}

// RunCollector collects copied run events for inspection.
type RunCollector struct {
	tracer *MemoryTracer
}

func NewRunCollector() *RunCollector {
	return &RunCollector{tracer: NewMemoryTracer()}
}

func (c *RunCollector) OnStart(event Event) { c.tracer.OnStart(event) }
func (c *RunCollector) OnEnd(event Event)   { c.tracer.OnEnd(event) }
func (c *RunCollector) OnError(event Event) { c.tracer.OnError(event) }
func (c *RunCollector) Events() []Event     { return c.tracer.Events() }

// Runs returns start events, which represent collected runs in this lightweight
// tracer model.
func (c *RunCollector) Runs() []Event {
	return c.tracer.Filter(func(event Event) bool { return event.EventType == "start" })
}

func cloneMap(values map[string]any) map[string]any {
	if values == nil {
		return nil
	}
	out := make(map[string]any, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

func cloneEvent(event Event) Event {
	event.Metadata = cloneMap(event.Metadata)
	return event
}
