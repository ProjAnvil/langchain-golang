package providerutil

import (
	"bufio"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/streamevents"
)

type recordingHandler struct {
	events []callbacks.Event
	err    error
}

func (h *recordingHandler) HandleEvent(_ context.Context, event callbacks.Event) error {
	if h.err != nil {
		return h.err
	}
	h.events = append(h.events, event)
	return nil
}

func TestNewSSEScannerAcceptsLinesBeyondDefaultLimit(t *testing.T) {
	// A tool-call argument delta larger than bufio.Scanner's 64KB default.
	line := `data: {"args":"` + strings.Repeat("a", 128*1024) + `"}`
	scanner := NewSSEScanner(strings.NewReader(line + "\n\n"))
	if !scanner.Scan() {
		t.Fatalf("scan failed on >64KB line: %v", scanner.Err())
	}
	if got := scanner.Text(); got != line {
		t.Fatalf("scanned line length = %d, want %d", len(got), len(line))
	}
}

func TestNewSSEScannerRejectsOversizedLines(t *testing.T) {
	scanner := NewSSEScanner(strings.NewReader(strings.Repeat("x", maxSSELineBytes+1) + "\n"))
	if scanner.Scan() {
		t.Fatal("expected scan to fail on line exceeding maxSSELineBytes")
	}
	if !errors.Is(scanner.Err(), bufio.ErrTooLong) {
		t.Fatalf("err = %v, want bufio.ErrTooLong", scanner.Err())
	}
}

func TestEmitNoCallbacksIsNoop(t *testing.T) {
	cfg := runnables.NewConfig()
	if err := Emit(context.Background(), cfg, callbacks.EventChatModelStart, nil, nil, nil); err != nil {
		t.Fatalf("Emit = %v", err)
	}
	if err := EmitStream(context.Background(), cfg, messages.AI("hi")); err != nil {
		t.Fatalf("EmitStream = %v", err)
	}
	var started bool
	if err := EmitProtocol(context.Background(), cfg, &started, streamevents.Event{Event: streamevents.EventContentBlockStart}); err != nil {
		t.Fatalf("EmitProtocol = %v", err)
	}
	if started {
		t.Fatal("protocol must not be marked started when callbacks are empty")
	}
}

func TestEmitCarriesErrorString(t *testing.T) {
	h := &recordingHandler{}
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(h)))
	boom := errors.New("boom")
	if err := Emit(context.Background(), cfg, callbacks.EventChatModelError, nil, nil, boom); err != nil {
		t.Fatalf("Emit = %v", err)
	}
	if len(h.events) != 1 {
		t.Fatalf("events = %d, want 1", len(h.events))
	}
	if h.events[0].Error != "boom" {
		t.Fatalf("event.Error = %q, want %q", h.events[0].Error, "boom")
	}
}

func TestEmitProtocolEmitsMessageStartExactlyOnce(t *testing.T) {
	h := &recordingHandler{}
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(h)))
	var started bool
	for i := 0; i < 3; i++ {
		if err := EmitProtocol(context.Background(), cfg, &started, streamevents.Event{
			Event: streamevents.EventContentBlockStart,
		}); err != nil {
			t.Fatalf("EmitProtocol #%d = %v", i, err)
		}
	}
	if len(h.events) != 4 {
		t.Fatalf("events = %d, want 4 (1 start + 3 content)", len(h.events))
	}
	first := h.events[0].Chunk.(streamevents.Event)
	if first.Event != streamevents.EventMessageStart {
		t.Fatalf("first chunk event = %v, want message_start", first.Event)
	}
}

func TestEmitProtocolEventHasNoStartBookkeeping(t *testing.T) {
	h := &recordingHandler{}
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(h)))
	if err := EmitProtocolEvent(context.Background(), cfg, streamevents.Event{Event: streamevents.EventContentBlockStart}); err != nil {
		t.Fatalf("EmitProtocolEvent = %v", err)
	}
	if err := EmitProtocolEvent(context.Background(), cfg, streamevents.Event{Event: streamevents.EventMessageFinish}); err != nil {
		t.Fatalf("EmitProtocolEvent = %v", err)
	}
	if len(h.events) != 2 {
		t.Fatalf("events = %d, want 2 with no implicit start", len(h.events))
	}
}

func TestEmitHandlerErrorPropagates(t *testing.T) {
	h := &recordingHandler{err: errors.New("handler down")}
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(h)))
	if err := EmitStream(context.Background(), cfg, messages.AI("hi")); err == nil {
		t.Fatal("EmitStream must propagate handler errors")
	}
}

func TestCloneMetadata(t *testing.T) {
	if got := CloneMetadata(nil); got != nil {
		t.Fatalf("CloneMetadata(nil) = %v, want nil", got)
	}
	original := map[string]any{"k": "v"}
	cloned := CloneMetadata(original)
	cloned["k2"] = "v2"
	if _, exists := original["k2"]; exists {
		t.Fatal("clone must not alias the original map")
	}
	if cloned["k"] != "v" {
		t.Fatalf("cloned[k] = %v, want v", cloned["k"])
	}
}
