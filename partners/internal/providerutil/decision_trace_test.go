package providerutil

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// TestDecisionStartEndNoCallbacks pins the no-op path: without registered
// callbacks neither function emits anything (the provider-event convention —
// Emit/EmitStream behave identically).
func TestDecisionStartEndNoCallbacks(t *testing.T) {
	recorder := callbacks.NewRecorder()
	// A manager with a recorder is required for emission; without options
	// (empty cfg) nothing may reach any handler.
	DecisionStart(t.Context(), runnables.NewConfig(), "p", "m", "in")
	DecisionEnd(t.Context(), runnables.NewConfig(), "p", "m", nil, nil, nil)
	if len(recorder.Events()) != 0 {
		t.Fatalf("no-callbacks path emitted events: %#v", recorder.Events())
	}
}

// TestDecisionStartCarriesProviderIdentity pins the start event: provider
// identity layered over the caller's config (run ID, parent, tags, caller
// metadata kept, ls_* added), mirroring _traced_config.
func TestDecisionStartCarriesProviderIdentity(t *testing.T) {
	recorder := callbacks.NewRecorder()
	cfg := runnables.NewConfig(
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
		runnables.WithRunID("run-1"),
		runnables.WithParentID("parent-1"),
		runnables.WithTags("team:x"),
		runnables.WithMetadata("team", "support"),
	)
	DecisionStart(t.Context(), cfg, "fake", "fake-model", "input")
	events := recorder.Events()
	if len(events) != 1 || events[0].Kind != callbacks.EventChatModelStart {
		t.Fatalf("start events: %#v", events)
	}
	event := events[0]
	if event.RunID != "run-1" || event.ParentID != "parent-1" {
		t.Fatalf("run identity: %#v", event)
	}
	if len(event.Tags) != 1 || event.Tags[0] != "team:x" {
		t.Fatalf("tags: %#v", event.Tags)
	}
	if event.Metadata["team"] != "support" || event.Metadata["ls_provider"] != "fake" ||
		event.Metadata["ls_model_name"] != "fake-model" || event.Metadata["ls_model_type"] != "chat" {
		t.Fatalf("metadata: %#v", event.Metadata)
	}
	if event.Input != "input" {
		t.Fatalf("input: %#v", event.Input)
	}
}

// TestDecisionEndSuccessAndUsage pins the end event: the typed output rides
// along, and reported usage lands on usage_metadata.
func TestDecisionEndSuccessAndUsage(t *testing.T) {
	recorder := callbacks.NewRecorder()
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(recorder)))
	DecisionEnd(t.Context(), cfg, "fake", "fake-model", "response", &messages.UsageMetadata{
		InputTokens: 3, OutputTokens: 4, TotalTokens: 7,
	}, nil)
	events := recorder.Events()
	if len(events) != 1 || events[0].Kind != callbacks.EventChatModelEnd {
		t.Fatalf("end events: %#v", events)
	}
	event := events[0]
	if event.Output != "response" || event.Error != "" {
		t.Fatalf("output/error: %#v", event)
	}
	usage, _ := event.Metadata["usage_metadata"].(map[string]any)
	if usage["input_tokens"] != 3 || usage["total_tokens"] != 7 {
		t.Fatalf("usage metadata: %#v", event.Metadata)
	}
}

// TestDecisionEndErrorWithoutUsage pins the error event: a failure emits
// EventChatModelError with the message and no usage block.
func TestDecisionEndErrorWithoutUsage(t *testing.T) {
	recorder := callbacks.NewRecorder()
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(recorder)))
	DecisionEnd(t.Context(), cfg, "fake", "fake-model", nil, nil, errors.New("boom"))
	events := recorder.Events()
	if len(events) != 1 || events[0].Kind != callbacks.EventChatModelError {
		t.Fatalf("error events: %#v", events)
	}
	if events[0].Error != "boom" {
		t.Fatalf("error text: %q", events[0].Error)
	}
	if _, has := events[0].Metadata["usage_metadata"]; has {
		t.Fatalf("error event must carry no usage: %#v", events[0].Metadata)
	}
}

// rejectingHandler rejects every event.
type rejectingHandler struct{}

func (*rejectingHandler) HandleEvent(_ context.Context, _ callbacks.Event) error {
	return errors.New("tracer down")
}

// TestDecisionEmitFailureIsSwallowed pins the tracing-failure contract: a
// handler rejecting the event never propagates — classification succeeds
// regardless of tracer health.
func TestDecisionEmitFailureIsSwallowed(t *testing.T) {
	cfg := runnables.NewConfig(runnables.WithCallbacks(callbacks.NewManager(&rejectingHandler{})))
	DecisionStart(t.Context(), cfg, "fake", "fake-model", "input")
	DecisionEnd(t.Context(), cfg, "fake", "fake-model", nil, nil, nil)
}

// TestDecisionStartEndZeroConfig covers the zero-value Config path: a
// config with nil metadata still emits, with the decision identity as the
// only metadata.
func TestDecisionStartEndZeroConfig(t *testing.T) {
	recorder := callbacks.NewRecorder()
	cfg := runnables.Config{Callbacks: callbacks.NewManager(recorder)}
	DecisionStart(t.Context(), cfg, "fake", "fake-model", nil)
	DecisionEnd(t.Context(), cfg, "fake", "fake-model", nil, nil, nil)
	events := recorder.Events()
	if len(events) != 2 {
		t.Fatalf("events: %d, want start+end", len(events))
	}
	if events[0].Metadata["ls_provider"] != "fake" || events[1].Metadata["ls_model_name"] != "fake-model" {
		t.Fatalf("decision identity on zero config: %#v %#v", events[0].Metadata, events[1].Metadata)
	}
}
