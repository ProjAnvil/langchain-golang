// Tests for the middleware TracePolicyProvider hook (langchain 1.3.15's
// middleware trace_policy, #38910): a middleware controls traced payloads for
// everything inside its model-call wrapping layer, applied on the emit side
// via callbacks.NewPayloadPolicyManager so every tracer sees scrubbed
// payloads.
//
// This is an external test package (middleware_test): the tests drive
// agents.CreateAgent end-to-end, and the agents package imports this one, so
// an in-package test would be an import cycle.
package middleware_test

import (
	"context"
	"maps"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/langchain/agents"
	"github.com/projanvil/langchain-golang/langchain/agents/middleware"
)

// tracePolicyModel is a minimal ChatModel double that plays the tracer
// attachment point: on every Invoke it emits a chat_model_start event — with
// a payload standing in for what a provider tracer would record, PII
// included — through the callback manager it finds in its context. The
// sequenceModel pattern (create_agent_test.go) minus the sequencing.
type tracePolicyModel struct {
	response messages.Message
}

func (m *tracePolicyModel) Invoke(ctx context.Context, input []messages.Message, _ ...runnables.Option) (messages.Message, error) {
	if manager, ok := callbacks.ManagerFromContext(ctx); ok {
		payload := map[string]any{
			"messages": input,
			"ssn":      "123-45-6789",
		}
		_ = manager.Emit(ctx, callbacks.Event{Kind: callbacks.EventChatModelStart, Name: "model", Input: payload})
	}
	return m.response, nil
}

func (m *tracePolicyModel) Batch(ctx context.Context, inputs [][]messages.Message, opts ...runnables.Option) ([]messages.Message, error) {
	out := make([]messages.Message, len(inputs))
	for i, in := range inputs {
		var err error
		out[i], err = m.Invoke(ctx, in, opts...)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *tracePolicyModel) Stream(context.Context, []messages.Message, ...runnables.Option) (runnables.Stream[messages.Message], error) {
	return nil, context.DeadlineExceeded
}

func (m *tracePolicyModel) InputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}
func (m *tracePolicyModel) OutputSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{})
}

func (m *tracePolicyModel) BindTools([]coretools.Tool) (language.ChatModel, error) { return m, nil }

func (m *tracePolicyModel) Capabilities() language.ChatModelCapabilities {
	return language.ChatModelCapabilities{}
}

// passthroughMiddleware wraps the model call without touching it and provides
// NO trace policy — the control: it must not affect traced payloads.
type passthroughMiddleware struct{}

func (m *passthroughMiddleware) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	return handler(ctx, request)
}

// tracePolicyMiddleware wraps the model call like passthroughMiddleware but
// also implements middleware.TracePolicyProvider.
type tracePolicyMiddleware struct {
	processInputs func(value any) any
}

func (m *tracePolicyMiddleware) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	return handler(ctx, request)
}

func (m *tracePolicyMiddleware) TracePolicy() middleware.TracePolicyConfig {
	return middleware.TracePolicyConfig{ProcessInputs: m.processInputs}
}

// omittingMiddleware is the outer layer of the nested-composition test: its
// policy replaces the whole payload with "<omitted>". A distinct type because
// CreateAgent rejects duplicate middleware instances (same-type dedupe).
type omittingMiddleware struct{}

func (m *omittingMiddleware) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	return handler(ctx, request)
}

func (m *omittingMiddleware) TracePolicy() middleware.TracePolicyConfig {
	return middleware.TracePolicyConfig{ProcessInputs: func(any) any { return "<omitted>" }}
}

// redactingMiddleware is the inner layer of the nested-composition test.
type redactingMiddleware struct{}

func (m *redactingMiddleware) WrapModelCall(ctx context.Context, request middleware.ModelRequest, handler middleware.ModelHandler) (middleware.ModelResponse, error) {
	return handler(ctx, request)
}

func (m *redactingMiddleware) TracePolicy() middleware.TracePolicyConfig {
	return middleware.TracePolicyConfig{ProcessInputs: redactSSN}
}

// redactSSN is a PII-scrubbing transform: it copies the payload map and
// replaces the "ssn" value with "[REDACTED]", leaving other values alone.
func redactSSN(value any) any {
	payload, ok := value.(map[string]any)
	if !ok {
		return value
	}
	out := maps.Clone(payload)
	out["ssn"] = "[REDACTED]"
	return out
}

// invokeTracePolicyAgent runs an agent built from model and mw over one human
// message with a recorder-backed tracer installed in the run context, and
// returns the events the tracer observed.
func invokeTracePolicyAgent(t *testing.T, model language.ChatModel, mw ...any) []callbacks.Event {
	t.Helper()
	agent, err := agents.CreateAgent(model, nil, agents.WithAgentMiddleware(mw...))
	if err != nil {
		t.Fatalf("CreateAgent: %v", err)
	}
	recorder := callbacks.NewRecorder()
	ctx := callbacks.ContextWithManager(t.Context(), callbacks.NewManager(recorder))
	out, err := agent.Invoke(ctx, []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if len(out) == 0 || out[len(out)-1].Content != "done" {
		t.Fatalf("agent output = %#v, want the model's response", out)
	}
	return recorder.Events()
}

// tracePolicyChatStart returns the single chat_model_start event.
func tracePolicyChatStart(t *testing.T, events []callbacks.Event) callbacks.Event {
	t.Helper()
	for _, event := range events {
		if event.Kind == callbacks.EventChatModelStart {
			return event
		}
	}
	t.Fatalf("no chat_model_start event in %d events", len(events))
	return callbacks.Event{}
}

func TestMiddlewareTracePolicyScrubsModelPayloads(t *testing.T) {
	// Control: a middleware without a trace policy leaves payloads untouched
	// (the raw SSN reaches the tracer).
	model := &tracePolicyModel{response: messages.AI("done")}
	control := tracePolicyChatStart(t, invokeTracePolicyAgent(t, model, &passthroughMiddleware{}))
	controlPayload, ok := control.Input.(map[string]any)
	if !ok {
		t.Fatalf("control Input = %#v, want a map", control.Input)
	}
	if got, want := controlPayload["ssn"], "123-45-6789"; got != want {
		t.Fatalf("control ssn = %v, want %q (no policy installed)", got, want)
	}

	// Scrubbed: the TracePolicyProvider middleware redacts the SSN before the
	// tracer observes it; the rest of the payload is untouched.
	scrubbed := tracePolicyChatStart(t, invokeTracePolicyAgent(t, model,
		&passthroughMiddleware{},
		&tracePolicyMiddleware{processInputs: redactSSN},
	))
	payload, ok := scrubbed.Input.(map[string]any)
	if !ok {
		t.Fatalf("scrubbed Input = %#v, want a map", scrubbed.Input)
	}
	if got, want := payload["ssn"], "[REDACTED]"; got != want {
		t.Fatalf("scrubbed ssn = %v, want %q", got, want)
	}
	msgs, ok := payload["messages"].([]messages.Message)
	if !ok || len(msgs) != 1 || msgs[0].Content != "hi" {
		t.Fatalf("scrubbed messages = %#v, want them preserved by the transform", payload["messages"])
	}
}

func TestNestedMiddlewarePoliciesCompose(t *testing.T) {
	// Onion composition: mws[0] is the outermost layer, so its policy is
	// installed in the context first and applied LAST at emit time. The inner
	// middleware's redaction runs first, then the outer policy replaces the
	// whole payload — the tracer only ever sees "<omitted>".
	model := &tracePolicyModel{response: messages.AI("done")}
	start := tracePolicyChatStart(t, invokeTracePolicyAgent(t, model,
		&omittingMiddleware{},  // mws[0]: outermost
		&redactingMiddleware{}, // inner
	))

	if got, want := start.Input, "<omitted>"; got != want {
		t.Fatalf("nested-policy Input = %#v, want %q (outermost transform wins)", got, want)
	}
}
