package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/callbacks"
)

// TestFinishReasonFields pins the stop_reason → shared-finish-fields mapping
// (Python langchain-anthropic #41926, _FINISH_REASONS): raw value kept,
// normalized vocabulary, suspended only for pause_turn.
func TestFinishReasonFields(t *testing.T) {
	cases := []struct {
		stopReason string
		normalized string
		state      string
	}{
		{"end_turn", "stop", "complete"},
		{"stop_sequence", "stop", "complete"},
		{"max_tokens", "length", "complete"},
		{"tool_use", "tool_call", "complete"},
		{"refusal", "refusal", "complete"},
		{"model_context_window_exceeded", "context_exceeded", "complete"},
		{"pause_turn", "other", "suspended"},
		{"something_new", "other", "complete"},
	}
	for _, tc := range cases {
		fields := finishReasonFields(tc.stopReason)
		if fields["finish_reason"] != tc.stopReason {
			t.Errorf("finish_reason(%q) = %v", tc.stopReason, fields["finish_reason"])
		}
		if fields["normalized_finish_reason"] != tc.normalized {
			t.Errorf("normalized(%q) = %v, want %q", tc.stopReason, fields["normalized_finish_reason"], tc.normalized)
		}
		if fields["finish_state"] != tc.state {
			t.Errorf("state(%q) = %v, want %q", tc.stopReason, fields["finish_state"], tc.state)
		}
	}
	if fields := finishReasonFields(""); fields != nil {
		t.Errorf("empty stop_reason: %#v, want nil", fields)
	}
}

// TestNonStreamingFinishReasonFields drives a non-streaming Messages API
// payload through Invoke and checks the assembled response_metadata carries
// the shared finish fields alongside the raw stop_reason (#41926).
func TestNonStreamingFinishReasonFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{
			"id":"msg_1",
			"type":"message",
			"role":"assistant",
			"model":"claude-sonnet-4-6",
			"stop_reason":"tool_use",
			"content":[{"type":"text","text":"hi"}],
			"usage":{"input_tokens":1,"output_tokens":2}
		}`)
	}))
	defer server.Close()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-sonnet-4-6"),
		modelconfig.WithAPIKey("k"),
	)
	out, err := model.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	md := out.ResponseMetadata
	if md["stop_reason"] != "tool_use" || md["finish_reason"] != "tool_use" ||
		md["normalized_finish_reason"] != "tool_call" || md["finish_state"] != "complete" {
		t.Fatalf("finish fields mismatch: %#v", md)
	}
}

// TestStreamFinishReasonFields drives a streamed message_delta stop_reason
// through Stream and checks the final message's response_metadata carries
// the shared finish fields (#41926).
func TestStreamFinishReasonFields(t *testing.T) {
	server := streamUsageServer(t, `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)
	defer server.Close()
	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		t.Context(),
		[]messages.Message{messages.Human("hi")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	chunks := drainStream(t, stream)
	// The final assembled message rides on EventChatModelEnd; the terminal
	// stop chunk carries the same fields for chunk-aggregating consumers.
	md := streamEndMessage(t, recorder).ResponseMetadata
	if md["stop_reason"] != "end_turn" || md["finish_reason"] != "end_turn" ||
		md["normalized_finish_reason"] != "stop" || md["finish_state"] != "complete" {
		t.Fatalf("stream finish fields mismatch: %#v", md)
	}
	var last messages.Message
	for _, chunk := range chunks {
		if chunk.ResponseMetadata["stop_reason"] == "end_turn" {
			last = chunk
		}
	}
	if last.ResponseMetadata["finish_reason"] != "end_turn" ||
		last.ResponseMetadata["normalized_finish_reason"] != "stop" {
		t.Fatalf("terminal stop chunk finish fields mismatch: %#v", last.ResponseMetadata)
	}
}

// streamEndMessage returns the message the recorder saw on EventChatModelEnd.
func streamEndMessage(t *testing.T, recorder *callbacks.Recorder) messages.Message {
	t.Helper()
	for _, event := range recorder.Events() {
		if event.Kind == callbacks.EventChatModelEnd {
			if msg, ok := event.Output.(messages.Message); ok {
				return msg
			}
			if raw, err := json.Marshal(event.Output); err == nil {
				var msg messages.Message
				if json.Unmarshal(raw, &msg) == nil {
					return msg
				}
			}
		}
	}
	t.Fatal("no EventChatModelEnd with message output recorded")
	return messages.Message{}
}
