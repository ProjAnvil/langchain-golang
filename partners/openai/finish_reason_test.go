package openai

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// TestChatFinishReasonFields pins the Chat Completions finish_reason →
// normalized mapping with the stream-observed corrections (#41926): refusal
// wins over a normal stop, and a forced tool_choice stop with tool calls
// becomes tool_call. Only normalized_finish_reason is produced — the raw
// value lands in response_metadata on its own.
func TestChatFinishReasonFields(t *testing.T) {
	cases := []struct {
		name         string
		finishReason string
		hasToolCalls bool
		refusal      bool
		normalized   string
	}{
		{"plain stop", "stop", false, false, "stop"},
		{"length", "length", false, false, "length"},
		{"tool_calls", "tool_calls", false, false, "tool_call"},
		{"content_filter", "content_filter", false, false, "content_filter"},
		{"forced choice stop with tools", "stop", true, false, "tool_call"},
		{"refusal wins over stop", "stop", false, true, "refusal"},
		{"refusal wins over tools", "stop", true, true, "refusal"},
		{"unknown", "funkychicken", false, false, "other"},
	}
	for _, tc := range cases {
		fields := chatFinishReasonFields(tc.finishReason, tc.hasToolCalls, tc.refusal)
		if got := fields["normalized_finish_reason"]; got != tc.normalized {
			t.Errorf("%s: normalized = %v, want %q", tc.name, got, tc.normalized)
		}
		if _, has := fields["finish_reason"]; has {
			t.Errorf("%s: unexpected finish_reason key", tc.name)
		}
	}
	if fields := chatFinishReasonFields("", false, false); fields != nil {
		t.Errorf("empty finish_reason: %#v, want nil", fields)
	}
}

// TestResponsesFinishReasonFields pins the Responses API status → shared
// finish fields mapping (#41926): finish_reason keeps the raw status, and an
// incomplete response resolves its incomplete_details.reason.
func TestResponsesFinishReasonFields(t *testing.T) {
	cases := []struct {
		name       string
		status     string
		incomplete map[string]any
		hasTools   bool
		refusal    bool
		raw        string
		normalized string
	}{
		{"completed", "completed", nil, false, false, "completed", "stop"},
		{"completed with tools", "completed", nil, true, false, "completed", "tool_call"},
		{"incomplete max output", "incomplete", map[string]any{"reason": "max_output_tokens"}, false, false, "incomplete", "length"},
		{"incomplete filter", "incomplete", map[string]any{"reason": "content_filter"}, false, false, "incomplete", "content_filter"},
		{"incomplete other", "incomplete", map[string]any{"reason": "weird"}, false, false, "incomplete", "other"},
		{"failed", "failed", nil, false, false, "failed", "error"},
		{"refusal", "completed", nil, false, true, "completed", "refusal"},
	}
	for _, tc := range cases {
		fields := responsesFinishReasonFields(tc.status, tc.incomplete, tc.hasTools, tc.refusal)
		if fields["finish_reason"] != tc.raw || fields["normalized_finish_reason"] != tc.normalized {
			t.Errorf("%s: %#v", tc.name, fields)
		}
	}
	if fields := responsesFinishReasonFields("", nil, false, false); fields != nil {
		t.Errorf("empty status: %#v, want nil", fields)
	}
}

// TestChatCompletionsFinishReasonFields drives a non-streaming Chat
// Completions payload with a forced tool_choice stop through Invoke and
// checks response_metadata carries the raw finish_reason plus the corrected
// normalized_finish_reason (#41926).
func TestChatCompletionsFinishReasonFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-1",
			"model":"gpt-test",
			"choices":[{
				"finish_reason":"stop",
				"message":{
					"role":"assistant",
					"content":"",
					"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":\"x\"}"}}]
				}
			}],
			"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	md := resp.ResponseMetadata
	if md["finish_reason"] != "stop" || md["normalized_finish_reason"] != "tool_call" {
		t.Fatalf("finish fields mismatch: %#v", md)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "search" {
		t.Fatalf("tool calls: %#v", resp.ToolCalls)
	}
}

// TestResponsesFinishReasonFieldsOnMessage drives a non-streaming Responses
// payload with status=incomplete through Invoke and checks response_metadata
// carries status plus the shared finish fields (#41926).
func TestResponsesFinishReasonFieldsOnMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_1",
			"model":"gpt-test",
			"status":"incomplete",
			"incomplete_details":{"reason":"max_output_tokens"},
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}],
			"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	md := resp.ResponseMetadata
	if md["status"] != "incomplete" || md["finish_reason"] != "incomplete" ||
		md["normalized_finish_reason"] != "length" {
		t.Fatalf("finish fields mismatch: %#v", md)
	}
	if resp.Content != "partial" {
		t.Fatalf("content = %q", resp.Content)
	}
}

// TestChatCompletionsStreamFinishReasonFields drives a streamed tool-call
// response whose final choice chunk carries finish_reason=stop (a forced
// tool_choice shape) and checks the terminal [DONE] chunk carries the shared
// finish fields with the tool_call correction (#41926).
func TestChatCompletionsStreamFinishReasonFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("go")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var terminal messages.Message
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		if chunk.ResponseMetadata["finish_reason"] != nil {
			terminal = chunk
		}
	}
	md := terminal.ResponseMetadata
	if md["finish_reason"] != "stop" || md["normalized_finish_reason"] != "tool_call" {
		t.Fatalf("terminal chunk finish fields mismatch: %#v", md)
	}
	// EventChatModelEnd carries the same fields on its output.
	for _, event := range recorder.Events() {
		if event.Kind == callbacks.EventChatModelEnd {
			if out, ok := event.Output.(messages.Message); ok {
				if out.ResponseMetadata["normalized_finish_reason"] != "tool_call" {
					t.Fatalf("end event finish fields: %#v", out.ResponseMetadata)
				}
			}
		}
	}
}

// TestChatCompletionsRefusalFinishReason ports the upstream
// test_create_chat_result_finish_reason refusal row: a normal stop with a
// refusal message normalizes to "refusal" (#41926).
func TestChatCompletionsRefusalFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-1",
			"model":"gpt-test",
			"choices":[{
				"finish_reason":"stop",
				"message":{"role":"assistant","content":null,"refusal":"I can't help."}
			}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	md := resp.ResponseMetadata
	if md["finish_reason"] != "stop" || md["normalized_finish_reason"] != "refusal" {
		t.Fatalf("finish fields mismatch: %#v", md)
	}
}

// TestChatCompletionsStreamRefusalFinishReason ports the upstream
// test_stream_refusal_finish_reason: a refusal delta followed by a stop
// finish_reason normalizes to "refusal" on the terminal chunk (#41926).
func TestChatCompletionsStreamRefusalFinishReason(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"refusal\":\"I can't help.\"}}]}\n\n")
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var terminal messages.Message
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
		if chunk.ResponseMetadata["finish_reason"] != nil {
			terminal = chunk
		}
	}
	md := terminal.ResponseMetadata
	if md["finish_reason"] != "stop" || md["normalized_finish_reason"] != "refusal" {
		t.Fatalf("terminal chunk finish fields mismatch: %#v", md)
	}
	// The refusal itself rides the terminal chunk as a RefusalBlock.
	foundRefusal := false
	for _, block := range terminal.ContentBlocks {
		if refusal, ok := block.(messages.RefusalBlock); ok && refusal.Refusal == "I can't help." {
			foundRefusal = true
		}
	}
	if !foundRefusal {
		t.Fatalf("terminal chunk refusal block: %#v", terminal.ContentBlocks)
	}
}

// TestResponsesUnknownStatusCoversDefault covers the unknown-status branch:
// anything but completed/incomplete/failed normalizes to "other".
func TestResponsesUnknownStatusCoversDefault(t *testing.T) {
	fields := responsesFinishReasonFields("queued", nil, false, false)
	if fields["normalized_finish_reason"] != "other" || fields["finish_reason"] != "queued" {
		t.Fatalf("unknown status: %#v", fields)
	}
}

// TestResponsesStreamIncompleteTerminal pins the response.incomplete
// terminal event (base.py:5749-5759): it assembles the final message like
// completed, with the incomplete_details.reason mapped onto the shared
// finish fields.
func TestResponsesStreamIncompleteTerminal(t *testing.T) {
	server := sseServer(t,
		"data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_i\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"partial\"}]}]}}\n\n")
	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(t.Context(), nil, runnables.WithCallbacks(callbacks.NewManager(recorder)))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	for {
		_, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
	}
	var end messages.Message
	for _, event := range recorder.Events() {
		if event.Kind == callbacks.EventChatModelEnd {
			if msg, ok := event.Output.(messages.Message); ok {
				end = msg
			}
		}
	}
	md := end.ResponseMetadata
	if md["finish_reason"] != "incomplete" || md["normalized_finish_reason"] != "length" {
		t.Fatalf("incomplete finish fields: %#v", md)
	}
	if end.Content != "partial" {
		t.Fatalf("content: %q", end.Content)
	}
}

// TestResponsesEmptyRefusalBlockNormalizesRefusal pins the presence-based
// refusal flag (base.py:5450-5453): a refusal block with empty text still
// normalizes to refusal.
func TestResponsesEmptyRefusalBlockNormalizesRefusal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"resp_1",
			"model":"gpt-test",
			"status":"completed",
			"output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":""}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("go")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if md := resp.ResponseMetadata; md["normalized_finish_reason"] != "refusal" {
		t.Fatalf("empty-text refusal block: %#v", md)
	}
}
