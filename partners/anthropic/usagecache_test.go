package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// invokeServer serves one canned non-streaming message response.
func invokeServer(t *testing.T, capture *map[string]any, responseBody string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if capture != nil {
			*capture = map[string]any{}
			if err := json.NewDecoder(r.Body).Decode(capture); err != nil {
				t.Errorf("decode request: %v", err)
			}
		}
		_, _ = fmt.Fprint(w, responseBody)
	}))
}

// invokeCacheModel runs one Invoke against server and returns the response.
func invokeCacheModel(t *testing.T, server *httptest.Server, opts ...modelconfig.Option) messages.Message {
	t.Helper()
	model := NewChatModel(append([]modelconfig.Option{
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("claude-test"),
	}, opts...)...)
	response, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return response
}

// TestInvokeUsageCacheDetailsEphemeral pins Python's _create_usage_metadata
// (langchain_anthropic/chat_models.py:2356) for the newer usage shape that
// carries the cache_creation TTL breakdown: Anthropic's input_tokens excludes
// cached tokens, so the true input total adds cache_read plus the cache
// creation count — and when specific ephemeral tokens are reported, the
// generic count is superseded to avoid double counting.
func TestInvokeUsageCacheDetailsEphemeral(t *testing.T) {
	server := invokeServer(t, nil, `{
		"id":"msg_cache","type":"message","role":"assistant","model":"claude-test",
		"stop_reason":"end_turn",
		"content":[{"type":"text","text":"ok"}],
		"usage":{
			"input_tokens":10,
			"output_tokens":5,
			"cache_read_input_tokens":100,
			"cache_creation_input_tokens":50,
			"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":10}
		}
	}`)
	defer server.Close()

	response := invokeCacheModel(t, server)
	usage := response.UsageMetadata
	if usage.InputTokens != 150 || usage.OutputTokens != 5 || usage.TotalTokens != 155 {
		t.Fatalf("usage: %+v", usage)
	}
	details := usage.InputTokenDetails
	if details == nil {
		t.Fatalf("usage details missing: %+v", usage)
	}
	if details.CacheReadInputTokens != 100 {
		t.Fatalf("cache read tokens: %+v", details)
	}
	// Go core's InputTokenDetails has no ephemeral_5m/1h fields, so the
	// specific ephemeral total (30+10=40 — deliberately distinct from the
	// generic 50 so the specific-supersedes-generic branch is pinned) folds
	// into the flat cache-creation count instead of Python's zeroed generic
	// count.
	if details.CacheCreationInputTokens != 40 {
		t.Fatalf("cache creation tokens: %+v", details)
	}
}

// TestInvokeUsageCacheDetailsGenericOnly covers the classic usage shape with
// flat cache counts and no cache_creation TTL dict.
func TestInvokeUsageCacheDetailsGenericOnly(t *testing.T) {
	server := invokeServer(t, nil, `{
		"id":"msg_cache","type":"message","role":"assistant","model":"claude-test",
		"content":[{"type":"text","text":"ok"}],
		"usage":{
			"input_tokens":10,
			"output_tokens":5,
			"cache_read_input_tokens":100,
			"cache_creation_input_tokens":40
		}
	}`)
	defer server.Close()

	response := invokeCacheModel(t, server)
	usage := response.UsageMetadata
	if usage.InputTokens != 150 || usage.OutputTokens != 5 || usage.TotalTokens != 155 {
		t.Fatalf("usage: %+v", usage)
	}
	details := usage.InputTokenDetails
	if details == nil {
		t.Fatalf("usage details missing: %+v", usage)
	}
	if details.CacheReadInputTokens != 100 || details.CacheCreationInputTokens != 40 {
		t.Fatalf("usage details: %+v", details)
	}
}

// TestInvokeUsageNoCacheFieldsUnchanged is the regression guard: responses
// without cache fields produce the same UsageMetadata as before (no
// InputTokenDetails).
func TestInvokeUsageNoCacheFieldsUnchanged(t *testing.T) {
	server := invokeServer(t, nil, `{
		"id":"msg_plain","type":"message","role":"assistant","model":"claude-test",
		"content":[{"type":"text","text":"ok"}],
		"usage":{"input_tokens":2,"output_tokens":3}
	}`)
	defer server.Close()

	response := invokeCacheModel(t, server)
	usage := response.UsageMetadata
	if usage.InputTokens != 2 || usage.OutputTokens != 3 || usage.TotalTokens != 5 {
		t.Fatalf("usage: %+v", usage)
	}
	if usage.InputTokenDetails != nil {
		t.Fatalf("usage details should stay nil without cache fields: %+v", usage.InputTokenDetails)
	}
}

// TestStreamUsageCacheDetailsFromMessageStart: the older API shape reports
// complete usage (with cache fields) at message_start and only output_tokens
// at message_delta; the final output must keep the cache details.
func TestStreamUsageCacheDetailsFromMessageStart(t *testing.T) {
	server := streamServer(
		sse("message_start", `{"type":"message_start","message":{"id":"msg_c1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":10,"output_tokens":0,"cache_read_input_tokens":100,"cache_creation_input_tokens":50,"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":20}}}}`),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)

	output := finalOutput(t, recorder)
	usage := output.UsageMetadata
	if usage.InputTokens != 160 || usage.OutputTokens != 7 || usage.TotalTokens != 167 {
		t.Fatalf("output usage: %+v", usage)
	}
	details := usage.InputTokenDetails
	if details == nil {
		t.Fatalf("output usage details missing: %+v", usage)
	}
	if details.CacheReadInputTokens != 100 || details.CacheCreationInputTokens != 50 {
		t.Fatalf("output usage details: %+v", details)
	}
}

// TestStreamUsageCacheDetailsFromMessageDelta: the newer API shape reports
// complete usage (with cache fields) at message_delta; those cache details
// must reach the final output too.
func TestStreamUsageCacheDetailsFromMessageDelta(t *testing.T) {
	server := streamServer(
		sse("message_start", streamMessageStart),
		sse("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		sse("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`),
		sse("content_block_stop", `{"type":"content_block_stop","index":0}`),
		sse("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":7,"cache_read_input_tokens":100,"cache_creation_input_tokens":50,"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":20}}}`),
		sse("message_stop", `{"type":"message_stop"}`),
	)
	defer server.Close()

	stream, recorder := streamWithRecorder(t, server)
	drainStream(t, stream)

	output := finalOutput(t, recorder)
	usage := output.UsageMetadata
	if usage.InputTokens != 160 || usage.OutputTokens != 7 || usage.TotalTokens != 167 {
		t.Fatalf("output usage: %+v", usage)
	}
	details := usage.InputTokenDetails
	if details == nil {
		t.Fatalf("output usage details missing: %+v", usage)
	}
	if details.CacheReadInputTokens != 100 || details.CacheCreationInputTokens != 50 {
		t.Fatalf("output usage details: %+v", details)
	}
}

// TestRequestStopSequences pins the stop_sequences constructor option
// (Python ChatAnthropic's stop_sequences field, chat_models.py:944) reaching
// the non-streaming request payload.
func TestRequestStopSequences(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	response := invokeCacheModel(t, server, WithStopSequences([]string{"END", "STOP"}))
	if response.Content != "ok" {
		t.Fatalf("response: %+v", response)
	}
	stops, ok := request["stop_sequences"].([]any)
	if !ok || len(stops) != 2 || stops[0] != "END" || stops[1] != "STOP" {
		t.Fatalf("stop_sequences: %#v", request["stop_sequences"])
	}
}

// TestRequestStopSequencesStream pins stop_sequences reaching the streaming
// request payload as well.
func TestRequestStopSequencesStream(t *testing.T) {
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		request = map[string]any{}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			sse("message_start", streamMessageStart)+sse("message_stop", `{"type":"message_stop"}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		WithStopSequences([]string{"DONE"}),
	)
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	for {
		_, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		if !ok {
			break
		}
	}

	stops, ok := request["stop_sequences"].([]any)
	if !ok || len(stops) != 1 || stops[0] != "DONE" {
		t.Fatalf("stop_sequences: %#v", request["stop_sequences"])
	}
	if request["stream"] != true {
		t.Fatalf("stream flag: %#v", request["stream"])
	}
}

// TestRequestStopSequencesOmittedByDefault is the regression guard: without
// the option the payload carries no stop_sequences key at all.
func TestRequestStopSequencesOmittedByDefault(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	_ = invokeCacheModel(t, server)
	if _, present := request["stop_sequences"]; present {
		t.Fatalf("stop_sequences should be omitted: %#v", request["stop_sequences"])
	}
}
