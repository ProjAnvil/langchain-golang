package openai

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// captureBodyServer spins up a server that records the decoded JSON request
// body and replies with fixed JSON (non-streaming endpoints).
func captureBodyServer(t *testing.T, reply string) (url string, gotBody func() map[string]any) {
	t.Helper()
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(server.Close)
	return server.URL, func() map[string]any { return body }
}

func contentParts(t *testing.T, msg map[string]any) []map[string]any {
	t.Helper()
	raw, ok := msg["content"].([]any)
	if !ok {
		t.Fatalf("content is not an array: %#v", msg["content"])
	}
	parts := make([]map[string]any, 0, len(raw))
	for _, p := range raw {
		m, ok := p.(map[string]any)
		if !ok {
			t.Fatalf("content part is not an object: %#v", p)
		}
		parts = append(parts, m)
	}
	return parts
}

func multimodalHuman() messages.Message {
	return messages.Human("ignored when blocks are present").WithContentBlocks([]messages.ContentBlock{
		messages.TextBlock{Text: "what is in this image?"},
		messages.ImageBlock{URL: "https://example.com/cat.png", MimeType: "image/png"},
		messages.ImageBlock{Base64: "aGVsbG8=", MimeType: "image/png"},
		messages.AudioBlock{Base64: "YXVkaW8=", MimeType: "audio/wav"},
	})
}

// TestResponsesMultimodalInput locks the Responses API request shape for
// human messages carrying content blocks: structured input content parts
// (input_text / input_image / input_audio), image_url as a bare string, and
// base64 images encoded as data URIs (chat_models/base.py payload construction).
func TestResponsesMultimodalInput(t *testing.T) {
	url, gotBody := captureBodyServer(t,
		`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`)
	model := NewChatModel(modelconfig.WithBaseURL(url), modelconfig.WithModel("gpt-test"))

	if _, err := model.Invoke(t.Context(), []messages.Message{multimodalHuman()}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	input, ok := gotBody()["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("input = %#v", gotBody()["input"])
	}
	msg, _ := input[0].(map[string]any)
	if msg["role"] != "user" {
		t.Fatalf("role = %v", msg["role"])
	}
	parts := contentParts(t, msg)
	if len(parts) != 4 {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[0]["type"] != "input_text" || parts[0]["text"] != "what is in this image?" {
		t.Fatalf("text part = %#v", parts[0])
	}
	if parts[1]["type"] != "input_image" || parts[1]["image_url"] != "https://example.com/cat.png" {
		t.Fatalf("image url part = %#v", parts[1])
	}
	if parts[2]["type"] != "input_image" || parts[2]["image_url"] != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("image base64 part = %#v", parts[2])
	}
	if parts[3]["type"] != "input_audio" {
		t.Fatalf("audio part = %#v", parts[3])
	}
	audio, _ := parts[3]["input_audio"].(map[string]any)
	if audio["data"] != "YXVkaW8=" || audio["format"] != "wav" {
		t.Fatalf("input_audio = %#v", audio)
	}
}

// TestResponsesPlainTextInputStaysString locks the no-blocks fast path: a
// plain human message must serialize content as a bare string so payloads do
// not grow a content-parts array for text-only conversations.
func TestResponsesPlainTextInputStaysString(t *testing.T) {
	url, gotBody := captureBodyServer(t,
		`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`)
	model := NewChatModel(modelconfig.WithBaseURL(url), modelconfig.WithModel("gpt-test"))

	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	input := gotBody()["input"].([]any)
	msg, _ := input[0].(map[string]any)
	if msg["content"] != "hi" {
		t.Fatalf("content = %#v, want bare string \"hi\"", msg["content"])
	}
}

// TestResponsesImageFileIDAndDetail covers the file_id image variant and the
// optional detail pass-through on Responses input_image parts.
func TestResponsesImageFileIDAndDetail(t *testing.T) {
	url, gotBody := captureBodyServer(t,
		`{"id":"resp_1","model":"gpt-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`)
	model := NewChatModel(modelconfig.WithBaseURL(url), modelconfig.WithModel("gpt-test"))

	human := messages.Human("look").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{FileID: "file-abc", Extras: map[string]any{"detail": "high"}},
	})
	if _, err := model.Invoke(t.Context(), []messages.Message{human}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	msg, _ := gotBody()["input"].([]any)[0].(map[string]any)
	parts := contentParts(t, msg)
	if len(parts) != 1 {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[0]["type"] != "input_image" || parts[0]["file_id"] != "file-abc" || parts[0]["detail"] != "high" {
		t.Fatalf("file_id image part = %#v", parts[0])
	}
}

// TestResponsesMultimodalErrors locks the loud failure modes: image blocks
// with no source and audio blocks without base64 data are rejected instead of
// being silently dropped (Python raises ValueError for these shapes).
func TestResponsesMultimodalErrors(t *testing.T) {
	url, _ := captureBodyServer(t, `{}`)
	model := NewChatModel(modelconfig.WithBaseURL(url), modelconfig.WithModel("gpt-test"))

	badImage := messages.Human("x").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{MimeType: "image/png"},
	})
	_, err := model.Invoke(t.Context(), []messages.Message{badImage})
	if err == nil || !strings.Contains(err.Error(), "image content block requires") {
		t.Fatalf("expected image-source error, got %v", err)
	}

	badAudio := messages.Human("x").WithContentBlocks([]messages.ContentBlock{
		messages.AudioBlock{URL: "https://example.com/a.wav"},
	})
	_, err = model.Invoke(t.Context(), []messages.Message{badAudio})
	if err == nil || !strings.Contains(err.Error(), "audio content block requires base64") {
		t.Fatalf("expected audio-data error, got %v", err)
	}
}

// TestChatCompletionsMultimodalInput locks the Chat Completions request shape:
// content parts with image_url objects ({"url": ...}) and input_audio parts,
// matching _convert_message_to_dict's multimodal branch.
func TestChatCompletionsMultimodalInput(t *testing.T) {
	url, gotBody := captureBodyServer(t,
		`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	model := NewChatModel(
		modelconfig.WithBaseURL(url),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	if _, err := model.Invoke(t.Context(), []messages.Message{multimodalHuman()}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	msgs := gotBody()["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v", msgs)
	}
	msg, _ := msgs[0].(map[string]any)
	parts := contentParts(t, msg)
	if len(parts) != 4 {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[0]["type"] != "text" || parts[0]["text"] != "what is in this image?" {
		t.Fatalf("text part = %#v", parts[0])
	}
	if parts[1]["type"] != "image_url" {
		t.Fatalf("image part = %#v", parts[1])
	}
	if iu, _ := parts[1]["image_url"].(map[string]any); iu["url"] != "https://example.com/cat.png" {
		t.Fatalf("image_url = %#v", parts[1]["image_url"])
	}
	if iu, _ := parts[2]["image_url"].(map[string]any); iu["url"] != "data:image/png;base64,aGVsbG8=" {
		t.Fatalf("base64 image_url = %#v", parts[2]["image_url"])
	}
	if parts[3]["type"] != "input_audio" {
		t.Fatalf("audio part = %#v", parts[3])
	}
	if audio, _ := parts[3]["input_audio"].(map[string]any); audio["data"] != "YXVkaW8=" || audio["format"] != "wav" {
		t.Fatalf("input_audio = %#v", audio)
	}
}

// TestChatCompletionsPlainTextStaysString locks the text-only fast path on the
// Chat Completions API too.
func TestChatCompletionsPlainTextStaysString(t *testing.T) {
	url, gotBody := captureBodyServer(t,
		`{"id":"x","model":"gpt-test","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{}}`)
	model := NewChatModel(
		modelconfig.WithBaseURL(url),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	if _, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	msg, _ := gotBody()["messages"].([]any)[0].(map[string]any)
	if msg["content"] != "hi" {
		t.Fatalf("content = %#v, want bare string \"hi\"", msg["content"])
	}
}

// TestChatCompletionsMultimodalErrors: Chat Completions has no file_id image
// form, so a file_id-only block must fail loudly rather than emit a broken part.
func TestChatCompletionsMultimodalErrors(t *testing.T) {
	url, _ := captureBodyServer(t, `{}`)
	model := NewChatModel(
		modelconfig.WithBaseURL(url),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()

	human := messages.Human("x").WithContentBlocks([]messages.ContentBlock{
		messages.ImageBlock{FileID: "file-abc"},
	})
	_, err := model.Invoke(t.Context(), []messages.Message{human})
	if err == nil || !strings.Contains(err.Error(), "url or base64") {
		t.Fatalf("expected file_id rejection, got %v", err)
	}
}

// TestResponsesReasoningEffortField locks WithReasoningEffort on the Responses
// API: it must serialize as reasoning: {"effort": ...} (not the Chat
// Completions reasoning_effort scalar) and be omitted when unset.
func TestResponsesReasoningEffortField(t *testing.T) {
	url, gotBody := captureBodyServer(t,
		`{"id":"resp_1","model":"o-test","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}`)
	base := NewChatModel(modelconfig.WithBaseURL(url), modelconfig.WithModel("o-test"))

	if _, err := base.Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("unset Invoke: %v", err)
	}
	if _, present := gotBody()["reasoning"]; present {
		t.Fatalf("unset model must omit reasoning, got %v", gotBody()["reasoning"])
	}
	if _, present := gotBody()["reasoning_effort"]; present {
		t.Fatalf("responses payload must not carry reasoning_effort scalar, got %v", gotBody()["reasoning_effort"])
	}

	if _, err := base.WithReasoningEffort("low").Invoke(t.Context(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("effort Invoke: %v", err)
	}
	reasoning, ok := gotBody()["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "low" {
		t.Fatalf("reasoning = %#v, want {effort: low}", gotBody()["reasoning"])
	}
}

// TestResponsesStreamUsageChunk locks Responses streaming usage extraction:
// response.completed's response.usage (including reasoning tokens) surfaces on
// a final usage-only chunk.
func TestResponsesStreamUsageChunk(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			`data: {"type":"response.output_text.delta","delta":"hi"}`+"\n\n"+
				`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","output":[],"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8,"output_tokens_details":{"reasoning_tokens":4}}}}`+"\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("gpt-test"))
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var chunks []messages.Message
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2 (text + usage)", len(chunks))
	}
	if chunks[0].Content != "hi" {
		t.Fatalf("first chunk = %#v", chunks[0])
	}
	um := chunks[1].UsageMetadata
	if um.InputTokens != 3 || um.OutputTokens != 5 || um.TotalTokens != 8 {
		t.Fatalf("usage = %#v", um)
	}
	if um.OutputTokenDetails == nil || um.OutputTokenDetails.ReasoningOutputTokens != 4 {
		t.Fatalf("output token details = %#v", um.OutputTokenDetails)
	}
}

// TestChatCompletionsStreamUsage locks Chat Completions streaming usage: the
// request must opt in via stream_options.include_usage and the final
// usage-only chunk (official prompt_tokens spelling) must surface as a
// usage-only message chunk.
func TestChatCompletionsStreamUsage(t *testing.T) {
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w,
			"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"+
				"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30,\"prompt_tokens_details\":{\"cached_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":6}}}\n\n"+
				"data: [DONE]\n\n",
		)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	stream, err := model.Stream(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	defer stream.Close()

	var chunks []messages.Message
	for {
		chunk, ok, err := stream.Next(t.Context())
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		chunks = append(chunks, chunk)
	}

	opts, ok := gotBody["stream_options"].(map[string]any)
	if !ok || opts["include_usage"] != true {
		t.Fatalf("stream_options = %#v, want {include_usage: true}", gotBody["stream_options"])
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2 (text + usage): %#v", len(chunks), chunks)
	}
	if chunks[0].Content != "hi" {
		t.Fatalf("first chunk = %#v", chunks[0])
	}
	um := chunks[1].UsageMetadata
	if um.InputTokens != 10 || um.OutputTokens != 20 || um.TotalTokens != 30 {
		t.Fatalf("usage = %#v", um)
	}
	if um.InputTokenDetails == nil || um.InputTokenDetails.CacheReadInputTokens != 2 {
		t.Fatalf("input token details = %#v", um.InputTokenDetails)
	}
	if um.OutputTokenDetails == nil || um.OutputTokenDetails.ReasoningOutputTokens != 6 {
		t.Fatalf("output token details = %#v", um.OutputTokenDetails)
	}
}

// TestChatCompletionsUsagePromptTokensSpelling guards the non-streaming usage
// normalization: the official Chat Completions usage field names
// (prompt_tokens/completion_tokens) must populate UsageMetadata identically to
// the Responses-style spellings covered by TestChatCompletionsUsageMetadataCacheTokens.
func TestChatCompletionsUsagePromptTokensSpelling(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-pt",
			"model":"gpt-test",
			"choices":[{"message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16,"completion_tokens_details":{"reasoning_tokens":3}}
		}`))
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("gpt-test"),
	).WithChatCompletions()
	resp, err := model.Invoke(t.Context(), []messages.Message{messages.Human("hi")})
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	um := resp.UsageMetadata
	if um.InputTokens != 7 || um.OutputTokens != 9 || um.TotalTokens != 16 {
		t.Fatalf("usage = %#v", um)
	}
	if um.OutputTokenDetails == nil || um.OutputTokenDetails.ReasoningOutputTokens != 3 {
		t.Fatalf("output token details = %#v", um.OutputTokenDetails)
	}
}

// TestCapabilitiesDeclareImplementedFeatures locks the capability declaration
// to what the adapter actually implements after multimodal input, streaming
// usage, and reasoning_effort landed: every flag below is exercised by both
// the Responses and Chat Completions paths, in Invoke and Stream.
func TestCapabilitiesDeclareImplementedFeatures(t *testing.T) {
	want := language.ChatModelCapabilities{
		ToolCalling:      true,
		ToolChoice:       true,
		StructuredOutput: true,
		JSONMode:         true,
		ImageInputs:      true,
		ImageURLs:        true,
		AudioInputs:      true,
		UsageMetadata:    true,
		Streaming:        true,
	}
	if got := NewChatModel().Capabilities(); got != want {
		t.Fatalf("capabilities = %+v, want %+v", got, want)
	}
}
