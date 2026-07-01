package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

// decodeRequest returns the raw decoded request body so parity tests can assert
// provider-native JSON shapes without depending on internal request structs.
func decodeRequest(t *testing.T, w http.ResponseWriter, r *http.Request, target any) {
	t.Helper()
	if err := json.NewDecoder(r.Body).Decode(target); err != nil {
		t.Fatalf("decode request: %v", err)
	}
}

func newTestServer(t *testing.T, capture *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var raw map[string]any
		decodeRequest(t, w, r, &raw)
		*capture = raw
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`)
	}))
}

func invokeWithBlocks(t *testing.T, model ChatModel, blocks []messages.ContentBlock) map[string]any {
	t.Helper()
	msg := messages.Human("")
	msg.ContentBlocks = blocks
	if _, err := model.Invoke(context.Background(), []messages.Message{msg}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	return map[string]any{}
}

// firstUserContent returns the parsed content blocks of the first user message.
func firstUserContent(t *testing.T, request map[string]any) []map[string]any {
	t.Helper()
	rawMessages, ok := request["messages"].([]any)
	if !ok || len(rawMessages) == 0 {
		t.Fatalf("missing messages: %+v", request)
	}
	first, ok := rawMessages[0].(map[string]any)
	if !ok {
		t.Fatalf("first message not an object: %+v", rawMessages[0])
	}
	content, ok := first["content"].([]any)
	if !ok {
		t.Fatalf("first message content not a list: %+v", first)
	}
	out := make([]map[string]any, len(content))
	for i, block := range content {
		out[i], ok = block.(map[string]any)
		if !ok {
			t.Fatalf("content block %d not an object: %+v", i, block)
		}
	}
	return out
}

func TestParityImageBase64ContentBlock(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	invokeWithBlocks(t, model, []messages.ContentBlock{
		{"type": "image", "source_type": "base64", "mime_type": "image/png", "base64": "Zm9v"},
		{"type": "text", "text": "what is this?"},
	})

	blocks := firstUserContent(t, request)
	image := blocks[0]
	if image["type"] != "image" {
		t.Fatalf("image type: %v", image)
	}
	source := image["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "image/png" || source["data"] != "Zm9v" {
		t.Fatalf("image source: %v", source)
	}
	text := blocks[1]
	if text["type"] != "text" || text["text"] != "what is this?" {
		t.Fatalf("text block: %v", text)
	}
}

func TestParityImageURLContentBlock(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	invokeWithBlocks(t, model, []messages.ContentBlock{
		{"type": "image", "source_type": "url", "url": "https://example.com/cat.png"},
	})

	source := firstUserContent(t, request)[0]["source"].(map[string]any)
	if source["type"] != "url" || source["url"] != "https://example.com/cat.png" {
		t.Fatalf("image url source: %v", source)
	}
}

func TestParityImageDataURIContentBlock(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	// data:image/jpeg;base64,Zm9v
	invokeWithBlocks(t, model, []messages.ContentBlock{
		{"type": "image", "source_type": "url", "url": "data:image/jpeg;base64,Zm9v"},
	})

	source := firstUserContent(t, request)[0]["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "image/jpeg" || source["data"] != "Zm9v" {
		t.Fatalf("data uri source: %v", source)
	}
}

func TestParityDocumentBase64ContentBlock(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	invokeWithBlocks(t, model, []messages.ContentBlock{
		{"type": "file", "source_type": "base64", "mime_type": "application/pdf", "base64": "QUJD"},
	})

	doc := firstUserContent(t, request)[0]
	if doc["type"] != "document" {
		t.Fatalf("document type: %v", doc)
	}
	source := doc["source"].(map[string]any)
	if source["type"] != "base64" || source["media_type"] != "application/pdf" || source["data"] != "QUJD" {
		t.Fatalf("document source: %v", source)
	}
}

func TestParityDocumentTextContentBlock(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	invokeWithBlocks(t, model, []messages.ContentBlock{
		{"type": "file", "source_type": "text", "mime_type": "text/plain", "text": "hello doc"},
	})

	doc := firstUserContent(t, request)[0]
	source := doc["source"].(map[string]any)
	if source["type"] != "text" || source["media_type"] != "text/plain" || source["data"] != "hello doc" {
		t.Fatalf("document text source: %v", source)
	}
}

func TestParityThinkingRequestParam(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	model = model.WithThinking(map[string]any{"type": "enabled", "budget_tokens": 1024})
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	thinking, ok := request["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("missing thinking param: %+v", request)
	}
	if thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(1024) {
		t.Fatalf("thinking param: %v", thinking)
	}
}

func TestParityTopPTopKRequestParams(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		WithTopP(0.75),
		WithTopK(64),
	)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("sample")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if request["top_p"] != 0.75 {
		t.Fatalf("top_p: %v", request["top_p"])
	}
	if request["top_k"] != float64(64) {
		t.Fatalf("top_k: %v", request["top_k"])
	}
}

func TestParityThinkingEnforcesSamplingConstraints(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		modelconfig.WithTemperature(0.2),
		WithTopP(0.75),
		WithTopK(64),
	).WithThinking(map[string]any{"type": "enabled", "budget_tokens": 1024})
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("think")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if request["temperature"] != float64(1) {
		t.Fatalf("temperature: %v", request["temperature"])
	}
	if _, ok := request["top_p"]; ok {
		t.Fatalf("top_p should be omitted with thinking: %+v", request)
	}
	if _, ok := request["top_k"]; ok {
		t.Fatalf("top_k should be omitted with thinking: %+v", request)
	}
}

func TestParityContextManagementRequestParam(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
	).WithContextManagement(map[string]any{
		"edits": []any{map[string]any{"type": "clear_tool_uses_20250919"}},
	})
	if _, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("Search for recent developments in AI"),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	contextManagement, ok := request["context_management"].(map[string]any)
	if !ok {
		t.Fatalf("context_management missing: %+v", request)
	}
	edits, ok := contextManagement["edits"].([]any)
	if !ok || len(edits) != 1 {
		t.Fatalf("context_management edits: %v", contextManagement["edits"])
	}
	edit, ok := edits[0].(map[string]any)
	if !ok || edit["type"] != "clear_tool_uses_20250919" {
		t.Fatalf("context_management edit: %v", edits[0])
	}
}

func TestParityInferenceGeoRequestParam(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
	).WithInferenceGeo("us")
	if _, err := model.Invoke(context.Background(), []messages.Message{
		messages.Human("Hello, world!"),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	if request["inference_geo"] != "us" {
		t.Fatalf("inference_geo: %v", request["inference_geo"])
	}
}

func TestParityStrictToolUsePayload(t *testing.T) {
	for _, tc := range []struct {
		name   string
		strict bool
	}{
		{name: "strict true", strict: true},
		{name: "strict false", strict: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var request map[string]any
			server := newTestServer(t, &request)
			defer server.Close()

			tool, err := tools.NewFunc(
				"get_weather",
				"Get the weather at a location.",
				schema.Object(map[string]schema.Schema{
					"location": schema.String("location"),
				}, "location"),
				func(_ context.Context, _ map[string]any) (tools.Result, error) {
					return tools.Result{Content: "Sunny"}, nil
				},
			)
			if err != nil {
				t.Fatalf("tool: %v", err)
			}
			model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
			modelWithTools, err := model.BindToolsStrict([]tools.Tool{tool}, tc.strict)
			if err != nil {
				t.Fatalf("bind tools strict: %v", err)
			}
			if _, err := modelWithTools.Invoke(context.Background(), []messages.Message{
				messages.Human("What's the weather?"),
			}); err != nil {
				t.Fatalf("invoke: %v", err)
			}

			rawTools, ok := request["tools"].([]any)
			if !ok || len(rawTools) != 1 {
				t.Fatalf("tools: %+v", request["tools"])
			}
			toolPayload, ok := rawTools[0].(map[string]any)
			if !ok {
				t.Fatalf("tool payload: %+v", rawTools[0])
			}
			if toolPayload["strict"] != tc.strict {
				t.Fatalf("strict: got %v want %v in %+v", toolPayload["strict"], tc.strict, toolPayload)
			}
		})
	}
}

func TestParityInvokeThinkingResponseBlock(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[{"type":"thinking","thinking":"let me reason","signature":"sig_123"},{"type":"text","text":"answer"}],"usage":{"input_tokens":1,"output_tokens":2}}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("q")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	var reasoning messages.ContentBlock
	for _, block := range resp.ContentBlocks {
		if block["type"] == "reasoning" {
			reasoning = block
		}
	}
	if reasoning == nil {
		t.Fatalf("no reasoning block: %+v", resp.ContentBlocks)
	}
	if reasoning["reasoning"] != "let me reason" || reasoning["signature"] != "sig_123" {
		t.Fatalf("reasoning block: %+v", reasoning)
	}
}

func TestParityInvokeRedactedThinkingResponseBlock(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"ZW5jcnlwdGVk"},{"type":"text","text":"ok"}],"usage":{}}`)
	}))
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	resp, err := model.Invoke(context.Background(), []messages.Message{messages.Human("q")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}

	found := false
	for _, block := range resp.ContentBlocks {
		if block["type"] == "reasoning" && block["data"] == "ZW5jcnlwdGVk" {
			found = true
		}
	}
	if !found {
		t.Fatalf("no redacted reasoning block: %+v", resp.ContentBlocks)
	}
}

func TestParityStreamThinkingBlock(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(event, data string) {
			fmt.Fprintf(w, "event: %s\n", event)
			fmt.Fprintf(w, "data: %s\n\n", data)
		}
		send("message_start", `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}`)
		send("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`)
		send("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"step by step"}}`)
		send("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_abc"}}`)
		send("content_block_stop", `{"type":"content_block_stop","index":0}`)
		send("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`)
		send("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`)
		send("content_block_stop", `{"type":"content_block_stop","index":1}`)
		send("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`)
		send("message_stop", `{"type":"message_stop"}`)
	}))
	defer server.Close()

	recorder := callbacks.NewRecorder()
	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	stream, err := model.Stream(
		context.Background(),
		[]messages.Message{messages.Human("q")},
		runnables.WithCallbacks(callbacks.NewManager(recorder)),
	)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()
	for {
		if _, ok, err := stream.Next(context.Background()); err != nil {
			t.Fatalf("next: %v", err)
		} else if !ok {
			break
		}
	}

	var output messages.Message
	for _, event := range recorder.Events() {
		if event.Kind == callbacks.EventChatModelEnd {
			if msg, ok := event.Output.(messages.Message); ok {
				output = msg
			}
		}
	}
	var reasoning messages.ContentBlock
	for _, block := range output.ContentBlocks {
		if block["type"] == "reasoning" {
			reasoning = block
		}
	}
	if reasoning == nil {
		t.Fatalf("no reasoning block in output: %+v", output.ContentBlocks)
	}
	if reasoning["reasoning"] != "step by step" || reasoning["signature"] != "sig_abc" {
		t.Fatalf("reasoning block: %+v", reasoning)
	}
	if output.Content != "answer" {
		t.Fatalf("text content: %q", output.Content)
	}
}

func TestParitySystemCacheControl(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	sys := messages.System("")
	sys.ContentBlocks = []messages.ContentBlock{
		{"type": "text", "text": "you are helpful", "cache_control": map[string]any{"type": "ephemeral"}},
	}
	if _, err := model.Invoke(context.Background(), []messages.Message{sys, messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	system, ok := request["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system not a block list: %+v", request["system"])
	}
	block := system[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "you are helpful" {
		t.Fatalf("system block: %v", block)
	}
	cc, _ := block["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Fatalf("cache_control: %v", cc)
	}
}

func TestParityContentBlockCacheControl(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	invokeWithBlocks(t, model, []messages.ContentBlock{
		{"type": "text", "text": "cache me", "cache_control": map[string]any{"type": "ephemeral"}},
	})

	block := firstUserContent(t, request)[0]
	cc, _ := block["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Fatalf("cache_control: %v", cc)
	}
}

func TestParityToolResultContentBlockCacheControlHoisted(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	toolResult := messages.Tool("toolu_123", "")
	toolResult.ContentBlocks = []messages.ContentBlock{
		{"type": "text", "text": "cached result", "cache_control": map[string]any{"type": "ephemeral"}},
	}
	if _, err := model.Invoke(context.Background(), []messages.Message{toolResult}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	block := firstUserContent(t, request)[0]
	if block["type"] != "tool_result" || block["tool_use_id"] != "toolu_123" {
		t.Fatalf("tool_result block: %v", block)
	}
	cc, _ := block["cache_control"].(map[string]any)
	if cc["type"] != "ephemeral" {
		t.Fatalf("tool_result cache_control: %v", cc)
	}
	content, ok := block["content"].([]any)
	if !ok || len(content) != 1 {
		t.Fatalf("tool_result content: %v", block["content"])
	}
	child, ok := content[0].(map[string]any)
	if !ok {
		t.Fatalf("tool_result child: %v", content[0])
	}
	if child["type"] != "text" || child["text"] != "cached result" {
		t.Fatalf("tool_result child: %v", child)
	}
	if _, ok := child["cache_control"]; ok {
		t.Fatalf("cache_control should be hoisted from child: %v", child)
	}
}

func TestParityToolChoiceRequestParam(t *testing.T) {
	var request map[string]any
	server := newTestServer(t, &request)
	defer server.Close()

	base := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithModel("m"))
	model := base.WithToolChoice(map[string]any{"type": "any", "disable_parallel_tool_use": true})
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	choice, ok := request["tool_choice"].(map[string]any)
	if !ok || choice["type"] != "any" {
		t.Fatalf("tool_choice: %+v", request["tool_choice"])
	}
	if choice["disable_parallel_tool_use"] != true {
		t.Fatalf("disable_parallel_tool_use: %v", choice["disable_parallel_tool_use"])
	}
}

func TestParityBetaHeaders(t *testing.T) {
	var betaHeader string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		betaHeader = r.Header.Get("anthropic-beta")
		_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],"usage":{}}`)
	}))
	defer server.Close()

	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithModel("m"),
		WithBetaHeaders("interleaved-thinking-2025-05-14", "prompt-caching-2024-07-31"),
	)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if betaHeader != "interleaved-thinking-2025-05-14,prompt-caching-2024-07-31" {
		t.Fatalf("anthropic-beta header: %q", betaHeader)
	}
}
