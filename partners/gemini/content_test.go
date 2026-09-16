package gemini

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
)

// TestBuildContentsMultimodal verifies image/audio blocks serialize as
// inline_data (base64 bytes) and URL blocks as file_data — Python's
// _convert_to_parts mapping.
func TestBuildContentsMultimodal(t *testing.T) {
	const tinyPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
	const tinyWAV = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
	const remoteURL = "https://example.com/cat.png"

	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"A cat."}]}}]}`))
	}))
	defer server.Close()

	human := messages.Human("")
	human = human.WithContentBlocks([]messages.ContentBlock{
		messages.TextBlock{Text: "describe"},
		messages.ImageBlock{Base64: tinyPNG, MimeType: "image/png"},
		messages.AudioBlock{Base64: tinyWAV, MimeType: "audio/wav"},
		messages.ImageBlock{URL: remoteURL, MimeType: "image/png"},
		messages.FileBlock{Base64: "JVBERi0=", MimeType: "application/pdf"},
	})
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	if _, err := model.Invoke(t.Context(), []messages.Message{human}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	contents, _ := body["contents"].([]any)
	if len(contents) != 1 {
		t.Fatalf("contents: %v", contents)
	}
	parts, _ := contents[0].(map[string]any)["parts"].([]any)
	if len(parts) != 5 {
		t.Fatalf("expected 5 parts, got %d: %v", len(parts), parts)
	}

	textPart := parts[0].(map[string]any)
	if textPart["text"] != "describe" {
		t.Errorf("text part: %v", textPart)
	}

	png, _ := parts[1].(map[string]any)["inlineData"].(map[string]any)
	if png == nil || png["mimeType"] != "image/png" {
		t.Fatalf("image inlineData: %v", parts[1])
	}
	// The SDK base64-encodes Blob.Data on the wire; it must round-trip to the
	// original block bytes.
	decoded, err := base64.StdEncoding.DecodeString(png["data"].(string))
	if err != nil {
		t.Fatalf("inlineData data not base64: %v", err)
	}
	if want, _ := base64.StdEncoding.DecodeString(tinyPNG); string(decoded) != string(want) {
		t.Errorf("inlineData data mismatch")
	}

	wav, _ := parts[2].(map[string]any)["inlineData"].(map[string]any)
	if wav == nil || wav["mimeType"] != "audio/wav" {
		t.Errorf("audio inlineData: %v", parts[2])
	}

	fileData, _ := parts[3].(map[string]any)["fileData"].(map[string]any)
	if fileData == nil || fileData["fileUri"] != remoteURL || fileData["mimeType"] != "image/png" {
		t.Errorf("image url fileData: %v", parts[3])
	}

	pdf, _ := parts[4].(map[string]any)["inlineData"].(map[string]any)
	if pdf == nil || pdf["mimeType"] != "application/pdf" {
		t.Errorf("pdf inlineData: %v", parts[4])
	}
}

// TestBuildContentsToolHistory verifies AI tool calls serialize as
// functionCall parts under role "model" and tool results as functionResponse
// parts under role "user" with the name resolved from the preceding AI tool
// call (Python's _get_ai_message_tool_messages_parts).
func TestBuildContentsToolHistory(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}`))
	}))
	defer server.Close()

	aiTurn := messages.AI("")
	aiTurn.ToolCalls = []messages.ToolCall{{
		ID:   "call_1",
		Name: "get_weather",
		Args: map[string]any{"location": "SF"},
	}}
	model := NewChatModel(
		modelconfig.WithBaseURL(server.URL),
		modelconfig.WithAPIKey("test-key"),
	)
	if _, err := model.Invoke(t.Context(), []messages.Message{
		messages.Human("weather in SF?"),
		aiTurn,
		messages.Tool("call_1", `{"result": "sunny"}`),
	}); err != nil {
		t.Fatalf("invoke: %v", err)
	}

	contents, _ := body["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents: %v", contents)
	}
	modelContent := contents[1].(map[string]any)
	if modelContent["role"] != "model" {
		t.Fatalf("ai role: %v", modelContent["role"])
	}
	fnCall, _ := modelContent["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	if fnCall == nil || fnCall["name"] != "get_weather" {
		t.Fatalf("functionCall part: %v", modelContent["parts"])
	}
	if args, _ := fnCall["args"].(map[string]any); args["location"] != "SF" {
		t.Errorf("functionCall args: %v", fnCall["args"])
	}

	toolContent := contents[2].(map[string]any)
	if toolContent["role"] != "user" {
		t.Fatalf("tool role: %v", toolContent["role"])
	}
	fnResponse, _ := toolContent["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if fnResponse == nil {
		t.Fatalf("functionResponse part: %v", toolContent["parts"])
	}
	// Name resolved from the AI tool call (not the call ID).
	if fnResponse["name"] != "get_weather" {
		t.Errorf("functionResponse name: %v", fnResponse["name"])
	}
	if resp, _ := fnResponse["response"].(map[string]any); resp["result"] != "sunny" {
		t.Errorf("functionResponse response: %v", fnResponse["response"])
	}
}

// TestBuildContentsToolResponseNonJSON verifies non-JSON tool content wraps
// as {"output": ...} (Python's _convert_tool_message_to_parts).
func TestBuildContentsToolResponseNonJSON(t *testing.T) {
	systemInstruction, contents, err := buildContents([]messages.Message{
		messages.Tool("call_9", "plain text result"),
	})
	if err != nil {
		t.Fatalf("buildContents: %v", err)
	}
	if systemInstruction != nil {
		t.Fatalf("unexpected systemInstruction: %v", systemInstruction)
	}
	response := contents[0].Parts[0].FunctionResponse
	if response.Name != "call_9" {
		t.Errorf("fallback name: %q", response.Name)
	}
	if response.Response["output"] != "plain text result" {
		t.Errorf("wrapped output: %v", response.Response)
	}
}

// TestBuildContentsSystemAggregation verifies multiple system messages merge
// into one systemInstruction Content (Python appends parts).
func TestBuildContentsSystemAggregation(t *testing.T) {
	systemInstruction, contents, err := buildContents([]messages.Message{
		messages.System("rule one"),
		messages.System("rule two"),
		messages.Human("hi"),
	})
	if err != nil {
		t.Fatalf("buildContents: %v", err)
	}
	if systemInstruction == nil || len(systemInstruction.Parts) != 2 {
		t.Fatalf("systemInstruction: %+v", systemInstruction)
	}
	if systemInstruction.Parts[0].Text != "rule one" || systemInstruction.Parts[1].Text != "rule two" {
		t.Errorf("system parts: %+v", systemInstruction.Parts)
	}
	if len(contents) != 1 || contents[0].Role != "user" {
		t.Fatalf("contents: %+v", contents)
	}
}

// TestBuildContentsEmptyAIMessage verifies an AI turn without content keeps
// a placeholder empty text part (the API rejects part-less Contents).
func TestBuildContentsEmptyAIMessage(t *testing.T) {
	_, contents, err := buildContents([]messages.Message{
		messages.Human("hi"),
		messages.AI(""),
	})
	if err != nil {
		t.Fatalf("buildContents: %v", err)
	}
	if len(contents) != 2 || contents[1].Role != "model" {
		t.Fatalf("contents: %+v", contents)
	}
	if len(contents[1].Parts) != 1 || contents[1].Parts[0].Text != "" {
		t.Fatalf("placeholder part missing: %+v", contents[1].Parts)
	}
}

// TestResponseToMessageThoughtPart verifies thought parts become reasoning
// blocks and stay out of the main content.
func TestResponseToMessageThoughtPart(t *testing.T) {
	server := newGeminiTestServer(t, func(t *testing.T, _ *http.Request, _ map[string]any) {},
		`{"candidates":[{"content":{"role":"model","parts":[
			{"text":"thinking...","thought":true},
			{"text":"the answer"}
		]},"finishReason":"STOP"}]}`)
	defer server.Close()

	model := NewChatModel(modelconfig.WithBaseURL(server.URL), modelconfig.WithAPIKey("test-key"))
	response, err := model.Invoke(t.Context(), []messages.Message{messages.Human("q")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if response.Content != "the answer" {
		t.Errorf("content: %q", response.Content)
	}
	var reasoning int
	for _, block := range response.ContentBlocks {
		if _, ok := block.(messages.ReasoningBlock); ok {
			reasoning++
		}
	}
	if reasoning != 1 {
		t.Errorf("reasoning blocks: %d (blocks: %+v)", reasoning, response.ContentBlocks)
	}
}
