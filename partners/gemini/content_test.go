package gemini

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/modelconfig"
	"google.golang.org/genai"
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

// TestBlockToPartVariants: the remaining block kinds map — video bytes and
// URLs, reasoning parts, and the streaming-only shapes are skipped on
// requests — while unsupported blocks fail the conversion.
func TestBlockToPartVariants(t *testing.T) {
	part, err := blockToPart(messages.VideoBlock{Base64: "AAAA", MimeType: "video/mp4"})
	if err != nil {
		t.Fatalf("blockToPart(video) error = %v", err)
	}
	if part.InlineData == nil || part.InlineData.MIMEType != "video/mp4" {
		t.Fatalf("video part = %+v", part)
	}

	part, err = blockToPart(messages.VideoBlock{URL: "https://example.test/clip.mp4", MimeType: "video/mp4"})
	if err != nil {
		t.Fatalf("blockToPart(video url) error = %v", err)
	}
	if part.FileData == nil || part.FileData.FileURI != "https://example.test/clip.mp4" {
		t.Fatalf("video url part = %+v", part)
	}

	part, err = blockToPart(messages.ReasoningBlock{Reasoning: "thinking hard"})
	if err != nil {
		t.Fatalf("blockToPart(reasoning) error = %v", err)
	}
	if part.Text != "thinking hard" || !part.Thought {
		t.Fatalf("reasoning part = %+v", part)
	}

	for _, block := range []messages.ContentBlock{
		messages.ToolCallChunkBlock{ID: "1", Name: "f"},
		messages.InvalidToolCallBlock{ID: "2"},
	} {
		part, err = blockToPart(block)
		if err != nil || part != nil {
			t.Fatalf("blockToPart(%T) = %+v, %v; want nil part", block, part, err)
		}
	}

	if _, err := blockToPart(messages.PlainTextBlock{Text: "x"}); err == nil {
		t.Fatal("blockToPart(unsupported) must fail")
	}
}

// TestDataPartFallbacks: inline data without a MIME type gets the kind's
// default; undecodable base64 and empty blocks fail with descriptive errors.
func TestDataPartFallbacks(t *testing.T) {
	cases := []struct {
		kind string
		want string
	}{
		{kind: "image", want: "image/png"},
		{kind: "audio", want: "audio/wav"},
		{kind: "video", want: "video/mp4"},
		{kind: "file", want: "application/octet-stream"},
	}
	for _, tc := range cases {
		part, err := dataPart(tc.kind, "", "NDI=", "")
		if err != nil {
			t.Fatalf("dataPart(%s, no mime) error = %v", tc.kind, err)
		}
		if part.InlineData.MIMEType != tc.want {
			t.Fatalf("dataPart(%s) mime = %q, want %q", tc.kind, part.InlineData.MIMEType, tc.want)
		}
	}

	if _, err := dataPart("image", "image/png", "%%not-base64%%", ""); err == nil {
		t.Fatal("dataPart(bad base64) must fail")
	}
	if _, err := dataPart("image", "", "", ""); err == nil {
		t.Fatal("dataPart(no data, no url) must fail")
	}
}

// TestBuildContentsRejectsBadBlocks: malformed multimodal blocks fail the
// history build for each message role (system / human / AI), and a tool
// message whose media blocks fail surfaces the same way.
func TestBuildContentsRejectsBadBlocks(t *testing.T) {
	bad := []messages.ContentBlock{messages.ImageBlock{Base64: "%%"}}
	if _, _, err := buildContents([]messages.Message{systemWithBlocks(bad)}); err == nil {
		t.Fatal("system message with bad block must fail")
	}
	if _, _, err := buildContents([]messages.Message{humanWithBlocks(bad)}); err == nil {
		t.Fatal("human message with bad block must fail")
	}
	if _, _, err := buildContents([]messages.Message{aiWithBlocks(bad)}); err == nil {
		t.Fatal("AI message with bad block must fail")
	}

	tool := messages.Tool("call-1", "output")
	tool = tool.WithContentBlocks([]messages.ContentBlock{messages.PlainTextBlock{Text: "?"}})
	if _, _, err := buildContents([]messages.Message{tool}); err == nil {
		t.Fatal("tool message with unsupported block must fail")
	}

	// Streaming-only blocks in a tool result are skipped, not fatal; media
	// blocks ride alongside the functionResponse part.
	toolOK := messages.Tool("call-1", "output")
	toolOK = toolOK.WithContentBlocks([]messages.ContentBlock{
		messages.ToolCallChunkBlock{ID: "1", Name: "f"},
		messages.ImageBlock{Base64: "aW1hZ2VkYXRh", MimeType: "image/png"},
	})
	_, contents, err := buildContents([]messages.Message{toolOK})
	if err != nil {
		t.Fatalf("buildContents(tool with chunk block) error = %v", err)
	}
	if len(contents) != 1 || len(contents[0].Parts) != 2 {
		t.Fatalf("contents = %+v, want inline-data + functionResponse parts", contents)
	}
	if contents[0].Parts[0].InlineData == nil || contents[0].Parts[1].FunctionResponse == nil {
		t.Fatalf("parts = %+v", contents[0].Parts)
	}

	// A tool result without content still sends an empty response object.
	empty := messages.Tool("call-2", "")
	_, contents, err = buildContents([]messages.Message{empty})
	if err != nil {
		t.Fatalf("buildContents(empty tool) error = %v", err)
	}
	response := contents[0].Parts[0].FunctionResponse
	if response.Name != "call-2" || response.Response == nil || len(response.Response) != 0 {
		t.Fatalf("functionResponse = %+v, want empty object under the id fallback name", response)
	}
}

func systemWithBlocks(blocks []messages.ContentBlock) messages.Message {
	m := messages.System("sys")
	return m.WithContentBlocks(blocks)
}

func humanWithBlocks(blocks []messages.ContentBlock) messages.Message {
	m := messages.Human("hi")
	return m.WithContentBlocks(blocks)
}

func aiWithBlocks(blocks []messages.ContentBlock) messages.Message {
	m := messages.AI("hi")
	return m.WithContentBlocks(blocks)
}

// TestResponseToMessageMediaParts: inline data parts decode into audio /
// video / image blocks by MIME prefix; nil parts and function calls carrying
// their own IDs pass through untouched; finishMessage lands in the metadata.
func TestResponseToMessageMediaParts(t *testing.T) {
	response := &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: "model", Parts: []*genai.Part{
				nil,
				{FunctionCall: &genai.FunctionCall{ID: "call_abc", Name: "search", Args: map[string]any{"q": "x"}}},
				{InlineData: &genai.Blob{Data: []byte("aud"), MIMEType: "audio/wav"}},
				{InlineData: &genai.Blob{Data: []byte("vid"), MIMEType: "video/mp4"}},
				{InlineData: &genai.Blob{Data: []byte("img"), MIMEType: "image/png"}},
				{InlineData: &genai.Blob{Data: []byte("???")}}, // no MIME type: skipped
			}},
			FinishReason:  genai.FinishReasonStop,
			FinishMessage: "all done",
		}},
		ModelVersion: "gemini-test",
		ResponseID:   "resp_9",
	}
	msg, err := responseToMessage(response)
	if err != nil {
		t.Fatalf("responseToMessage() error = %v", err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_abc" {
		t.Fatalf("tool calls = %+v, want the API-provided id", msg.ToolCalls)
	}
	var sawAudio, sawVideo, sawImage bool
	for _, block := range msg.ContentBlocks {
		switch b := block.(type) {
		case messages.AudioBlock:
			sawAudio = b.Base64 == base64.StdEncoding.EncodeToString([]byte("aud"))
		case messages.VideoBlock:
			sawVideo = b.Base64 == base64.StdEncoding.EncodeToString([]byte("vid"))
		case messages.ImageBlock:
			sawImage = b.Base64 == base64.StdEncoding.EncodeToString([]byte("img"))
		}
	}
	if !sawAudio || !sawVideo || !sawImage {
		t.Fatalf("media blocks = %#v, want audio+video+image", msg.ContentBlocks)
	}
	if msg.ResponseMetadata["finish_message"] != "all done" {
		t.Fatalf("finish_message = %v", msg.ResponseMetadata["finish_message"])
	}
	if msg.ID != "resp_9" {
		t.Fatalf("ID = %q, want resp_9", msg.ID)
	}
}

// TestOrSyntheticCallID: an API-provided id passes through; a missing one is
// synthesized from the per-message index.
func TestOrSyntheticCallID(t *testing.T) {
	if got := orSyntheticCallID("call_xyz", 3); got != "call_xyz" {
		t.Fatalf("orSyntheticCallID(id) = %q", got)
	}
	if got := orSyntheticCallID("", 3); got != "call_3" {
		t.Fatalf("orSyntheticCallID(missing) = %q, want call_3", got)
	}
}
