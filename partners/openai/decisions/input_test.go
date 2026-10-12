package decisions

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestToDecisionInputString(t *testing.T) {
	got, err := toDecisionInput("Production is down. Please help immediately.")
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	if got != "Production is down. Please help immediately." {
		t.Fatalf("toDecisionInput() = %#v, want the string unchanged", got)
	}
}

func TestToDecisionInputSingleHumanWithBlocks(t *testing.T) {
	msg := messages.Message{
		Role: messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{
			messages.TextBlock{Text: "look"},
			messages.ImageBlock{Base64: "abcd", MimeType: "image/png"},
			messages.ImageBlock{URL: "https://example.com/img.png"},
		},
	}
	got, err := toDecisionInput(msg)
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	list, ok := got.([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("toDecisionInput() = %#v, want one native user message", got)
	}
	m := list[0].(map[string]any)
	if m["role"] != "user" {
		t.Fatalf("role = %v, want user", m["role"])
	}
	content, ok := m["content"].([]any)
	if !ok || len(content) != 3 {
		t.Fatalf("content = %#v, want 3 parts", m["content"])
	}
	wantParts := []map[string]any{
		{"type": "input_text", "text": "look"},
		{"type": "input_image", "image_url": "data:image/png;base64,abcd"},
		{"type": "input_image", "image_url": "https://example.com/img.png"},
	}
	for i, want := range wantParts {
		cm, _ := content[i].(map[string]any)
		if !mapsEqual(cm, want) {
			t.Fatalf("content[%d] = %#v, want %#v", i, cm, want)
		}
	}
}

func TestToDecisionInputAllHumanSequence(t *testing.T) {
	withBlocks := messages.Message{
		Role:          messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{messages.ImageBlock{Base64: "zz"}},
	}
	got, err := toDecisionInput([]messages.Message{messages.Human("hi"), withBlocks})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	list, ok := got.([]any)
	if !ok || len(list) != 2 {
		t.Fatalf("toDecisionInput() = %#v, want two native user messages", got)
	}
	first := list[0].(map[string]any)
	if first["content"] != "hi" {
		t.Fatalf("first content = %#v, want plain string \"hi\"", first["content"])
	}
	second := list[1].(map[string]any)
	content := second["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("second content = %#v, want 1 part", second["content"])
	}
	part := content[0].(map[string]any)
	if part["type"] != "input_image" || part["image_url"] != "data:image/jpeg;base64,zz" {
		t.Fatalf("second content part = %#v", part)
	}
}

func TestToDecisionInputMixedRolesJSONPath(t *testing.T) {
	human := messages.Message{
		Role: messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{
			messages.TextBlock{Text: "a cat"},
			messages.ImageBlock{Base64: "cat", MimeType: "image/png"},
		},
	}
	got, err := toDecisionInput([]messages.Message{messages.System("be nice"), human})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	if m["role"] != "user" {
		t.Fatalf("role = %v, want user", m["role"])
	}
	content, ok := m["content"].([]any)
	if !ok || len(content) != 3 {
		t.Fatalf("content = %#v, want [json text, image, trailing text]", m["content"])
	}
	textPart := content[0].(map[string]any)
	if textPart["type"] != "input_text" {
		t.Fatalf("content[0] = %#v, want input_text", textPart)
	}
	var jsonText strings.Builder
	for _, part := range content {
		if pm, ok := part.(map[string]any); ok && pm["type"] == "input_text" {
			jsonText.WriteString(pm["text"].(string))
		}
	}
	joined := jsonText.String()
	if !strings.Contains(joined, `"role":"system"`) || !strings.Contains(joined, `"role":"user"`) {
		t.Fatalf("json text = %q, want both roles", joined)
	}
	if strings.Contains(joined, "data:image") {
		t.Fatalf("json text = %q, base64 image must be lifted out of the JSON", joined)
	}
	imagePart := content[1].(map[string]any)
	if imagePart["type"] != "input_image" || imagePart["image_url"] != "data:image/png;base64,cat" {
		t.Fatalf("content[1] = %#v, want the base64 image part", imagePart)
	}
	if tail := content[2].(map[string]any); tail["type"] != "input_text" {
		t.Fatalf("content[2] = %#v, want trailing input_text", tail)
	}
}

func TestToDecisionInputMapStateJSONPath(t *testing.T) {
	human := messages.Human("delete everything")
	state := map[string]any{
		"messages": []messages.Message{human},
		"tool_call": map[string]any{
			"id":   "call_1",
			"name": "delete_file",
			"args": map[string]any{"path": "/tmp/x"},
		},
		"count": 2,
	}
	got, err := toDecisionInput(state)
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	if m["role"] != "user" {
		t.Fatalf("role = %v, want user", m["role"])
	}
	content := m["content"].([]any)
	textPart := content[0].(map[string]any)
	jsonText := textPart["text"].(string)
	for _, want := range []string{`"tool_call"`, `"delete_file"`, `"messages"`, `"count":2`} {
		if !strings.Contains(jsonText, want) {
			t.Fatalf("json text = %q, want it to contain %q", jsonText, want)
		}
	}
}

func TestToDecisionInputHostedImageKeptInJSON(t *testing.T) {
	human := messages.Message{
		Role: messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{
			messages.ImageBlock{URL: "https://example.com/cat.png"},
		},
	}
	got, err := toDecisionInput([]messages.Message{messages.AI("hello"), human})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	content := m["content"].([]any)
	textPart := content[0].(map[string]any)
	jsonText := textPart["text"].(string)
	if !strings.Contains(jsonText, `"image_url"`) || !strings.Contains(jsonText, "https://example.com/cat.png") {
		t.Fatalf("json text = %q, hosted image should stay inside the JSON as an image_url block", jsonText)
	}
	if len(content) != 1 {
		t.Fatalf("content = %#v, want a single text part (no base64 images to lift)", content)
	}
}

func TestToDecisionInputMediaErrorInJSONPath(t *testing.T) {
	human := messages.Message{
		Role: messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{
			messages.AudioBlock{Base64: "x"},
		},
	}
	_, err := toDecisionInput([]messages.Message{messages.AI("hi"), human})
	if err == nil {
		t.Fatal("toDecisionInput() expected an error for audio blocks")
	}
	if !errorsIsInvalidInput(err) {
		t.Fatalf("toDecisionInput() error = %v, want ErrInvalidInput", err)
	}
	if !strings.Contains(err.Error(), "audio") {
		t.Fatalf("toDecisionInput() error = %v, want it to mention the block type", err)
	}
}

func TestToDecisionInputNonStringMapKeys(t *testing.T) {
	_, err := toDecisionInput(map[int]string{1: "x"}) //nolint:staticcheck // intentionally wrong key type
	if err == nil {
		t.Fatal("toDecisionInput() expected an error for non-string keys")
	}
	if !strings.Contains(err.Error(), "keys must be strings") {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
}

func TestToDecisionInputUnsupportedValue(t *testing.T) {
	_, err := toDecisionInput(map[string]any{"bad": make(chan int)})
	if err == nil {
		t.Fatal("toDecisionInput() expected an error for unsupported values")
	}
	if !strings.Contains(err.Error(), "Unsupported OpenAI Decisions state value") {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	if !strings.Contains(err.Error(), "chan int") {
		t.Fatalf("toDecisionInput() error = %v, want the Go type name", err)
	}
}

func TestToDecisionInputAIToolCallsInJSON(t *testing.T) {
	ai := messages.Message{
		Role: messages.RoleAI,
		ToolCalls: []messages.ToolCall{
			{ID: "call_1", Name: "delete_file", Args: map[string]any{"path": "/tmp/x"}},
		},
	}
	got, err := toDecisionInput([]messages.Message{ai, messages.Human("ok")})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	content := m["content"].([]any)
	var jsonText strings.Builder
	for _, part := range content {
		if pm, ok := part.(map[string]any); ok && pm["type"] == "input_text" {
			jsonText.WriteString(pm["text"].(string))
		}
	}
	joined := jsonText.String()
	for _, want := range []string{`"tool_calls"`, `"function"`, `"delete_file"`, `"arguments":"{\"path\":\"/tmp/x\"}"`} {
		if !strings.Contains(joined, want) {
			t.Fatalf("json text = %q, want it to contain %q", joined, want)
		}
	}
}

func TestToDecisionInputFileIDImageErrorInJSON(t *testing.T) {
	human := messages.Message{
		Role:          messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{messages.ImageBlock{FileID: "file-9"}},
	}
	_, err := toDecisionInput([]messages.Message{messages.AI("hi"), human})
	if err == nil || !strings.Contains(err.Error(), "image") {
		t.Fatalf("toDecisionInput() error = %v, want an error mentioning the image block", err)
	}
}

func TestToDecisionInputAnySequenceAllHuman(t *testing.T) {
	got, err := toDecisionInput([]any{messages.Human("a"), messages.Human("b")})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	list := got.([]any)
	if len(list) != 2 {
		t.Fatalf("toDecisionInput() = %#v, want two native user messages", got)
	}
	if list[0].(map[string]any)["content"] != "a" {
		t.Fatalf("first content = %#v", list[0])
	}
}

func TestToDecisionInputAnySequenceMixedGoesJSON(t *testing.T) {
	got, err := toDecisionInput([]any{messages.Human("a"), "just a string"})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	if m["role"] != "user" {
		t.Fatalf("role = %v, want user (JSON path)", m["role"])
	}
	content := m["content"].([]any)
	text := content[0].(map[string]any)["text"].(string)
	if !strings.Contains(text, `"just a string"`) {
		t.Fatalf("json text = %q", text)
	}
}

func TestToDecisionInputHumanSequenceMediaError(t *testing.T) {
	human := messages.Message{
		Role:          messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{messages.VideoBlock{URL: "https://example.com/v.mp4"}},
	}
	_, err := toDecisionInput([]messages.Message{messages.Human("ok"), human})
	if err == nil || !strings.Contains(err.Error(), "video") {
		t.Fatalf("toDecisionInput() error = %v, want an error mentioning the video block", err)
	}
}

func TestToDecisionInputNamedToolMessageAndReasoningInJSON(t *testing.T) {
	tool := messages.Message{
		Role:       messages.RoleTool,
		Name:       "search",
		ToolCallID: "call_1",
		Content:    "result",
	}
	ai := messages.Message{
		Role:          messages.RoleAI,
		ContentBlocks: []messages.ContentBlock{messages.ReasoningBlock{Reasoning: "thinking"}},
	}
	got, err := toDecisionInput([]messages.Message{tool, ai, messages.Human("next")})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	content := m["content"].([]any)
	var jsonText strings.Builder
	for _, part := range content {
		if pm, ok := part.(map[string]any); ok && pm["type"] == "input_text" {
			jsonText.WriteString(pm["text"].(string))
		}
	}
	joined := jsonText.String()
	for _, want := range []string{`"name":"search"`, `"tool_call_id":"call_1"`, `"reasoning":"thinking"`} {
		if !strings.Contains(joined, want) {
			t.Fatalf("json text = %q, want it to contain %q", joined, want)
		}
	}
}

func TestToDecisionInputNullValuesPassThrough(t *testing.T) {
	got, err := toDecisionInput(map[string]any{"k": nil})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	m := jsonPathUserMessage(t, got)
	text := m["content"].([]any)[0].(map[string]any)["text"].(string)
	if !strings.Contains(text, `"k":null`) {
		t.Fatalf("json text = %q", text)
	}
}

func TestToDecisionInputJSONIsDeterministic(t *testing.T) {
	state := map[string]any{"b": 1, "a": 2}
	first, err := toDecisionInput(state)
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	second, err := toDecisionInput(map[string]any{"a": 2, "b": 1})
	if err != nil {
		t.Fatalf("toDecisionInput() error = %v", err)
	}
	fm := jsonPathUserMessage(t, first)
	sm := jsonPathUserMessage(t, second)
	fc := fm["content"].([]any)[0].(map[string]any)["text"].(string)
	sc := sm["content"].([]any)[0].(map[string]any)["text"].(string)
	if fc != sc {
		t.Fatalf("json text differs between runs: %q vs %q", fc, sc)
	}
	var decoded any
	if err := json.Unmarshal([]byte(fc), &decoded); err != nil {
		t.Fatalf("json text is not valid JSON: %v", err)
	}
}

// jsonPathUserMessage unwraps the single-element message list produced by
// the JSON serialization path.
func jsonPathUserMessage(t *testing.T, got any) map[string]any {
	t.Helper()
	list, ok := got.([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("toDecisionInput() = %#v, want a one-element user message list", got)
	}
	m, ok := list[0].(map[string]any)
	if !ok {
		t.Fatalf("toDecisionInput() = %#v, want a user message map", list[0])
	}
	return m
}

func mapsEqual(a, b map[string]any) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if bv, ok := b[k]; !ok || v != bv {
			return false
		}
	}
	return true
}
