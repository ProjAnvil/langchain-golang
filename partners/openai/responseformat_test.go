package openai

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Mirrors libs/partners/openai/tests/unit_tests/chat_models/test_base.py
// ::test__convert_to_openai_response_format and the json_mode branch of
// ::test_with_structured_output, plus chat_models/base.py:4344-4376
// (response_format -> text.format translation on the Responses API).

func TestJSONModeResponsesAPIRequest(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := toolChoiceBoundModel(t, server.URL, false).WithJSONMode()
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	text, ok := (*got)["text"].(map[string]any)
	if !ok {
		t.Fatalf("text = %v, want object", (*got)["text"])
	}
	format, ok := text["format"].(map[string]any)
	if !ok || format["type"] != "json_object" {
		t.Fatalf("text.format = %v, want {type:json_object}", text["format"])
	}
	if _, leaked := (*got)["response_format"]; leaked {
		t.Fatal("response_format must not leak into a Responses API payload")
	}
}

func TestJSONModeChatCompletionsRequest(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceChatBody)
	model := toolChoiceBoundModel(t, server.URL, true).WithJSONMode()
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	format, ok := (*got)["response_format"].(map[string]any)
	if !ok || format["type"] != "json_object" {
		t.Fatalf("response_format = %v, want {type:json_object}", (*got)["response_format"])
	}
}

func TestResponseFormatRawPassthroughChatCompletions(t *testing.T) {
	raw := map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "math_reasoning",
			"schema": map[string]any{"type": "object"},
			"strict": true,
		},
	}
	server, got := toolChoiceServer(t, toolChoiceChatBody)
	model := toolChoiceBoundModel(t, server.URL, true).WithResponseFormat(raw)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	format, ok := (*got)["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format = %v, want object", (*got)["response_format"])
	}
	nested, ok := format["json_schema"].(map[string]any)
	if !ok || format["type"] != "json_schema" || nested["name"] != "math_reasoning" || nested["strict"] != true {
		t.Fatalf("response_format = %v, want verbatim passthrough of %v", format, raw)
	}
}

func TestResponseFormatJSONSchemaResponsesAPIFlattens(t *testing.T) {
	// base.py:4361-4374: {"type":"json_schema","json_schema":{...}} flattens
	// into text.format = {"type":"json_schema", **json_schema}.
	raw := map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "math_reasoning",
			"schema": map[string]any{"type": "object"},
			"strict": true,
		},
	}
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := toolChoiceBoundModel(t, server.URL, false).WithResponseFormat(raw)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	text, ok := (*got)["text"].(map[string]any)
	if !ok {
		t.Fatalf("text = %v, want object", (*got)["text"])
	}
	format, ok := text["format"].(map[string]any)
	if !ok || format["type"] != "json_schema" || format["name"] != "math_reasoning" || format["strict"] != true {
		t.Fatalf("text.format = %v, want flattened json_schema", text["format"])
	}
	if _, nested := format["json_schema"]; nested {
		t.Fatalf("nested json_schema key must be flattened away: %v", format)
	}
}

func TestStructuredOutputWinsOverResponseFormat(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceResponsesBody)
	model := toolChoiceBoundModel(t, server.URL, false).
		WithJSONMode().
		WithStructuredOutput("joke", map[string]any{"type": "object"}, true)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	text, ok := (*got)["text"].(map[string]any)
	if !ok {
		t.Fatalf("text = %v, want object", (*got)["text"])
	}
	format, ok := text["format"].(map[string]any)
	if !ok || format["type"] != "json_schema" || format["name"] != "joke" {
		t.Fatalf("text.format = %v, want structured-output json_schema to win", text["format"])
	}
}
