package openai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
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

// TestStructuredOutputChatCompletionsRequest pins the Chat Completions wire
// shape for WithStructuredOutput: the structuredOutput branch of the Responses
// builder (buildRequest, chatmodel.go) must reach /chat/completions too, in the
// CC-native nested form
// {"type":"json_schema","json_schema":{"name":...,"schema":...,"strict":...}}.
// Regression: buildChatCompletionsRequest used to read only responseFormat,
// dropping the schema from the payload entirely.
func TestStructuredOutputChatCompletionsRequest(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceChatBody)
	model := toolChoiceBoundModel(t, server.URL, true).
		WithStructuredOutput("joke", map[string]any{"type": "object"}, true)
	if _, err := model.Invoke(context.Background(), []messages.Message{messages.Human("hi")}); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	format, ok := (*got)["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format = %v, want object", (*got)["response_format"])
	}
	nested, ok := format["json_schema"].(map[string]any)
	if !ok || format["type"] != "json_schema" {
		t.Fatalf("response_format = %v, want {type:json_schema json_schema:{name,schema,strict}}", format)
	}
	if nested["name"] != "joke" || nested["strict"] != true {
		t.Fatalf("json_schema = %v, want {name:joke strict:true}", nested)
	}
	schemaObj, ok := nested["schema"].(map[string]any)
	if !ok || schemaObj["type"] != "object" {
		t.Fatalf("json_schema.schema = %v, want {type:object}", nested["schema"])
	}
}

// TestInvokeStructuredChatCompletionsRequest drives the
// language.StructuredCaller native path in Chat Completions mode:
// InvokeStructured's json_schema binding (name derived from the schema
// "title", strict=true) must serialize into the CC request payload, matching
// the Responses-API behavior pinned by TestChatModelInvokeStructured.
func TestInvokeStructuredChatCompletionsRequest(t *testing.T) {
	server, got := toolChoiceServer(t, toolChoiceChatBody)
	model := toolChoiceBoundModel(t, server.URL, true)

	sch := schema.Object(map[string]schema.Schema{
		"answer": schema.String("yes/no answer"),
	}, "answer")
	sch["title"] = "answer_schema"

	if _, err := model.InvokeStructured(context.Background(), []messages.Message{
		messages.Human("answer yes"),
	}, sch); err != nil {
		t.Fatalf("InvokeStructured: %v", err)
	}
	format, ok := (*got)["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format = %v, want object", (*got)["response_format"])
	}
	nested, ok := format["json_schema"].(map[string]any)
	if !ok || format["type"] != "json_schema" {
		t.Fatalf("response_format = %v, want {type:json_schema json_schema:{name,schema,strict}}", format)
	}
	if nested["name"] != "answer_schema" || nested["strict"] != true {
		t.Fatalf("json_schema = %v, want {name:answer_schema strict:true}", nested)
	}
	// Compare the schema as JSON because the request round-trips through JSON
	// (so []string required-keys decode back as []any, breaking reflect.DeepEqual).
	gotSchemaJSON, err := json.Marshal(nested["schema"])
	if err != nil {
		t.Fatalf("marshal got schema: %v", err)
	}
	wantSchemaJSON, err := json.Marshal(sch)
	if err != nil {
		t.Fatalf("marshal want schema: %v", err)
	}
	if string(gotSchemaJSON) != string(wantSchemaJSON) {
		t.Fatalf("json_schema.schema deep-equal:\n got: %s\nwant: %s",
			gotSchemaJSON, wantSchemaJSON)
	}
}
