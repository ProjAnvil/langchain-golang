package openai

import "github.com/projanvil/langchain-golang/core/schema"

// WithJSONMode returns a copy of the model constrained to emit valid JSON,
// mirroring Python model.bind(response_format={"type": "json_object"})
// (chat_models/base.py:3094). On the Responses API it becomes
// text.format={"type":"json_object"}; on Chat Completions it is sent as
// response_format={"type":"json_object"}.
func (m ChatModel) WithJSONMode() ChatModel {
	return m.WithResponseFormat(map[string]any{"type": "json_object"})
}

// WithResponseFormat returns a copy of the model sending a raw
// provider-native response_format dict (Python's dict passthrough form).
// Supported shapes: {"type":"json_object"} and
// {"type":"json_schema","json_schema":{"name":...,"schema":...,"strict":...}}
// (flattened into text.format on the Responses API, base.py:4356-4374).
// WithStructuredOutput (schema-typed config) takes precedence when both are
// set.
func (m ChatModel) WithResponseFormat(format map[string]any) ChatModel {
	next := m
	next.responseFormat = format
	return next
}

// toResponseFormat converts a raw response_format dict into the Responses API
// text.format shape (chat_models/base.py:4356-4374).
func toResponseFormat(raw map[string]any) responseFormat {
	switch raw["type"] {
	case "json_object":
		return responseFormat{Type: "json_object"}
	case "json_schema":
		// Accept both the nested chat shape
		// {"type":"json_schema","json_schema":{...}} and a flat
		// {"type":"json_schema","name":...,"schema":...} form.
		src := raw
		if nested, ok := raw["json_schema"].(map[string]any); ok {
			src = nested
		}
		out := responseFormat{Type: "json_schema"}
		out.Name, _ = src["name"].(string)
		if sch, ok := src["schema"].(map[string]any); ok {
			out.Schema = schema.Schema(sch)
		}
		out.Strict, _ = src["strict"].(bool)
		return out
	default:
		typ, _ := raw["type"].(string)
		return responseFormat{Type: typ}
	}
}
