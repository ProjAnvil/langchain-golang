package openai

// ToolChoice constrains which tool the model must call, mirroring Python
// ChatOpenAI.bind_tools(tool_choice=...) (chat_models/base.py:2268-2294).
// Construct it with one of the ToolChoice* helpers and attach it with
// ChatModel.WithToolChoice.
type ToolChoice struct {
	// value is either a string mode ("auto"|"none"|"required") or a
	// map[string]any provider object ({"type":"function","function":{...}} or
	// a raw passthrough dict).
	value any
}

// ToolChoiceAuto lets the model decide whether to call a tool ("auto").
func ToolChoiceAuto() ToolChoice { return ToolChoice{value: "auto"} }

// ToolChoiceNone forbids tool calls ("none").
func ToolChoiceNone() ToolChoice { return ToolChoice{value: "none"} }

// ToolChoiceRequired forces at least one tool call ("required"). Python's
// bind_tools maps "any" and True to "required"; the same applies here.
func ToolChoiceRequired() ToolChoice { return ToolChoice{value: "required"} }

// ToolChoiceFunction forces the named function tool, mirroring Python's
// {"type": "function", "function": {"name": <<tool_name>>}} form.
func ToolChoiceFunction(name string) ToolChoice {
	return ToolChoice{value: map[string]any{
		"type":     "function",
		"function": map[string]any{"name": name},
	}}
}

// ToolChoiceRaw passes a provider-native tool_choice object through verbatim
// (Python's dict form), e.g. {"type": "file_search"} for well-known tools.
func ToolChoiceRaw(raw map[string]any) ToolChoice { return ToolChoice{value: raw} }

// responsesToolChoice translates the Chat-Completions-shaped tool_choice into
// the Responses API shape: {"type":"function","function":{"name":X}} becomes
// {"type":"function","name":X} (chat_models/base.py:4331-4341). Strings and
// other dicts pass through unchanged.
func responsesToolChoice(value any) any {
	dict, ok := value.(map[string]any)
	if !ok {
		return value
	}
	fn, ok := dict["function"].(map[string]any)
	if dict["type"] != "function" || !ok {
		return value
	}
	out := map[string]any{"type": "function"}
	for key, val := range fn {
		out[key] = val
	}
	return out
}
