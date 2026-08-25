package messages

import "encoding/json"

// DefaultToolParser is a best-effort parser for raw tool call dicts,
// mirroring Python's default_tool_parser (messages/tool.py:349-383). It
// handles ONLY the OpenAI "function" wire shape: entries without a
// "function" key are skipped (this is deliberately stricter than the
// lenient parseToolCalls used by ConvertToMessages, which also accepts the
// LangChain {"name","args"} shape and keeps raw argument strings on JSON
// failure). Entries whose function arguments are not valid JSON become
// InvalidToolCallBlocks (Python's InvalidToolCall, with error=None → the
// zero Error string). Divergence: Python indexes
// raw_tool_call["function"]["arguments"]/["name"] directly, so a missing
// key raises KeyError; Go is lenient — missing arguments parse as an
// invalid call (the empty string fails json.Unmarshal) and a missing name
// becomes "". Divergence: non-string or non-object JSON arguments
// (e.g. a JSON array) become InvalidToolCallBlocks, because ToolCall.Args is
// a map; Python's json.loads would accept them into a dict-typed field.
func DefaultToolParser(rawToolCalls []map[string]any) ([]ToolCall, []InvalidToolCallBlock) {
	toolCalls := []ToolCall{}
	invalidToolCalls := []InvalidToolCallBlock{}
	for _, rawToolCall := range rawToolCalls {
		functionMap, ok := rawToolCall["function"].(map[string]any)
		if !ok {
			continue
		}
		functionName, _ := functionMap["name"].(string)
		rawArguments, _ := functionMap["arguments"].(string)
		id, _ := rawToolCall["id"].(string)
		args := map[string]any{}
		if err := json.Unmarshal([]byte(rawArguments), &args); err != nil {
			invalidToolCalls = append(invalidToolCalls, InvalidToolCallBlock{
				Name: functionName,
				Args: rawArguments,
				ID:   id,
			})
			continue
		}
		if args == nil { // "null" arguments → {} (Python: args or {})
			args = map[string]any{}
		}
		toolCalls = append(toolCalls, ToolCall{
			Name: functionName,
			Args: args,
			ID:   id,
		})
	}
	return toolCalls, invalidToolCalls
}

// DefaultToolChunkParser is the streaming sibling of DefaultToolParser,
// mirroring Python's default_tool_chunk_parser (messages/tool.py:386-412).
// Every entry yields a chunk — entries without "function" produce zero
// name/args (Python None) with id/index still passed through — and function
// arguments stay raw, possibly partial, JSON strings.
func DefaultToolChunkParser(rawToolCalls []map[string]any) []ToolCallChunkBlock {
	chunks := make([]ToolCallChunkBlock, 0, len(rawToolCalls))
	for _, rawToolCall := range rawToolCalls {
		chunk := ToolCallChunkBlock{}
		if functionMap, ok := rawToolCall["function"].(map[string]any); ok {
			chunk.Name, _ = functionMap["name"].(string)
			chunk.Args, _ = functionMap["arguments"].(string)
		}
		chunk.ID, _ = rawToolCall["id"].(string)
		chunk.Index = rawToolCall["index"]
		chunks = append(chunks, chunk)
	}
	return chunks
}
