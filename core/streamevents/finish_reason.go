package streamevents

import "github.com/projanvil/langchain-golang/core/messages"

// finishReasonKeys are the shared finish fields providers set on
// response_metadata (Python langchain-core 1.8.4, _finish_reason.py #41926):
// finish_reason keeps the provider's raw value, normalized_finish_reason the
// shared vocabulary, finish_state whether the turn completed or suspended.
var finishReasonKeys = [3]string{"finish_reason", "normalized_finish_reason", "finish_state"}

// FinishReasonFields returns the finish fields present in responseMetadata
// with the two provider-agnostic corrections applied (#41926): a refusal
// wins over any normalized reason, and a normalized "stop" with tool calls
// becomes "tool_call". Refusals and tool calls can arrive in earlier stream
// chunks than the finish reason, so core corrects them on the assembled
// message. An empty map means the provider set none of the keys.
func FinishReasonFields(responseMetadata map[string]any, hasToolCalls bool, hasRefusal bool) map[string]any {
	fields := make(map[string]any, len(finishReasonKeys))
	for _, key := range finishReasonKeys {
		if value, ok := responseMetadata[key]; ok && value != nil && value != "" {
			fields[key] = value
		}
	}
	normalized, _ := fields["normalized_finish_reason"].(string)
	if normalized == "" {
		return fields
	}
	if hasRefusal {
		fields["normalized_finish_reason"] = "refusal"
	} else if normalized == "stop" && hasToolCalls {
		fields["normalized_finish_reason"] = "tool_call"
	}
	return fields
}

// hasRefusalBlock reports whether the assembled message carries a refusal:
// a RefusalBlock in its content blocks or an additional_kwargs refusal
// (the Chat Completions spelling).
func hasRefusalBlock(message messages.Message) bool {
	// Truthiness, matching the Chat Completions spelling upstream
	// (base.py:2208): an empty refusal string does not count.
	if s, ok := message.AdditionalKwargs["refusal"].(string); ok && s != "" {
		return true
	}
	for _, block := range message.ContentBlocks {
		if _, ok := block.(messages.RefusalBlock); ok {
			return true
		}
	}
	return false
}
