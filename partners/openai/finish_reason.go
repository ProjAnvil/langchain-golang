package openai

// Finish-reason fields on response_metadata, porting Python
// langchain-openai's #41926 helpers: providers own the mapping from their
// raw finish values onto the shared normalized vocabulary, and core reads
// the shared keys when assembling message-finish stream events.

// chatFinishReasonValues maps Chat Completions finish_reason values onto the
// shared normalized_finish_reason vocabulary (base.py _CHAT_FINISH_REASONS).
var chatFinishReasonValues = map[string]string{
	"stop":           "stop",
	"length":         "length",
	"tool_calls":     "tool_call",
	"content_filter": "content_filter",
}

// responsesIncompleteReasons maps Responses API incomplete_details.reason
// values onto the shared vocabulary (base.py _RESPONSES_INCOMPLETE_REASONS).
var responsesIncompleteReasons = map[string]string{
	"max_output_tokens": "length",
	"content_filter":    "content_filter",
}

// chatFinishReasonFields maps a Chat Completions finish_reason onto the
// shared finish fields (#41926). Only normalized_finish_reason is added: the
// raw finish_reason already lands in response_metadata on its own. OpenAI
// refuses with a normal stop — the refusal is the signal — and a forced
// tool_choice ends with stop even though the model called a tool.
func chatFinishReasonFields(finishReason string, hasToolCalls, refusal bool) map[string]any {
	if finishReason == "" {
		return nil
	}
	normalized := "other"
	if refusal {
		normalized = "refusal"
	} else if value, ok := chatFinishReasonValues[finishReason]; ok {
		normalized = value
	}
	if normalized == "stop" && hasToolCalls {
		normalized = "tool_call"
	}
	return map[string]any{"normalized_finish_reason": normalized}
}

// responsesFinishReasonFields maps a Responses API status onto the shared
// finish fields (#41926): finish_reason keeps the raw status, and an
// incomplete response resolves its incomplete_details.reason.
func responsesFinishReasonFields(status string, incompleteDetails map[string]any, hasToolCalls, refusal bool) map[string]any {
	if status == "" {
		return nil
	}
	var normalized string
	switch {
	case refusal:
		normalized = "refusal"
	case status == "completed":
		if hasToolCalls {
			normalized = "tool_call"
		} else {
			normalized = "stop"
		}
	case status == "incomplete":
		reason, _ := incompleteDetails["reason"].(string)
		if value, ok := responsesIncompleteReasons[reason]; ok {
			normalized = value
		} else {
			normalized = "other"
		}
	case status == "failed":
		normalized = "error"
	default:
		normalized = "other"
	}
	return map[string]any{
		"finish_reason":            status,
		"normalized_finish_reason": normalized,
	}
}
