package anthropic

// Finish-reason fields on response_metadata, porting Python
// langchain-anthropic's _finish_reason_fields (#41926): providers own the
// mapping from their raw stop values onto the shared vocabulary, and core
// reads the three keys when assembling message-finish stream events.

// finishReasonValues maps Messages API stop_reason values onto the shared
// normalized_finish_reason vocabulary (chat_models.py _FINISH_REASONS).
var finishReasonValues = map[string]string{
	"end_turn":                     "stop",
	"stop_sequence":                "stop",
	"max_tokens":                   "length",
	"tool_use":                     "tool_call",
	"refusal":                      "refusal",
	"model_context_window_exceeded": "context_exceeded",
}

// finishReasonFields maps a Messages API stop_reason onto the shared finish
// fields in response_metadata: finish_reason keeps the raw value,
// normalized_finish_reason the shared vocabulary, and finish_state reports
// whether the turn ended complete or suspended (pause_turn means the caller
// must send another request to finish). An empty stopReason yields nil.
func finishReasonFields(stopReason string) map[string]any {
	if stopReason == "" {
		return nil
	}
	normalized, ok := finishReasonValues[stopReason]
	if !ok {
		normalized = "other"
	}
	state := "complete"
	if stopReason == "pause_turn" {
		state = "suspended"
	}
	return map[string]any{
		"finish_reason":           stopReason,
		"normalized_finish_reason": normalized,
		"finish_state":            state,
	}
}
