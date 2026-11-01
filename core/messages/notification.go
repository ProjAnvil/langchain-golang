// This file ports Python's langchain_core.messages.notification (#41540):
// helpers for marking a human message as added by the agent harness — a
// background task update, a conversation summary — so downstream consumers
// can tell harness-written content apart from what the user typed.
//
// The marker lives in AdditionalKwargs under "lc_source" (the feature that
// added the message) and, when the feature has fields, "lc_data". Chat model
// integrations do not send these keys to the model.

package messages

import (
	"maps"
	"reflect"
	"slices"
)

// NotificationMetadata reports which feature added a message, and that
// feature's fields. It mirrors Python's NotificationMetadata TypedDict:
// Source is always present; Data is nil unless the message carries a map
// lc_data.
type NotificationMetadata struct {
	// Source names the feature that added the message, e.g. "async_subagent".
	Source string
	// Data holds the feature's fields, documented by that feature. It is a
	// copy of lc_data and is nil when lc_data is absent or not a map.
	Data map[string]any
}

// AddNotificationMetadata returns a copy of message marked as added by the
// agent harness. Use it for a human message the user didn't type, such as a
// background task update or a conversation summary. It sets
// AdditionalKwargs["lc_source"] and, when data is non-nil,
// AdditionalKwargs["lc_data"] (a copy of data). Any earlier lc_source and
// lc_data are removed first. Content is unchanged — write the content the
// model reads yourself.
//
// It does not validate the message role, but the marker is only meaningful
// on human messages. Anyone who can send input to a thread can set these
// keys; don't use them to decide whether to trust a message.
//
// MergeRuns joins a notification with adjacent mergeable messages; the
// merged message keeps the metadata and concatenates lc_source.
func AddNotificationMetadata(message Message, source string, data map[string]any) Message {
	message = Clone(message)
	kwargs := make(map[string]any, len(message.AdditionalKwargs)+2)
	maps.Copy(kwargs, message.AdditionalKwargs)
	delete(kwargs, "lc_data")
	kwargs["lc_source"] = source
	if data != nil {
		kwargs["lc_data"] = maps.Clone(data)
	}
	message.AdditionalKwargs = kwargs
	return message
}

// GetNotificationMetadata returns the notification metadata of message: the
// metadata AddNotificationMetadata set, or nil for a message without a
// string lc_source. Data is a copy of lc_data and is left out unless
// lc_data is a map.
func GetNotificationMetadata(message Message) *NotificationMetadata {
	source, ok := message.AdditionalKwargs["lc_source"].(string)
	if !ok {
		return nil
	}
	metadata := &NotificationMetadata{Source: source}
	if data, ok := message.AdditionalKwargs["lc_data"].(map[string]any); ok {
		metadata.Data = maps.Clone(data)
	}
	return metadata
}

// mergeNotificationKwargs carries notification metadata across MergeRuns.
// Python's merge_message_runs joins messages through chunk addition, which
// merges additional_kwargs with merge_dicts; scoped to the notification
// keys, a notification merged with an adjacent human message keeps the
// metadata with lc_source concatenated and lc_data deep-merged. Keys the
// first message lacks are copied from the incoming one (merge_dicts'
// missing-key rule), so an lc_data without an lc_source is carried too.
// Divergence: where merge_dicts raises TypeError (lc_source or lc_data type
// conflicts) MergeRuns cannot fail — for lc_source, a string first value
// with a non-string incoming value keeps the first, and any non-string
// first value is replaced by the incoming one; for lc_data leaves, the
// first value wins. Both arguments are clones owned by the caller, so
// values may be moved between them.
func mergeNotificationKwargs(last *Message, current Message) {
	if len(current.AdditionalKwargs) == 0 {
		return
	}
	if last.AdditionalKwargs == nil {
		last.AdditionalKwargs = make(map[string]any, 2)
	}
	if currentSource, ok := current.AdditionalKwargs["lc_source"]; ok {
		if leftSource, ok := last.AdditionalKwargs["lc_source"].(string); ok {
			if rightSource, ok := currentSource.(string); ok {
				last.AdditionalKwargs["lc_source"] = leftSource + rightSource
			}
		} else {
			last.AdditionalKwargs["lc_source"] = currentSource
		}
	}
	if currentData, ok := current.AdditionalKwargs["lc_data"]; ok {
		if leftData, exists := last.AdditionalKwargs["lc_data"]; exists {
			if leftMap, lok := leftData.(map[string]any); lok {
				if rightMap, rok := currentData.(map[string]any); rok {
					last.AdditionalKwargs["lc_data"] = mergeNotificationData(leftMap, rightMap)
				}
			}
		} else {
			last.AdditionalKwargs["lc_data"] = currentData
		}
	}
}

// mergeNotificationData deep-merges two lc_data maps the way merge_dicts
// merges dict+dict: right-only keys are copied in, equal values are kept
// (Python's `continue` — this is what keeps an equal "id" from doubling),
// nested maps recurse, strings concatenate (merge_dicts' `+=`), and lists
// append (merge_lists' plain append). Any other leaf conflict — type
// mismatch, differing bools or numbers — keeps the left value where Python
// raises TypeError or sums ints; MergeRuns has no error channel and
// accumulating user data would be more surprising than keeping the first
// value.
func mergeNotificationData(left map[string]any, right map[string]any) map[string]any {
	merged := maps.Clone(left)
	if merged == nil {
		merged = make(map[string]any, len(right))
	}
	for key, rightValue := range right {
		leftValue, exists := merged[key]
		if !exists || equalAny(leftValue, rightValue) {
			merged[key] = rightValue
			continue
		}
		switch leftTyped := leftValue.(type) {
		case map[string]any:
			if rightMap, ok := rightValue.(map[string]any); ok {
				merged[key] = mergeNotificationData(leftTyped, rightMap)
			}
		case string:
			if rightString, ok := rightValue.(string); ok {
				merged[key] = leftTyped + rightString
			}
		case []any:
			if rightList, ok := rightValue.([]any); ok {
				merged[key] = append(slices.Clone(leftTyped), rightList...)
			}
		}
		// Other conflicts keep the left value (see the doc comment).
	}
	return merged
}

// equalAny reports shallow equality of two any values, mirroring the
// `merged[k] == v` check merge_dicts applies before its type branches.
func equalAny(left, right any) bool {
	return reflect.DeepEqual(left, right)
}
