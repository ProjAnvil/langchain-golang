// This file ports Python's langchain_core.messages.notification (#41540):
// helpers for marking a human message as added by the agent harness — a
// background task update, a conversation summary — so downstream consumers
// can tell harness-written content apart from what the user typed.
//
// The marker lives in AdditionalKwargs under "lc_source" (the feature that
// added the message) and, when the feature has fields, "lc_data". Chat model
// integrations do not send these keys to the model.

package messages

import "maps"

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
// merges additional_kwargs with merge_dicts; that concatenates conflicting
// string values (merge_dicts' `merged[k] += v` for str) and copies keys the
// first message lacks. Scoped to the notification keys, a notification
// merged with an adjacent human message keeps the metadata with lc_source
// concatenated. Divergence: on a type conflict Python raises TypeError;
// MergeRuns cannot fail, so the first message's value wins. Both arguments
// are clones owned by the caller, so values may be moved between them.
func mergeNotificationKwargs(last *Message, current Message) {
	currentSource, hasSource := current.AdditionalKwargs["lc_source"]
	if !hasSource {
		return
	}
	if last.AdditionalKwargs == nil {
		last.AdditionalKwargs = make(map[string]any, 2)
	}
	if leftSource, ok := last.AdditionalKwargs["lc_source"].(string); ok {
		if rightSource, ok := currentSource.(string); ok {
			last.AdditionalKwargs["lc_source"] = leftSource + rightSource
		}
	} else {
		last.AdditionalKwargs["lc_source"] = currentSource
	}
	if _, ok := last.AdditionalKwargs["lc_data"]; !ok {
		if data, ok := current.AdditionalKwargs["lc_data"]; ok {
			last.AdditionalKwargs["lc_data"] = data
		}
	}
}
