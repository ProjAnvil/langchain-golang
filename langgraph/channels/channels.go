// Package channels implements the reducer/merge semantics behind LangGraph's
// "channels" concept (see Python's `langgraph.channels`).
//
// Scope note: Python's channels are pluggable stateful objects (LastValue,
// Topic, BinaryOperatorAggregate, EphemeralValue, ...) with their own
// checkpoint serialization boundary. This Go port simplifies that to a single
// concept: a Reducer function that combines an existing state value with an
// incoming update. Graph state itself is a plain map[string]any (see the
// graph package); each key's Reducer determines how concurrent/successive
// writes to that key are combined. This is sufficient to reproduce the two
// channel behaviors LangChain's agent loop actually depends on: "last write
// wins" (the default, equivalent to Python's LastValue) and "accumulate"
// (equivalent to Python's BinaryOperatorAggregate, most commonly used via
// `add_messages` for a conversation's `messages` key).
package channels

import (
	"fmt"
	"reflect"

	"github.com/projanvil/langchain-golang/core/messages"
)

// Reducer combines an existing channel value with an incoming update and
// returns the new value. It mirrors the binary operator behind Python's
// `BinaryOperatorAggregate` channel (e.g. `operator.add` for list
// concatenation, or `add_messages` for ID-aware message merging).
//
// existing is the zero value (nil) the first time a key is written.
type Reducer func(existing any, update any) (any, error)

// LastValueReducer overwrites the existing value with update, mirroring
// Python's `LastValue` channel: a key using this reducer is expected to
// receive at most one update per super-step. It is the default reducer for
// any state key without an explicit registration.
func LastValueReducer(_ any, update any) (any, error) {
	return update, nil
}

// AppendSliceReducer concatenates existing and update, treating both as
// slices of the same element type, mirroring Python's common
// `Annotated[list[T], operator.add]` pattern. existing may be nil (treated as
// an empty slice). Both values must be slices (or nil); anything else is a
// reducer error.
func AppendSliceReducer(existing any, update any) (any, error) {
	if existing == nil {
		return update, nil
	}
	if update == nil {
		return existing, nil
	}

	existingVal := reflect.ValueOf(existing)
	updateVal := reflect.ValueOf(update)
	if existingVal.Kind() != reflect.Slice || updateVal.Kind() != reflect.Slice {
		return nil, fmt.Errorf("channels: AppendSliceReducer requires slice values, got %T and %T", existing, update)
	}
	if existingVal.Type() != updateVal.Type() {
		return nil, fmt.Errorf("channels: AppendSliceReducer requires matching slice types, got %s and %s", existingVal.Type(), updateVal.Type())
	}

	out := reflect.AppendSlice(reflect.MakeSlice(existingVal.Type(), 0, existingVal.Len()+updateVal.Len()), existingVal)
	out = reflect.AppendSlice(out, updateVal)
	return out.Interface(), nil
}

// MessagesReducer merges message lists by ID, mirroring Python's
// `add_messages` (minus its `RemoveMessage`/`REMOVE_ALL_MESSAGES` deletion
// support and OpenAI content-format conversion, both out of scope here): a
// message in update whose ID matches a message already in existing replaces
// it in place; messages with new or empty IDs are appended in order. Both
// arguments must be []messages.Message (or nil).
func MessagesReducer(existing any, update any) (any, error) {
	existingMsgs, err := asMessageSlice(existing, "existing")
	if err != nil {
		return nil, err
	}
	updateMsgs, err := asMessageSlice(update, "update")
	if err != nil {
		return nil, err
	}

	merged := make([]messages.Message, len(existingMsgs), len(existingMsgs)+len(updateMsgs))
	copy(merged, existingMsgs)
	indexByID := make(map[string]int, len(merged))
	for i, m := range merged {
		if m.ID != "" {
			indexByID[m.ID] = i
		}
	}

	for _, m := range updateMsgs {
		if m.ID != "" {
			if idx, ok := indexByID[m.ID]; ok {
				merged[idx] = m
				continue
			}
		}
		merged = append(merged, m)
		if m.ID != "" {
			indexByID[m.ID] = len(merged) - 1
		}
	}

	return merged, nil
}

func asMessageSlice(value any, label string) ([]messages.Message, error) {
	if value == nil {
		return nil, nil
	}
	msgs, ok := value.([]messages.Message)
	if !ok {
		return nil, fmt.Errorf("channels: MessagesReducer requires []messages.Message for %s, got %T", label, value)
	}
	return msgs, nil
}
