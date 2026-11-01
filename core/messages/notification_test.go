package messages

import (
	"reflect"
	"testing"
)

// TestAddNotificationMetadataReturnsCopy ports Python's
// test_metadata_is_added_to_a_copy: the marker lands on a copy with the same
// id, and the original message is untouched.
func TestAddNotificationMetadataReturnsCopy(t *testing.T) {
	original := Message{
		Role:             RoleHuman,
		Content:          "done",
		ID:               "msg_1",
		AdditionalKwargs: map[string]any{"other": 1},
	}
	message := AddNotificationMetadata(original, "async_subagent", map[string]any{"task_id": "t_42"})

	if message.ID != "msg_1" || message.Content != "done" || message.Role != RoleHuman {
		t.Fatalf("copy changed identity fields: %#v", message)
	}
	want := map[string]any{
		"other":     1,
		"lc_source": "async_subagent",
		"lc_data":   map[string]any{"task_id": "t_42"},
	}
	if !reflect.DeepEqual(message.AdditionalKwargs, want) {
		t.Fatalf("kwargs = %#v, want %#v", message.AdditionalKwargs, want)
	}
	if !reflect.DeepEqual(original.AdditionalKwargs, map[string]any{"other": 1}) {
		t.Fatalf("original mutated: %#v", original.AdditionalKwargs)
	}
}

// TestGetNotificationMetadata ports Python's test_metadata_is_read_back.
func TestGetNotificationMetadata(t *testing.T) {
	message := AddNotificationMetadata(Human("done"), "async_subagent", map[string]any{"task_id": "t_42"})

	got := GetNotificationMetadata(message)
	if got == nil || got.Source != "async_subagent" {
		t.Fatalf("metadata = %#v, want source async_subagent", got)
	}
	if !reflect.DeepEqual(got.Data, map[string]any{"task_id": "t_42"}) {
		t.Fatalf("data = %#v", got.Data)
	}
}

// TestAddNotificationMetadataReplaces ports Python's
// test_marking_again_replaces_source_and_data: re-marking replaces lc_source
// and drops the earlier lc_data when no new data is given.
func TestAddNotificationMetadataReplaces(t *testing.T) {
	message := AddNotificationMetadata(Human("x"), "first", map[string]any{"a": 1})
	remarked := AddNotificationMetadata(message, "second", nil)

	got := GetNotificationMetadata(remarked)
	if got == nil || got.Source != "second" || got.Data != nil {
		t.Fatalf("metadata = %#v, want source second without data", got)
	}
}

// TestAddNotificationMetadataCopiesData ports Python's test_data_is_copied:
// the caller's map is cloned, later mutations don't reach the message.
func TestAddNotificationMetadataCopiesData(t *testing.T) {
	data := map[string]any{"status": "success"}
	message := AddNotificationMetadata(Human("x"), "s", data)
	data["status"] = "error"

	got := GetNotificationMetadata(message)
	if got == nil || !reflect.DeepEqual(got.Data, map[string]any{"status": "success"}) {
		t.Fatalf("data = %#v, want the copied value", got)
	}
}

// TestGetNotificationMetadataWithoutSource ports Python's
// test_messages_without_a_string_source_have_no_metadata.
func TestGetNotificationMetadataWithoutSource(t *testing.T) {
	if got := GetNotificationMetadata(Human("hi")); got != nil {
		t.Fatalf("plain message metadata = %#v, want nil", got)
	}
	if got := GetNotificationMetadata(Message{
		Role:             RoleHuman,
		AdditionalKwargs: map[string]any{"lc_source": 1},
	}); got != nil {
		t.Fatalf("non-string source metadata = %#v, want nil", got)
	}
}

// TestGetNotificationMetadataNonMapData ports Python's
// test_non_dict_data_is_left_out: a non-map lc_data is dropped from the
// result while the source survives.
func TestGetNotificationMetadataNonMapData(t *testing.T) {
	got := GetNotificationMetadata(Message{
		Role: RoleHuman,
		AdditionalKwargs: map[string]any{
			"lc_source": "s",
			"lc_data":   "not a map",
		},
	})
	if got == nil || got.Source != "s" || got.Data != nil {
		t.Fatalf("metadata = %#v, want source s without data", got)
	}
}

// TestNotificationMetadataSerialization ports Python's
// test_metadata_survives_serialization: lc_source/lc_data round-trip through
// the stable dict form.
func TestNotificationMetadataSerialization(t *testing.T) {
	message := AddNotificationMetadata(
		Message{Role: RoleHuman, Content: "done", ID: "msg_1"},
		"async_subagent",
		map[string]any{"a": 1},
	)
	dict, err := MessagesToDict([]Message{message})
	if err != nil {
		t.Fatalf("MessagesToDict: %v", err)
	}
	restored, err := MessagesFromDict(dict)
	if err != nil {
		t.Fatalf("MessagesFromDict: %v", err)
	}
	if len(restored) != 1 {
		t.Fatalf("restored %d messages", len(restored))
	}
	// Divergence from Python's `restored == message`: Go's encoding/json
	// decodes numbers as float64, so whole-struct DeepEqual against the
	// original (int 1) cannot hold. Compare the fields that matter.
	if restored[0].Role != RoleHuman || restored[0].Content != "done" || restored[0].ID != "msg_1" {
		t.Fatalf("restored = %#v", restored[0])
	}
	got := GetNotificationMetadata(restored[0])
	if got == nil || got.Source != "async_subagent" || !reflect.DeepEqual(got.Data, map[string]any{"a": float64(1)}) {
		t.Fatalf("restored metadata = %#v", got)
	}
}

// TestMergeRunsConcatenatesNotificationSource pins the MergeRuns behavior the
// Python helpers document: chunk addition merges additional_kwargs with
// merge_dicts, which concatenates conflicting string values, so a
// notification merged with an adjacent human message keeps the metadata with
// lc_source concatenated.
func TestMergeRunsConcatenatesNotificationSource(t *testing.T) {
	merged := MergeRuns([]Message{
		{Role: RoleHuman, Content: "a", AdditionalKwargs: map[string]any{"lc_source": "first"}},
		{Role: RoleHuman, Content: "b", AdditionalKwargs: map[string]any{"lc_source": "second"}},
	})
	if len(merged) != 1 {
		t.Fatalf("merged %d messages, want 1: %#v", len(merged), merged)
	}
	if source, ok := merged[0].AdditionalKwargs["lc_source"].(string); !ok || source != "firstsecond" {
		t.Fatalf("lc_source = %#v, want firstsecond", merged[0].AdditionalKwargs["lc_source"])
	}
	if merged[0].Content != "a\nb" {
		t.Fatalf("content = %q", merged[0].Content)
	}
}

// TestMergeRunsCarriesNotificationSource covers merge_dicts' missing-key
// rule: lc_source only on the incoming message is carried onto the merged
// one (a notification joined with a plain human message keeps the metadata).
func TestMergeRunsCarriesNotificationSource(t *testing.T) {
	merged := MergeRuns([]Message{
		Human("question"),
		AddNotificationMetadata(Human("update"), "async_subagent", map[string]any{"task": "t"}),
	})
	if len(merged) != 1 {
		t.Fatalf("merged %d messages, want 1: %#v", len(merged), merged)
	}
	got := GetNotificationMetadata(merged[0])
	if got == nil || got.Source != "async_subagent" || !reflect.DeepEqual(got.Data, map[string]any{"task": "t"}) {
		t.Fatalf("merged metadata = %#v, want carried source/data", got)
	}
}

// TestMergeRunsMergesNotificationData pins merge_dicts' dict branch scoped to
// lc_data (#41540 review): when both merged messages carry map lc_data, the
// result deep-merges — nested maps recurse, right-only keys are copied,
// equal values are kept and conflicting strings concatenate (merge_dicts'
// `+=`, so "ok" + "done" merges to "okdone").
func TestMergeRunsMergesNotificationData(t *testing.T) {
	merged := MergeRuns([]Message{
		AddNotificationMetadata(Human("a"), "s", map[string]any{
			"task": map[string]any{"id": "t1", "status": "ok"},
			"keep": "x",
		}),
		AddNotificationMetadata(Human("b"), "s2", map[string]any{
			"task":  map[string]any{"id": "t1", "status": "done", "note": "n"},
			"extra": true,
		}),
	})
	if len(merged) != 1 {
		t.Fatalf("merged %d messages, want 1: %#v", len(merged), merged)
	}
	// "id" is equal on both sides and stays single (merge_dicts' `continue`
	// before the str concat).
	wantData := map[string]any{
		"task":  map[string]any{"id": "t1", "status": "okdone", "note": "n"},
		"keep":  "x",
		"extra": true,
	}
	if !reflect.DeepEqual(merged[0].AdditionalKwargs["lc_data"], wantData) {
		t.Fatalf("lc_data = %#v, want deep-merged %#v", merged[0].AdditionalKwargs["lc_data"], wantData)
	}
	if source, _ := merged[0].AdditionalKwargs["lc_source"].(string); source != "ss2" {
		t.Fatalf("lc_source = %#v, want ss2", merged[0].AdditionalKwargs["lc_source"])
	}
}

// TestMergeRunsCarriesNotificationDataWithoutSource covers merge_dicts'
// missing-key rule for lc_data alone: an incoming lc_data without an
// lc_source is still carried onto the merged message (it is notification
// data even though the message is not itself marked).
func TestMergeRunsCarriesNotificationDataWithoutSource(t *testing.T) {
	merged := MergeRuns([]Message{
		Human("a"),
		{Role: RoleHuman, Content: "b", AdditionalKwargs: map[string]any{
			"lc_data": map[string]any{"k": 1},
		}},
	})
	if len(merged) != 1 {
		t.Fatalf("merged %d messages, want 1: %#v", len(merged), merged)
	}
	if !reflect.DeepEqual(merged[0].AdditionalKwargs["lc_data"], map[string]any{"k": 1}) {
		t.Fatalf("lc_data = %#v, want carried", merged[0].AdditionalKwargs["lc_data"])
	}
	if got := GetNotificationMetadata(merged[0]); got != nil {
		t.Fatalf("metadata = %#v, want nil without lc_source", got)
	}
}
