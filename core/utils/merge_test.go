package utils

import (
	"reflect"
	"strings"
	"testing"
)

// Mirrors test_utils.py:442 test_merge_lists (all parametrized cases),
// including the no-mutation assertion (Python deep-copies before merging).
func TestMergeLists(t *testing.T) {
	cases := []struct {
		name     string
		left     []any
		right    []any
		expected []any
	}{
		{"both nil", nil, nil, nil},
		{"left nil", nil, []any{1, 2}, []any{1, 2}},
		{"right nil", []any{1, 2}, nil, []any{1, 2}},
		{"simple merge", []any{1, 2}, []any{3, 4}, []any{1, 2, 3, 4}},
		{"empty lists", []any{}, []any{}, []any{}},
		{"empty left", []any{}, []any{1}, []any{1}},
		{"empty right", []any{1}, []any{}, []any{1}},
		{
			"merge with index handling",
			[]any{map[string]any{"index": 0, "text": "hello"}},
			[]any{map[string]any{"index": 0, "text": " world"}},
			[]any{map[string]any{"index": 0, "text": "hello world"}},
		},
		{
			"multiple elements with different indexes",
			[]any{map[string]any{"index": 0, "a": "x"}},
			[]any{map[string]any{"index": 1, "b": "y"}},
			[]any{map[string]any{"index": 0, "a": "x"}, map[string]any{"index": 1, "b": "y"}},
		},
		{
			"elements without index key",
			[]any{map[string]any{"no_index": "a"}},
			[]any{map[string]any{"no_index": "b"}},
			[]any{map[string]any{"no_index": "a"}, map[string]any{"no_index": "b"}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			leftCopy := deepCloneSlice(tc.left)
			rightCopy := deepCloneSlice(tc.right)
			actual, err := MergeLists(tc.left, tc.right)
			if err != nil {
				t.Fatalf("MergeLists: %v", err)
			}
			if !reflect.DeepEqual(actual, tc.expected) {
				t.Fatalf("MergeLists(%v, %v) = %v, want %v", tc.left, tc.right, actual, tc.expected)
			}
			if !reflect.DeepEqual(tc.left, leftCopy) {
				t.Fatalf("left mutated: %v", tc.left)
			}
			if !reflect.DeepEqual(tc.right, rightCopy) {
				t.Fatalf("right mutated: %v", tc.right)
			}
		})
	}
}

// deepCloneSlice round-trips through JSON-free copying sufficient for these
// fixtures (maps, slices, scalars).
func deepCloneSlice(in []any) []any {
	if in == nil {
		return nil
	}
	out := make([]any, len(in))
	for i, value := range in {
		out[i] = deepCloneValue(value)
	}
	return out
}

func deepCloneValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(typed))
		for k, v := range typed {
			out[k] = deepCloneValue(v)
		}
		return out
	case []any:
		return deepCloneSlice(typed)
	default:
		return typed
	}
}

// Mirrors test_merge_lists_multiple_others (test_utils.py:454).
func TestMergeListsMultipleOthers(t *testing.T) {
	result, err := MergeLists([]any{1}, []any{2}, []any{3})
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if !reflect.DeepEqual(result, []any{1, 2, 3}) {
		t.Fatalf("result: %v", result)
	}
}

// Mirrors test_merge_lists_all_none (test_utils.py:460).
func TestMergeListsAllNil(t *testing.T) {
	result, err := MergeLists(nil, nil, nil)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if result != nil {
		t.Fatalf("result: %v, want nil", result)
	}
}

// Index matching is numeric-type-insensitive so JSON-decoded chunks (float64
// indexes) merge with int-indexed chunks — the Go analog of Python's
// e_left["index"] == e["index"] comparison.
func TestMergeListsJSONDecodedIndex(t *testing.T) {
	left := []any{map[string]any{"index": 0, "text": "hello"}}
	right := []any{map[string]any{"index": float64(0), "text": " world"}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{map[string]any{"index": 0, "text": "hello world"}}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// int64 indexes merge like int indexes (Go-built chunks may carry int64
// values, e.g. from an int64 counter); also covers the int64 arms of
// numericAsFloat/coerceNumeric/indexAsFloat.
func TestMergeListsInt64Index(t *testing.T) {
	left := []any{map[string]any{"index": int64(0), "text": "hello"}}
	right := []any{map[string]any{"index": int64(0), "text": " world"}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{map[string]any{"index": int64(0), "text": "hello world"}}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// Elements with the same index but conflicting non-empty IDs do not merge
// (Python's ID-consistency rule, _merge.py:123-128).
func TestMergeListsConflictingIDsAppend(t *testing.T) {
	left := []any{map[string]any{"index": 0, "id": "a", "text": "hello"}}
	right := []any{map[string]any{"index": 0, "id": "b", "text": " world"}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{
		map[string]any{"index": 0, "id": "a", "text": "hello"},
		map[string]any{"index": 0, "id": "b", "text": " world"},
	}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// lc_-prefixed string indexes merge like integer indexes
// (_merge.py:111-115).
func TestMergeListsLCStringIndex(t *testing.T) {
	left := []any{map[string]any{"index": "lc_1", "text": "hello"}}
	right := []any{map[string]any{"index": "lc_1", "text": " world"}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{map[string]any{"index": "lc_1", "text": "hello world"}}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// When the existing entry has a "type" and the incoming chunk is
// non_standard, the incoming "value" folds into "extras"
// (_merge.py:133-144).
func TestMergeListsStandardPlusNonStandard(t *testing.T) {
	left := []any{map[string]any{"index": 0, "type": "text", "text": "hello"}}
	right := []any{map[string]any{
		"index": 0,
		"type":  "non_standard",
		"value": map[string]any{"type": "special", "payload": 42},
	}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{map[string]any{
		"index":  0,
		"type":   "text",
		"text":   "hello",
		"extras": map[string]any{"payload": 42},
	}}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// non_standard + non_standard folds the incoming value into "value" and keeps
// the index (_merge.py:145-155).
func TestMergeListsNonStandardPlusNonStandard(t *testing.T) {
	left := []any{map[string]any{
		"index": 0,
		"type":  "non_standard",
		"value": map[string]any{"a": 1},
	}}
	right := []any{map[string]any{
		"index": 0,
		"type":  "non_standard",
		"value": map[string]any{"type": "special", "b": 2},
	}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{map[string]any{
		"index": 0,
		"type":  "non_standard",
		"value": map[string]any{"a": 1, "b": 2},
	}}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// An incoming chunk's own "type" key is stripped when merging into an
// existing entry (_merge.py:156-161).
func TestMergeListsStripsIncomingType(t *testing.T) {
	left := []any{map[string]any{"index": 0, "text": "hello"}}
	right := []any{map[string]any{"index": 0, "type": "text", "text": " world"}}
	actual, err := MergeLists(left, right)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	want := []any{map[string]any{"index": 0, "text": "hello world"}}
	if !reflect.DeepEqual(actual, want) {
		t.Fatalf("result: %v, want %v", actual, want)
	}
}

// merge_dicts semantics exercised directly (Python covers them via
// test_merge_obj and the TypeError paths).
func TestMergeDicts(t *testing.T) {
	// None-value adoption + None right-values: a None right value is ASSIGNED
	// when the key is absent from the merged result (_merge.py:33-36 — the
	// skip branch only applies to pre-existing keys), so "skip": nil lands in
	// the output; None right-values for pre-existing keys are skipped.
	got, err := MergeDicts(
		map[string]any{"function_call": map[string]any{"arguments": nil}, "keep": 1},
		map[string]any{"function_call": map[string]any{"arguments": "{\n"}, "skip": nil},
	)
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	want := map[string]any{"function_call": map[string]any{"arguments": "{\n"}, "keep": 1, "skip": nil}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("result: %v, want %v", got, want)
	}

	// String concatenation; equal id values skipped.
	got, err = MergeDicts(
		map[string]any{"content": "hello", "id": "lc_1"},
		map[string]any{"content": " world", "id": "lc_1"},
	)
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"content": "hello world", "id": "lc_1"}) {
		t.Fatalf("result: %v", got)
	}

	// lc_-prefixed "index" strings are not concatenated.
	got, err = MergeDicts(
		map[string]any{"index": "lc_abc"},
		map[string]any{"index": "lc_def"},
	)
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"index": "lc_abc"}) {
		t.Fatalf("result: %v", got)
	}

	// Ints sum; index/created/timestamp are last-wins (_merge.py:71-79).
	got, err = MergeDicts(
		map[string]any{"tokens": 1, "index": 0},
		map[string]any{"tokens": 2, "index": 3},
	)
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"tokens": 3, "index": 3}) {
		t.Fatalf("result: %v", got)
	}

	// Equal values pass through unchanged.
	got, err = MergeDicts(map[string]any{"ok": true}, map[string]any{"ok": true})
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"ok": true}) {
		t.Fatalf("result: %v", got)
	}

	// Type mismatch raises (Python TypeError, _merge.py:39-44).
	_, err = MergeDicts(map[string]any{"key": "string"}, map[string]any{"key": 1})
	if err == nil || !strings.Contains(err.Error(), "different type") {
		t.Fatalf("expected type mismatch error, got %v", err)
	}

	// Unequal bools raise (_merge.py:80-85). DIVERGENCE (see Task 11 log):
	// in Python bool is a subclass of int, so True+False reaches the int
	// branch and sums to 1 (_merge.py:71-79) with no error; Go's strict
	// typing errors instead, which is intentional.
	_, err = MergeDicts(map[string]any{"key": true}, map[string]any{"key": false})
	if err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected unsupported type error, got %v", err)
	}
}

// Additional branch coverage for the merge helpers, all mirroring _merge.py
// semantics: nil-skip for pre-existing keys, nested slice recursion, numeric
// coercion arms, non-mergeable indexes, empty/nil IDs, and error propagation.
func TestMergeDictsBranches(t *testing.T) {
	// nil right value for a pre-existing key is skipped (_merge.py:33-36).
	got, err := MergeDicts(map[string]any{"a": 1}, map[string]any{"a": nil})
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"a": 1}) {
		t.Fatalf("result: %v", got)
	}

	// Nested slices recurse through merge_lists (_merge.py:57-62).
	got, err = MergeDicts(
		map[string]any{"chunks": []any{map[string]any{"index": 0, "text": "he"}}},
		map[string]any{"chunks": []any{map[string]any{"index": 0, "text": "llo"}}},
	)
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	want := map[string]any{"chunks": []any{map[string]any{"index": 0, "text": "hello"}}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("result: %v, want %v", got, want)
	}

	// Nested errors propagate from the recursive calls.
	if _, err = MergeDicts(
		map[string]any{"nested": map[string]any{"k": "s"}},
		map[string]any{"nested": map[string]any{"k": 1}},
	); err == nil {
		t.Fatal("expected nested type mismatch error")
	}
	if _, err = MergeDicts(
		map[string]any{"chunks": []any{map[string]any{"index": 0, "text": "a"}}},
		map[string]any{"chunks": []any{map[string]any{"index": 0, "text": 1}}},
	); err == nil {
		t.Fatal("expected nested list merge error")
	}

	// int64 existing + integral float right: coerced, then unequal non-int
	// numerics are an unsupported-type error (Python has no int64/float split).
	_, err = MergeDicts(map[string]any{"n": int64(1)}, map[string]any{"n": float64(2)})
	if err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected unsupported type error, got %v", err)
	}

	// float64 existing + int right: coerced to float64, same unsupported path.
	_, err = MergeDicts(map[string]any{"t": 1.5}, map[string]any{"t": 2})
	if err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("expected unsupported type error, got %v", err)
	}

	// Equal float64 values pass through (DeepEqual arm after coercion).
	got, err = MergeDicts(map[string]any{"t": 1.5}, map[string]any{"t": float64(1.5)})
	if err != nil {
		t.Fatalf("MergeDicts: %v", err)
	}
	if !reflect.DeepEqual(got, map[string]any{"t": 1.5}) {
		t.Fatalf("result: %v", got)
	}
}

func TestMergeListsBranches(t *testing.T) {
	// Non-mergeable indexes append: non-lc_ string index, non-integral float.
	got, err := MergeLists(
		[]any{map[string]any{"index": "plain", "v": 1}},
		[]any{map[string]any{"index": "plain", "v": 2}},
	)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("non-lc_ string index should append: %v", got)
	}
	got, err = MergeLists(
		[]any{map[string]any{"index": 1.5, "v": 1}},
		[]any{map[string]any{"index": 1.5, "v": 2}},
	)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("non-integral float index should append: %v", got)
	}

	// Empty-string and nil IDs are not "in conflict" and merge.
	got, err = MergeLists(
		[]any{map[string]any{"index": 0, "id": "", "text": "a"}},
		[]any{map[string]any{"index": 0, "id": "x", "text": "b"}},
	)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if len(got) != 1 || got[0].(map[string]any)["text"] != "ab" {
		t.Fatalf("empty existing id should merge: %v", got)
	}
	got, err = MergeLists(
		[]any{map[string]any{"index": 0, "id": nil, "text": "a"}},
		[]any{map[string]any{"index": 0, "text": "b"}},
	)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if len(got) != 1 || got[0].(map[string]any)["text"] != "ab" {
		t.Fatalf("nil existing id should merge: %v", got)
	}

	// A merge_dicts failure inside the loop propagates out of MergeLists.
	_, err = MergeLists(
		[]any{map[string]any{"index": 0, "text": "a"}},
		[]any{map[string]any{"index": 0, "text": 1}},
	)
	if err == nil {
		t.Fatal("expected merge error to propagate")
	}

	// Non-map elements in the merged list are skipped when matching indexes.
	got, err = MergeLists(
		[]any{"plain", map[string]any{"index": 0, "text": "a"}},
		[]any{map[string]any{"index": 0, "text": "b"}},
	)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if len(got) != 2 || got[1].(map[string]any)["text"] != "ab" {
		t.Fatalf("result: %v", got)
	}
}

func TestMergeListsNilIDValueMerges(t *testing.T) {
	// Both sides present "id" but the existing value is nil: no conflict.
	got, err := MergeLists(
		[]any{map[string]any{"index": 0, "id": nil, "text": "a"}},
		[]any{map[string]any{"index": 0, "id": "x", "text": "b"}},
	)
	if err != nil {
		t.Fatalf("MergeLists: %v", err)
	}
	if len(got) != 1 || got[0].(map[string]any)["text"] != "ab" {
		t.Fatalf("nil id value should merge: %v", got)
	}
}

func TestMergeDictsNonLosslessNumericMismatch(t *testing.T) {
	// int existing + non-integral float right cannot be coerced losslessly, so
	// the type check rejects it (Python's TypeError path, _merge.py:39-44).
	_, err := MergeDicts(map[string]any{"n": 1}, map[string]any{"n": 1.5})
	if err == nil || !strings.Contains(err.Error(), "different type") {
		t.Fatalf("expected type mismatch error, got %v", err)
	}
}
