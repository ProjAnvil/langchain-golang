package vectorstores

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func docWith(metadata map[string]any) documents.Document {
	return documents.New("content", metadata)
}

func TestMatchFilter(t *testing.T) {
	tests := []struct {
		name   string
		filter map[string]any
		metal  map[string]any
		want   bool
	}{
		// Top-level behavior.
		{"nil filter matches everything", nil, map[string]any{"group": "a"}, true},
		{"empty filter matches everything", map[string]any{}, map[string]any{"group": "a"}, true},
		{
			"multiple fields are ANDed",
			map[string]any{"group": "a", "page": map[string]any{FilterGt: 1}},
			map[string]any{"group": "a", "page": 2},
			true,
		},
		{
			"multiple fields one miss",
			map[string]any{"group": "a", "page": map[string]any{FilterGt: 5}},
			map[string]any{"group": "a", "page": 2},
			false,
		},

		// $eq (shorthand and explicit).
		{"eq shorthand match", map[string]any{"group": "a"}, map[string]any{"group": "a"}, true},
		{"eq shorthand miss", map[string]any{"group": "a"}, map[string]any{"group": "b"}, false},
		{"eq explicit match", map[string]any{"group": map[string]any{FilterEq: "a"}}, map[string]any{"group": "a"}, true},
		{"eq missing field", map[string]any{"group": "a"}, nil, false},
		{"eq nil metadata", map[string]any{"group": "a"}, map[string]any{}, false},
		{"eq bool value", map[string]any{"ok": true}, map[string]any{"ok": true}, true},
		{"eq bool miss", map[string]any{"ok": true}, map[string]any{"ok": false}, false},
		{"eq int vs float64", map[string]any{"page": 5}, map[string]any{"page": float64(5)}, true},
		{"eq int64 vs int", map[string]any{"page": int64(5)}, map[string]any{"page": 5}, true},
		{"eq float mismatch", map[string]any{"page": 5.5}, map[string]any{"page": 5}, false},
		{"eq type mismatch", map[string]any{"page": 5}, map[string]any{"page": "5"}, false},

		// $ne: langchain-postgres parity — a missing field never matches $ne
		// (jsonb_path_match yields NULL for absent keys).
		{"ne different value", map[string]any{"group": map[string]any{FilterNe: "a"}}, map[string]any{"group": "b"}, true},
		{"ne same value", map[string]any{"group": map[string]any{FilterNe: "a"}}, map[string]any{"group": "a"}, false},
		{"ne missing field", map[string]any{"group": map[string]any{FilterNe: "a"}}, nil, false},
		{"ne numeric normalization", map[string]any{"page": map[string]any{FilterNe: 5}}, map[string]any{"page": float64(5)}, false},

		// Ordered comparisons.
		{"gt numeric", map[string]any{"page": map[string]any{FilterGt: 2}}, map[string]any{"page": 3}, true},
		{"gt equal boundary", map[string]any{"page": map[string]any{FilterGt: 2}}, map[string]any{"page": 2}, false},
		{"gte boundary", map[string]any{"page": map[string]any{FilterGte: 2}}, map[string]any{"page": 2}, true},
		{"lt numeric", map[string]any{"page": map[string]any{FilterLt: 2}}, map[string]any{"page": 1}, true},
		{"lte boundary", map[string]any{"page": map[string]any{FilterLte: 2}}, map[string]any{"page": 2}, true},
		{"gt missing field", map[string]any{"page": map[string]any{FilterGt: 0}}, nil, false},
		{"gt string compare", map[string]any{"name": map[string]any{FilterGt: "apple"}}, map[string]any{"name": "banana"}, true},
		{"gt string boundary", map[string]any{"name": map[string]any{FilterGt: "apple"}}, map[string]any{"name": "apple"}, false},
		{"gt type mismatch", map[string]any{"page": map[string]any{FilterGt: 2}}, map[string]any{"page": "3"}, false},
		{"gt int vs float64", map[string]any{"page": map[string]any{FilterGt: 2.5}}, map[string]any{"page": 3}, true},

		// $in / $nin.
		{"in match", map[string]any{"group": map[string]any{FilterIn: []any{"a", "b"}}}, map[string]any{"group": "b"}, true},
		{"in miss", map[string]any{"group": map[string]any{FilterIn: []any{"a", "b"}}}, map[string]any{"group": "c"}, false},
		{"in missing field", map[string]any{"group": map[string]any{FilterIn: []any{"a"}}}, nil, false},
		{"in empty list", map[string]any{"group": map[string]any{FilterIn: []any{}}}, map[string]any{"group": "a"}, false},
		{"in numeric normalization", map[string]any{"page": map[string]any{FilterIn: []any{1, 2, 3}}}, map[string]any{"page": 2.0}, true},
		// Bool elements are rejected by ValidateFilter (langchain-postgres
		// parity), so they never contribute a match either.
		{"in bool element never matches", map[string]any{"ok": map[string]any{FilterIn: []any{true}}}, map[string]any{"ok": true}, false},
		{"nin present not in", map[string]any{"group": map[string]any{FilterNin: []any{"a", "b"}}}, map[string]any{"group": "c"}, true},
		{"nin present in", map[string]any{"group": map[string]any{FilterNin: []any{"a", "b"}}}, map[string]any{"group": "a"}, false},
		{"nin missing field", map[string]any{"group": map[string]any{FilterNin: []any{"a"}}}, nil, false},
		{"nin empty list", map[string]any{"group": map[string]any{FilterNin: []any{}}}, map[string]any{"group": "a"}, true},

		// $between: inclusive bounds (pgvector: >= low AND <= high).
		{"between low bound", map[string]any{"page": map[string]any{FilterBetween: []any{1, 5}}}, map[string]any{"page": 1}, true},
		{"between high bound", map[string]any{"page": map[string]any{FilterBetween: []any{1, 5}}}, map[string]any{"page": 5}, true},
		{"between middle", map[string]any{"page": map[string]any{FilterBetween: []any{1, 5}}}, map[string]any{"page": 3}, true},
		{"between below", map[string]any{"page": map[string]any{FilterBetween: []any{1, 5}}}, map[string]any{"page": 0}, false},
		{"between above", map[string]any{"page": map[string]any{FilterBetween: []any{1, 5}}}, map[string]any{"page": 6}, false},
		{"between missing field", map[string]any{"page": map[string]any{FilterBetween: []any{1, 5}}}, nil, false},
		{"between strings", map[string]any{"name": map[string]any{FilterBetween: []any{"a", "c"}}}, map[string]any{"name": "b"}, true},
		{"between string outside", map[string]any{"name": map[string]any{FilterBetween: []any{"a", "b"}}}, map[string]any{"name": "d"}, false},

		// $exists.
		{"exists true present", map[string]any{"group": map[string]any{FilterExists: true}}, map[string]any{"group": "a"}, true},
		{"exists true missing", map[string]any{"group": map[string]any{FilterExists: true}}, nil, false},
		{"exists false missing", map[string]any{"group": map[string]any{FilterExists: false}}, nil, true},
		{"exists false present", map[string]any{"group": map[string]any{FilterExists: false}}, map[string]any{"group": "a"}, false},
		{"exists true nil value", map[string]any{"group": map[string]any{FilterExists: true}}, map[string]any{"group": nil}, true},

		// $like: SQL LIKE wildcards, case-sensitive.
		{"like exact", map[string]any{"name": map[string]any{FilterLike: "alpha"}}, map[string]any{"name": "alpha"}, true},
		{"like prefix wildcard", map[string]any{"name": map[string]any{FilterLike: "%pha"}}, map[string]any{"name": "alpha"}, true},
		{"like suffix wildcard", map[string]any{"name": map[string]any{FilterLike: "al%"}}, map[string]any{"name": "alpha"}, true},
		{"like inner wildcard", map[string]any{"name": map[string]any{FilterLike: "%lp%"}}, map[string]any{"name": "alpha"}, true},
		{"like single char", map[string]any{"name": map[string]any{FilterLike: "a_c"}}, map[string]any{"name": "abc"}, true},
		{"like single char too long", map[string]any{"name": map[string]any{FilterLike: "a_c"}}, map[string]any{"name": "abcc"}, false},
		{"like case sensitive", map[string]any{"name": map[string]any{FilterLike: "%Alp%"}}, map[string]any{"name": "alpha"}, false},
		{"like miss", map[string]any{"name": map[string]any{FilterLike: "zeta"}}, map[string]any{"name": "alpha"}, false},
		{"like missing field", map[string]any{"name": map[string]any{FilterLike: "%a%"}}, nil, false},
		{"like non-string metadata", map[string]any{"page": map[string]any{FilterLike: "%1%"}}, map[string]any{"page": 1}, false},

		// Invalid filters evaluate to false rather than panicking; use
		// ValidateFilter for explicit error reporting.
		{"unknown operator", map[string]any{"group": map[string]any{"$bogus": "a"}}, map[string]any{"group": "a"}, false},
		{"operator as field", map[string]any{"$and": "x"}, map[string]any{"$and": "x"}, false},
		{"eq list value", map[string]any{"group": []any{"a"}}, map[string]any{"group": "a"}, false},
		{"empty condition map", map[string]any{"page": map[string]any{}}, map[string]any{"page": 3}, false},
		{
			"condition with multiple operators",
			map[string]any{"page": map[string]any{FilterGt: 1, FilterLt: 5}},
			map[string]any{"page": 3},
			false,
		},
		{"in not a list", map[string]any{"group": map[string]any{FilterIn: "a"}}, map[string]any{"group": "a"}, false},
		{"nin not a list", map[string]any{"group": map[string]any{FilterNin: "a"}}, map[string]any{"group": "a"}, false},
		{
			// Bool elements are rejected by ValidateFilter, so they never
			// contribute to $nin either (the field is "not in" the empty set).
			"nin bool element ignored",
			map[string]any{"ok": map[string]any{FilterNin: []any{true}}},
			map[string]any{"ok": true},
			true,
		},
		{"between wrong arity", map[string]any{"page": map[string]any{FilterBetween: []any{1}}}, map[string]any{"page": 1}, false},
		{
			"between low bound type mismatch",
			map[string]any{"page": map[string]any{FilterBetween: []any{"a", 5}}},
			map[string]any{"page": 3},
			false,
		},
		{
			"between high bound type mismatch",
			map[string]any{"page": map[string]any{FilterBetween: []any{1, "c"}}},
			map[string]any{"page": 3},
			false,
		},
		{"exists non-bool operand", map[string]any{"group": map[string]any{FilterExists: "yes"}}, map[string]any{"group": 1}, false},
		{"like non-string operand", map[string]any{"name": map[string]any{FilterLike: 42}}, map[string]any{"name": "x"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MatchFilter(docWith(tt.metal), tt.filter); got != tt.want {
				t.Fatalf("MatchFilter: got %v want %v", got, tt.want)
			}
		})
	}
}

func TestValidateFilterValid(t *testing.T) {
	valid := []map[string]any{
		nil,
		{},
		{"group": "a"},
		{"group": nil},
		{"group": true},
		{"page": 3},
		{"group": map[string]any{FilterEq: "a"}},
		{"group": map[string]any{FilterNe: "a"}},
		{"page": map[string]any{FilterGt: 1}},
		{"page": map[string]any{FilterGte: int64(1)}},
		{"page": map[string]any{FilterLt: 1.5}},
		{"page": map[string]any{FilterLte: float32(2)}},
		{"name": map[string]any{FilterGt: "apple"}},
		{"group": map[string]any{FilterIn: []any{"a", "b"}}},
		{"page": map[string]any{FilterIn: []any{1, 2.5, "3"}}},
		{"group": map[string]any{FilterNin: []any{}}},
		{"page": map[string]any{FilterBetween: []any{1, 5}}},
		{"page": map[string]any{FilterBetween: []any{1.5, 5.5}}},
		{"name": map[string]any{FilterBetween: []any{"a", "c"}}},
		{"group": map[string]any{FilterExists: true}},
		{"group": map[string]any{FilterExists: false}},
		{"name": map[string]any{FilterLike: "%alpha%"}},
		{"group": "a", "page": map[string]any{FilterBetween: []any{1, 5}}},
	}
	for _, filter := range valid {
		if err := ValidateFilter(filter); err != nil {
			t.Fatalf("ValidateFilter(%#v): unexpected error %v", filter, err)
		}
	}
}

func TestValidateFilterErrors(t *testing.T) {
	tests := []struct {
		name   string
		filter map[string]any
	}{
		{"empty field name", map[string]any{"": "a"}},
		{"operator as field", map[string]any{"$and": []any{}}},
		{"condition with no operators", map[string]any{"group": map[string]any{}}},
		{
			"condition with multiple operators",
			map[string]any{"group": map[string]any{FilterGte: 1, FilterLte: 5}},
		},
		{"unknown operator", map[string]any{"group": map[string]any{"$bogus": "a"}}},
		{"eq list value", map[string]any{"group": []any{"a"}}},
		{"eq map value", map[string]any{"group": map[string]any{"nested": "a"}}},
		{"eq explicit list operand", map[string]any{"group": map[string]any{FilterEq: []any{"a"}}}},
		{"ne explicit list operand", map[string]any{"group": map[string]any{FilterNe: []any{"a"}}}},
		{"between not a list", map[string]any{"page": map[string]any{FilterBetween: "scalar"}}},
		{"gt bool value", map[string]any{"page": map[string]any{FilterGt: true}}},
		{"gt nil value", map[string]any{"page": map[string]any{FilterGt: nil}}},
		{"gt list value", map[string]any{"page": map[string]any{FilterGt: []any{1}}}},
		{"in not a list", map[string]any{"group": map[string]any{FilterIn: "a"}}},
		{"in bool element", map[string]any{"ok": map[string]any{FilterIn: []any{true}}}},
		{"in nil element", map[string]any{"group": map[string]any{FilterIn: []any{nil}}}},
		{"in map element", map[string]any{"group": map[string]any{FilterIn: []any{map[string]any{"a": 1}}}}},
		{"nin not a list", map[string]any{"group": map[string]any{FilterNin: "a"}}},
		{"between too few bounds", map[string]any{"page": map[string]any{FilterBetween: []any{1}}}},
		{"between too many bounds", map[string]any{"page": map[string]any{FilterBetween: []any{1, 2, 3}}}},
		{"between non-numeric bounds", map[string]any{"page": map[string]any{FilterBetween: []any{"a", 5}}}},
		{"between bool bound", map[string]any{"page": map[string]any{FilterBetween: []any{true, false}}}},
		{"between mixed string and bool bounds", map[string]any{"page": map[string]any{FilterBetween: []any{"a", true}}}},
		{"exists non-bool", map[string]any{"group": map[string]any{FilterExists: "yes"}}},
		{"like non-string", map[string]any{"name": map[string]any{FilterLike: 42}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateFilter(tt.filter); err == nil {
				t.Fatalf("ValidateFilter(%#v): expected error", tt.filter)
			}
		})
	}
}

func TestValidateFilterErrorMessageMentionsOperator(t *testing.T) {
	err := ValidateFilter(map[string]any{"group": map[string]any{"$bogus": "a"}})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "$bogus") {
		t.Fatalf("error should mention the invalid operator: %v", err)
	}
}

// matchOperator's default arm is unreachable through MatchFilter (matchCondition
// rejects unknown operators first), so it is exercised directly.
func TestMatchOperator_UnknownOperator(t *testing.T) {
	if matchOperator("$bogus", "a", "a", true) {
		t.Fatal("unknown operator must never match")
	}
}

// switchOrder's default arm is unreachable through the comparison operators,
// so it is exercised directly.
func TestSwitchOrder_UnknownOperator(t *testing.T) {
	if switchOrder("$bogus", 1) {
		t.Fatal("unknown comparison operator must never match")
	}
}

func TestNumericValue(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  float64
	}{
		{"int", 7, 7},
		{"int8", int8(7), 7},
		{"int16", int16(7), 7},
		{"int32", int32(7), 7},
		{"int64", int64(7), 7},
		{"uint", uint(7), 7},
		{"uint8", uint8(7), 7},
		{"uint16", uint16(7), 7},
		{"uint32", uint32(7), 7},
		{"uint64", uint64(7), 7},
		{"float32", float32(1.5), 1.5},
		{"float64", 1.5, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := numericValue(tt.value)
			if !ok || got != tt.want {
				t.Fatalf("numericValue(%v): got (%v, %v) want (%v, true)", tt.value, got, ok, tt.want)
			}
		})
	}
	t.Run("non_numeric", func(t *testing.T) {
		if _, ok := numericValue("7"); ok {
			t.Fatal("string must not be numeric")
		}
		if _, ok := numericValue(true); ok {
			t.Fatal("bool must not be numeric")
		}
		if _, ok := numericValue(nil); ok {
			t.Fatal("nil must not be numeric")
		}
	})
}

func TestToAnySlice(t *testing.T) {
	t.Run("any_slice_passthrough", func(t *testing.T) {
		got, ok := toAnySlice([]any{"a", 1})
		if !ok || len(got) != 2 || got[0] != "a" || got[1] != 1 {
			t.Fatalf("got %v ok %v", got, ok)
		}
	})
	t.Run("typed_string_slice_widens", func(t *testing.T) {
		got, ok := toAnySlice([]string{"a", "b"})
		if !ok || len(got) != 2 || got[0] != "a" || got[1] != "b" {
			t.Fatalf("got %v ok %v", got, ok)
		}
	})
	t.Run("typed_int_slice_widens", func(t *testing.T) {
		got, ok := toAnySlice([]int{3, 4})
		if !ok || len(got) != 2 || got[0] != 3 || got[1] != 4 {
			t.Fatalf("got %v ok %v", got, ok)
		}
	})
	t.Run("non_slice_rejected", func(t *testing.T) {
		if got, ok := toAnySlice("scalar"); ok {
			t.Fatalf("scalar must not widen, got %v", got)
		}
		if got, ok := toAnySlice(nil); ok {
			t.Fatalf("nil must not widen, got %v", got)
		}
	})
}
