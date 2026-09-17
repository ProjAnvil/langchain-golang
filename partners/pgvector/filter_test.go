package pgvector

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// BuildFilterClause must reject membership filters that pass structural
// validation but cannot render (empty lists), and must propagate the error
// out of the field loop.
func TestBuildFilterClause_RejectsEmptyMembership(t *testing.T) {
	for name, filter := range map[string]map[string]any{
		"empty_in_list":  {"group": map[string]any{vectorstores.FilterIn: []any{}}},
		"empty_nin_list": {"group": map[string]any{vectorstores.FilterNin: []any{}}},
	} {
		t.Run(name, func(t *testing.T) {
			args := []any{}
			if _, err := buildFilterClause(filter, &args); err == nil {
				t.Fatal("expected error for empty membership list")
			}
		})
	}
}

// Typed Go slices (for example []string) must widen to []any inside $in so
// filters written with natural literals translate.
func TestBuildFilterClause_WidensTypedSlices(t *testing.T) {
	args := []any{}
	got, err := buildFilterClause(
		map[string]any{"group": map[string]any{vectorstores.FilterIn: []string{"a", "b"}}},
		&args,
	)
	if err != nil {
		t.Fatalf("buildFilterClause: %v", err)
	}
	want := `((metadata->>'group' = $1 OR metadata->>'group' = $2))`
	if got != want {
		t.Fatalf("clause:\n got %q\nwant %q", got, want)
	}
	if len(args) != 2 || args[0] != "a" || args[1] != "b" {
		t.Fatalf("args: got %v want [a b]", args)
	}
}

// Defensive rendering branches that ValidateFilter usually blocks stay
// reachable through direct condition rendering.
func TestBuildConditionClause_RejectsInvalidOperands(t *testing.T) {
	tests := []struct {
		name      string
		condition any
	}{
		{"like_with_non_string_pattern", map[string]any{vectorstores.FilterLike: 42}},
		{"unsupported_operator", map[string]any{"$bogus": 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := []any{}
			if _, err := buildConditionClause("field", tt.condition, &args); err == nil {
				t.Fatal("expected rendering error")
			}
		})
	}
}

func TestEqualityPredicate_RejectsNonScalarOperand(t *testing.T) {
	args := []any{}
	if _, err := equalityPredicate("field", struct{}{}, "=", &args); err == nil {
		t.Fatal("expected error for non-scalar equality operand")
	}
}

func TestComparisonPredicate_RejectsInvalidInput(t *testing.T) {
	t.Run("unsupported_operator", func(t *testing.T) {
		args := []any{}
		if _, err := comparisonPredicate("field", "$bogus", 1, &args); err == nil {
			t.Fatal("expected error for unsupported comparison operator")
		}
	})
	t.Run("non_string_non_numeric_operand", func(t *testing.T) {
		args := []any{}
		if _, err := comparisonPredicate("field", vectorstores.FilterGt, struct{}{}, &args); err == nil {
			t.Fatal("expected error for non-string non-numeric comparison operand")
		}
	})
}

func TestInPredicate_RejectsInvalidInput(t *testing.T) {
	t.Run("non_list_operand", func(t *testing.T) {
		args := []any{}
		if _, err := inPredicate("field", "scalar", &args, false); err == nil {
			t.Fatal("expected error for non-list membership operand")
		}
	})
	t.Run("empty_list_operand", func(t *testing.T) {
		args := []any{}
		if _, err := inPredicate("field", []any{}, &args, false); err == nil {
			t.Fatal("expected error for empty membership list")
		}
	})
	t.Run("non_scalar_item", func(t *testing.T) {
		args := []any{}
		if _, err := inPredicate("field", []any{struct{}{}}, &args, false); err == nil {
			t.Fatal("expected error for non-scalar membership item")
		}
	})
}

func TestBetweenPredicate_RejectsInvalidBounds(t *testing.T) {
	t.Run("wrong_arity", func(t *testing.T) {
		args := []any{}
		if _, err := betweenPredicate("field", []any{1, 2, 3}, &args); err == nil {
			t.Fatal("expected error for three bounds")
		}
	})
	t.Run("invalid_low_bound", func(t *testing.T) {
		args := []any{}
		if _, err := betweenPredicate("field", []any{struct{}{}, 2}, &args); err == nil {
			t.Fatal("expected error for invalid low bound")
		}
	})
	t.Run("invalid_high_bound", func(t *testing.T) {
		args := []any{}
		if _, err := betweenPredicate("field", []any{1, struct{}{}}, &args); err == nil {
			t.Fatal("expected error for invalid high bound")
		}
	})
}

func TestNumericOperand(t *testing.T) {
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
			got, ok := numericOperand(tt.value)
			if !ok || got != tt.want {
				t.Fatalf("numericOperand(%v): got (%v, %v) want (%v, true)", tt.value, got, ok, tt.want)
			}
		})
	}
	t.Run("non_numeric", func(t *testing.T) {
		if _, ok := numericOperand(struct{}{}); ok {
			t.Fatal("struct must not be numeric")
		}
		if _, ok := numericOperand("7"); ok {
			t.Fatal("string must not be numeric")
		}
	})
}

func TestToAnySlice(t *testing.T) {
	t.Run("any_slice_passthrough", func(t *testing.T) {
		items := []any{"a", 1}
		got, ok := toAnySlice(items)
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
	t.Run("typed_array_widens", func(t *testing.T) {
		got, ok := toAnySlice([2]int{3, 4})
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
