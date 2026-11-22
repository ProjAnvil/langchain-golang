package utils

import (
	"reflect"
	"testing"
)

// TestSmallAUsage exercises DictIntOp. A single package-level test with
// subtests keeps the namespace small in a package shared with other agents.
func TestSmallAUsage(t *testing.T) {
	add := func(a, b int) int { return a + b }
	sub := func(a, b int) int { return a - b }
	mul := func(a, b int) int { return a * b }

	// smallABuildNested wraps {"x": 1} under "n" the given number of times.
	smallABuildNested := func(levels int) map[string]any {
		var m = map[string]any{"x": 1}
		for i := 0; i < levels; i++ {
			m = map[string]any{"n": m}
		}
		return m
	}

	t.Run("success", func(t *testing.T) {
		cases := []struct {
			name     string
			left     map[string]any
			right    map[string]any
			op       func(int, int) int
			maxDepth int
			want     map[string]any
		}{
			{
				name:     "add flat maps",
				left:     map[string]any{"a": 1, "b": 2},
				right:    map[string]any{"a": 10, "b": 20},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{"a": 11, "b": 22},
			},
			{
				name:     "subtract flat maps",
				left:     map[string]any{"a": 5},
				right:    map[string]any{"a": 3},
				op:       sub,
				maxDepth: 10,
				want:     map[string]any{"a": 2},
			},
			{
				name:     "custom multiply op",
				left:     map[string]any{"a": 3},
				right:    map[string]any{"a": 4},
				op:       mul,
				maxDepth: 10,
				want:     map[string]any{"a": 12},
			},
			{
				name:     "missing keys use zero default",
				left:     map[string]any{"a": 1},
				right:    map[string]any{"b": 2},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{"a": 1, "b": 2},
			},
			{
				name:     "missing key with subtraction",
				left:     map[string]any{"a": 5},
				right:    map[string]any{},
				op:       sub,
				maxDepth: 10,
				want:     map[string]any{"a": 5},
			},
			{
				name:     "additional integer types",
				left:     map[string]any{"a": int64(5)},
				right:    map[string]any{"a": int32(2)},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{"a": 7},
			},
			{
				name:     "nested maps",
				left:     map[string]any{"a": map[string]any{"x": 1, "y": 2}},
				right:    map[string]any{"a": map[string]any{"x": 10}},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{"a": map[string]any{"x": 11, "y": 2}},
			},
			{
				name:     "nested map missing entirely on right",
				left:     map[string]any{"a": map[string]any{"x": 5}},
				right:    map[string]any{},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{"a": map[string]any{"x": 5}},
			},
			{
				name:     "nested map missing entirely on left",
				left:     map[string]any{},
				right:    map[string]any{"a": map[string]any{"x": 5}},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{"a": map[string]any{"x": 5}},
			},
			{
				name:     "empty maps",
				left:     map[string]any{},
				right:    map[string]any{},
				op:       add,
				maxDepth: 10,
				want:     map[string]any{},
			},
			{
				name:     "nil maps",
				left:     nil,
				right:    nil,
				op:       add,
				maxDepth: 10,
				want:     map[string]any{},
			},
			{
				name:     "flat map with maxDepth 1",
				left:     map[string]any{"a": 1},
				right:    map[string]any{"a": 2},
				op:       add,
				maxDepth: 1,
				want:     map[string]any{"a": 3},
			},
			{
				name:     "one nested level with maxDepth 2",
				left:     smallABuildNested(1),
				right:    smallABuildNested(1),
				op:       add,
				maxDepth: 2,
				want:     map[string]any{"n": map[string]any{"x": 2}},
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got, err := DictIntOp(tc.left, tc.right, tc.op, tc.maxDepth)
				if err != nil {
					t.Fatalf("DictIntOp unexpected error: %v", err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("DictIntOp = %#v, want %#v", got, tc.want)
				}
			})
		}
	})

	t.Run("does not mutate inputs", func(t *testing.T) {
		left := map[string]any{"a": 1}
		right := map[string]any{"a": 2}
		if _, err := DictIntOp(left, right, add, 10); err != nil {
			t.Fatalf("DictIntOp unexpected error: %v", err)
		}
		if !reflect.DeepEqual(left, map[string]any{"a": 1}) || !reflect.DeepEqual(right, map[string]any{"a": 2}) {
			t.Fatalf("DictIntOp mutated its inputs: left=%#v right=%#v", left, right)
		}
	})

	t.Run("errors", func(t *testing.T) {
		cases := []struct {
			name     string
			left     map[string]any
			right    map[string]any
			op       func(int, int) int
			maxDepth int
		}{
			{
				name:     "nil op",
				left:     map[string]any{"a": 1},
				right:    map[string]any{"a": 2},
				op:       nil,
				maxDepth: 10,
			},
			{
				name:     "maxDepth zero",
				left:     map[string]any{"a": 1},
				right:    map[string]any{"a": 2},
				op:       add,
				maxDepth: 0,
			},
			{
				name:     "negative maxDepth",
				left:     map[string]any{"a": 1},
				right:    map[string]any{"a": 2},
				op:       add,
				maxDepth: -1,
			},
			{
				name:     "nested exceeds maxDepth 1",
				left:     smallABuildNested(1),
				right:    smallABuildNested(1),
				op:       add,
				maxDepth: 1,
			},
			{
				name:     "two nested levels exceed maxDepth 2",
				left:     smallABuildNested(2),
				right:    smallABuildNested(2),
				op:       add,
				maxDepth: 2,
			},
			{
				name:     "string value type mismatch",
				left:     map[string]any{"a": "text"},
				right:    map[string]any{"a": 1},
				op:       add,
				maxDepth: 10,
			},
			{
				name:     "float value type mismatch",
				left:     map[string]any{"a": 1.5},
				right:    map[string]any{"a": 2},
				op:       add,
				maxDepth: 10,
			},
			{
				name:     "bool value type mismatch",
				left:     map[string]any{"a": true},
				right:    map[string]any{"a": 2},
				op:       add,
				maxDepth: 10,
			},
			{
				name:     "nil value type mismatch",
				left:     map[string]any{"a": nil},
				right:    map[string]any{"a": 2},
				op:       add,
				maxDepth: 10,
			},
			{
				name:     "int versus dict mismatch",
				left:     map[string]any{"a": 1},
				right:    map[string]any{"a": map[string]any{"x": 1}},
				op:       add,
				maxDepth: 10,
			},
			{
				name:     "dict versus int mismatch",
				left:     map[string]any{"a": map[string]any{"x": 1}},
				right:    map[string]any{"a": 1},
				op:       add,
				maxDepth: 10,
			},
			{
				name:     "nested type mismatch",
				left:     map[string]any{"a": map[string]any{"x": "text"}},
				right:    map[string]any{"a": map[string]any{"x": 1}},
				op:       add,
				maxDepth: 10,
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := DictIntOp(tc.left, tc.right, tc.op, tc.maxDepth); err == nil {
					t.Fatalf("DictIntOp expected an error, got nil")
				}
			})
		}
	})
}
