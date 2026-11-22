package utils

import "testing"

// TestSmallAStrings exercises StringifyDict, CommaList, and
// SanitizeForPostgres. A single package-level test with subtests keeps the
// namespace small in a package shared with other agents.
func TestSmallAStrings(t *testing.T) {
	t.Run("StringifyDict", func(t *testing.T) {
		cases := []struct {
			name string
			data map[string]any
			want string
		}{
			{
				name: "sorted keys",
				data: map[string]any{"b": 2, "a": 1},
				want: "a: 1\nb: 2\n",
			},
			{
				name: "string values",
				data: map[string]any{"name": "World", "greeting": "Hello"},
				want: "greeting: Hello\nname: World\n",
			},
			{
				name: "empty map",
				data: map[string]any{},
				want: "",
			},
			{
				name: "nil map",
				data: nil,
				want: "",
			},
			{
				name: "nested map",
				data: map[string]any{"outer": map[string]any{"x": 1}},
				want: "outer: \nx: 1\n\n",
			},
			{
				name: "slice of any",
				data: map[string]any{"items": []any{"a", "b"}},
				want: "items: a\nb\n",
			},
			{
				name: "slice of strings",
				data: map[string]any{"tags": []string{"x", "y"}},
				want: "tags: x\ny\n",
			},
			{
				name: "numeric value",
				data: map[string]any{"count": 42},
				want: "count: 42\n",
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if got := StringifyDict(tc.data); got != tc.want {
					t.Fatalf("StringifyDict = %q, want %q", got, tc.want)
				}
			})
		}
	})

	t.Run("CommaList", func(t *testing.T) {
		cases := []struct {
			name  string
			items []any
			want  string
		}{
			{"strings", []any{"a", "b", "c"}, "a, b, c"},
			{"numbers", []any{1, 2, 3}, "1, 2, 3"},
			{"mixed", []any{"x", 2, true}, "x, 2, true"},
			{"single", []any{"only"}, "only"},
			{"empty", []any{}, ""},
			{"nil", nil, ""},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if got := CommaList(tc.items); got != tc.want {
					t.Fatalf("CommaList = %q, want %q", got, tc.want)
				}
			})
		}
	})

	t.Run("SanitizeForPostgres", func(t *testing.T) {
		cases := []struct {
			name string
			text string
			want string
		}{
			{"unchanged", "hello world", "hello world"},
			{"embedded nul", "a\x00b", "ab"},
			{"only nul", "\x00", ""},
			{"multiple nul", "\x00\x00", ""},
			{"nul at start", "\x00end", "end"},
			{"nul at end", "start\x00", "start"},
			{"interleaved nul", "a\x00b\x00c", "abc"},
			{"empty", "", ""},
			{"other control bytes preserved", "a\x01b\x00c", "a\x01bc"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				if got := SanitizeForPostgres(tc.text); got != tc.want {
					t.Fatalf("SanitizeForPostgres(%q) = %q, want %q", tc.text, got, tc.want)
				}
			})
		}
	})
}
