package utils

import "testing"

// TestSmallAFormatting exercises StrictFormatter.Format and
// StrictFormatter.ValidateInputVariables. A single package-level test with
// subtests keeps the namespace small in a package shared with other agents.
func TestSmallAFormatting(t *testing.T) {
	var formatter StrictFormatter

	t.Run("Format", func(t *testing.T) {
		success := []struct {
			name     string
			template string
			kwargs   map[string]any
			want     string
		}{
			{
				name:     "simple substitution",
				template: "Hello, {name}!",
				kwargs:   map[string]any{"name": "World"},
				want:     "Hello, World!",
			},
			{
				name:     "multiple and repeated fields",
				template: "A {x} B {y} C {x}",
				kwargs:   map[string]any{"x": 1, "y": 2},
				want:     "A 1 B 2 C 1",
			},
			{
				name:     "escaped braces",
				template: "{{literal}} and {name}",
				kwargs:   map[string]any{"name": "World"},
				want:     "{literal} and World",
			},
			{
				name:     "triple braces",
				template: "{{{name}}}",
				kwargs:   map[string]any{"name": "X"},
				want:     "{X}",
			},
			{
				name:     "empty template",
				template: "",
				kwargs:   nil,
				want:     "",
			},
			{
				name:     "conversion repr",
				template: "{v!r}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "\"hi\"",
			},
			{
				name:     "conversion str",
				template: "{v!s}",
				kwargs:   map[string]any{"v": 42},
				want:     "42",
			},
			{
				name:     "conversion ascii",
				template: "{v!a}",
				kwargs:   map[string]any{"v": "caf" + string(rune(0xE9))},
				want:     "\"caf\\u00e9\"",
			},
			{
				name:     "spec right align",
				template: "{v:>6}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "    hi",
			},
			{
				name:     "spec left align",
				template: "{v:<6}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "hi    ",
			},
			{
				name:     "spec center",
				template: "{v:^6}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "  hi  ",
			},
			{
				name:     "spec custom fill",
				template: "{v:*>6}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "****hi",
			},
			{
				name:     "spec width numeric default right",
				template: "{v:6}",
				kwargs:   map[string]any{"v": 42},
				want:     "    42",
			},
			{
				name:     "spec width string default left",
				template: "{v:6}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "hi    ",
			},
			{
				name:     "conversion and spec combined",
				template: "{v!r:>8}",
				kwargs:   map[string]any{"v": "hi"},
				want:     "    \"hi\"",
			},
			{
				name:     "equals alignment keeps sign",
				template: "{v:=6}",
				kwargs:   map[string]any{"v": -42},
				want:     "-   42",
			},
			{
				name:     "extra kwargs are ignored",
				template: "{name}",
				kwargs:   map[string]any{"name": "X", "unused": 1},
				want:     "X",
			},
			{
				name:     "spec wider than value is unchanged",
				template: "{v:>2}",
				kwargs:   map[string]any{"v": "hello"},
				want:     "hello",
			},
		}
		for _, tc := range success {
			t.Run(tc.name, func(t *testing.T) {
				got, err := formatter.Format(tc.template, tc.kwargs)
				if err != nil {
					t.Fatalf("Format(%q) unexpected error: %v", tc.template, err)
				}
				if got != tc.want {
					t.Fatalf("Format(%q) = %q, want %q", tc.template, got, tc.want)
				}
			})
		}

		failures := []struct {
			name     string
			template string
			kwargs   map[string]any
		}{
			{"positional empty", "{}", map[string]any{}},
			{"positional index", "{0}", map[string]any{}},
			{"positional with spec", "{0:>5}", map[string]any{}},
			{"unknown variable", "{name}", map[string]any{}},
			{"missing key among others", "{a} {b}", map[string]any{"a": 1}},
			{"nil kwargs", "{name}", nil},
			{"unsupported conversion", "{v!q}", map[string]any{"v": 1}},
			{"empty conversion", "{v!}", map[string]any{"v": 1}},
			{"conversion without colon", "{v!rs}", map[string]any{"v": 1}},
			{"unsupported spec precision", "{v:.2f}", map[string]any{"v": 1}},
			{"unsupported spec type", "{v:d}", map[string]any{"v": 1}},
			{"invalid dotted name", "{a.b}", map[string]any{"a.b": 1}},
			{"invalid index name", "{a[0]}", map[string]any{"a[0]": 1}},
			{"leading digit name", "{1a}", map[string]any{"1a": 1}},
			{"unmatched brace", "{name", map[string]any{"name": 1}},
			{"single closing brace", "oops }", nil},
		}
		for _, tc := range failures {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := formatter.Format(tc.template, tc.kwargs); err == nil {
					t.Fatalf("Format(%q) expected an error, got nil", tc.template)
				}
			})
		}
	})

	t.Run("ValidateInputVariables", func(t *testing.T) {
		success := []struct {
			name      string
			template  string
			variables []string
		}{
			{"valid", "Hello {name}", []string{"name"}},
			{"multiple valid", "{a} {b}", []string{"a", "b"}},
			{"extra variables allowed", "{name}", []string{"name", "extra"}},
			{"conversion and spec placeholder", "{v!r:>8}", []string{"v"}},
			{"empty template", "", []string{"a"}},
			{"escaped braces only", "{{x}}", nil},
		}
		for _, tc := range success {
			t.Run(tc.name, func(t *testing.T) {
				if err := formatter.ValidateInputVariables(tc.template, tc.variables); err != nil {
					t.Fatalf("ValidateInputVariables(%q) unexpected error: %v", tc.template, err)
				}
			})
		}

		failures := []struct {
			name      string
			template  string
			variables []string
		}{
			{"unknown variable", "{a} {b}", []string{"a"}},
			{"no variables", "{a}", nil},
			{"positional", "{}", []string{"name"}},
			{"unsupported spec", "{v:.2f}", []string{"v"}},
			{"unsupported conversion", "{v!q}", []string{"v"}},
			{"unmatched brace", "{a", []string{"a"}},
		}
		for _, tc := range failures {
			t.Run(tc.name, func(t *testing.T) {
				if err := formatter.ValidateInputVariables(tc.template, tc.variables); err == nil {
					t.Fatalf("ValidateInputVariables(%q) expected an error, got nil", tc.template)
				}
			})
		}
	})
}
