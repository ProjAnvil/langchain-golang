package checkpoint

import "testing"

// TestMetadataMatchesFilter pins the @>-style containment semantics of
// MetadataMatchesFilter: JSON-normalized numeric equality, recursive object
// containment (a subset filter matches), and closed source/step/parents keys.
func TestMetadataMatchesFilter(t *testing.T) {
	md := Metadata{
		Source:  "loop",
		Step:    1,
		Parents: map[string]string{"": "p1", "sub": "p2"},
	}
	cases := []struct {
		name   string
		filter map[string]any
		want   bool
	}{
		{"nil filter matches", nil, true},
		{"empty filter matches", map[string]any{}, true},
		{"source equal", map[string]any{"source": "loop"}, true},
		{"step equal as int", map[string]any{"step": 1}, true},
		{"step equal as float64", map[string]any{"step": 1.0}, true},
		{"parents subset contained", map[string]any{"parents": map[string]string{"": "p1"}}, true},
		{"parents exact", map[string]any{"parents": map[string]string{"": "p1", "sub": "p2"}}, true},
		{"source mismatch", map[string]any{"source": "input"}, false},
		{"step mismatch", map[string]any{"step": 2}, false},
		{"missing key", map[string]any{"missing": "x"}, false},
		{"parents value mismatch", map[string]any{"parents": map[string]string{"": "other"}}, false},
		{"parents extra key absent from doc", map[string]any{"parents": map[string]string{"extra": "x"}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := MetadataMatchesFilter(md, tc.filter); got != tc.want {
				t.Fatalf("MetadataMatchesFilter(%+v, %v) = %v, want %v", md, tc.filter, got, tc.want)
			}
		})
	}
}

// TestMetadataMatchesFilterUnmarshalableFilter verifies a filter value that
// json.Marshal rejects (here a func) fails to match rather than panicking:
// normalizeJSON falls back to the raw value, and the func compares unequal
// to any JSON scalar in the metadata document.
func TestMetadataMatchesFilterUnmarshalableFilter(t *testing.T) {
	md := Metadata{Source: "loop", Step: 1}
	if MetadataMatchesFilter(md, map[string]any{"step": func() {}}) {
		t.Fatalf("MetadataMatchesFilter with func filter value = true, want false")
	}
}

// TestJSONContains pins the recursive @>-style containment cases that
// Metadata's closed source/step/parents shape cannot reach through
// MetadataMatchesFilter: array filters (element-wise containment against an
// array target) and type-mismatched nesting (a map filter against a scalar
// target, an array filter against a map target).
func TestJSONContains(t *testing.T) {
	cases := []struct {
		name   string
		doc    any
		filter any
		want   bool
	}{
		{"map filter against scalar target", "x", map[string]any{"k": "v"}, false},
		{"array filter against map target", map[string]any{"k": "v"}, []any{"k"}, false},
		{"array filter against scalar target", "x", []any{"x"}, false},
		{"array element contained", []any{"a", "b", "c"}, []any{"b"}, true},
		{"array all elements contained", []any{"a", "b", "c"}, []any{"c", "a"}, true},
		{"array element missing", []any{"a", "b"}, []any{"z"}, false},
		{"array against empty target", []any{}, []any{"a"}, false},
		{"empty array filter matches any array", []any{"a"}, []any{}, true},
		{"object element contained in array element",
			[]any{map[string]any{"k": "v", "extra": 1}},
			[]any{map[string]any{"k": "v"}}, true},
		{"object element not contained in any array element",
			[]any{map[string]any{"k": "other"}},
			[]any{map[string]any{"k": "v"}}, false},
		{"nested array containment",
			map[string]any{"tags": []any{"x", "y"}},
			map[string]any{"tags": []any{"y"}}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := jsonContains(tc.doc, tc.filter); got != tc.want {
				t.Fatalf("jsonContains(%v, %v) = %v, want %v", tc.doc, tc.filter, got, tc.want)
			}
		})
	}
}

// TestNormalizeJSON verifies the JSON round-trip normalization: numbers
// become float64, maps/slices become map[string]any/[]any, and a value
// json.Marshal rejects is returned unchanged.
func TestNormalizeJSON(t *testing.T) {
	t.Run("numbers and containers normalized", func(t *testing.T) {
		in := map[string]any{"n": 1, "m": map[string]int{"k": 2}, "s": []int{3}}
		out, ok := normalizeJSON(in).(map[string]any)
		if !ok {
			t.Fatalf("normalizeJSON(%v) did not return a map", in)
		}
		if out["n"] != 1.0 {
			t.Fatalf("normalized number = %#v, want float64(1)", out["n"])
		}
		if m, ok := out["m"].(map[string]any); !ok || m["k"] != 2.0 {
			t.Fatalf("normalized nested map = %#v, want map[string]any{k: float64(2)}", out["m"])
		}
		if s, ok := out["s"].([]any); !ok || s[0] != 3.0 {
			t.Fatalf("normalized slice = %#v, want []any{float64(3)}", out["s"])
		}
	})

	t.Run("unmarshalable value returned unchanged", func(t *testing.T) {
		in := map[string]any{"n": 1, "bad": func() {}}
		out, ok := normalizeJSON(in).(map[string]any)
		if !ok {
			t.Fatalf("normalizeJSON(%v) did not return the input map", in)
		}
		// No normalization happened: the number stays an int.
		if n, isInt := out["n"].(int); !isInt || n != 1 {
			t.Fatalf("unmarshalable input was normalized anyway: n = %#v, want int(1)", out["n"])
		}
		if _, isFunc := out["bad"].(func()); !isFunc {
			t.Fatalf("unmarshalable input lost its func value: %#v", out["bad"])
		}
	})
}
