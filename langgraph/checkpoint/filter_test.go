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
