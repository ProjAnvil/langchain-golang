package graph

import (
	"fmt"
	"hash/fnv"
	"testing"
)

func TestFnTaskIDDeterministic(t *testing.T) {
	a := FnTaskID("cp", "", 3, "mapper", "a@0", 1)
	b := FnTaskID("cp", "", 3, "mapper", "a@0", 1)
	if a != b {
		t.Fatalf("same inputs must hash identically: %q vs %q", a, b)
	}
}

func TestFnTaskIDFieldSensitivity(t *testing.T) {
	base := FnTaskID("cp", "ns", 3, "mapper", "a@0", 1)
	variants := map[string]string{
		"cpID":       FnTaskID("cp2", "ns", 3, "mapper", "a@0", 1),
		"step":       FnTaskID("cp", "ns", 4, "mapper", "a@0", 1),
		"name":       FnTaskID("cp", "ns", 3, "reducer", "a@0", 1),
		"parentPath": FnTaskID("cp", "ns", 3, "mapper", "a@1", 1),
		"callIdx":    FnTaskID("cp", "ns", 3, "mapper", "a@0", 2),
	}
	seen := map[string]string{"base": base}
	for field, id := range variants {
		if id == base {
			t.Errorf("perturbing %s did not change the ID: %q", field, id)
		}
		if prev, dup := seen[id]; dup {
			t.Errorf("%s collides with %s: %q", field, prev, id)
		}
		seen[id] = field
	}
}

func TestFnTaskIDFormat(t *testing.T) {
	id := FnTaskID("cp", "", 0, "mapper", "", 0)
	if len(id) != 16 {
		t.Fatalf("want 16 hex chars, got %d (%q)", len(id), id)
	}
	for _, c := range id {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			t.Fatalf("non-lowercase-hex char %q in %q", c, id)
		}
	}
}

func TestFnTaskIDMatchesManualHash(t *testing.T) {
	h := fnv.New64a()
	fmt.Fprintf(h, "%s\x00%s\x00%d\x00%s\x00%s\x00%d\x00", "cp", "", 0, "mapper", "", 0)
	want := fmt.Sprintf("%016x", h.Sum64())
	if got := FnTaskID("cp", "", 0, "mapper", "", 0); got != want {
		t.Fatalf("FnTaskID = %q, manual fnv-1a recompute = %q", got, want)
	}
}
