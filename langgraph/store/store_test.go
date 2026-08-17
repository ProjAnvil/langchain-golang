package store

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TestPutRejectsInvalidNamespaces: Put must fail with ErrInvalidNamespace for
// an empty namespace, a namespace containing an empty label, a label with a
// period, and the reserved "langgraph" root label — mirroring Python's
// _validate_namespace.
func TestPutRejectsInvalidNamespaces(t *testing.T) {
	s := NewInMemoryStore()
	cases := []struct {
		name      string
		namespace []string
	}{
		{"empty namespace", nil},
		{"empty namespace slice", []string{}},
		{"empty label", []string{"a", ""}},
		{"label with period", []string{"a.b"}},
		{"reserved langgraph root", []string{"langgraph", "sub"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := s.Put(context.Background(), tc.namespace, "k", map[string]any{}, nil)
			if !errors.Is(err, ErrInvalidNamespace) {
				t.Fatalf("Put(%v) error = %v, want ErrInvalidNamespace", tc.namespace, err)
			}
		})
	}
}

// TestPutAcceptsLanggraphNonRoot: "langgraph" is only reserved as the root
// label; it is valid deeper in the path.
func TestPutAcceptsLanggraphNonRoot(t *testing.T) {
	s := NewInMemoryStore()
	if err := s.Put(context.Background(), []string{"a", "langgraph"}, "k", map[string]any{}, nil); err != nil {
		t.Fatalf("Put with non-root langgraph label: %v", err)
	}
}

// TestNamespaceHasPrefix exercises the element-wise prefix matcher directly.
func TestNamespaceHasPrefix(t *testing.T) {
	cases := []struct {
		name   string
		ns     []string
		prefix []string
		want   bool
	}{
		{"empty prefix matches everything", []string{"a", "b"}, nil, true},
		{"exact match", []string{"a", "b"}, []string{"a", "b"}, true},
		{"proper prefix", []string{"a", "b", "c"}, []string{"a", "b"}, true},
		{"prefix longer than namespace", []string{"a"}, []string{"a", "b"}, false},
		{"mismatched segment", []string{"a", "x"}, []string{"a", "b"}, false},
		{"first segment mismatch", []string{"x", "b"}, []string{"a"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := namespaceHasPrefix(tc.ns, tc.prefix); got != tc.want {
				t.Errorf("namespaceHasPrefix(%v, %v) = %v, want %v", tc.ns, tc.prefix, got, tc.want)
			}
		})
	}
}

// TestNamespaceHasSuffix exercises the element-wise suffix matcher directly.
func TestNamespaceHasSuffix(t *testing.T) {
	cases := []struct {
		name   string
		ns     []string
		suffix []string
		want   bool
	}{
		{"empty suffix matches everything", []string{"a", "b"}, nil, true},
		{"exact match", []string{"a", "b"}, []string{"a", "b"}, true},
		{"proper suffix", []string{"a", "b", "c"}, []string{"b", "c"}, true},
		{"suffix longer than namespace", []string{"b"}, []string{"a", "b"}, false},
		{"mismatched trailing segment", []string{"a", "x"}, []string{"a", "b"}, false},
		{"matches only as prefix not suffix", []string{"a", "b"}, []string{"a"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := namespaceHasSuffix(tc.ns, tc.suffix); got != tc.want {
				t.Errorf("namespaceHasSuffix(%v, %v) = %v, want %v", tc.ns, tc.suffix, got, tc.want)
			}
		})
	}
}

// TestJoinNSSeparator: joinNS joins segments with a NUL byte. NOTE: a label
// containing a literal NUL would collide with a two-segment namespace
// (validation does not reject NUL) — that suspected production bug is left
// unfixed here and reported separately; this test only pins the rendering.
func TestJoinNSSeparator(t *testing.T) {
	if got := joinNS([]string{"a", "b"}); got != "a\x00b" {
		t.Errorf("joinNS([a b]) = %q, want %q", got, "a\x00b")
	}
	if got := joinNS(nil); got != "" {
		t.Errorf("joinNS(nil) = %q, want empty", got)
	}
	if !strings.Contains(joinNS([]string{"x", "y", "z"}), "\x00") {
		t.Errorf("joinNS should join with NUL separator, got %q", joinNS([]string{"x", "y", "z"}))
	}
}
