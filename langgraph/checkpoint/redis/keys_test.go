package redis

import (
	"reflect"
	"testing"
)

// The management methods (DeleteForRuns/Prune) must recover (thread, ns,
// checkpoint id) from scanned keys, so keys.go grows a parser for the
// escaped colon-joined format (escapeKeyComponent, keys.go:46).
func TestParsePrefixedKey(t *testing.T) {
	key := checkpointKey("th:r", "", "cp1")
	parts, ok := parsePrefixedKey(checkpointPrefix, key)
	if !ok {
		t.Fatalf("parsePrefixedKey(%q): ok = false", key)
	}
	if want := []string{"th:r", "", "cp1"}; !reflect.DeepEqual(parts, want) {
		t.Fatalf("parsePrefixedKey(%q) = %q, want %q", key, parts, want)
	}

	// Wrong prefix is rejected.
	if _, ok := parsePrefixedKey(checkpointWritePrefix, key); ok {
		t.Fatalf("parsePrefixedKey with wrong prefix: ok = true")
	}

	// Backslash escapes round-trip.
	key2 := zsetKey(`a\b`, "ns:1")
	parts2, ok := parsePrefixedKey(checkpointZSetPrefix, key2)
	if !ok || !reflect.DeepEqual(parts2, []string{`a\b`, "ns:1"}) {
		t.Fatalf("parsePrefixedKey(%q) = %q, %v", key2, parts2, ok)
	}
}

func TestSplitEscapedEdgeCases(t *testing.T) {
	if got := splitEscaped(""); !reflect.DeepEqual(got, []string{""}) {
		t.Fatalf("splitEscaped(\"\") = %q, want [\"\"]", got)
	}
	if got := splitEscaped(`a\:b:c`); !reflect.DeepEqual(got, []string{"a:b", "c"}) {
		t.Fatalf("splitEscaped = %q, want [\"a:b\" \"c\"]", got)
	}
	// Trailing backslash (cannot occur from escapeKeyComponent, but the
	// parser must not panic): kept literally.
	if got := splitEscaped(`a\`); !reflect.DeepEqual(got, []string{`a\`}) {
		t.Fatalf("splitEscaped trailing backslash = %q", got)
	}
}
