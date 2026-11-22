package indexing_test

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/indexing"
)

// assertUUIDv5 checks the canonical form, version nibble, and RFC 4122
// variant nibble of a UUID string.
func assertUUIDv5(t *testing.T, value string) {
	t.Helper()
	if len(value) != 36 || value[8] != '-' || value[13] != '-' || value[18] != '-' || value[23] != '-' {
		t.Fatalf("value %q is not a canonical UUID string", value)
	}
	if value[14] != '5' {
		t.Fatalf("value %q has version nibble %q, want 5", value, value[14])
	}
	switch value[19] {
	case '8', '9', 'a', 'b':
	default:
		t.Fatalf("value %q has invalid RFC 4122 variant nibble %q", value, value[19])
	}
}

func TestHashStringToUUIDKnownAnswers(t *testing.T) {
	// Values computed from Python's
	// str(uuid.uuid5(uuid.UUID(int=1984), hashlib.sha1(s.encode()).hexdigest())).
	cases := []struct {
		input string
		want  string
	}{
		{"hello", "80e14d7a-1380-58de-8284-42517bed090a"},
		{"", "1a895976-2fcc-5396-9e10-96ea24efaeed"},
		{"hello world", "593d8b9e-e6ee-566a-90e9-d51f267e6523"},
		{"langchain", "7d2e169f-0d33-5739-b1bb-481f3d759a34"},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			got := indexing.HashStringToUUID(tc.input)
			if got != tc.want {
				t.Fatalf("HashStringToUUID(%q) = %q, want %q", tc.input, got, tc.want)
			}
			assertUUIDv5(t, got)
		})
	}
}

func TestHashStringToUUIDDeterministic(t *testing.T) {
	a := indexing.HashStringToUUID("stable")
	b := indexing.HashStringToUUID("stable")
	if a != b {
		t.Fatalf("deterministic hash changed: %q != %q", a, b)
	}
	if a == indexing.HashStringToUUID("different") {
		t.Fatalf("distinct inputs produced the same UUID")
	}
}

func TestHashStringMatchesHashStringToUUIDForSHA1(t *testing.T) {
	for _, algorithm := range []indexing.KeyEncoder{"", indexing.KeyEncoderSHA1} {
		got, err := indexing.HashString("hello", algorithm)
		if err != nil {
			t.Fatalf("HashString: %v", err)
		}
		if want := indexing.HashStringToUUID("hello"); got != want {
			t.Fatalf("HashString(%q) = %q, want %q", algorithm, got, want)
		}
	}
}

func TestHashStringAlgorithms(t *testing.T) {
	algorithms := []indexing.KeyEncoder{
		indexing.KeyEncoderSHA1,
		indexing.KeyEncoderSHA256,
		indexing.KeyEncoderSHA512,
		indexing.KeyEncoderBlake2b,
	}
	seen := map[string]bool{}
	for _, algorithm := range algorithms {
		first, err := indexing.HashString("payload", algorithm)
		if err != nil {
			t.Fatalf("HashString(%q): %v", algorithm, err)
		}
		second, _ := indexing.HashString("payload", algorithm)
		if first != second {
			t.Fatalf("HashString(%q) is not deterministic", algorithm)
		}
		assertUUIDv5(t, first)
		seen[first] = true
	}
	if len(seen) != len(algorithms) {
		t.Fatalf("different algorithms produced colliding UUIDs: %v", seen)
	}
	if _, err := indexing.HashString("payload", indexing.KeyEncoder("bogus")); err == nil ||
		!strings.Contains(err.Error(), "unsupported hashing algorithm") {
		t.Fatalf("error = %v, want an unsupported-algorithm error", err)
	}
}

func TestHashNestedDictIgnoresInsertionOrder(t *testing.T) {
	// Two maps with the same content but different construction/insertion
	// order, including nested maps, must hash identically.
	a := map[string]any{
		"z": 1,
		"a": map[string]any{"x": "first", "b": "second"},
		"m": []any{3, 1, 2},
	}
	b := map[string]any{
		"m": []any{3, 1, 2},
		"a": map[string]any{"b": "second", "x": "first"},
		"z": 1,
	}
	ha, err := indexing.HashNestedDict(a, indexing.KeyEncoderSHA1)
	if err != nil {
		t.Fatalf("HashNestedDict(a): %v", err)
	}
	hb, err := indexing.HashNestedDict(b, indexing.KeyEncoderSHA1)
	if err != nil {
		t.Fatalf("HashNestedDict(b): %v", err)
	}
	if ha != hb {
		t.Fatalf("nested-map ordering changed the hash: %q != %q", ha, hb)
	}
	assertUUIDv5(t, ha)

	// Changing a value changes the hash.
	mutated := map[string]any{
		"z": 2,
		"a": map[string]any{"x": "first", "b": "second"},
		"m": []any{3, 1, 2},
	}
	hc, err := indexing.HashNestedDict(mutated, indexing.KeyEncoderSHA1)
	if err != nil {
		t.Fatalf("HashNestedDict(mutated): %v", err)
	}
	if hc == ha {
		t.Fatalf("different content produced the same hash")
	}
}

func TestHashNestedDictUnserializable(t *testing.T) {
	_, err := indexing.HashNestedDict(map[string]any{"bad": make(chan int)}, indexing.KeyEncoderSHA1)
	if err == nil || !strings.Contains(err.Error(), "failed to hash metadata") {
		t.Fatalf("error = %v, want a serialization error", err)
	}
}

func TestSourceIDAssigner(t *testing.T) {
	doc := documents.Document{
		PageContent: "content",
		Metadata:    map[string]any{"source": "doc-1", "count": 42},
	}

	// nil spec: no source assigned.
	resolver, err := indexing.SourceIDAssigner(nil)
	if err != nil {
		t.Fatalf("SourceIDAssigner(nil): %v", err)
	}
	if got, _ := resolver(doc); got != "" {
		t.Fatalf("nil resolver = %q, want empty", got)
	}

	// string spec: read from metadata.
	resolver, err = indexing.SourceIDAssigner("source")
	if err != nil {
		t.Fatalf("SourceIDAssigner(string): %v", err)
	}
	if got, _ := resolver(doc); got != "doc-1" {
		t.Fatalf("string resolver = %q, want doc-1", got)
	}

	// missing key produces a clear error.
	resolver, _ = indexing.SourceIDAssigner("missing")
	if _, err := resolver(doc); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("error = %v, want a not-found error", err)
	}

	// non-string value produces a clear error.
	resolver, _ = indexing.SourceIDAssigner("count")
	if _, err := resolver(doc); err == nil || !strings.Contains(err.Error(), "must be a string") {
		t.Fatalf("error = %v, want a type error", err)
	}

	// func(documents.Document) (string, error).
	resolver, err = indexing.SourceIDAssigner(func(documents.Document) (string, error) {
		return "custom", nil
	})
	if err != nil {
		t.Fatalf("SourceIDAssigner(func): %v", err)
	}
	if got, _ := resolver(doc); got != "custom" {
		t.Fatalf("func resolver = %q, want custom", got)
	}

	// func(documents.Document) string.
	resolver, err = indexing.SourceIDAssigner(func(d documents.Document) string {
		return d.PageContent
	})
	if err != nil {
		t.Fatalf("SourceIDAssigner(func string): %v", err)
	}
	if got, _ := resolver(doc); got != "content" {
		t.Fatalf("func-string resolver = %q, want content", got)
	}

	// KeyEncoderFunc (a defined func type) is accepted.
	var keyFunc indexing.KeyEncoderFunc = func(documents.Document) (string, error) {
		return "keyfunc", nil
	}
	resolver, err = indexing.SourceIDAssigner(keyFunc)
	if err != nil {
		t.Fatalf("SourceIDAssigner(KeyEncoderFunc): %v", err)
	}
	if got, _ := resolver(doc); got != "keyfunc" {
		t.Fatalf("KeyEncoderFunc resolver = %q, want keyfunc", got)
	}

	// invalid spec type.
	if _, err := indexing.SourceIDAssigner(123); err == nil ||
		!strings.Contains(err.Error(), "should be nil, a string, or a func") {
		t.Fatalf("error = %v, want an invalid-spec error", err)
	}
}

func TestBatch(t *testing.T) {
	if _, err := indexing.Batch(0, []int{1, 2, 3}); err == nil {
		t.Fatalf("expected an error for size 0")
	}
	if _, err := indexing.Batch(-1, []int{1, 2, 3}); err == nil {
		t.Fatalf("expected an error for negative size")
	}

	empty, err := indexing.Batch(2, []int{})
	if err != nil || empty != nil {
		t.Fatalf("Batch(empty) = %v, %v; want nil, nil", empty, err)
	}

	got, err := indexing.Batch(2, []int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	want := [][]int{{1, 2}, {3, 4}, {5}}
	if len(got) != len(want) {
		t.Fatalf("Batch = %v, want %v", got, want)
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("Batch[%d] = %v, want %v", i, got[i], want[i])
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("Batch = %v, want %v", got, want)
			}
		}
	}

	// Chunks are copies: mutating a chunk does not affect the input.
	items := []int{10, 20, 30}
	chunks, _ := indexing.Batch(2, items)
	chunks[0][0] = 99
	if items[0] != 10 {
		t.Fatalf("Batch did not copy chunks: input changed to %v", items)
	}
}
