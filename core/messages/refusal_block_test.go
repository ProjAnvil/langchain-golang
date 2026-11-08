package messages

import (
	"reflect"
	"testing"
)

// Ports the block-level cases of core/tests/unit_tests/messages/test_refusal.py (#41582).

func TestRefusalBlockRoundTrip(t *testing.T) {
	m := map[string]any{
		"type":     "refusal",
		"refusal":  "I cannot help with that.",
		"id":       "ref-1",
		"index":    "lc_0",
		"category": "safety", // unknown key -> Extras
	}
	block := ParseContentBlock(m)
	rb, ok := block.(RefusalBlock)
	if !ok {
		t.Fatalf("ParseContentBlock returned %T, want RefusalBlock", block)
	}
	if rb.Refusal != "I cannot help with that." || rb.ID != "ref-1" || rb.Index != "lc_0" {
		t.Fatalf("fields mismatch: %+v", rb)
	}
	if !reflect.DeepEqual(rb.Extras, map[string]any{"category": "safety"}) {
		t.Fatalf("extras mismatch: %+v", rb.Extras)
	}
	back := BlockToMap(rb)
	if back["type"] != "refusal" || back["refusal"] != "I cannot help with that." {
		t.Fatalf("BlockToMap lost fields: %v", back)
	}
	if back["id"] != "ref-1" || back["index"] != "lc_0" {
		t.Fatalf("BlockToMap lost id/index: %v", back)
	}
	// Extras flatten to the top level (mergeExtras convention, no "extras" key).
	if back["category"] != "safety" {
		t.Fatalf("BlockToMap must flatten extras, got: %v", back)
	}
	if _, hasExtrasKey := back["extras"]; hasExtrasKey {
		t.Fatalf("BlockToMap must not emit an \"extras\" key: %v", back)
	}
}

func TestRefusalBlockMinimalOmitsOptionalFields(t *testing.T) {
	back := BlockToMap(RefusalBlock{Refusal: ""})
	if back["type"] != "refusal" || back["refusal"] != "" {
		t.Fatalf("minimal block mismatch: %v", back)
	}
	if _, ok := back["id"]; ok {
		t.Fatalf("empty id must be omitted: %v", back)
	}
	if _, ok := back["index"]; ok {
		t.Fatalf("nil index must be omitted: %v", back)
	}
}

func TestRefusalBlockCloneIsDeep(t *testing.T) {
	orig := RefusalBlock{Refusal: "no", Extras: map[string]any{"category": "safety"}}
	cloned, ok := CloneBlock(orig).(RefusalBlock)
	if !ok {
		t.Fatalf("CloneBlock returned %T", CloneBlock(orig))
	}
	cloned.Extras["category"] = "mutated"
	if orig.Extras["category"] != "safety" {
		t.Fatal("CloneBlock must deep-copy Extras")
	}
}

func TestRefusalBlockKnownType(t *testing.T) {
	if !KNOWN_BLOCK_TYPES["refusal"] {
		t.Fatal("KNOWN_BLOCK_TYPES must include \"refusal\"")
	}
}
