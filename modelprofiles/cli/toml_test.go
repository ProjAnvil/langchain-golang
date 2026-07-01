package cli

import (
	"reflect"
	"testing"
)

func TestParseTOMLSubset(t *testing.T) {
	data := []byte(`provider = "anthropic"

[overrides]
image_url_inputs = true
pdf_inputs = true
structured_output = false
tool_call_streaming = true

[overrides."claude-haiku-4-5"]
structured_output = true

[overrides."claude-opus-4-5"]
structured_output = true
`)

	doc, err := parseTOMLSubset(data)
	if err != nil {
		t.Fatalf("parseTOMLSubset() error = %v", err)
	}

	if doc["provider"] != "anthropic" {
		t.Fatalf("expected provider = anthropic, got %v", doc["provider"])
	}

	overrides, ok := doc["overrides"].(map[string]any)
	if !ok {
		t.Fatalf("expected overrides table, got %T", doc["overrides"])
	}
	if overrides["image_url_inputs"] != true {
		t.Errorf("expected image_url_inputs = true, got %v", overrides["image_url_inputs"])
	}
	if overrides["structured_output"] != false {
		t.Errorf("expected structured_output = false, got %v", overrides["structured_output"])
	}

	haiku, ok := overrides["claude-haiku-4-5"].(map[string]any)
	if !ok {
		t.Fatalf("expected claude-haiku-4-5 sub-table, got %T", overrides["claude-haiku-4-5"])
	}
	if haiku["structured_output"] != true {
		t.Errorf("expected nested structured_output = true, got %v", haiku["structured_output"])
	}
}

func TestParseTOMLSubsetComments(t *testing.T) {
	data := []byte(`# top comment
[overrides] # trailing comment
tool_call_streaming = true # another comment
`)
	doc, err := parseTOMLSubset(data)
	if err != nil {
		t.Fatalf("parseTOMLSubset() error = %v", err)
	}
	overrides := doc["overrides"].(map[string]any)
	if overrides["tool_call_streaming"] != true {
		t.Errorf("expected tool_call_streaming = true, got %v", overrides["tool_call_streaming"])
	}
}

func TestParseTOMLSubsetMalformed(t *testing.T) {
	cases := []string{
		"[unterminated",
		"key without equals",
		`[overrides."unterminated]`,
	}
	for _, tc := range cases {
		if _, err := parseTOMLSubset([]byte(tc)); err == nil {
			t.Errorf("expected error for input %q", tc)
		}
	}
}

func TestParseAugmentations(t *testing.T) {
	data := []byte(`provider = "anthropic"

[overrides]
tool_call_streaming = true
structured_output = false

[overrides."claude-haiku-4-5"]
structured_output = true
`)

	providerAug, modelAugs, err := ParseAugmentations(data)
	if err != nil {
		t.Fatalf("ParseAugmentations() error = %v", err)
	}

	want := Profile{"tool_call_streaming": true, "structured_output": false}
	if !reflect.DeepEqual(providerAug, want) {
		t.Errorf("providerAug = %v, want %v", providerAug, want)
	}

	wantModelAugs := map[string]Profile{
		"claude-haiku-4-5": {"structured_output": true},
	}
	if !reflect.DeepEqual(modelAugs, wantModelAugs) {
		t.Errorf("modelAugs = %v, want %v", modelAugs, wantModelAugs)
	}
}

func TestParseAugmentationsNoOverrides(t *testing.T) {
	providerAug, modelAugs, err := ParseAugmentations([]byte(`provider = "openai"`))
	if err != nil {
		t.Fatalf("ParseAugmentations() error = %v", err)
	}
	if len(providerAug) != 0 || len(modelAugs) != 0 {
		t.Errorf("expected empty overrides, got providerAug=%v modelAugs=%v", providerAug, modelAugs)
	}
}
