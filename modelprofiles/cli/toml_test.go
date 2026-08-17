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
		"key = [1, 2]", // unsupported value type
		"key = ",       // empty value
	}
	for _, tc := range cases {
		if _, err := parseTOMLSubset([]byte(tc)); err == nil {
			t.Errorf("expected error for input %q", tc)
		}
	}
}

func TestParseTOMLValue(t *testing.T) {
	cases := []struct {
		input string
		want  any
	}{
		{"true", true},
		{"false", false},
		{`"hello"`, "hello"},
		{`""`, ""},
		{"42", int64(42)},
		{"-7", int64(-7)},
		{"3.14", 3.14},
		{"1e3", 1000.0},
	}
	for _, tc := range cases {
		got, err := parseTOMLValue(tc.input)
		if err != nil {
			t.Errorf("parseTOMLValue(%q) error = %v", tc.input, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseTOMLValue(%q) = %#v, want %#v", tc.input, got, tc.want)
		}
	}
}

func TestParseTOMLValueErrors(t *testing.T) {
	for _, input := range []string{
		"",              // empty value
		"[1, 2]",        // arrays unsupported
		"2024-01-01",    // dates unsupported
		`"unterminated`, // unterminated string
	} {
		if _, err := parseTOMLValue(input); err == nil {
			t.Errorf("expected error for parseTOMLValue(%q)", input)
		}
	}
}

func TestParseTOMLSubsetQuotedKey(t *testing.T) {
	doc, err := parseTOMLSubset([]byte(`"my key" = 1`))
	if err != nil {
		t.Fatalf("parseTOMLSubset() error = %v", err)
	}
	if doc["my key"] != int64(1) {
		t.Errorf("expected quoted key to be unquoted, got %v", doc)
	}
}

func TestParseTOMLSubsetCommentInsideQuotes(t *testing.T) {
	doc, err := parseTOMLSubset([]byte(`key = "value # not a comment"`))
	if err != nil {
		t.Fatalf("parseTOMLSubset() error = %v", err)
	}
	if doc["key"] != "value # not a comment" {
		t.Errorf("expected # inside quotes preserved, got %v", doc["key"])
	}
}

func TestParseTOMLSubsetSegmentConflictsWithScalar(t *testing.T) {
	// `a = 1` then `[a]` — the path segment is a scalar, not a table.
	if _, err := parseTOMLSubset([]byte("a = 1\n[a]\n")); err == nil {
		t.Errorf("expected error when table path collides with scalar")
	}
}

func TestSplitTOMLTablePathEmptySegment(t *testing.T) {
	for _, header := range []string{"a..b", ".a", "a.", "  "} {
		if _, err := splitTOMLTablePath(header); err == nil {
			t.Errorf("expected error for header %q", header)
		}
	}
}

func TestSplitTOMLTablePathWhitespaceTrimmed(t *testing.T) {
	got, err := splitTOMLTablePath(` overrides . "claude-x" `)
	if err != nil {
		t.Fatalf("splitTOMLTablePath() error = %v", err)
	}
	want := []string{"overrides", "claude-x"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("splitTOMLTablePath() = %v, want %v", got, want)
	}
}

func TestUnquoteTOMLKey(t *testing.T) {
	cases := map[string]string{
		`"quoted"`:    "quoted",
		"plain":       "plain",
		`"x"`:         "x",
		`"unbalanced`: `"unbalanced`, // missing closing quote: left alone
	}
	for input, want := range cases {
		if got := unquoteTOMLKey(input); got != want {
			t.Errorf("unquoteTOMLKey(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestParseAugmentationsInvalidTOML(t *testing.T) {
	if _, _, err := ParseAugmentations([]byte("[unterminated")); err == nil {
		t.Errorf("expected error for invalid TOML")
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
