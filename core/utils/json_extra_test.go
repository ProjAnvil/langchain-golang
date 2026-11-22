package utils

import (
	"strings"
	"testing"
)

func TestParseAndCheckJSONMarkdownBareObject(t *testing.T) {
	got, err := ParseAndCheckJSONMarkdown(`{"a": 1, "b": "x"}`)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got["b"] != "x" {
		t.Errorf("parsed b = %v, want %q", got["b"], "x")
	}
	if got["a"] != float64(1) {
		t.Errorf("parsed a = %v, want 1", got["a"])
	}
}

func TestParseAndCheckJSONMarkdownBareObjectWhitespaceAndBackticks(t *testing.T) {
	got, err := ParseAndCheckJSONMarkdown("` \n\t{\"a\": 1}\t\n `")
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got["a"] != float64(1) {
		t.Fatalf("parsed a = %v", got["a"])
	}
}

func TestParseAndCheckJSONMarkdownFencedJSON(t *testing.T) {
	text := "Here is the result:\n```json\n{\"action\": \"search\", \"query\": \"go\"}\n```\nDone."
	got, err := ParseAndCheckJSONMarkdown(text)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got["action"] != "search" || got["query"] != "go" {
		t.Fatalf("parsed object = %v", got)
	}
}

func TestParseAndCheckJSONMarkdownFencedNoLangTag(t *testing.T) {
	got, err := ParseAndCheckJSONMarkdown("```\n{\"a\": 1}\n```")
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got["a"] != float64(1) {
		t.Fatalf("parsed object = %v", got)
	}
}

func TestParseAndCheckJSONMarkdownFencedInline(t *testing.T) {
	got, err := ParseAndCheckJSONMarkdown("```{\"a\": 1}```")
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got["a"] != float64(1) {
		t.Fatalf("parsed object = %v", got)
	}
}

func TestParseAndCheckJSONMarkdownEmptyObject(t *testing.T) {
	got, err := ParseAndCheckJSONMarkdown("{}")
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got == nil || len(got) != 0 {
		t.Fatalf("parsed object = %v, want a non-nil empty map", got)
	}
}

func TestParseAndCheckJSONMarkdownNestedObject(t *testing.T) {
	text := `{"a": {"b": [1, 2, {"c": 3}]}, "d": "x"}`
	got, err := ParseAndCheckJSONMarkdown(text)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	inner, ok := got["a"].(map[string]any)
	if !ok {
		t.Fatalf("nested object missing: %v", got)
	}
	array, ok := inner["b"].([]any)
	if !ok || len(array) != 3 {
		t.Fatalf("nested array = %v", inner["b"])
	}
	deep, ok := array[2].(map[string]any)
	if !ok || deep["c"] != float64(3) {
		t.Fatalf("deep value = %v", array[2])
	}
	if got["d"] != "x" {
		t.Fatalf("parsed d = %v", got["d"])
	}
}

func TestParseAndCheckJSONMarkdownActionInputNewlineAndTab(t *testing.T) {
	// The action_input value contains real newline and tab bytes.
	text := "{\"action\": \"search\", \"action_input\": \"line1\nline2\tend\"}"
	got, err := ParseAndCheckJSONMarkdown(text)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	want := "line1\nline2\tend"
	if got["action_input"] != want {
		t.Fatalf("parsed action_input = %q, want %q", got["action_input"], want)
	}
}

func TestParseAndCheckJSONMarkdownActionInputControlChars(t *testing.T) {
	// NUL (0x00) and SOH (0x01) are raw control bytes inside the value.
	text := "{\"action_input\": \"a\x00b\x01c\"}"
	got, err := ParseAndCheckJSONMarkdown(text)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	want := "a\x00b\x01c"
	if got["action_input"] != want {
		t.Fatalf("parsed action_input = %q, want %q", got["action_input"], want)
	}
}

func TestParseAndCheckJSONMarkdownActionInputCarriageReturn(t *testing.T) {
	text := "{\"action_input\": \"one\rtwo\"}"
	got, err := ParseAndCheckJSONMarkdown(text)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	if got["action_input"] != "one\rtwo" {
		t.Fatalf("parsed action_input = %q", got["action_input"])
	}
}

func TestParseAndCheckJSONMarkdownActionInputAlreadyValid(t *testing.T) {
	text := `{"action_input": "say \"hi\"\nnext"}`
	got, err := ParseAndCheckJSONMarkdown(text)
	if err != nil {
		t.Fatalf("ParseAndCheckJSONMarkdown returned error: %v", err)
	}
	want := "say \"hi\"\nnext"
	if got["action_input"] != want {
		t.Fatalf("parsed action_input = %q, want %q", got["action_input"], want)
	}
}

func TestParseAndCheckJSONMarkdownTopLevelArrayRejected(t *testing.T) {
	for _, text := range []string{
		`[1, 2, 3]`,
		"```json\n[1, 2, 3]\n```",
		`[]`,
	} {
		_, err := ParseAndCheckJSONMarkdown(text)
		if err == nil {
			t.Errorf("expected a non-object error for %q", text)
			continue
		}
		if !strings.Contains(err.Error(), "object") {
			t.Errorf("error for %q = %q, want a non-object error", text, err.Error())
		}
	}
}

func TestParseAndCheckJSONMarkdownScalarsRejected(t *testing.T) {
	for _, text := range []string{`42`, `"hello"`, `true`, `null`, `3.14`} {
		_, err := ParseAndCheckJSONMarkdown(text)
		if err == nil {
			t.Errorf("expected a non-object error for %q", text)
			continue
		}
		if !strings.Contains(err.Error(), "object") {
			t.Errorf("error for %q = %q, want a non-object error", text, err.Error())
		}
	}
}

func TestParseAndCheckJSONMarkdownInvalidJSON(t *testing.T) {
	for _, text := range []string{
		`{invalid}`,
		`not json at all`,
		"```json\n{broken\n```",
	} {
		_, err := ParseAndCheckJSONMarkdown(text)
		if err == nil {
			t.Errorf("expected an invalid-JSON error for %q", text)
			continue
		}
		if !strings.Contains(err.Error(), "invalid JSON") {
			t.Errorf("error for %q = %q, want an invalid-JSON error", text, err.Error())
		}
	}
}

func TestParseAndCheckJSONMarkdownMissingFences(t *testing.T) {
	for _, text := range []string{
		"```json\n{\"a\": 1}",
		"```json",
		"```",
	} {
		_, err := ParseAndCheckJSONMarkdown(text)
		if err == nil {
			t.Errorf("expected a missing-fences error for %q", text)
			continue
		}
		if !strings.Contains(err.Error(), "missing fences") {
			t.Errorf("error for %q = %q, want a missing-fences error", text, err.Error())
		}
	}
}

func TestParseAndCheckJSONMarkdownFencedEmptyContentIsInvalidJSON(t *testing.T) {
	_, err := ParseAndCheckJSONMarkdown("```json\n```")
	if err == nil || !strings.Contains(err.Error(), "invalid JSON") {
		t.Fatalf("empty fenced content error = %v, want an invalid-JSON error", err)
	}
}
