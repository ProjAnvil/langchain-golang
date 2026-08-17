package modelprofiles

import (
	"strings"
	"testing"
)

func oldRegistry() Registry {
	return Registry{
		"gpt-4": Profile{
			"name":              "GPT-4",
			"max_input_tokens":  8192,
			"max_output_tokens": 4096,
			"image_inputs":      false,
			"tool_calling":      true,
		},
		"old-model": Profile{
			"name":             "Old",
			"max_input_tokens": 1000,
		},
	}
}

func newRegistry() Registry {
	return Registry{
		"gpt-4": Profile{
			"name":              "GPT-4",
			"max_input_tokens":  8192,
			"max_output_tokens": 16384,
			"image_inputs":      true,
			"tool_calling":      true,
		},
		"gpt-5": Profile{
			"name":              "GPT-5",
			"max_input_tokens":  400000,
			"max_output_tokens": 128000,
			"image_inputs":      true,
			"reasoning_output":  true,
			"tool_calling":      true,
		},
	}
}

func TestDiffProfiles(t *testing.T) {
	diff := DiffProfiles(oldRegistry(), newRegistry())
	if len(diff.Added) != 1 || diff.Added[0] != "gpt-5" {
		t.Fatalf("added: %#v", diff.Added)
	}
	if len(diff.Removed) != 1 || diff.Removed[0] != "old-model" {
		t.Fatalf("removed: %#v", diff.Removed)
	}
	change := diff.Changed["gpt-4"]["max_output_tokens"]
	if change.Old != 4096 || change.New != 16384 {
		t.Fatalf("change: %#v", change)
	}
	if diff.IsEmpty() {
		t.Fatal("diff should not be empty")
	}
	if !DiffProfiles(oldRegistry(), oldRegistry()).IsEmpty() {
		t.Fatal("identical registries should be empty")
	}
}

func TestRenderProviderSection(t *testing.T) {
	section := RenderProviderSection("openai", DiffProfiles(oldRegistry(), newRegistry()))
	for _, want := range []string{
		"### openai",
		"1 added",
		"`gpt-5`",
		"400,000 ctx",
		"reasoning",
		"1 removed",
		"`old-model`",
		"1 changed",
		"max output tokens 4,096 -> 16,384",
		"added image input",
	} {
		if !strings.Contains(section, want) {
			t.Fatalf("section missing %q:\n%s", want, section)
		}
	}
	if RenderProviderSection("openai", Diff{}) != "" {
		t.Fatal("empty diff should render empty section")
	}
}

func TestBuildSummary(t *testing.T) {
	summary := BuildSummary(map[string]Diff{"openai": DiffProfiles(oldRegistry(), newRegistry())})
	if !strings.HasPrefix(summary, "## Summary of changes") {
		t.Fatalf("summary: %s", summary)
	}
	if !strings.Contains(summary, "1 added · 1 removed · 1 changed") {
		t.Fatalf("summary headline: %s", summary)
	}
	if BuildSummary(map[string]Diff{"openai": {}}) != "No model profile data changed." {
		t.Fatal("expected no-change summary")
	}
}

func TestFormatValueAndDescriptors(t *testing.T) {
	if FormatValue("x", nil) != "unset" {
		t.Fatal("nil format")
	}
	if FormatValue("tool_calling", true) != "yes" || FormatValue("tool_calling", false) != "no" {
		t.Fatal("bool format")
	}
	if FormatValue("max_input_tokens", 200000) != "200,000" {
		t.Fatal("token format")
	}
	if FormatValue("name", "GPT") != "`GPT`" {
		t.Fatal("string format")
	}
	descriptor := DescribeNewModel(Profile{
		"max_input_tokens":  200000,
		"max_output_tokens": 64000,
		"image_inputs":      true,
		"audio_inputs":      true,
		"video_inputs":      true,
		"pdf_inputs":        true,
		"tool_calling":      true,
	})
	for _, want := range []string{"200,000 ctx", "64,000 out", "text+image+audio+video+pdf in", "tools"} {
		if !strings.Contains(descriptor, want) {
			t.Fatalf("descriptor missing %q: %s", want, descriptor)
		}
	}
	if DescribeNewModel(Profile{"name": "x"}) != "" {
		t.Fatal("expected empty descriptor")
	}
}

func TestFormatValueEdgeCases(t *testing.T) {
	cases := []struct {
		Name  string
		Field string
		Value any
		Want  string
	}{
		{"float token", "max_input_tokens", 128000.0, "128,000"},
		{"int64 token", "max_output_tokens", int64(8192), "8,192"},
		{"small token", "max_input_tokens", 999, "999"},
		{"negative token", "max_input_tokens", -12345, "-12,345"},
		{"non-integral float token", "max_input_tokens", 128000.5, "128000.5"},
		{"string token", "max_input_tokens", "unknown", "`unknown`"},
		{"generic value", "limits", []string{"a", "b"}, "[a b]"},
		{"non-token bool", "image_inputs", true, "yes"},
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			if got := FormatValue(tc.Field, tc.Value); got != tc.Want {
				t.Fatalf("FormatValue(%q, %#v) = %q, want %q", tc.Field, tc.Value, got, tc.Want)
			}
		})
	}
}

func TestDescribeFieldChange(t *testing.T) {
	cases := []struct {
		Name  string
		Field string
		Old   any
		New   any
		Want  string
	}{
		{"added capability", "tool_calling", false, true, "added tool calling"},
		{"removed capability", "tool_calling", true, false, "removed tool calling"},
		{"bool stays true", "tool_calling", true, true, "tool calling yes -> yes"},
		{"unknown field label", "custom_field", "a", "b", "custom_field `a` -> `b`"},
		{"token change", "max_input_tokens", 8192, 200000, "max input tokens 8,192 -> 200,000"},
		{"unset to value", "name", nil, "GPT", "display name unset -> `GPT`"},
		{"old bool new string", "tool_calling", true, "yes", "removed tool calling"},
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			if got := DescribeFieldChange(tc.Field, tc.Old, tc.New); got != tc.Want {
				t.Fatalf("DescribeFieldChange(%q, %#v, %#v) = %q, want %q", tc.Field, tc.Old, tc.New, got, tc.Want)
			}
		})
	}
}

func TestTruncate(t *testing.T) {
	rows := make([]string, maxRows+1)
	for i := range rows {
		rows[i] = "- row"
	}
	got := Truncate(rows)
	if len(got) != maxRows+1 || got[len(got)-1] != "- ...and 1 more" {
		t.Fatalf("truncate: %#v", got)
	}
}
