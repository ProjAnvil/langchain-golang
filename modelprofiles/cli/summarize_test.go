package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func summarizeOldRegistry() Registry {
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

func summarizeNewRegistry() Registry {
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

func TestSummarize(t *testing.T) {
	got := Summarize("openai", summarizeOldRegistry(), summarizeNewRegistry())
	for _, want := range []string{
		"## Summary of changes",
		"1 added · 1 removed · 1 changed",
		"### openai",
		"**+ 1 added**",
		"`gpt-5`",
		"400,000 ctx",
		"**- 1 removed**",
		"`old-model`",
		"**~ 1 changed**",
		"`gpt-4`",
		"max output tokens 4,096 -> 16,384",
		"added image input",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("Summarize() output missing %q:\n%s", want, got)
		}
	}
}

func TestSummarizeNoChanges(t *testing.T) {
	reg := summarizeOldRegistry()
	if got := Summarize("openai", reg, reg); got != "No model profile data changed." {
		t.Fatalf("Summarize() no-change output = %q", got)
	}
}

func TestRunSummarizeEndToEnd(t *testing.T) {
	dir := t.TempDir()
	beforePath := filepath.Join(dir, "before.json")
	afterPath := filepath.Join(dir, "after.json")
	writeProfilesFile(t, beforePath, summarizeOldRegistry())
	writeProfilesFile(t, afterPath, summarizeNewRegistry())

	var stdout, stderr bytes.Buffer
	err := RunSummarize(SummarizeOptions{
		Provider: "openai",
		Before:   beforePath,
		After:    afterPath,
		Stdout:   &stdout,
		Stderr:   &stderr,
	})
	if err != nil {
		t.Fatalf("RunSummarize() error = %v, stderr = %s", err, stderr.String())
	}
	for _, want := range []string{"### openai", "`gpt-5`", "`old-model`", "max output tokens 4,096 -> 16,384"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("RunSummarize() output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestRunSummarizeDefaultsAfterToDataDir(t *testing.T) {
	dir := t.TempDir()
	beforePath := filepath.Join(dir, "before.json")
	writeProfilesFile(t, beforePath, summarizeOldRegistry())
	writeProfilesFile(t, filepath.Join(dir, profilesFileName), summarizeNewRegistry())

	var stdout, stderr bytes.Buffer
	err := RunSummarize(SummarizeOptions{
		Provider: "openai",
		DataDir:  dir,
		Before:   beforePath,
		Stdout:   &stdout,
		Stderr:   &stderr,
	})
	if err != nil {
		t.Fatalf("RunSummarize() error = %v, stderr = %s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), "### openai") || !strings.Contains(stdout.String(), "`gpt-5`") {
		t.Fatalf("RunSummarize() output missing expected content:\n%s", stdout.String())
	}
}

func writeProfilesFile(t *testing.T, path string, profiles Registry) {
	t.Helper()
	contents, err := BuildProfilesJSON(profiles)
	if err != nil {
		t.Fatalf("BuildProfilesJSON() error = %v", err)
	}
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatalf("failed to write %s: %v", path, err)
	}
}
