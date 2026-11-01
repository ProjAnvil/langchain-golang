package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// tempFile returns an open *os.File backed by a temp file plus a function that
// closes it and returns its contents.
func tempFile(t *testing.T) (*os.File, func() string) {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "out")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	return f, func() string {
		name := f.Name()
		if err := f.Close(); err != nil {
			t.Fatalf("failed to close temp file: %v", err)
		}
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("failed to read temp file: %v", err)
		}
		return string(data)
	}
}

// stdinWith returns an *os.File readable as stdin containing the given input.
func stdinWith(t *testing.T, input string) *os.File {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Fatalf("failed to create stdin file: %v", err)
	}
	if _, err := f.WriteString(input); err != nil {
		t.Fatalf("failed to write stdin file: %v", err)
	}
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatalf("failed to rewind stdin file: %v", err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

// invoke runs the CLI with fresh temp files for stdin/stdout/stderr and
// returns the exit code along with what was written to stdout and stderr.
func invoke(t *testing.T, args []string, stdin *os.File) (int, string, string) {
	t.Helper()
	if stdin == nil {
		stdin = stdinWith(t, "")
	}
	stdout, readStdout := tempFile(t)
	stderr, readStderr := tempFile(t)
	code := run(args, stdin, stdout, stderr)
	return code, readStdout(), readStderr()
}

func TestNoArgsPrintsUsageToStderr(t *testing.T) {
	code, stdout, stderr := invoke(t, nil, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want empty", stdout)
	}
	if !strings.Contains(stderr, "expected a subcommand") || !strings.Contains(stderr, "Usage:") {
		t.Errorf("stderr = %q, want a subcommand error plus usage", stderr)
	}
}

func TestHelpVariants(t *testing.T) {
	for _, arg := range []string{"-h", "--help", "help"} {
		t.Run(arg, func(t *testing.T) {
			code, stdout, stderr := invoke(t, []string{arg}, nil)
			if code != 0 {
				t.Errorf("exit code = %d, want 0", code)
			}
			if !strings.Contains(stdout, "Usage:") {
				t.Errorf("stdout = %q, want usage text", stdout)
			}
			if stderr != "" {
				t.Errorf("stderr = %q, want empty", stderr)
			}
		})
	}
}

func TestUnknownCommand(t *testing.T) {
	code, _, stderr := invoke(t, []string{"bogus"}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, `unknown command "bogus"`) {
		t.Errorf("stderr = %q, want unknown command error", stderr)
	}
}

func TestRefreshRequiresProviderAndDataDir(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{"NoFlags", []string{"refresh"}},
		{"ProviderOnly", []string{"refresh", "--provider", "anthropic"}},
		{"DataDirOnly", []string{"refresh", "--data-dir", "/tmp/data"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, _, stderr := invoke(t, tt.args, nil)
			if code != 1 {
				t.Errorf("exit code = %d, want 1", code)
			}
			if !strings.Contains(stderr, "both --provider and --data-dir are required") {
				t.Errorf("stderr = %q, want missing-flags error", stderr)
			}
		})
	}
}

func TestRefreshRejectsUnknownFlag(t *testing.T) {
	code, _, _ := invoke(t, []string{"refresh", "--nope"}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
}

func TestRefreshDeclinedConfirmationAbortsBeforeDownload(t *testing.T) {
	// A data dir outside the working directory requires confirmation; answering
	// "n" must abort before any network access happens.
	outsideDir := filepath.Join(t.TempDir(), "profiles")
	stdin := stdinWith(t, "n\n")
	code, _, stderr := invoke(t, []string{"refresh", "--provider", "anthropic", "--data-dir", outsideDir}, stdin)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "Continue?") {
		t.Errorf("stderr = %q, want a confirmation prompt", stderr)
	}
	if !strings.Contains(stderr, "error:") {
		t.Errorf("stderr = %q, want the abort error to be reported", stderr)
	}
}

func TestSummarizeRequiresFlags(t *testing.T) {
	code, _, stderr := invoke(t, []string{"summarize", "--provider", "anthropic"}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "--provider, --data-dir, and --before are required") {
		t.Errorf("stderr = %q, want missing-flags error", stderr)
	}
}

func TestSummarizeRejectsUnknownFlag(t *testing.T) {
	code, _, _ := invoke(t, []string{"summarize", "--nope"}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
}

func writeProfiles(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write %s: %v", path, err)
	}
}

func TestSummarizeSuccess(t *testing.T) {
	dir := t.TempDir()
	before := filepath.Join(dir, "before.json")
	after := filepath.Join(dir, "after.json")
	writeProfiles(t, before, `{}`)
	writeProfiles(t, after, `{"model-1": {"name": "Model One", "max_input_tokens": 1000}}`)

	code, stdout, stderr := invoke(t, []string{
		"summarize", "--provider", "testprovider", "--data-dir", dir,
		"--before", before, "--after", after,
	}, nil)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr: %q)", code, stderr)
	}
	if !strings.Contains(stdout, "### testprovider") {
		t.Errorf("stdout = %q, want a section heading for the provider", stdout)
	}
	if !strings.Contains(stdout, "model-1") {
		t.Errorf("stdout = %q, want the added model listed", stdout)
	}
}

func TestSummarizeDefaultsAfterToDataDirProfiles(t *testing.T) {
	dir := t.TempDir()
	before := filepath.Join(dir, "before.json")
	writeProfiles(t, before, `{}`)
	writeProfiles(t, filepath.Join(dir, "profiles.json"), `{}`)

	code, stdout, stderr := invoke(t, []string{
		"summarize", "--provider", "testprovider", "--data-dir", dir, "--before", before,
	}, nil)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr: %q)", code, stderr)
	}
	if !strings.Contains(stdout, "No model profile data changed.") {
		t.Errorf("stdout = %q, want the no-changes summary", stdout)
	}
}

func TestSummarizeMissingBeforeFileFails(t *testing.T) {
	dir := t.TempDir()
	code, _, stderr := invoke(t, []string{
		"summarize", "--provider", "testprovider", "--data-dir", dir,
		"--before", filepath.Join(dir, "does-not-exist.json"),
	}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "error:") {
		t.Errorf("stderr = %q, want the load error to be reported", stderr)
	}
}

func TestMaterializeRejectsUnknownFlag(t *testing.T) {
	code, _, _ := invoke(t, []string{"materialize", "--nope"}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
}

// materializeServer serves a minimal models.dev-shaped payload with one model
// per provider.
func materializeServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := `{
			"anthropic": {"models": {"claude-sonnet-4-6": {"name": "Claude Sonnet 4.6", "limit": {"context": 200000, "output": 64000}}}},
			"openai": {"models": {"gpt-5": {"name": "GPT-5", "limit": {"context": 400000, "output": 128000}}}}
		}`
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

func writeAugmentations(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("failed to create augmentation dir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write %s: %v", path, err)
	}
}

func TestMaterializeWritesCombinedBundle(t *testing.T) {
	server := materializeServer(t)
	dataRoot := t.TempDir()
	writeAugmentations(t, filepath.Join(dataRoot, "anthropic", "profile_augmentations.toml"), `provider = "anthropic"

[overrides."claude-sonnet-4-6"]
mid_conversation_system_messages = true
mid_conversation_tools = true
`)
	writeAugmentations(t, filepath.Join(dataRoot, "openai", "profile_augmentations.toml"), `provider = "openai"

[overrides]
tool_choice = true

[overrides."gpt-5"]
mid_conversation_system_messages = true
mid_conversation_tools = false
`)

	out := filepath.Join(t.TempDir(), "nested", "profiles.json")
	code, stdout, stderr := invoke(t, []string{
		"materialize",
		"--providers", "anthropic,openai",
		"--data-root", dataRoot,
		"--out", out,
		"--api-url", server.URL,
	}, nil)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr: %q)", code, stderr)
	}

	contents, err := os.ReadFile(out)
	if err != nil {
		t.Fatalf("failed to read output bundle: %v", err)
	}
	text := string(contents)
	for _, want := range []string{
		`"anthropic"`,
		`"openai"`,
		`"claude-sonnet-4-6"`,
		`"mid_conversation_system_messages": true`,
		`"mid_conversation_tools": true`,
		`"mid_conversation_tools": false`,
		`"tool_choice": true`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("bundle missing %q; got:\n%s", want, text)
		}
	}
	if !strings.HasSuffix(text, "\n") {
		t.Errorf("bundle should end with a newline, got tail %q", text[len(text)-2:])
	}
	for _, provider := range []string{"anthropic", "openai"} {
		if !strings.Contains(stdout, provider) {
			t.Errorf("stdout = %q, want progress output naming %s", stdout, provider)
		}
	}
}

func TestMaterializeUnknownProviderFails(t *testing.T) {
	server := materializeServer(t)
	out := filepath.Join(t.TempDir(), "profiles.json")
	code, _, stderr := invoke(t, []string{
		"materialize",
		"--providers", "not-a-provider",
		"--data-root", t.TempDir(),
		"--out", out,
		"--api-url", server.URL,
	}, nil)
	if code != 1 {
		t.Errorf("exit code = %d, want 1", code)
	}
	if !strings.Contains(stderr, "error:") {
		t.Errorf("stderr = %q, want the fetch error to be reported", stderr)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Errorf("materialize wrote %s despite failure, want no output", out)
	}
}
