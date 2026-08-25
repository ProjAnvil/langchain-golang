package middleware

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

// TestCodexSandboxExecutionPolicySpawnsCLI mirrors
// test_codex_policy_spawns_codex_cli (test_shell_execution_policies.py:192).
func TestCodexSandboxExecutionPolicySpawnsCLI(t *testing.T) {
	stubLookPath(t, func(name string) (string, error) { return "/usr/bin/" + name, nil })
	policy := CodexSandboxExecutionPolicy{
		Platform:        "linux",
		ConfigOverrides: map[string]any{"sandbox_permissions": []string{"disk-full-read-access"}},
	}
	got := policy.BuildCommand([]string{"/bin/bash"}, "/ws")
	want := []string{
		"/usr/bin/codex", "sandbox", "linux",
		"-c", `sandbox_permissions=["disk-full-read-access"]`,
		"--", "/bin/bash",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
}

// TestCodexResolvePlatformAuto mirrors test_codex_policy_auto_platform_linux
// and test_codex_policy_auto_platform_macos
// (test_shell_execution_policies.py:234-243). Python monkeypatches sys.platform;
// Go tests the unexported goos-parameterized helper directly.
func TestCodexResolvePlatformAuto(t *testing.T) {
	policy := CodexSandboxExecutionPolicy{}
	if got, err := policy.resolvePlatform("linux"); err != nil || got != "linux" {
		t.Fatalf("resolvePlatform(linux) = %q, %v; want linux, nil", got, err)
	}
	if got, err := policy.resolvePlatform("darwin"); err != nil || got != "macos" {
		t.Fatalf("resolvePlatform(darwin) = %q, %v; want macos, nil", got, err)
	}
	// Explicit platform passes through regardless of GOOS.
	explicit := CodexSandboxExecutionPolicy{Platform: "macos"}
	if got, err := explicit.resolvePlatform("linux"); err != nil || got != "macos" {
		t.Fatalf("explicit platform = %q, %v; want macos, nil", got, err)
	}
	// ResolvePlatform uses the real runtime.GOOS: on a supported dev platform
	// auto resolution succeeds.
	if _, err := policy.ResolvePlatform(); err != nil {
		t.Skipf("auto platform unsupported on this GOOS: %v", err)
	}
}

// TestCodexResolvePlatformAutoFailure mirrors
// test_codex_policy_auto_platform_failure (test_shell_execution_policies.py:253).
func TestCodexResolvePlatformAutoFailure(t *testing.T) {
	policy := CodexSandboxExecutionPolicy{}
	if _, err := policy.resolvePlatform("win32"); err == nil ||
		!strings.Contains(err.Error(), "could not determine a supported platform") {
		t.Fatalf("expected unsupported-platform error, got %v", err)
	}
}

// TestCodexResolveBinaryMissing mirrors test_codex_policy_resolve_missing_binary
// (test_shell_execution_policies.py:246).
func TestCodexResolveBinaryMissing(t *testing.T) {
	stubLookPath(t, func(name string) (string, error) { return "", fmt.Errorf("not found") })
	if _, err := (CodexSandboxExecutionPolicy{Binary: "codex"}).ResolveBinary(); err == nil ||
		!strings.Contains(err.Error(), "Codex sandbox policy requires the 'codex' CLI to be installed") {
		t.Fatalf("expected missing-binary error, got %v", err)
	}
	stubLookPath(t, func(name string) (string, error) { return "/usr/bin/" + name, nil })
	path, err := (CodexSandboxExecutionPolicy{}).ResolveBinary()
	if err != nil || path != "/usr/bin/codex" {
		t.Fatalf("ResolveBinary = %q, %v; want /usr/bin/codex, nil", path, err)
	}
}

// TestCodexConfigOverridesSorted mirrors test_codex_policy_sorts_config_overrides
// (test_shell_execution_policies.py:271).
func TestCodexConfigOverridesSorted(t *testing.T) {
	stubLookPath(t, identityLookPath)
	policy := CodexSandboxExecutionPolicy{
		Platform:        "linux",
		ConfigOverrides: map[string]any{"b": 2, "a": 1},
	}
	got := policy.BuildCommand([]string{"echo"}, "/ws")
	var overrides []string
	for i, part := range got {
		if part == "-c" && i+1 < len(got) {
			overrides = append(overrides, got[i+1])
		}
	}
	want := []string{"a=1", "b=2"}
	if !reflect.DeepEqual(overrides, want) {
		t.Fatalf("overrides = %#v, want %#v", overrides, want)
	}
}

// TestCodexFormatOverride mirrors test_codex_policy_formats_override_values
// (test_shell_execution_policies.py:260). DIVERGENCE: encoding/json emits
// compact JSON (`{"a":1}`) where Python's json.dumps uses `{"a": 1}` — a
// whitespace-only difference; both are valid JSON the Codex CLI accepts.
func TestCodexFormatOverride(t *testing.T) {
	if got := formatCodexOverride(map[string]any{"a": 1}); got != `{"a":1}` {
		t.Fatalf("formatCodexOverride(map) = %q, want %q", got, `{"a":1}`)
	}
	if got := formatCodexOverride("x"); got != `"x"` {
		t.Fatalf("formatCodexOverride(string) = %q, want %q", got, `"x"`)
	}
	// Unmarshalable values fall back to fmt.Sprint (Python falls back to str()).
	if got := formatCodexOverride(math.NaN()); got != "NaN" {
		t.Fatalf("formatCodexOverride(NaN) = %q, want %q", got, "NaN")
	}
}
