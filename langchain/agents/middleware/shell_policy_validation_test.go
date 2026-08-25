package middleware

import (
	"strings"
	"testing"
)

// TestNewShellToolMiddlewareValidatesExecutionPolicy mirrors Python's
// __post_init__ construction-time validation (`_execution.py:113/295`): Go
// policies are plain structs, so validation runs in NewShellToolMiddleware.
func TestNewShellToolMiddlewareValidatesExecutionPolicy(t *testing.T) {
	if _, err := NewShellToolMiddleware(t.TempDir(),
		WithShellExecutionPolicyRunner(DockerExecutionPolicy{CPUTimeSeconds: 1})); err == nil ||
		!strings.Contains(err.Error(), "does not support cpu_time_seconds") {
		t.Fatalf("expected docker cpu_time_seconds rejection, got %v", err)
	}
	if _, err := NewShellToolMiddleware(t.TempDir(),
		WithShellExecutionPolicyRunner(HostExecutionPolicy{MemoryBytes: -5})); err == nil ||
		!strings.Contains(err.Error(), "memory_bytes must be positive if provided") {
		t.Fatalf("expected host memory_bytes rejection, got %v", err)
	}
	// Valid policies still construct.
	mw, err := NewShellToolMiddleware(t.TempDir(),
		WithShellExecutionPolicyRunner(DockerExecutionPolicy{Image: "img", MemoryBytes: 1024, CPUs: "1.0"}))
	if err != nil || mw == nil {
		t.Fatalf("valid docker policy must construct: %v", err)
	}
	if _, err := NewShellToolMiddleware(t.TempDir()); err != nil {
		t.Fatalf("default host policy must construct: %v", err)
	}
}
