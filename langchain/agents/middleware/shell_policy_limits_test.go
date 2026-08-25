package middleware

import (
	"reflect"
	"strings"
	"testing"
)

// TestHostExecutionPolicyValidate mirrors the limits portion of
// test_host_policy_validations
// (libs/langchain_v1/tests/unit_tests/agents/middleware/implementations/test_shell_execution_policies.py:57).
// Python additionally rejects an explicit cpu_time_seconds=0 / memory_bytes=0
// at construction; Go treats 0 as unset (divergence #3), so the zero-value
// policy must validate and only negatives are expected to fail.
func TestHostExecutionPolicyValidate(t *testing.T) {
	if err := (HostExecutionPolicy{}).Validate(); err != nil {
		t.Fatalf("zero-value policy must validate: %v", err)
	}
	if err := (HostExecutionPolicy{CPUTimeSeconds: 2, MemoryBytes: 4096}).Validate(); err != nil {
		t.Fatalf("positive limits must validate: %v", err)
	}
	if err := (HostExecutionPolicy{CPUTimeSeconds: -1}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "cpu_time_seconds must be positive if provided") {
		t.Fatalf("expected cpu_time_seconds validation error, got %v", err)
	}
	if err := (HostExecutionPolicy{MemoryBytes: -1}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "memory_bytes must be positive if provided") {
		t.Fatalf("expected memory_bytes validation error, got %v", err)
	}
}

// TestHostExecutionPolicyLimitsPreamble mirrors test_host_policy_applies_prlimit
// (test_shell_execution_policies.py:74) at the argv level: Go has no pre-exec
// rlimit hook, so limits are enforced by a `ulimit` preamble wrapping the
// spawned command. `ulimit -t` is seconds; `ulimit -v` is KiB (4096 bytes = 4 KiB).
func TestHostExecutionPolicyLimitsPreamble(t *testing.T) {
	got := HostExecutionPolicy{CPUTimeSeconds: 2, MemoryBytes: 4096}.BuildCommand(
		[]string{"/bin/sh", "-c", "echo hi"}, "/ws")
	want := []string{"/bin/sh", "-c", `ulimit -t 2; ulimit -v 4; exec "$@"`,
		"langchain-host-limits", "/bin/sh", "-c", "echo hi"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
}

func TestHostExecutionPolicyCPUOnlyPreamble(t *testing.T) {
	got := HostExecutionPolicy{CPUTimeSeconds: 5}.BuildCommand([]string{"make"}, "/ws")
	want := []string{"/bin/sh", "-c", `ulimit -t 5; exec "$@"`, "langchain-host-limits", "make"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
}

// TestHostExecutionPolicyMemoryRoundsUpToKiB pins divergence #2: 1025 bytes
// rounds UP to 2 KiB so the effective limit is never stricter than requested.
func TestHostExecutionPolicyMemoryRoundsUpToKiB(t *testing.T) {
	got := HostExecutionPolicy{MemoryBytes: 1025}.BuildCommand([]string{"make"}, "/ws")
	want := []string{"/bin/sh", "-c", `ulimit -v 2; exec "$@"`, "langchain-host-limits", "make"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
}

func TestHostExecutionPolicyNoLimitsUnchanged(t *testing.T) {
	got := HostExecutionPolicy{}.BuildCommand([]string{"/bin/sh", "-c", "echo hi"}, "/ws")
	want := []string{"/bin/sh", "-c", "echo hi"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
}
