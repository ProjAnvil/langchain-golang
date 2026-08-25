package middleware

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// stubLookPath replaces the binary-resolution seam for the duration of a test,
// mirroring the Python tests' monkeypatch.setattr(shutil.which, ...).
func stubLookPath(t *testing.T, fn func(string) (string, error)) {
	t.Helper()
	orig := execLookPath
	execLookPath = fn
	t.Cleanup(func() { execLookPath = orig })
}

func identityLookPath(name string) (string, error) { return name, nil }

// TestDockerExecutionPolicySpawnsDockerRun mirrors
// test_docker_policy_spawns_docker_run (test_shell_execution_policies.py:287).
func TestDockerExecutionPolicySpawnsDockerRun(t *testing.T) {
	stubLookPath(t, func(name string) (string, error) { return "/usr/bin/" + name, nil })
	policy := DockerExecutionPolicy{
		Image:        "ubuntu:22.04",
		MemoryBytes:  4096,
		ExtraRunArgs: []string{"--ipc", "host"},
		Env:          map[string]string{"PATH": "/bin"},
	}
	got := policy.BuildCommand([]string{"/bin/bash"}, "/workspace-parity")
	want := []string{
		"/usr/bin/docker", "run", "-i", "--rm",
		"--network", "none",
		"--memory", "4096",
		"-v", "/workspace-parity:/workspace-parity",
		"-w", "/workspace-parity",
		"-e", "PATH=/bin",
		"--ipc", "host",
		"ubuntu:22.04", "/bin/bash",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
}

// TestDockerExecutionPolicySkipsMountForTempWorkspace mirrors
// test_docker_policy_skips_mount_for_temp_workspace
// (test_shell_execution_policies.py:342): a workspace whose basename starts
// with ShellTempPrefix is NOT bind-mounted and the workdir becomes "/".
// The Python test also asserts `command[-2:] == [image, '/bin/sh']`; that
// assertion is dropped here because Go's zero-value Image is omitted from the
// argv entirely (see the Image field doc), so the argv ends with the command.
func TestDockerExecutionPolicySkipsMountForTempWorkspace(t *testing.T) {
	stubLookPath(t, identityLookPath)
	policy := DockerExecutionPolicy{CPUs: "1.5"}
	got := policy.BuildCommand([]string{"/bin/sh"}, "/tmp/"+ShellTempPrefix+"case")
	want := []string{
		"docker", "run", "-i", "--rm",
		"--network", "none",
		"-w", "/",
		"--cpus", "1.5",
		"/bin/sh",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildCommand = %#v, want %#v", got, want)
	}
	for _, part := range got {
		if part == "-v" {
			t.Fatalf("temp workspace must not be bind-mounted: %#v", got)
		}
	}
}

// TestDockerExecutionPolicyValidate mirrors test_docker_policy_rejects_cpu_limit,
// test_docker_policy_validates_memory, test_docker_policy_validates_cpus and
// test_docker_policy_validates_user (test_shell_execution_policies.py:332-380).
func TestDockerExecutionPolicyValidate(t *testing.T) {
	if err := (DockerExecutionPolicy{}).Validate(); err != nil {
		t.Fatalf("zero-value policy must validate: %v", err)
	}
	if err := (DockerExecutionPolicy{CPUTimeSeconds: 1}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "does not support cpu_time_seconds") {
		t.Fatalf("expected cpu_time_seconds rejection, got %v", err)
	}
	if err := (DockerExecutionPolicy{MemoryBytes: -1}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "memory_bytes must be positive if provided") {
		t.Fatalf("expected memory_bytes validation error, got %v", err)
	}
	if err := (DockerExecutionPolicy{CPUs: "  "}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "cpus must be a non-empty string when provided") {
		t.Fatalf("expected cpus validation error, got %v", err)
	}
	if err := (DockerExecutionPolicy{User: "  "}).Validate(); err == nil ||
		!strings.Contains(err.Error(), "user must be a non-empty string when provided") {
		t.Fatalf("expected user validation error, got %v", err)
	}
}

// TestDockerExecutionPolicyReadOnlyAndUser mirrors
// test_docker_policy_read_only_and_user (test_shell_execution_policies.py:383).
func TestDockerExecutionPolicyReadOnlyAndUser(t *testing.T) {
	stubLookPath(t, identityLookPath)
	policy := DockerExecutionPolicy{ReadOnlyRootfs: true, User: "1000:1000"}
	got := policy.BuildCommand([]string{"/bin/sh"}, "/ws")
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "--read-only") {
		t.Fatalf("expected --read-only in %#v", got)
	}
	for i, part := range got {
		if part == "--user" {
			if i+1 >= len(got) || got[i+1] != "1000:1000" {
				t.Fatalf("expected --user 1000:1000 in %#v", got)
			}
			return
		}
	}
	t.Fatalf("expected --user in %#v", got)
}

// TestDockerExecutionPolicyKeepContainer covers Python's
// remove_container_on_exit=False (`_execution.py:339-340`).
func TestDockerExecutionPolicyKeepContainer(t *testing.T) {
	stubLookPath(t, identityLookPath)
	got := DockerExecutionPolicy{KeepContainer: true, Image: "img"}.BuildCommand([]string{"true"}, "/ws")
	for _, part := range got {
		if part == "--rm" {
			t.Fatalf("KeepContainer must omit --rm: %#v", got)
		}
	}
}

// TestDockerExecutionPolicyNetworkEnabled covers Python's
// network_enabled=True (`_execution.py:341-342`).
func TestDockerExecutionPolicyNetworkEnabled(t *testing.T) {
	stubLookPath(t, identityLookPath)
	got := DockerExecutionPolicy{NetworkEnabled: true, Image: "img"}.BuildCommand([]string{"true"}, "/ws")
	for _, part := range got {
		if part == "--network" {
			t.Fatalf("NetworkEnabled must omit --network none: %#v", got)
		}
	}
}

// TestDockerExecutionPolicyEnvSorted pins divergence #5: env entries render in
// sorted-key order for determinism (Go maps iterate randomly).
func TestDockerExecutionPolicyEnvSorted(t *testing.T) {
	stubLookPath(t, identityLookPath)
	policy := DockerExecutionPolicy{Image: "img", Env: map[string]string{"B": "2", "A": "1"}}
	got := policy.BuildCommand([]string{"true"}, "/ws")
	var envParts []string
	for i, part := range got {
		if part == "-e" && i+1 < len(got) {
			envParts = append(envParts, got[i+1])
		}
	}
	want := []string{"A=1", "B=2"}
	if !reflect.DeepEqual(envParts, want) {
		t.Fatalf("env parts = %#v, want %#v", envParts, want)
	}
}

// TestDockerExecutionPolicyResolveBinary mirrors
// test_docker_policy_resolve_missing_binary (test_shell_execution_policies.py:405).
func TestDockerExecutionPolicyResolveBinary(t *testing.T) {
	stubLookPath(t, func(name string) (string, error) { return "", fmt.Errorf("not found") })
	if _, err := (DockerExecutionPolicy{}).ResolveBinary(); err == nil ||
		!strings.Contains(err.Error(), "Docker execution policy requires the 'docker' CLI to be installed") {
		t.Fatalf("expected missing-binary error, got %v", err)
	}
	stubLookPath(t, func(name string) (string, error) { return "/usr/bin/" + name, nil })
	path, err := (DockerExecutionPolicy{}).ResolveBinary()
	if err != nil || path != "/usr/bin/docker" {
		t.Fatalf("ResolveBinary = %q, %v; want /usr/bin/docker, nil", path, err)
	}
}
