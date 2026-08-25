package middleware

import (
	"reflect"
	"testing"
)

func TestHostExecutionPolicyIdentity(t *testing.T) {
	got := HostExecutionPolicy{}.BuildCommand([]string{"/bin/sh", "-c", "echo hi"}, "/ws")
	want := []string{"/bin/sh", "-c", "echo hi"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("HostExecutionPolicy = %#v, want %#v", got, want)
	}
}

func TestDockerExecutionPolicyBuildCommand(t *testing.T) {
	stubLookPath(t, identityLookPath)
	got := DockerExecutionPolicy{Image: "python:3.12-slim"}.BuildCommand([]string{"/bin/sh", "-c", "pwd"}, "/ws")
	want := []string{"docker", "run", "-i", "--rm", "--network", "none", "-v", "/ws:/ws", "-w", "/ws", "python:3.12-slim", "/bin/sh", "-c", "pwd"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DockerExecutionPolicy = %#v, want %#v", got, want)
	}
}

func TestCodexSandboxExecutionPolicyBuildCommand(t *testing.T) {
	got := CodexSandboxExecutionPolicy{Platform: "linux"}.BuildCommand([]string{"/bin/sh", "-c", "pwd"}, "/ws")
	want := []string{"codex", "exec", "sandbox", "--platform", "linux", "--", "/bin/sh", "-c", "pwd"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CodexSandboxExecutionPolicy = %#v, want %#v", got, want)
	}
}

func TestCodexSandboxExecutionPolicyNoPlatform(t *testing.T) {
	got := CodexSandboxExecutionPolicy{}.BuildCommand([]string{"sh"}, "/ws")
	want := []string{"codex", "exec", "sandbox", "--", "sh"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("CodexSandboxExecutionPolicy = %#v, want %#v", got, want)
	}
}
