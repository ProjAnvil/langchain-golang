package middleware

import (
	"fmt"
	"strings"
)

// ExecutionPolicy builds the argv used to launch a shell or command in a
// workspace. It mirrors Python's `BaseExecutionPolicy.spawn` decision at the
// command-construction level: each policy wraps the base command with the
// appropriate runner (host / docker / codex sandbox). The construction is a
// pure function so it is testable without actually spawning a process.
type ExecutionPolicy interface {
	// BuildCommand returns the argv for running command in workspace.
	BuildCommand(command []string, workspace string) []string
}

// HostExecutionPolicy runs commands directly on the host, mirroring Python's
// `HostExecutionPolicy` (`_execution.py:91`).
//
// DIVERGENCE (enforcement mechanism): Python applies cpu_time_seconds /
// memory_bytes via `resource.setrlimit` in a preexec_fn (macOS) or `prlimit`
// after spawn (Linux). Go's os/exec has no pre-exec hook and SysProcAttr
// exposes no rlimit on any platform, so limits are enforced by wrapping the
// spawned argv in a `/bin/sh -c` preamble that runs `ulimit` before
// `exec "$@"` replaces the wrapper with the real command. `prlimit` wrapping
// is not implemented.
type HostExecutionPolicy struct {
	// CPUTimeSeconds, when > 0, caps the shell's CPU time in seconds
	// (RLIMIT_CPU / `ulimit -t`). Zero (the default) disables the limit.
	// Mirrors Python's `cpu_time_seconds`; note Go treats 0 as unset where
	// Python rejects 0 at construction.
	CPUTimeSeconds int
	// MemoryBytes, when > 0, caps the shell's address space (RLIMIT_AS /
	// `ulimit -v`, which takes KiB — the value is rounded UP to the nearest
	// KiB so the effective limit is never stricter than requested). Zero (the
	// default) disables the limit. Mirrors Python's `memory_bytes`.
	MemoryBytes int64
}

// Validate mirrors Python's `HostExecutionPolicy.__post_init__` limit checks
// (`_execution.py:113-129`). It is called by NewShellToolMiddleware when this
// policy is installed via WithShellExecutionPolicyRunner.
func (p HostExecutionPolicy) Validate() error {
	if p.CPUTimeSeconds < 0 {
		return fmt.Errorf("cpu_time_seconds must be positive if provided")
	}
	if p.MemoryBytes < 0 {
		return fmt.Errorf("memory_bytes must be positive if provided")
	}
	return nil
}

func (p HostExecutionPolicy) BuildCommand(command []string, _ string) []string {
	if p.CPUTimeSeconds <= 0 && p.MemoryBytes <= 0 {
		return append([]string(nil), command...)
	}
	// `exec "$@"` replaces the wrapper shell with the real command, so the
	// ulimits (inherited across exec) apply to the command itself.
	// "langchain-host-limits" is $0 for the wrapper shell.
	var preamble strings.Builder
	if p.CPUTimeSeconds > 0 {
		fmt.Fprintf(&preamble, "ulimit -t %d; ", p.CPUTimeSeconds)
	}
	if p.MemoryBytes > 0 {
		fmt.Fprintf(&preamble, "ulimit -v %d; ", (p.MemoryBytes+1023)/1024)
	}
	preamble.WriteString(`exec "$@"`)
	argv := []string{"/bin/sh", "-c", preamble.String(), "langchain-host-limits"}
	return append(argv, command...)
}

// DockerExecutionPolicy runs commands inside a Docker container, mirroring
// Python's `DockerExecutionPolicy._build_command`.
type DockerExecutionPolicy struct {
	// Image is the container image to run (e.g. "python:3.12-slim").
	Image string
}

func (d DockerExecutionPolicy) BuildCommand(command []string, workspace string) []string {
	args := []string{
		"docker", "run", "--rm",
		"-v", workspace + ":" + workspace,
		"-w", workspace,
	}
	if d.Image != "" {
		args = append(args, d.Image)
	}
	return append(args, command...)
}

// CodexSandboxExecutionPolicy runs commands inside a Codex CLI sandbox,
// mirroring Python's `CodexSandboxExecutionPolicy._build_command`.
type CodexSandboxExecutionPolicy struct {
	// Platform is the sandbox platform hint passed to the Codex CLI ("" is
	// valid — the CLI autodetects).
	Platform string
}

func (c CodexSandboxExecutionPolicy) BuildCommand(command []string, _ string) []string {
	args := []string{"codex", "exec", "sandbox"}
	if c.Platform != "" {
		args = append(args, "--platform", c.Platform)
	}
	args = append(args, "--")
	return append(args, command...)
}
