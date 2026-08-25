package middleware

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// execLookPath is a test seam over exec.LookPath, mirroring the Python tests'
// monkeypatch.setattr(shutil.which, ...) (test_shell_execution_policies.py).
var execLookPath = exec.LookPath

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
// Python's `DockerExecutionPolicy` (`_execution.py:266`).
//
// Parity notes:
//   - The argv starts `docker run -i` (Python `_execution.py:338`); the old Go
//     port emitted `docker run --rm` without `-i`.
//   - The container network is disabled by default (`--network none`), matching
//     Python's `network_enabled=False` default.
//   - A workspace whose basename starts with ShellTempPrefix is an ephemeral
//     session workspace and is NOT bind-mounted (Python `_should_mount_workspace`,
//     `_execution.py:366`); the workdir falls back to "/".
//   - Python passes the session env into `_build_command`; Go's ExecutionPolicy
//     interface receives no env, so container env is configured via the Env
//     field and rendered as sorted `-e KEY=VALUE` flags (sorted for determinism).
type DockerExecutionPolicy struct {
	// Binary is the docker CLI name/path (default "docker"), resolved via PATH
	// lookup like Python's `_resolve_binary` (`_execution.py:369`).
	Binary string
	// Image is the container image to run (e.g. "python:3.12-slim"). Empty
	// omits the image argument (preserves the pre-parity Go behavior for the
	// zero value; Python defaults to "python:3.12-alpine3.19").
	Image string
	// KeepContainer, when false (default), emits --rm (Python
	// `remove_container_on_exit=True`).
	KeepContainer bool
	// NetworkEnabled, when false (default), emits --network none (Python
	// `network_enabled=False`).
	NetworkEnabled bool
	// MemoryBytes, when > 0, emits --memory <bytes> (Python `memory_bytes`).
	MemoryBytes int64
	// CPUTimeSeconds is rejected by Validate: Docker has no per-second CPU-time
	// limit; use CPUs instead (Python raises RuntimeError, `_execution.py:300-305`).
	CPUTimeSeconds int
	// CPUs, when non-empty, emits --cpus <value> (Python `cpus`).
	CPUs string
	// ReadOnlyRootfs emits --read-only (Python `read_only_rootfs`).
	ReadOnlyRootfs bool
	// User, when non-empty, emits --user <value> (Python `user`).
	User string
	// ExtraRunArgs are appended after all other flags, before the image
	// (Python `extra_run_args`).
	ExtraRunArgs []string
	// Env renders as -e KEY=VALUE flags (sorted by key). See the parity note
	// above for why this is a struct field rather than a BuildCommand argument.
	Env map[string]string
}

// Validate mirrors Python's `DockerExecutionPolicy.__post_init__`
// (`_execution.py:295-312`). It is called by NewShellToolMiddleware when this
// policy is installed via WithShellExecutionPolicyRunner.
func (d DockerExecutionPolicy) Validate() error {
	if d.MemoryBytes < 0 {
		return fmt.Errorf("memory_bytes must be positive if provided")
	}
	if d.CPUTimeSeconds != 0 {
		return fmt.Errorf("DockerExecutionPolicy does not support cpu_time_seconds; configure CPU limits using Docker run options such as '--cpus'")
	}
	if d.CPUs != "" && strings.TrimSpace(d.CPUs) == "" {
		return fmt.Errorf("cpus must be a non-empty string when provided")
	}
	if d.User != "" && strings.TrimSpace(d.User) == "" {
		return fmt.Errorf("user must be a non-empty string when provided")
	}
	return nil
}

// ResolveBinary mirrors Python's `DockerExecutionPolicy._resolve_binary`
// (`_execution.py:369-377`): it resolves Binary (default "docker") against
// PATH and errors when the CLI is unavailable. Python raises at spawn time;
// in Go the error surfaces here and from os/exec at process start.
func (d DockerExecutionPolicy) ResolveBinary() (string, error) {
	binary := d.Binary
	if binary == "" {
		binary = "docker"
	}
	path, err := execLookPath(binary)
	if err != nil {
		return "", fmt.Errorf("Docker execution policy requires the '%s' CLI to be installed and available on PATH", binary)
	}
	return path, nil
}

// BuildCommand mirrors Python's `DockerExecutionPolicy._build_command`
// (`_execution.py:331-363`), including argument order.
func (d DockerExecutionPolicy) BuildCommand(command []string, workspace string) []string {
	binary := d.Binary
	if binary == "" {
		binary = "docker"
	}
	// Best-effort PATH resolution (keeps BuildCommand a total function); the
	// hard error path lives in ResolveBinary, mirroring Python's spawn-time
	// RuntimeError.
	if path, err := execLookPath(binary); err == nil {
		binary = path
	}
	args := []string{binary, "run", "-i"}
	if !d.KeepContainer {
		args = append(args, "--rm")
	}
	if !d.NetworkEnabled {
		args = append(args, "--network", "none")
	}
	if d.MemoryBytes > 0 {
		args = append(args, "--memory", strconv.FormatInt(d.MemoryBytes, 10))
	}
	if !strings.HasPrefix(filepath.Base(workspace), ShellTempPrefix) {
		args = append(args, "-v", workspace+":"+workspace, "-w", workspace)
	} else {
		args = append(args, "-w", "/")
	}
	if d.ReadOnlyRootfs {
		args = append(args, "--read-only")
	}
	envKeys := make([]string, 0, len(d.Env))
	for k := range d.Env {
		envKeys = append(envKeys, k)
	}
	sort.Strings(envKeys)
	for _, k := range envKeys {
		args = append(args, "-e", k+"="+d.Env[k])
	}
	if d.CPUs != "" {
		args = append(args, "--cpus", d.CPUs)
	}
	if d.User != "" {
		args = append(args, "--user", d.User)
	}
	args = append(args, d.ExtraRunArgs...)
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
