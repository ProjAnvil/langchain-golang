package middleware

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
// `HostExecutionPolicy` (no runner wrapper).
type HostExecutionPolicy struct{}

func (HostExecutionPolicy) BuildCommand(command []string, _ string) []string {
	return append([]string(nil), command...)
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
