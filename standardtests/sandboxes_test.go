package standardtests

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// localSandbox adapts the host filesystem plus /bin/sh to the SandboxBackend
// interface. The Go port has no SandboxBackend implementation today — the
// middleware ShellToolMiddleware (langchain/agents/middleware/shell.go:74)
// exposes only command execution, not this file-operations surface — so
// localSandbox is a test-only reference adapter and the conformance suite is
// the contract future backends must satisfy.
type localSandbox struct{}

// newLocalSandbox returns a clean local sandbox plus its absolute root
// directory, matching the SandboxFactory contract.
func newLocalSandbox(t testing.TB) (SandboxBackend, string) {
	t.Helper()
	if _, err := exec.LookPath("sh"); err != nil {
		t.Skipf("POSIX shell required for sandbox self-test: %v", err)
	}
	return &localSandbox{}, t.TempDir()
}

func (l *localSandbox) Write(_ context.Context, path string, content string) SandboxWriteResult {
	if !filepath.IsAbs(path) {
		return SandboxWriteResult{Path: path, Error: "invalid_path"}
	}
	if _, err := os.Stat(path); err == nil {
		return SandboxWriteResult{Path: path, Error: fmt.Sprintf("File '%s' already exists", path)}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return SandboxWriteResult{Path: path, Error: err.Error()}
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return SandboxWriteResult{Path: path, Error: err.Error()}
	}
	return SandboxWriteResult{Path: path}
}

func (l *localSandbox) Execute(ctx context.Context, command string) SandboxExecuteResponse {
	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	result := SandboxExecuteResponse{}
	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = 1
			fmt.Fprintf(&buf, "%v", err)
		}
	}
	result.Output = buf.String()
	return result
}

// The methods below are stubs in Task 3; Tasks 4-6 replace them one group at
// a time (TDD: each group's subtests fail against the stub first).
func (l *localSandbox) Read(context.Context, string, SandboxReadOptions) SandboxReadResult {
	return SandboxReadResult{Error: "unimplemented"}
}

func (l *localSandbox) Edit(context.Context, string, string, string, bool) SandboxEditResult {
	return SandboxEditResult{Error: "unimplemented"}
}

func (l *localSandbox) Ls(context.Context, string) SandboxLsResult {
	return SandboxLsResult{Error: "unimplemented"}
}

func (l *localSandbox) Glob(context.Context, string, string) SandboxGlobResult {
	return SandboxGlobResult{Error: "unimplemented"}
}

func (l *localSandbox) Grep(context.Context, string, string, string) SandboxGrepResult {
	return SandboxGrepResult{Error: "unimplemented"}
}

func (l *localSandbox) UploadFiles(context.Context, []SandboxFileUpload) []SandboxUploadResponse {
	return []SandboxUploadResponse{{Error: "unimplemented"}}
}

func (l *localSandbox) DownloadFiles(context.Context, []string) []SandboxDownloadResponse {
	return []SandboxDownloadResponse{{Error: "unimplemented"}}
}

// TestRunSandboxConformanceWithLocalSandbox wires the conformance suite to
// the local sandbox adapter, mirroring how Python implementers subclass
// SandboxIntegrationTests with a sandbox fixture.
func TestRunSandboxConformanceWithLocalSandbox(t *testing.T) {
	RunSandboxConformance(t, newLocalSandbox)
}
