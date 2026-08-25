package standardtests

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
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
// sandboxMaxBinaryPreviewBytes is the binary preview size limit fixed by the
// Python conformance suite (test_read_binary_file_1_mib_returns_error).
const sandboxMaxBinaryPreviewBytes = 512000

func (l *localSandbox) Read(_ context.Context, path string, opts SandboxReadOptions) SandboxReadResult {
	data, err := os.ReadFile(path)
	if err != nil {
		return SandboxReadResult{Error: fmt.Sprintf("File not found: '%s'", path)}
	}
	if !utf8.Valid(data) {
		if len(data) > sandboxMaxBinaryPreviewBytes {
			return SandboxReadResult{Error: fmt.Sprintf(
				"File '%s': Binary file exceeds maximum preview size of 512000 bytes", path)}
		}
		return SandboxReadResult{FileData: &SandboxFileData{
			Content:  base64.StdEncoding.EncodeToString(data),
			Encoding: "base64",
		}}
	}
	var lines []string
	if text := string(data); text != "" {
		lines = strings.Split(text, "\n")
	}
	if opts.Offset > 0 {
		if opts.Offset >= len(lines) {
			lines = nil
		} else {
			lines = lines[opts.Offset:]
		}
	}
	if opts.Limit != nil && *opts.Limit < len(lines) {
		lines = lines[:*opts.Limit]
	}
	return SandboxReadResult{FileData: &SandboxFileData{
		Content:  strings.Join(lines, "\n"),
		Encoding: "utf-8",
	}}
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

func (l *localSandbox) UploadFiles(_ context.Context, files []SandboxFileUpload) []SandboxUploadResponse {
	responses := make([]SandboxUploadResponse, 0, len(files))
	for _, f := range files {
		resp := SandboxUploadResponse{Path: f.Path}
		switch {
		case !filepath.IsAbs(f.Path):
			resp.Error = "invalid_path"
		default:
			if err := os.MkdirAll(filepath.Dir(f.Path), 0o755); err != nil {
				resp.Error = err.Error()
			} else if err := os.WriteFile(f.Path, f.Content, 0o644); err != nil {
				resp.Error = err.Error()
			}
		}
		responses = append(responses, resp)
	}
	return responses
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
