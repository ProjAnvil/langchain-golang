package standardtests

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
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

func (l *localSandbox) Edit(_ context.Context, path string, old string, new string, replaceAll bool) SandboxEditResult {
	data, err := os.ReadFile(path)
	if err != nil {
		return SandboxEditResult{Error: fmt.Sprintf("File not found: '%s'", path)}
	}
	content := string(data)
	count := strings.Count(content, old)
	if count == 0 {
		return SandboxEditResult{Error: fmt.Sprintf("String not found in '%s'", path)}
	}
	if count > 1 && !replaceAll {
		return SandboxEditResult{Error: fmt.Sprintf("Multiple occurrences (%d) of %q in '%s'", count, old, path)}
	}
	n := 1
	if replaceAll {
		n = -1
	}
	if err := os.WriteFile(path, []byte(strings.Replace(content, old, new, n)), 0o644); err != nil {
		return SandboxEditResult{Error: err.Error()}
	}
	return SandboxEditResult{Occurrences: count}
}

func (l *localSandbox) Ls(_ context.Context, path string) SandboxLsResult {
	entries, err := os.ReadDir(path)
	if err != nil {
		return SandboxLsResult{Error: err.Error(), Entries: []SandboxEntry{}}
	}
	out := make([]SandboxEntry, 0, len(entries))
	for _, e := range entries {
		out = append(out, SandboxEntry{Path: filepath.Join(path, e.Name()), IsDir: e.IsDir()})
	}
	return SandboxLsResult{Entries: out}
}

func (l *localSandbox) Glob(_ context.Context, pattern string, dir string) SandboxGlobResult {
	matches := []SandboxEntry{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		rel, relErr := filepath.Rel(dir, p)
		if relErr != nil || rel == "." {
			return nil
		}
		ok, matchErr := matchSandboxGlob(pattern, rel)
		if matchErr != nil {
			return matchErr
		}
		if ok {
			matches = append(matches, SandboxEntry{Path: rel, IsDir: d.IsDir()})
		}
		return nil
	})
	if err != nil {
		return SandboxGlobResult{Error: err.Error()}
	}
	return SandboxGlobResult{Matches: matches}
}

// matchSandboxGlob matches a slash-separated glob pattern against a relative
// path; the segment "**" matches any number of path segments. POSIX paths
// only (the suite assumes a POSIX userland).
func matchSandboxGlob(pattern string, name string) (bool, error) {
	return matchGlobSegments(strings.Split(pattern, "/"), strings.Split(name, "/"))
}

func matchGlobSegments(pat []string, name []string) (bool, error) {
	for len(pat) > 0 {
		if pat[0] == "**" {
			if len(pat) == 1 {
				return true, nil
			}
			for i := 0; i <= len(name); i++ {
				ok, err := matchGlobSegments(pat[1:], name[i:])
				if err != nil {
					return false, err
				}
				if ok {
					return true, nil
				}
			}
			return false, nil
		}
		if len(name) == 0 {
			return false, nil
		}
		ok, err := path.Match(pat[0], name[0])
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
		pat = pat[1:]
		name = name[1:]
	}
	return len(name) == 0, nil
}

func (l *localSandbox) Grep(_ context.Context, pattern string, dir string, glob string) SandboxGrepResult {
	matches := []SandboxGrepMatch{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if glob != "" {
			ok, matchErr := path.Match(glob, d.Name())
			if matchErr != nil {
				return matchErr
			}
			if !ok {
				return nil
			}
		}
		data, readErr := os.ReadFile(p)
		if readErr != nil || !utf8.Valid(data) {
			return nil // skip unreadable and binary files
		}
		for i, line := range strings.Split(string(data), "\n") {
			if strings.Contains(line, pattern) {
				matches = append(matches, SandboxGrepMatch{Path: p, Line: i + 1, Text: line})
			}
		}
		return nil
	})
	if err != nil {
		return SandboxGrepResult{Error: err.Error()}
	}
	return SandboxGrepResult{Matches: matches}
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

func (l *localSandbox) DownloadFiles(_ context.Context, paths []string) []SandboxDownloadResponse {
	responses := make([]SandboxDownloadResponse, 0, len(paths))
	for _, p := range paths {
		resp := SandboxDownloadResponse{Path: p}
		if !filepath.IsAbs(p) {
			resp.Error = "invalid_path"
			responses = append(responses, resp)
			continue
		}
		info, err := os.Stat(p)
		switch {
		case errors.Is(err, fs.ErrNotExist):
			resp.Error = "file_not_found"
		case err != nil && os.IsPermission(err):
			resp.Error = "permission_denied"
		case err != nil:
			resp.Error = err.Error()
		case info.IsDir():
			resp.Error = "is_directory"
		default:
			data, readErr := os.ReadFile(p)
			switch {
			case readErr != nil && os.IsPermission(readErr):
				resp.Error = "permission_denied"
			case readErr != nil:
				resp.Error = readErr.Error()
			default:
				resp.Content = data
			}
		}
		responses = append(responses, resp)
	}
	return responses
}

// TestRunSandboxConformanceWithLocalSandbox wires the conformance suite to
// the local sandbox adapter, mirroring how Python implementers subclass
// SandboxIntegrationTests with a sandbox fixture.
func TestRunSandboxConformanceWithLocalSandbox(t *testing.T) {
	RunSandboxConformance(t, newLocalSandbox)
}
