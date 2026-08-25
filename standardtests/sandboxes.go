package standardtests

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
)

// SandboxWriteResult mirrors deepagents' WriteResult: Error is empty on
// success.
type SandboxWriteResult struct {
	Path  string
	Error string
}

// SandboxFileData mirrors deepagents' file_data payload. Encoding is "utf-8"
// for text content and "base64" for binary content.
type SandboxFileData struct {
	Content  string
	Encoding string
}

// SandboxReadResult mirrors deepagents' ReadResult.
type SandboxReadResult struct {
	Error    string
	FileData *SandboxFileData
}

// SandboxReadOptions carries the optional line window for Read. Offset skips
// that many leading lines; Limit caps the number of returned lines (nil means
// no limit; a pointer to 0 returns no content lines).
type SandboxReadOptions struct {
	Offset int
	Limit  *int
}

// SandboxEditResult mirrors deepagents' EditResult. Occurrences is the number
// of replacements made.
type SandboxEditResult struct {
	Error       string
	Occurrences int
}

// SandboxEntry is one ls/glob entry. Path is absolute for Ls and relative to
// the searched directory for Glob, mirroring deepagents.
type SandboxEntry struct {
	Path  string
	IsDir bool
}

// SandboxLsResult mirrors deepagents' LsResult.
type SandboxLsResult struct {
	Error   string
	Entries []SandboxEntry
}

// SandboxGlobResult mirrors deepagents' GlobResult.
type SandboxGlobResult struct {
	Error   string
	Matches []SandboxEntry
}

// SandboxGrepMatch is one literal grep hit with a 1-based line number.
type SandboxGrepMatch struct {
	Path string
	Line int
	Text string
}

// SandboxGrepResult mirrors deepagents' GrepResult.
type SandboxGrepResult struct {
	Error   string
	Matches []SandboxGrepMatch
}

// SandboxFileUpload is one (path, content) upload pair.
type SandboxFileUpload struct {
	Path    string
	Content []byte
}

// SandboxUploadResponse mirrors deepagents' FileUploadResponse.
type SandboxUploadResponse struct {
	Path  string
	Error string
}

// SandboxDownloadResponse mirrors deepagents' FileDownloadResponse. Content
// is nil when Error is set.
type SandboxDownloadResponse struct {
	Path    string
	Content []byte
	Error   string
}

// SandboxExecuteResponse mirrors deepagents' ExecuteResponse.
type SandboxExecuteResponse struct {
	Output    string
	ExitCode  int
	Truncated bool
}

// SandboxBackend is the Go port of deepagents' SandboxBackendProtocol: the
// sandbox surface a backend must expose to pass the conformance suite. Go has
// no sync/async split (a declared divergence), so the protocol collapses to
// one method set.
type SandboxBackend interface {
	Write(ctx context.Context, path string, content string) SandboxWriteResult
	Read(ctx context.Context, path string, opts SandboxReadOptions) SandboxReadResult
	Edit(ctx context.Context, path string, old string, new string, replaceAll bool) SandboxEditResult
	Ls(ctx context.Context, path string) SandboxLsResult
	Glob(ctx context.Context, pattern string, path string) SandboxGlobResult
	Grep(ctx context.Context, pattern string, path string, glob string) SandboxGrepResult
	UploadFiles(ctx context.Context, files []SandboxFileUpload) []SandboxUploadResponse
	DownloadFiles(ctx context.Context, paths []string) []SandboxDownloadResponse
	Execute(ctx context.Context, command string) SandboxExecuteResponse
}

// SandboxFactory creates a clean sandbox backend and returns it together with
// the absolute root directory the suite may use for file operations (the Go
// equivalent of the Python suite's sandbox fixture plus sandbox_root_dir).
type SandboxFactory func(t testing.TB) (SandboxBackend, string)

// sandboxPath joins rel onto root the way the Python suite's sandbox_path
// helper does.
func sandboxPath(root string, rel string) string {
	return strings.TrimRight(root, "/") + "/" + strings.TrimLeft(rel, "/")
}

// shellQuote single-quotes s for POSIX shells, mirroring shlex.quote for the
// inputs this suite produces.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// intPtr returns a pointer to i, for SandboxReadOptions.Limit.
func intPtr(i int) *int { return &i }

// numberedLines renders format for i in [from, to] and joins with newlines.
func numberedLines(format string, from, to int) string {
	lines := make([]string, 0, to-from+1)
	for i := from; i <= to; i++ {
		lines = append(lines, fmt.Sprintf(format, i))
	}
	return strings.Join(lines, "\n")
}

// requireNoSandboxError fails the test if err is non-empty.
func requireNoSandboxError(t *testing.T, op string, err string) {
	t.Helper()
	if err != "" {
		t.Fatalf("%s: unexpected error: %s", op, err)
	}
}

// requireSandboxErrorContaining fails the test unless err contains at least
// one of the substrings (case-insensitive).
func requireSandboxErrorContaining(t *testing.T, op string, err string, substrings ...string) {
	t.Helper()
	lower := strings.ToLower(err)
	for _, sub := range substrings {
		if strings.Contains(lower, sub) {
			return
		}
	}
	t.Fatalf("%s: expected an error containing one of %v, got %q", op, substrings, err)
}

// requireContains fails the test unless haystack contains every needle.
func requireContains(t *testing.T, what string, haystack string, needles ...string) {
	t.Helper()
	for _, n := range needles {
		if !strings.Contains(haystack, n) {
			t.Fatalf("%s: %q does not contain %q", what, haystack, n)
		}
	}
}

// requireNotContains fails the test if haystack contains any needle.
func requireNotContains(t *testing.T, what string, haystack string, needles ...string) {
	t.Helper()
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			t.Fatalf("%s: %q unexpectedly contains %q", what, haystack, n)
		}
	}
}

// requireFileData fails the test unless res carries file data with no error,
// and returns the data.
func requireFileData(t *testing.T, op string, res SandboxReadResult) *SandboxFileData {
	t.Helper()
	requireNoSandboxError(t, op, res.Error)
	if res.FileData == nil {
		t.Fatalf("%s: file_data is nil", op)
	}
	return res.FileData
}

// requireContainsPath fails the test unless entries contains every path.
func requireContainsPath(t *testing.T, what string, entries []SandboxEntry, paths ...string) {
	t.Helper()
	set := map[string]bool{}
	for _, e := range entries {
		set[e.Path] = true
	}
	for _, p := range paths {
		if !set[p] {
			t.Fatalf("%s: %q not found in %v", what, p, entries)
		}
	}
}

// requireEmptyOrNotice fails the test unless content is empty/blank or
// contains an emptiness notice (mirrors test_read_empty_file).
func requireEmptyOrNotice(t *testing.T, what string, content string) {
	t.Helper()
	if strings.Contains(strings.ToLower(content), "empty") || strings.TrimSpace(content) == "" {
		return
	}
	t.Fatalf("%s: expected empty content or an emptiness notice, got %q", what, content)
}

// RunSandboxConformance verifies a SandboxBackend implementation against the
// sandbox behaviors the Go port exposes. It mirrors the sync tests of
// SandboxIntegrationTests in
// libs/standard-tests/langchain_tests/integration_tests/sandboxes.py, scoped
// per the parity spec: the async variants (no sync/async duality in Go) and
// test_execute_capture_at_source_offload (requires deepagents'
// create_deep_agent/CompositeBackend, which the Go port does not have) are
// out of scope. Like the Python suite, it assumes a POSIX userland (sh, cat,
// wc, awk, seq, chmod) inside the sandbox. The 512000-byte binary preview
// limit and its exact error message are part of the conformance contract
// (sandboxes.py test_read_binary_file_1_mib_returns_error).
func RunSandboxConformance(t *testing.T, factory SandboxFactory) {
	t.Helper()
	ctx := context.Background()

	t.Run("write new file", func(t *testing.T) { // test_write_new_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "new_file.txt")
		content := "Hello, sandbox!\nLine 2\nLine 3"
		result := backend.Write(ctx, testPath, content)
		requireNoSandboxError(t, "write", result.Error)
		requireEqual(t, "write path", result.Path, testPath)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), content)
	})

	t.Run("write creates parent dirs", func(t *testing.T) { // test_write_creates_parent_dirs
		backend, root := factory(t)
		testPath := sandboxPath(root, "deep/nested/dir/file.txt")
		content := "Nested file content"
		result := backend.Write(ctx, testPath, content)
		requireNoSandboxError(t, "write", result.Error)
		requireEqual(t, "write path", result.Path, testPath)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), content)
	})

	t.Run("write existing file fails", func(t *testing.T) { // test_write_existing_file_fails
		backend, root := factory(t)
		testPath := sandboxPath(root, "existing.txt")
		first := backend.Write(ctx, testPath, "First content")
		requireNoSandboxError(t, "first write", first.Error)
		result := backend.Write(ctx, testPath, "Second content")
		requireSandboxErrorContaining(t, "second write", result.Error, "already exists")
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), "First content")
	})

	t.Run("write empty file", func(t *testing.T) { // test_write_empty_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "empty.txt")
		result := backend.Write(ctx, testPath, "")
		requireNoSandboxError(t, "write", result.Error)
		execResult := backend.Execute(ctx, "[ -f "+shellQuote(testPath)+" ] && echo exists || echo missing")
		requireContains(t, "probe output", execResult.Output, "exists")
	})

	t.Run("write path with spaces", func(t *testing.T) { // test_write_path_with_spaces
		backend, root := factory(t)
		testPath := sandboxPath(root, "dir with spaces/file name.txt")
		content := "Content in file with spaces"
		result := backend.Write(ctx, testPath, content)
		requireNoSandboxError(t, "write", result.Error)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), content)
	})

	t.Run("write unicode content", func(t *testing.T) { // test_write_unicode_content
		backend, root := factory(t)
		testPath := sandboxPath(root, "unicode.txt")
		content := "Hello 👋 世界 مرحبا Привет 🌍\nLine with émojis 🎉"
		result := backend.Write(ctx, testPath, content)
		requireNoSandboxError(t, "write", result.Error)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), content)
	})

	t.Run("write consecutive slashes in path", func(t *testing.T) { // test_write_consecutive_slashes_in_path
		backend, root := factory(t)
		// Python's sandbox_path normalizes consecutive slashes away; the Go
		// sandboxPath helper does the same, so this is a plain write roundtrip.
		testPath := sandboxPath(root, "file.txt")
		content := "Content"
		result := backend.Write(ctx, testPath, content)
		requireNoSandboxError(t, "write", result.Error)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), content)
	})

	t.Run("write special characters", func(t *testing.T) { // test_write_special_characters
		backend, root := factory(t)
		testPath := sandboxPath(root, "special.txt")
		content := "Special chars: $VAR, `command`, $(subshell), 'quotes', \"quotes\"\n" +
			"Tab\there\n" +
			"Backslash: \\\\"
		result := backend.Write(ctx, testPath, content)
		requireNoSandboxError(t, "write", result.Error)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), content)
	})

	t.Run("write content with only newlines", func(t *testing.T) { // test_write_content_with_only_newlines
		backend, root := factory(t)
		testPath := sandboxPath(root, "only_newlines.txt")
		result := backend.Write(ctx, testPath, "\n\n\n\n\n")
		requireNoSandboxError(t, "write", result.Error)
		execResult := backend.Execute(ctx, "wc -l "+shellQuote(testPath))
		requireContains(t, "wc output", execResult.Output, "5")
	})

	t.Run("execute large stdout", func(t *testing.T) { // test_execute_large_stdout_payload (awk instead of python -c; same 500 KiB payload)
		backend, _ := factory(t)
		result := backend.Execute(ctx, `awk 'BEGIN{for(i=0;i<500*1024;i++)printf "x"}'`)
		requireEqual(t, "exit code", result.ExitCode, 0)
		if result.Truncated {
			t.Fatalf("execute large stdout: output unexpectedly truncated")
		}
		if len(result.Output) < 500*1024 {
			t.Fatalf("output length: got %d want >= %d", len(result.Output), 500*1024)
		}
		requireContains(t, "output prefix", result.Output[:1], "x")
	})

	t.Run("read basic file", func(t *testing.T) { // test_read_basic_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "read_test.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Line 1\nLine 2\nLine 3").Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "read content", data.Content, "Line 1", "Line 2", "Line 3")
	})

	t.Run("read unicode content", func(t *testing.T) { // test_read_unicode_content
		backend, root := factory(t)
		testPath := sandboxPath(root, "unicode_read.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Hello 👋 世界\nПривет мир\nمرحبا العالم").Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "read content", data.Content, "👋", "世界", "Привет")
	})

	t.Run("read empty file", func(t *testing.T) { // test_read_empty_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "empty_read.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "").Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireEmptyOrNotice(t, "empty file content", data.Content)
	})

	t.Run("read nonexistent file", func(t *testing.T) { // test_read_nonexistent_file
		backend, root := factory(t)
		result := backend.Read(ctx, sandboxPath(root, "nonexistent.txt"), SandboxReadOptions{})
		requireSandboxErrorContaining(t, "read", result.Error, "not_found", "not found")
	})

	t.Run("read path is sanitized", func(t *testing.T) { // test_read_path_is_sanitized
		backend, _ := factory(t)
		malicious := "'; import os; os.system('echo INJECTED'); #"
		result := backend.Read(ctx, malicious, SandboxReadOptions{})
		if result.Error == "" {
			t.Fatalf("read of injected path: expected an error, got none")
		}
		if result.FileData != nil {
			t.Fatalf("read of injected path: expected nil file_data, got %#v", result.FileData)
		}
	})

	t.Run("read binary file", func(t *testing.T) { // test_read_binary_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "binary.png")
		raw := make([]byte, 256)
		for i := range raw {
			raw[i] = byte(i)
		}
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: raw}})
		requireLen(t, "upload responses", uploads, 1)
		requireNoSandboxError(t, "upload", uploads[0].Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireEqual(t, "encoding", data.Encoding, "base64")
		decoded, err := base64.StdEncoding.DecodeString(data.Content)
		requireNoErr(t, "base64 decode", err)
		requireDeepEqual(t, "decoded content", decoded, raw)
	})

	t.Run("read binary file 100 kib", func(t *testing.T) { // test_read_binary_file_100_kib
		backend, root := factory(t)
		testPath := sandboxPath(root, "binary_100kib.png")
		chunk := make([]byte, 256)
		for i := range chunk {
			chunk[i] = byte(i)
		}
		raw := bytes.Repeat(chunk, 400)
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: raw}})
		requireLen(t, "upload responses", uploads, 1)
		requireNoSandboxError(t, "upload", uploads[0].Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireEqual(t, "encoding", data.Encoding, "base64")
		decoded, err := base64.StdEncoding.DecodeString(data.Content)
		requireNoErr(t, "base64 decode", err)
		requireDeepEqual(t, "decoded content", decoded, raw)
	})

	t.Run("read binary file 1 mib returns error", func(t *testing.T) { // test_read_binary_file_1_mib_returns_error
		backend, root := factory(t)
		testPath := sandboxPath(root, "binary_1mib.png")
		chunk := make([]byte, 256)
		for i := range chunk {
			chunk[i] = byte(i)
		}
		raw := bytes.Repeat(chunk, 4096)
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: raw}})
		requireLen(t, "upload responses", uploads, 1)
		requireNoSandboxError(t, "upload", uploads[0].Error)
		result := backend.Read(ctx, testPath, SandboxReadOptions{})
		if result.FileData != nil {
			t.Fatalf("read of 1 MiB binary: expected nil file_data, got %#v", result.FileData)
		}
		requireEqual(t, "error", result.Error,
			"File '"+testPath+"': Binary file exceeds maximum preview size of 512000 bytes")
	})

	t.Run("read with offset", func(t *testing.T) { // test_read_with_offset
		backend, root := factory(t)
		testPath := sandboxPath(root, "offset_test.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, numberedLines("Row_%d_content", 1, 10)).Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{Offset: 5}))
		requireContains(t, "content", data.Content, "Row_6_content")
		requireNotContains(t, "content", data.Content, "Row_1_content")
	})

	t.Run("read with limit", func(t *testing.T) { // test_read_with_limit
		backend, root := factory(t)
		testPath := sandboxPath(root, "limit_test.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, numberedLines("Row_%d_content", 1, 100)).Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{Limit: intPtr(5)}))
		requireContains(t, "content", data.Content, "Row_1_content", "Row_5_content")
		requireNotContains(t, "content", data.Content, "Row_6_content")
	})

	t.Run("read with offset and limit", func(t *testing.T) { // test_read_with_offset_and_limit
		backend, root := factory(t)
		testPath := sandboxPath(root, "offset_limit_test.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, numberedLines("Row_%d_content", 1, 20)).Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{Offset: 10, Limit: intPtr(5)}))
		requireContains(t, "content", data.Content, "Row_11_content", "Row_15_content")
		requireNotContains(t, "content", data.Content, "Row_10_content", "Row_16_content")
	})

	t.Run("read with zero limit", func(t *testing.T) { // test_read_with_zero_limit
		backend, root := factory(t)
		testPath := sandboxPath(root, "zero_limit.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Line 1\nLine 2\nLine 3").Error)
		result := backend.Read(ctx, testPath, SandboxReadOptions{Limit: intPtr(0)})
		content := ""
		if result.FileData != nil {
			content = result.FileData.Content
		}
		requireNotContains(t, "zero-limit content", content, "Line 1")
	})

	t.Run("read offset beyond file length", func(t *testing.T) { // test_read_offset_beyond_file_length
		backend, root := factory(t)
		testPath := sandboxPath(root, "offset_beyond.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Line 1\nLine 2\nLine 3").Error)
		result := backend.Read(ctx, testPath, SandboxReadOptions{Offset: 100, Limit: intPtr(10)})
		content := ""
		if result.FileData != nil {
			content = result.FileData.Content
		}
		requireNotContains(t, "content", content, "Line 1", "Line 2", "Line 3")
		requireNotContains(t, "error", result.Error, "Line 1", "Line 2", "Line 3")
	})

	t.Run("read offset at exact file length", func(t *testing.T) { // test_read_offset_at_exact_file_length
		backend, root := factory(t)
		testPath := sandboxPath(root, "offset_exact.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, numberedLines("Line %d", 1, 5)).Error)
		result := backend.Read(ctx, testPath, SandboxReadOptions{Offset: 5, Limit: intPtr(10)})
		content := ""
		if result.FileData != nil {
			content = result.FileData.Content
		}
		requireNotContains(t, "content", content, "Line 1", "Line 5")
		requireNotContains(t, "error", result.Error, "Line 1", "Line 5")
	})

	t.Run("read very large file in chunks", func(t *testing.T) { // test_read_very_large_file_in_chunks
		backend, root := factory(t)
		testPath := sandboxPath(root, "large_chunked.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, numberedLines("Line_%04d_content", 0, 999)).Error)
		first := requireFileData(t, "read first chunk", backend.Read(ctx, testPath, SandboxReadOptions{Offset: 0, Limit: intPtr(100)}))
		requireContains(t, "first chunk", first.Content, "Line_0000_content", "Line_0099_content")
		requireNotContains(t, "first chunk", first.Content, "Line_0100_content")
		middle := requireFileData(t, "read middle chunk", backend.Read(ctx, testPath, SandboxReadOptions{Offset: 500, Limit: intPtr(100)}))
		requireContains(t, "middle chunk", middle.Content, "Line_0500_content", "Line_0599_content")
		requireNotContains(t, "middle chunk", middle.Content, "Line_0499_content")
		last := requireFileData(t, "read last chunk", backend.Read(ctx, testPath, SandboxReadOptions{Offset: 900, Limit: intPtr(100)}))
		requireContains(t, "last chunk", last.Content, "Line_0900_content", "Line_0999_content")
	})

	t.Run("read file with very long lines", func(t *testing.T) { // test_read_file_with_very_long_lines
		backend, root := factory(t)
		testPath := sandboxPath(root, "long_lines.txt")
		content := "Short line\n" + strings.Repeat("x", 3000) + "\nAnother short line"
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, content).Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "content", data.Content, "Short line")
	})

	t.Run("write very long content", func(t *testing.T) { // test_write_very_long_content
		backend, root := factory(t)
		testPath := sandboxPath(root, "very_long.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, numberedLines("Line %d with some content here", 0, 999)).Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "content", data.Content, "Line 0 with some content here")
	})
}
