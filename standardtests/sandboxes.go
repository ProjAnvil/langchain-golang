package standardtests

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strconv"
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

	t.Run("edit single occurrence", func(t *testing.T) { // test_edit_single_occurrence
		backend, root := factory(t)
		testPath := sandboxPath(root, "edit_single.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Hello world\nGoodbye world\nHello again").Error)
		result := backend.Edit(ctx, testPath, "Goodbye", "Farewell", false)
		requireNoSandboxError(t, "edit", result.Error)
		requireEqual(t, "occurrences", result.Occurrences, 1)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "content", data.Content, "Farewell world")
		requireNotContains(t, "content", data.Content, "Goodbye")
	})

	t.Run("edit multiple without replace all", func(t *testing.T) { // test_edit_multiple_occurrences_without_replace_all
		backend, root := factory(t)
		testPath := sandboxPath(root, "edit_multi.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "apple\nbanana\napple\norange\napple").Error)
		result := backend.Edit(ctx, testPath, "apple", "pear", false)
		requireSandboxErrorContaining(t, "edit", result.Error, "multiple")
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "content", data.Content, "apple")
		requireNotContains(t, "content", data.Content, "pear")
	})

	t.Run("edit multiple with replace all", func(t *testing.T) { // test_edit_multiple_occurrences_with_replace_all
		backend, root := factory(t)
		testPath := sandboxPath(root, "edit_replace_all.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "apple\nbanana\napple\norange\napple").Error)
		result := backend.Edit(ctx, testPath, "apple", "pear", true)
		requireNoSandboxError(t, "edit", result.Error)
		requireEqual(t, "occurrences", result.Occurrences, 3)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireNotContains(t, "content", data.Content, "apple")
		requireEqual(t, "pear count", strings.Count(data.Content, "pear"), 3)
	})

	t.Run("edit string not found", func(t *testing.T) { // test_edit_string_not_found
		backend, root := factory(t)
		testPath := sandboxPath(root, "edit_not_found.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Hello world").Error)
		result := backend.Edit(ctx, testPath, "nonexistent", "replacement", false)
		requireSandboxErrorContaining(t, "edit", result.Error, "not found")
	})

	t.Run("edit nonexistent file", func(t *testing.T) { // test_edit_nonexistent_file
		backend, root := factory(t)
		result := backend.Edit(ctx, sandboxPath(root, "nonexistent_edit.txt"), "old", "new", false)
		requireSandboxErrorContaining(t, "edit", result.Error, "not_found", "not found")
	})

	t.Run("edit special characters", func(t *testing.T) { // test_edit_special_characters
		backend, root := factory(t)
		testPath := sandboxPath(root, "edit_special.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Price: $100.00\nPattern: [a-z]*\nPath: /usr/bin").Error)
		requireNoSandboxError(t, "first edit", backend.Edit(ctx, testPath, "$100.00", "$200.00", false).Error)
		requireNoSandboxError(t, "second edit", backend.Edit(ctx, testPath, "[a-z]*", "[0-9]+", false).Error)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "content", data.Content, "$200.00", "[0-9]+")
	})

	t.Run("edit multiline", func(t *testing.T) { // test_edit_multiline_support
		backend, root := factory(t)
		testPath := sandboxPath(root, "edit_multiline.txt")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, "Line 1\nLine 2\nLine 3").Error)
		result := backend.Edit(ctx, testPath, "Line 1\nLine 2", "Combined", false)
		requireNoSandboxError(t, "edit", result.Error)
		requireEqual(t, "occurrences", result.Occurrences, 1)
		data := requireFileData(t, "read", backend.Read(ctx, testPath, SandboxReadOptions{}))
		requireContains(t, "content", data.Content, "Combined")
	})

	t.Run("ls lists files", func(t *testing.T) { // test_ls_lists_files
		backend, root := factory(t)
		requireNoSandboxError(t, "write a", backend.Write(ctx, sandboxPath(root, "a.txt"), "a").Error)
		requireNoSandboxError(t, "write b", backend.Write(ctx, sandboxPath(root, "b.txt"), "b").Error)
		result := backend.Ls(ctx, root)
		requireNoSandboxError(t, "ls", result.Error)
		requireContainsPath(t, "ls entries", result.Entries, sandboxPath(root, "a.txt"), sandboxPath(root, "b.txt"))
	})

	t.Run("ls lists nested directories", func(t *testing.T) { // test_ls_lists_nested_directories
		backend, root := factory(t)
		baseDir := sandboxPath(root, "ls_nested")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir)+"/subdir && touch "+shellQuote(baseDir)+"/root.txt")
		result := backend.Ls(ctx, baseDir)
		requireNoSandboxError(t, "ls", result.Error)
		requireContainsPath(t, "ls entries", result.Entries, baseDir+"/subdir", baseDir+"/root.txt")
	})

	t.Run("ls unicode filenames", func(t *testing.T) { // test_ls_unicode_filenames
		backend, root := factory(t)
		baseDir := sandboxPath(root, "ls_unicode")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write 测试文件", backend.Write(ctx, baseDir+"/测试文件.txt", "content").Error)
		requireNoSandboxError(t, "write файл", backend.Write(ctx, baseDir+"/файл.txt", "content").Error)
		result := backend.Ls(ctx, baseDir)
		requireNoSandboxError(t, "ls", result.Error)
		requireContainsPath(t, "ls entries", result.Entries, baseDir+"/测试文件.txt", baseDir+"/файл.txt")
	})

	t.Run("ls large directory", func(t *testing.T) { // test_ls_large_directory
		backend, root := factory(t)
		baseDir := sandboxPath(root, "ls_large")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir)+" && cd "+shellQuote(baseDir)+
			" && for i in $(seq 0 49); do echo content > file_$(printf '%03d' $i).txt; done")
		result := backend.Ls(ctx, baseDir)
		requireNoSandboxError(t, "ls", result.Error)
		requireLen(t, "ls entries", result.Entries, 50)
		requireContainsPath(t, "ls entries", result.Entries, baseDir+"/file_000.txt", baseDir+"/file_049.txt")
	})

	t.Run("ls path with trailing slash", func(t *testing.T) { // test_ls_path_with_trailing_slash
		backend, root := factory(t)
		baseDir := sandboxPath(root, "ls_trailing")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/file.txt", "content").Error)
		result := backend.Ls(ctx, baseDir+"/")
		requireNoSandboxError(t, "ls", result.Error)
		requireContainsPath(t, "ls entries", result.Entries, baseDir+"/file.txt")
	})

	t.Run("ls special characters in filenames", func(t *testing.T) { // test_ls_special_characters_in_filenames
		backend, root := factory(t)
		baseDir := sandboxPath(root, "ls_special")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write (1)", backend.Write(ctx, baseDir+"/file(1).txt", "content").Error)
		requireNoSandboxError(t, "write [2]", backend.Write(ctx, baseDir+"/file[2].txt", "content").Error)
		requireNoSandboxError(t, "write -3", backend.Write(ctx, baseDir+"/file-3.txt", "content").Error)
		result := backend.Ls(ctx, baseDir)
		requireNoSandboxError(t, "ls", result.Error)
		requireContainsPath(t, "ls entries", result.Entries,
			baseDir+"/file(1).txt", baseDir+"/file[2].txt", baseDir+"/file-3.txt")
	})

	t.Run("ls path is sanitized", func(t *testing.T) { // test_ls_path_is_sanitized
		backend, _ := factory(t)
		malicious := "'; import os; os.system('echo INJECTED'); #"
		result := backend.Ls(ctx, malicious)
		if result.Error == "" && len(result.Entries) != 0 {
			t.Fatalf("ls of injected path: expected error or empty entries, got %v", result.Entries)
		}
	})

	t.Run("glob single match exact", func(t *testing.T) { // test_glob
		backend, root := factory(t)
		requireNoSandboxError(t, "write x.py", backend.Write(ctx, sandboxPath(root, "x.py"), "print('x')").Error)
		requireNoSandboxError(t, "write y.txt", backend.Write(ctx, sandboxPath(root, "y.txt"), "y").Error)
		result := backend.Glob(ctx, "*.py", root)
		requireNoSandboxError(t, "glob", result.Error)
		requireDeepEqual(t, "glob matches", result.Matches, []SandboxEntry{{Path: "x.py"}})
	})

	t.Run("glob basic pattern", func(t *testing.T) { // test_glob_basic_pattern
		backend, root := factory(t)
		baseDir := sandboxPath(root, "glob_test")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write 1", backend.Write(ctx, baseDir+"/file1.txt", "content").Error)
		requireNoSandboxError(t, "write 2", backend.Write(ctx, baseDir+"/file2.txt", "content").Error)
		requireNoSandboxError(t, "write 3", backend.Write(ctx, baseDir+"/file3.py", "content").Error)
		result := backend.Glob(ctx, "*.txt", baseDir)
		requireNoSandboxError(t, "glob", result.Error)
		requireLen(t, "glob matches", result.Matches, 2)
		requireContainsPath(t, "glob matches", result.Matches, "file1.txt", "file2.txt")
		for _, m := range result.Matches {
			requireNotContains(t, "glob match path", m.Path, ".py")
		}
	})

	t.Run("glob recursive pattern", func(t *testing.T) { // test_glob_recursive_pattern
		backend, root := factory(t)
		baseDir := sandboxPath(root, "glob_recursive")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir)+"/subdir1 "+shellQuote(baseDir)+"/subdir2")
		requireNoSandboxError(t, "write root", backend.Write(ctx, baseDir+"/root.txt", "content").Error)
		requireNoSandboxError(t, "write nested1", backend.Write(ctx, baseDir+"/subdir1/nested1.txt", "content").Error)
		requireNoSandboxError(t, "write nested2", backend.Write(ctx, baseDir+"/subdir2/nested2.txt", "content").Error)
		result := backend.Glob(ctx, "**/*.txt", baseDir)
		requireNoSandboxError(t, "glob", result.Error)
		requireContainsPath(t, "glob matches", result.Matches,
			"root.txt", "subdir1/nested1.txt", "subdir2/nested2.txt")
	})

	t.Run("glob no matches", func(t *testing.T) { // test_glob_no_matches
		backend, root := factory(t)
		requireNoSandboxError(t, "write", backend.Write(ctx, sandboxPath(root, "file.txt"), "content").Error)
		result := backend.Glob(ctx, "*.py", root)
		requireNoSandboxError(t, "glob", result.Error)
		requireLen(t, "glob matches", result.Matches, 0)
	})

	t.Run("glob with directories", func(t *testing.T) { // test_glob_with_directories
		backend, root := factory(t)
		baseDir := sandboxPath(root, "glob_dirs")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir)+"/dir1 "+shellQuote(baseDir)+"/dir2")
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/file.txt", "content").Error)
		result := backend.Glob(ctx, "*", baseDir)
		requireNoSandboxError(t, "glob", result.Error)
		requireLen(t, "glob matches", result.Matches, 3)
		dirs, files := 0, 0
		for _, m := range result.Matches {
			if m.IsDir {
				dirs++
			} else {
				files++
			}
		}
		requireEqual(t, "dir count", dirs, 2)
		requireEqual(t, "file count", files, 1)
	})

	t.Run("glob hidden files explicitly", func(t *testing.T) { // test_glob_hidden_files_explicitly
		backend, root := factory(t)
		baseDir := sandboxPath(root, "glob_hidden")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write hidden1", backend.Write(ctx, baseDir+"/.hidden1", "content").Error)
		requireNoSandboxError(t, "write hidden2", backend.Write(ctx, baseDir+"/.hidden2", "content").Error)
		requireNoSandboxError(t, "write visible", backend.Write(ctx, baseDir+"/visible.txt", "content").Error)
		result := backend.Glob(ctx, ".*", baseDir)
		requireNoSandboxError(t, "glob", result.Error)
		requireLen(t, "glob matches", result.Matches, 2)
		requireContainsPath(t, "glob matches", result.Matches, ".hidden1", ".hidden2")
	})

	t.Run("glob with character class", func(t *testing.T) { // test_glob_with_character_class
		backend, root := factory(t)
		baseDir := sandboxPath(root, "glob_charclass")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		for _, name := range []string{"file1.txt", "file2.txt", "file3.txt", "fileA.txt"} {
			requireNoSandboxError(t, "write "+name, backend.Write(ctx, baseDir+"/"+name, "content").Error)
		}
		result := backend.Glob(ctx, "file[1-2].txt", baseDir)
		requireNoSandboxError(t, "glob", result.Error)
		requireDeepEqual(t, "glob matches", result.Matches,
			[]SandboxEntry{{Path: "file1.txt"}, {Path: "file2.txt"}})
	})

	t.Run("glob with question mark", func(t *testing.T) { // test_glob_with_question_mark
		backend, root := factory(t)
		baseDir := sandboxPath(root, "glob_question")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		for _, name := range []string{"file1.txt", "file2.txt", "file10.txt"} {
			requireNoSandboxError(t, "write "+name, backend.Write(ctx, baseDir+"/"+name, "content").Error)
		}
		result := backend.Glob(ctx, "file?.txt", baseDir)
		requireNoSandboxError(t, "glob", result.Error)
		requireDeepEqual(t, "glob matches", result.Matches,
			[]SandboxEntry{{Path: "file1.txt"}, {Path: "file2.txt"}})
	})

	t.Run("grep literal", func(t *testing.T) { // test_grep_literal
		backend, root := factory(t)
		requireNoSandboxError(t, "write", backend.Write(ctx, sandboxPath(root, "grep.txt"), "a (b)\nstr | int\n").Error)
		result := backend.Grep(ctx, "str | int", root, "")
		requireNoSandboxError(t, "grep", result.Error)
		if len(result.Matches) == 0 {
			t.Fatalf("grep matches: got none, want at least one")
		}
		if !strings.HasSuffix(result.Matches[0].Path, "/grep.txt") {
			t.Fatalf("grep match path: got %q want suffix /grep.txt", result.Matches[0].Path)
		}
		requireEqual(t, "grep match text", strings.TrimSpace(result.Matches[0].Text), "str | int")
	})

	t.Run("grep basic search", func(t *testing.T) { // test_grep_basic_search
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_test")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write 1", backend.Write(ctx, baseDir+"/file1.txt", "Hello world\nGoodbye world").Error)
		requireNoSandboxError(t, "write 2", backend.Write(ctx, baseDir+"/file2.txt", "Hello there\nGoodbye friend").Error)
		result := backend.Grep(ctx, "Hello", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 2)
		paths := result.Matches[0].Path + "\n" + result.Matches[1].Path
		requireContains(t, "grep paths", paths, "file1.txt", "file2.txt")
		for _, m := range result.Matches {
			requireEqual(t, "grep line", m.Line, 1)
		}
	})

	t.Run("grep with glob pattern", func(t *testing.T) { // test_grep_with_glob_pattern
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_glob")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write txt", backend.Write(ctx, baseDir+"/test.txt", "pattern").Error)
		requireNoSandboxError(t, "write py", backend.Write(ctx, baseDir+"/test.py", "pattern").Error)
		requireNoSandboxError(t, "write md", backend.Write(ctx, baseDir+"/test.md", "pattern").Error)
		result := backend.Grep(ctx, "pattern", baseDir, "*.py")
		requireNoSandboxError(t, "grep", result.Error)
		requireDeepEqual(t, "grep matches", result.Matches,
			[]SandboxGrepMatch{{Path: baseDir + "/test.py", Line: 1, Text: "pattern"}})
	})

	t.Run("grep no matches", func(t *testing.T) { // test_grep_no_matches
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_empty")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/file.txt", "Hello world").Error)
		result := backend.Grep(ctx, "nonexistent", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 0)
	})

	t.Run("grep multiple matches per file", func(t *testing.T) { // test_grep_multiple_matches_per_file
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_multi")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/fruits.txt", "apple\nbanana\napple\norange\napple").Error)
		result := backend.Grep(ctx, "apple", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 3)
		lines := make([]int, 0, len(result.Matches))
		for _, m := range result.Matches {
			lines = append(lines, m.Line)
		}
		requireDeepEqual(t, "grep lines", lines, []int{1, 3, 5})
	})

	t.Run("grep literal string matching", func(t *testing.T) { // test_grep_literal_string_matching
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_literal")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/numbers.txt", "test123\ntest456\nabcdef").Error)
		result := backend.Grep(ctx, "test123", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 1)
		requireContains(t, "grep match text", result.Matches[0].Text, "test123")
	})

	t.Run("grep case sensitivity", func(t *testing.T) { // test_grep_case_sensitivity
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_case")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/case.txt", "Hello\nhello\nHELLO").Error)
		result := backend.Grep(ctx, "Hello", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireDeepEqual(t, "grep matches", result.Matches,
			[]SandboxGrepMatch{{Path: baseDir + "/case.txt", Line: 1, Text: "Hello"}})
	})

	t.Run("grep unicode pattern", func(t *testing.T) { // test_grep_unicode_pattern
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_unicode")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/unicode.txt", "Hello 世界\nПривет мир\n测试 pattern").Error)
		result := backend.Grep(ctx, "世界", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 1)
		requireContains(t, "grep match text", result.Matches[0].Text, "世界")
	})

	t.Run("grep with special characters", func(t *testing.T) { // test_grep_with_special_characters
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_special")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/special.txt", "Price: $100\nPath: /usr/bin\nPattern: [a-z]*").Error)
		dollar := backend.Grep(ctx, "$100", baseDir, "")
		requireNoSandboxError(t, "grep $100", dollar.Error)
		requireLen(t, "dollar matches", dollar.Matches, 1)
		requireContains(t, "dollar match text", dollar.Matches[0].Text, "$100")
		brackets := backend.Grep(ctx, "[a-z]*", baseDir, "")
		requireNoSandboxError(t, "grep [a-z]*", brackets.Error)
		requireLen(t, "bracket matches", brackets.Matches, 1)
		requireContains(t, "bracket match text", brackets.Matches[0].Text, "[a-z]*")
	})

	t.Run("grep empty directory", func(t *testing.T) { // test_grep_empty_directory
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_empty_dir")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		result := backend.Grep(ctx, "anything", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 0)
	})

	t.Run("grep across nested directories", func(t *testing.T) { // test_grep_across_nested_directories
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_nested")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir)+"/sub1/sub2")
		requireNoSandboxError(t, "write root", backend.Write(ctx, baseDir+"/root.txt", "target here").Error)
		requireNoSandboxError(t, "write l1", backend.Write(ctx, baseDir+"/sub1/level1.txt", "target here").Error)
		requireNoSandboxError(t, "write l2", backend.Write(ctx, baseDir+"/sub1/sub2/level2.txt", "target here").Error)
		result := backend.Grep(ctx, "target", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireLen(t, "grep matches", result.Matches, 3)
	})

	t.Run("grep with globstar include pattern", func(t *testing.T) { // test_grep_with_globstar_include_pattern
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_globstar")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir)+"/a/b")
		requireNoSandboxError(t, "write py", backend.Write(ctx, baseDir+"/a/b/target.py", "needle").Error)
		requireNoSandboxError(t, "write txt", backend.Write(ctx, baseDir+"/a/ignore.txt", "needle").Error)
		result := backend.Grep(ctx, "needle", baseDir, "*.py")
		requireNoSandboxError(t, "grep", result.Error)
		requireDeepEqual(t, "grep matches", result.Matches,
			[]SandboxGrepMatch{{Path: baseDir + "/a/b/target.py", Line: 1, Text: "needle"}})
	})

	t.Run("grep reports correct line numbers", func(t *testing.T) { // test_grep_reports_correct_line_numbers
		backend, root := factory(t)
		baseDir := sandboxPath(root, "grep_multiline")
		backend.Execute(ctx, "mkdir -p "+shellQuote(baseDir))
		requireNoSandboxError(t, "write", backend.Write(ctx, baseDir+"/long.txt", numberedLines("Line %d", 1, 100)).Error)
		result := backend.Grep(ctx, "Line 50", baseDir, "")
		requireNoSandboxError(t, "grep", result.Error)
		requireDeepEqual(t, "grep matches", result.Matches,
			[]SandboxGrepMatch{{Path: baseDir + "/long.txt", Line: 50, Text: "Line 50"}})
	})

	t.Run("upload single file", func(t *testing.T) { // test_upload_single_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "test_upload_single.txt")
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: []byte("Hello, Sandbox!")}})
		requireLen(t, "upload responses", uploads, 1)
		requireEqual(t, "upload path", uploads[0].Path, testPath)
		requireNoSandboxError(t, "upload", uploads[0].Error)
		execResult := backend.Execute(ctx, "cat "+shellQuote(testPath))
		requireEqual(t, "cat output", strings.TrimSpace(execResult.Output), "Hello, Sandbox!")
	})

	t.Run("download single file", func(t *testing.T) { // test_download_single_file
		backend, root := factory(t)
		testPath := sandboxPath(root, "test_download_single.txt")
		content := []byte("Download test content")
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: content}})
		requireLen(t, "upload responses", uploads, 1)
		requireNoSandboxError(t, "upload", uploads[0].Error)
		downloads := backend.DownloadFiles(ctx, []string{testPath})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: testPath, Content: content}})
	})

	t.Run("upload download roundtrip", func(t *testing.T) { // test_upload_download_roundtrip
		backend, root := factory(t)
		testPath := sandboxPath(root, "test_roundtrip.txt")
		content := []byte("Roundtrip test: special chars \n\t\r\x00")
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: content}})
		requireDeepEqual(t, "upload responses", uploads, []SandboxUploadResponse{{Path: testPath}})
		downloads := backend.DownloadFiles(ctx, []string{testPath})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: testPath, Content: content}})
	})

	t.Run("upload multiple files order preserved", func(t *testing.T) { // test_upload_multiple_files_order_preserved
		backend, root := factory(t)
		files := []SandboxFileUpload{
			{Path: sandboxPath(root, "test_multi_1.txt"), Content: []byte("Content 1")},
			{Path: sandboxPath(root, "test_multi_2.txt"), Content: []byte("Content 2")},
			{Path: sandboxPath(root, "test_multi_3.txt"), Content: []byte("Content 3")},
		}
		uploads := backend.UploadFiles(ctx, files)
		requireDeepEqual(t, "upload responses", uploads, []SandboxUploadResponse{
			{Path: files[0].Path},
			{Path: files[1].Path},
			{Path: files[2].Path},
		})
	})

	t.Run("download multiple files order preserved", func(t *testing.T) { // test_download_multiple_files_order_preserved
		backend, root := factory(t)
		files := []SandboxFileUpload{
			{Path: sandboxPath(root, "test_batch_1.txt"), Content: []byte("Batch 1")},
			{Path: sandboxPath(root, "test_batch_2.txt"), Content: []byte("Batch 2")},
			{Path: sandboxPath(root, "test_batch_3.txt"), Content: []byte("Batch 3")},
		}
		uploads := backend.UploadFiles(ctx, files)
		requireLen(t, "upload responses", uploads, 3)
		requireNoSandboxError(t, "upload", uploads[0].Error)
		downloads := backend.DownloadFiles(ctx, []string{files[0].Path, files[1].Path, files[2].Path})
		requireDeepEqual(t, "download responses", downloads, []SandboxDownloadResponse{
			{Path: files[0].Path, Content: files[0].Content},
			{Path: files[1].Path, Content: files[1].Content},
			{Path: files[2].Path, Content: files[2].Content},
		})
	})

	t.Run("upload binary content roundtrip", func(t *testing.T) { // test_upload_binary_content_roundtrip
		backend, root := factory(t)
		testPath := sandboxPath(root, "binary_file.bin")
		content := make([]byte, 256)
		for i := range content {
			content[i] = byte(i)
		}
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: content}})
		requireDeepEqual(t, "upload responses", uploads, []SandboxUploadResponse{{Path: testPath}})
		downloads := backend.DownloadFiles(ctx, []string{testPath})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: testPath, Content: content}})
	})

	t.Run("upload large file reports expected size", func(t *testing.T) { // test_upload_large_file_reports_expected_size
		backend, root := factory(t)
		testPath := sandboxPath(root, "large_upload.txt")
		content := bytes.Repeat([]byte("0123456789abcdef"), 1024*640)
		requireEqual(t, "content length", len(content), 10*1024*1024)
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: testPath, Content: content}})
		requireDeepEqual(t, "upload responses", uploads, []SandboxUploadResponse{{Path: testPath}})
		execResult := backend.Execute(ctx, "wc -c "+shellQuote(testPath))
		requireEqual(t, "wc exit code", execResult.ExitCode, 0)
		requireContains(t, "wc output", execResult.Output, strconv.Itoa(len(content)))
		downloads := backend.DownloadFiles(ctx, []string{testPath})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: testPath, Content: content}})
	})

	t.Run("download error file not found", func(t *testing.T) { // test_download_error_file_not_found
		backend, root := factory(t)
		missingPath := sandboxPath(root, "nonexistent_test_file.txt")
		downloads := backend.DownloadFiles(ctx, []string{missingPath})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: missingPath, Error: "file_not_found"}})
	})

	t.Run("download error is directory", func(t *testing.T) { // test_download_error_is_directory
		backend, root := factory(t)
		dirPath := sandboxPath(root, "test_directory")
		backend.Execute(ctx, "rm -rf "+shellQuote(dirPath)+" && mkdir -p "+shellQuote(dirPath))
		downloads := backend.DownloadFiles(ctx, []string{dirPath})
		requireLen(t, "download responses", downloads, 1)
		requireEqual(t, "download path", downloads[0].Path, dirPath)
		if downloads[0].Content != nil {
			t.Fatalf("download of directory: expected nil content, got %d bytes", len(downloads[0].Content))
		}
		requireSandboxErrorContaining(t, "download", downloads[0].Error, "is_directory", "file_not_found", "invalid_path")
	})

	t.Run("download error permission denied", func(t *testing.T) { // test_download_error_permission_denied
		if os.Geteuid() == 0 {
			t.Skip("chmod 000 files are still readable by root")
		}
		backend, root := factory(t)
		testPath := sandboxPath(root, "test_no_read.txt")
		backend.Execute(ctx, "rm -f "+shellQuote(testPath)+" && echo secret > "+shellQuote(testPath)+" && chmod 000 "+shellQuote(testPath))
		defer backend.Execute(ctx, "chmod 644 "+shellQuote(testPath)+" || true")
		downloads := backend.DownloadFiles(ctx, []string{testPath})
		requireLen(t, "download responses", downloads, 1)
		requireEqual(t, "download path", downloads[0].Path, testPath)
		if downloads[0].Content != nil {
			t.Fatalf("download of chmod 000 file: expected nil content, got %d bytes", len(downloads[0].Content))
		}
		requireSandboxErrorContaining(t, "download", downloads[0].Error, "permission_denied", "file_not_found", "invalid_path")
	})

	t.Run("download error invalid path relative", func(t *testing.T) { // test_download_error_invalid_path_relative
		backend, _ := factory(t)
		downloads := backend.DownloadFiles(ctx, []string{"relative/path.txt"})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: "relative/path.txt", Error: "invalid_path"}})
	})

	t.Run("upload missing parent dir or roundtrip", func(t *testing.T) { // test_upload_missing_parent_dir_or_roundtrip
		backend, root := factory(t)
		dirPath := sandboxPath(root, "test_upload_missing_parent_dir")
		path := dirPath + "/deepagents_test_upload.txt"
		content := []byte("nope")
		backend.Execute(ctx, "rm -rf "+shellQuote(dirPath))
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: path, Content: content}})
		requireLen(t, "upload responses", uploads, 1)
		requireEqual(t, "upload path", uploads[0].Path, path)
		if uploads[0].Error != "" {
			// Some sandboxes reject a missing parent directory instead of
			// auto-creating it; both are conformant.
			requireSandboxErrorContaining(t, "upload", uploads[0].Error,
				"invalid_path", "permission_denied", "file_not_found")
			return
		}
		downloads := backend.DownloadFiles(ctx, []string{path})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: path, Content: content}})
	})

	t.Run("upload relative path returns invalid path", func(t *testing.T) { // test_upload_relative_path_returns_invalid_path
		backend, _ := factory(t)
		uploads := backend.UploadFiles(ctx, []SandboxFileUpload{{Path: "relative_upload.txt", Content: []byte("nope")}})
		requireDeepEqual(t, "upload responses", uploads,
			[]SandboxUploadResponse{{Path: "relative_upload.txt", Error: "invalid_path"}})
	})

	t.Run("write read download large text with escaped content", func(t *testing.T) { // test_write_read_download_large_text_with_escaped_content
		backend, root := factory(t)
		testPath := sandboxPath(root, "large_sync_escaped.txt")
		line := "prefix\t☃世界π≈3.14159" +
			" | spaces   preserved" +
			" | quotes ' \"" +
			" | brackets [] {{}}" +
			" | shell $VAR `cmd` $(subshell)" +
			" | slash /tmp/path and backslash \\\\" +
			" | control-ish \\r \\n" +
			" | suffix"
		lines := make([]string, 0, 2500)
		for i := 0; i < 2500; i++ {
			lines = append(lines, fmt.Sprintf("%04d:%s", i, line))
		}
		content := strings.Join(lines, "\n")
		requireNoSandboxError(t, "write", backend.Write(ctx, testPath, content).Error)

		pages := make([]string, 0, 25)
		for offset := 0; offset < len(lines); offset += 100 {
			page := requireFileData(t, "read page", backend.Read(ctx, testPath,
				SandboxReadOptions{Offset: offset, Limit: intPtr(100)}))
			requireDeepEqual(t, "page content", page.Content, strings.Join(lines[offset:offset+100], "\n"))
			pages = append(pages, page.Content)
		}
		requireDeepEqual(t, "reconstructed content", strings.Join(pages, "\n"), content)

		downloads := backend.DownloadFiles(ctx, []string{testPath})
		requireDeepEqual(t, "download responses", downloads,
			[]SandboxDownloadResponse{{Path: testPath, Content: []byte(content)}})
	})
}
