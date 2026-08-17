package middleware

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFilesystemFileSearchDefaultsAndInvalidRoot(t *testing.T) {
	middleware, err := NewFilesystemFileSearchMiddleware(t.TempDir(), 0)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	if middleware.MaxFileSizeBytes != 10*1024*1024 {
		t.Fatalf("default max size mismatch: %d", middleware.MaxFileSizeBytes)
	}

	if _, err := NewFilesystemFileSearchMiddleware(filepath.Join(t.TempDir(), "does-not-exist"), 1); err == nil {
		t.Fatal("expected error for nonexistent root")
	}
}

func TestFilesystemFileSearchGlobSearchPaths(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\n")
	writeTestFile(t, root, "src/util/helper.go", "package util\n")
	writeTestFile(t, root, "note.txt", "hi\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	// Empty path defaults to the root.
	if got := middleware.GlobSearch("*.txt", ""); got != "/note.txt" {
		t.Fatalf("empty path mismatch: %q", got)
	}
	// Relative path is resolved against the root.
	if got := middleware.GlobSearch("*.go", "src"); got != "/src/main.go" {
		t.Fatalf("relative path mismatch: %q", got)
	}
	// Nonexistent directory.
	if got := middleware.GlobSearch("*.go", "/missing"); got != "No files found" {
		t.Fatalf("missing dir mismatch: %q", got)
	}
	// Path pointing at a regular file.
	if got := middleware.GlobSearch("*.go", "/note.txt"); got != "No files found" {
		t.Fatalf("file path mismatch: %q", got)
	}
	// Invalid patterns.
	for _, pattern := range []string{"", "/absolute", "..", "a\x00b"} {
		if got := middleware.GlobSearch(pattern, "/"); got != "No files found" {
			t.Fatalf("pattern %q should be rejected, got %q", pattern, got)
		}
	}
	// "?" matches a single character within one path segment.
	if got := middleware.GlobSearch("note.tx?", "/"); got != "/note.txt" {
		t.Fatalf("question mark mismatch: %q", got)
	}
	// "*" does not cross directory boundaries; "**" does.
	if got := middleware.GlobSearch("*/*.go", "/"); got != "/src/main.go" {
		t.Fatalf("single star mismatch: %q", got)
	}
	got := middleware.GlobSearch("**/*.go", "/")
	if !strings.Contains(got, "/src/main.go") || !strings.Contains(got, "/src/util/helper.go") {
		t.Fatalf("double star mismatch: %q", got)
	}
	// No matches at all.
	if got := middleware.GlobSearch("*.rs", "/"); got != "No files found" {
		t.Fatalf("no match mismatch: %q", got)
	}
}

func TestFilesystemFileSearchGlobSkipsEscapingSymlink(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "secret.txt")
	if err := os.WriteFile(outside, []byte("secret\n"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "link.txt")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	writeTestFile(t, root, "real.txt", "hi\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	got := middleware.GlobSearch("*.txt", "/")
	if got != "/real.txt" {
		t.Fatalf("symlink escaping the root must be skipped: %q", got)
	}
}

func TestFilesystemFileSearchToolsInvoke(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}

	globResult, err := middleware.Tools[0].Invoke(context.Background(), map[string]any{"pattern": "*.go", "path": "/src"})
	if err != nil {
		t.Fatalf("glob invoke: %v", err)
	}
	if globResult.Content != "/src/main.go" {
		t.Fatalf("glob invoke mismatch: %q", globResult.Content)
	}

	// Empty path and output_mode fall back to defaults.
	grepResult, err := middleware.Tools[1].Invoke(context.Background(), map[string]any{"pattern": "package"})
	if err != nil {
		t.Fatalf("grep invoke: %v", err)
	}
	if grepResult.Content != "/src/main.go" {
		t.Fatalf("grep invoke mismatch: %q", grepResult.Content)
	}
}

func TestFilesystemFileSearchTildePathRejected(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "main.go", "package main\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	middleware.UseRipgrep = false
	if got := middleware.GrepSearch("package", "~/", "", GrepContent); got != "No matches found" {
		t.Fatalf("tilde path should be rejected, got %q", got)
	}
	if got := middleware.GlobSearch("*.go", "~/"); got != "No files found" {
		t.Fatalf("tilde glob path should be rejected, got %q", got)
	}
}

func TestFilesystemFileSearchPythonSearchSkipsLargeAndUnreadableFiles(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "big.go", "package big\n")
	writeTestFile(t, root, "small.go", "package small\n")
	unreadable := filepath.Join(root, "unreadable.go")
	if err := os.WriteFile(unreadable, []byte("package x\n"), 0o000); err != nil {
		t.Fatalf("write unreadable file: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(unreadable, 0o644) })

	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	middleware.UseRipgrep = false
	// big.go (12 bytes) fits; small.go (14 bytes) exceeds the limit and the
	// unreadable file (within the limit) fails to read.
	middleware.MaxFileSizeBytes = 13
	got := middleware.GrepSearch("package", "/", "*.go", GrepFilesWithMatches)
	if strings.Contains(got, "small.go") || strings.Contains(got, "unreadable.go") {
		t.Fatalf("oversized/unreadable files must be skipped: %q", got)
	}
	if !strings.Contains(got, "big.go") {
		t.Fatalf("small-enough file should be searched: %q", got)
	}
}

func TestFilesystemFileSearchIncludePatternValidation(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "main.go", "package main\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	for _, include := range []string{"\x00", "a\nb", "a}", "{}"} {
		if got := middleware.GrepSearch("package", "/", include, GrepContent); got != "Invalid include pattern" {
			t.Fatalf("include %q should be invalid, got %q", include, got)
		}
	}
}

func TestExpandIncludePatterns(t *testing.T) {
	got := expandIncludePatterns("a{x,y}b")
	want := []string{"axb", "ayb"}
	if len(got) != len(want) {
		t.Fatalf("expansion mismatch: %#v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expansion mismatch: %#v, want %#v", got, want)
		}
	}
	if got := expandIncludePatterns("plain"); len(got) != 1 || got[0] != "plain" {
		t.Fatalf("plain pattern mismatch: %#v", got)
	}
	if got := expandIncludePatterns("{a,b}"); len(got) != 2 {
		t.Fatalf("brace expansion mismatch: %#v", got)
	}
}

func TestMatchIncludePattern(t *testing.T) {
	if !matchIncludePattern("main.go", "*.{go,py}") {
		t.Fatal("expected brace pattern to match")
	}
	if matchIncludePattern("main.txt", "*.{go,py}") {
		t.Fatal("expected brace pattern to reject .txt")
	}
	if !matchIncludePattern("main.go", "*.go") {
		t.Fatal("expected plain pattern to match")
	}
}

func TestFilesystemFileSearchRipgrepParsesEdgeCases(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep not installed")
	}
	root := t.TempDir()
	writeTestFile(t, root, "main.go", "package main\n")
	// A colon in the file name makes rg's path:line:content output unparseable;
	// the entry is skipped rather than misreported.
	writeTestFile(t, root, "weird:name.go", "package weird\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 1)
	if err != nil {
		t.Fatalf("new middleware: %v", err)
	}
	middleware.UseRipgrep = true
	got := middleware.GrepSearch("package", "/", "*.go", GrepFilesWithMatches)
	if !strings.Contains(got, "/main.go") {
		t.Fatalf("expected main.go in results: %q", got)
	}
}
