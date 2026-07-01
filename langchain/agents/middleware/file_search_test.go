package middleware

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFilesystemFileSearchGlobSearch(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\n")
	writeTestFile(t, root, "src/app.py", "print('hi')\n")
	writeTestFile(t, root, "README.md", "hello\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 10)
	if err != nil {
		t.Fatalf("new file search middleware: %v", err)
	}

	got := middleware.GlobSearch("**/*.go", "/")
	if got != "/src/main.go" {
		t.Fatalf("glob result mismatch: %q", got)
	}

	got = middleware.GlobSearch("../*.go", "/")
	if got != "No files found" {
		t.Fatalf("expected traversal rejection, got %q", got)
	}
}

func TestFilesystemFileSearchGrepSearchModes(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\nfunc main() {}\n")
	writeTestFile(t, root, "src/app.py", "def main():\n    pass\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 10)
	if err != nil {
		t.Fatalf("new file search middleware: %v", err)
	}

	files := middleware.GrepSearch("main", "/", "*.{go,py}", GrepFilesWithMatches)
	if !strings.Contains(files, "/src/main.go") || !strings.Contains(files, "/src/app.py") {
		t.Fatalf("files mode mismatch: %q", files)
	}

	content := middleware.GrepSearch("func", "/", "*.go", GrepContent)
	if content != "/src/main.go:2:func main() {}" {
		t.Fatalf("content mode mismatch: %q", content)
	}

	count := middleware.GrepSearch("main", "/", "*.go", GrepCount)
	if count != "/src/main.go:2" {
		t.Fatalf("count mode mismatch: %q", count)
	}
}

func TestFilesystemFileSearchInvalidInputs(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 10)
	if err != nil {
		t.Fatalf("new file search middleware: %v", err)
	}

	if got := middleware.GrepSearch("[", "/", "", GrepContent); !strings.HasPrefix(got, "Invalid regex pattern:") {
		t.Fatalf("expected invalid regex, got %q", got)
	}
	if got := middleware.GrepSearch("main", "/", "*.{go", GrepContent); got != "Invalid include pattern" {
		t.Fatalf("expected invalid include, got %q", got)
	}
	if got := middleware.GrepSearch("missing", "/", "", GrepContent); got != "No matches found" {
		t.Fatalf("expected no matches, got %q", got)
	}
	if got := middleware.GrepSearch("main", "/../", "", GrepContent); got != "No matches found" {
		t.Fatalf("expected traversal path rejection, got %q", got)
	}
}

func TestFilesystemFileSearchGrepSearchWithoutRipgrepMatchesPythonFallback(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\nfunc main() {}\n")
	writeTestFile(t, root, "src/app.py", "def main():\n    pass\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 10)
	if err != nil {
		t.Fatalf("new file search middleware: %v", err)
	}
	middleware.UseRipgrep = false

	files := middleware.GrepSearch("main", "/", "*.{go,py}", GrepFilesWithMatches)
	if !strings.Contains(files, "/src/main.go") || !strings.Contains(files, "/src/app.py") {
		t.Fatalf("files mode mismatch: %q", files)
	}

	content := middleware.GrepSearch("func", "/", "*.go", GrepContent)
	if content != "/src/main.go:2:func main() {}" {
		t.Fatalf("content mode mismatch: %q", content)
	}

	count := middleware.GrepSearch("main", "/", "*.go", GrepCount)
	if count != "/src/main.go:2" {
		t.Fatalf("count mode mismatch: %q", count)
	}
}

func TestFilesystemFileSearchGrepSearchFallsBackWhenRipgrepUnavailable(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\nfunc main() {}\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 10)
	if err != nil {
		t.Fatalf("new file search middleware: %v", err)
	}
	middleware.UseRipgrep = true
	t.Setenv("PATH", "")

	content := middleware.GrepSearch("func", "/", "*.go", GrepContent)
	if content != "/src/main.go:2:func main() {}" {
		t.Fatalf("expected fallback search to still find match, got %q", content)
	}
}

func TestFilesystemFileSearchTools(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "src/main.go", "package main\n")
	middleware, err := NewFilesystemFileSearchMiddleware(root, 10)
	if err != nil {
		t.Fatalf("new file search middleware: %v", err)
	}
	if len(middleware.Tools) != 2 {
		t.Fatalf("tool count mismatch: %d", len(middleware.Tools))
	}
	if middleware.Tools[0].Name() != "glob_search" || middleware.Tools[1].Name() != "grep_search" {
		t.Fatalf("tool names mismatch: %q %q", middleware.Tools[0].Name(), middleware.Tools[1].Name())
	}
}

func writeTestFile(t *testing.T, root string, rel string, content string) {
	t.Helper()
	path := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
}
