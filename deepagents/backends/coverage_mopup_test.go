package backends

// Task 9 release-gate coverage mop-up: focused tests for the error and edge
// branches of Tasks 1-3 left uncovered by the task suites, so the
// deepagents/... packages clear the 95% Makefile gate.

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNewFilesystemBackendMissingRoot(t *testing.T) {
	_, err := NewFilesystemBackend(filepath.Join(t.TempDir(), "missing"))
	if err == nil || !strings.Contains(err.Error(), "root directory does not exist") {
		t.Fatalf("missing root must error, got %v", err)
	}
}

func TestFilesystemBackendLsEdgePaths(t *testing.T) {
	dir := t.TempDir()
	backend, err := NewFilesystemBackend(dir)
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	if _, err := backend.Write(t.Context(), "/note.md", "x"); err != nil {
		t.Fatalf("seed write: %v", err)
	}

	// Ls on a file is not_a_directory.
	_, err = backend.Ls(t.Context(), "/note.md")
	var lsErr *Error
	if !errors.As(err, &lsErr) || lsErr.Code != ErrInvalidPath || !strings.Contains(lsErr.Message, "not_a_directory") {
		t.Fatalf("ls on a file = %v, want not_a_directory", err)
	}

	// A broken symlink child is skipped, and a valid symlink + dir list.
	if err := os.Symlink(filepath.Join(dir, "gone"), filepath.Join(dir, "broken")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	res, err := backend.Ls(t.Context(), "/")
	if err != nil {
		t.Fatalf("ls: %v", err)
	}
	names := make([]string, 0, len(res.Entries))
	for _, e := range res.Entries {
		names = append(names, e.Name)
	}
	if strings.Join(names, ",") != "note.md,sub" {
		t.Fatalf("ls entries = %v, want the file and dir only (broken symlink skipped)", names)
	}
}

func TestFilesystemBackendReadEditErrorPaths(t *testing.T) {
	dir := t.TempDir()
	backend, err := NewFilesystemBackend(dir)
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	if _, err := backend.Write(t.Context(), "/note.md", "x"); err != nil {
		t.Fatalf("seed write: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}

	// Read on a directory reads as file-not-found.
	if _, err := backend.Read(t.Context(), "/sub", 0, 10); err == nil ||
		!strings.Contains(err.Error(), "File '/sub' not found") {
		t.Fatalf("read on dir = %v, want not found", err)
	}
	// Edit on a missing file and on a directory.
	for _, path := range []string{"/missing.md", "/sub"} {
		if _, err := backend.Edit(t.Context(), path, "a", "b", false); err == nil ||
			!strings.Contains(err.Error(), "Error: File '") {
			t.Fatalf("edit %s = %v, want not-found literal", path, err)
		}
	}
	// Write under a file component fails at MkdirAll.
	if _, err := backend.Write(t.Context(), "/note.md/deep.md", "x"); err == nil ||
		!strings.Contains(err.Error(), "Error writing file") {
		t.Fatalf("write under a file = %v, want write error", err)
	}
}

func TestFilesystemBackendPermissionDeniedMapsToCode(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: permission bits do not block I/O")
	}
	dir := t.TempDir()
	backend, err := NewFilesystemBackend(dir)
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	if _, err := backend.Write(t.Context(), "/secret.md", "x"); err != nil {
		t.Fatalf("seed write: %v", err)
	}
	if err := os.Chmod(filepath.Join(dir, "secret.md"), 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(dir, "secret.md"), 0o644) })

	_, err = backend.Read(t.Context(), "/secret.md", 0, 10)
	var backendErr *Error
	if !errors.As(err, &backendErr) || backendErr.Code != ErrPermissionDenied {
		t.Fatalf("read of unreadable file = %v, want permission_denied code", err)
	}

	// A writable-denied file: the read succeeds, the write-back fails.
	if err := os.Chmod(filepath.Join(dir, "secret.md"), 0o444); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(dir, "secret.md"), 0o644) })
	if _, err := backend.Edit(t.Context(), "/secret.md", "x", "y", true); err == nil ||
		!strings.Contains(err.Error(), "Error editing file") {
		t.Fatalf("edit of read-only file = %v, want edit error", err)
	}
}

func TestFilesystemBackendDeleteSymlinkAndDirectory(t *testing.T) {
	dir := t.TempDir()
	backend, err := NewFilesystemBackend(dir)
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	if _, err := backend.Write(t.Context(), "/target.md", "x"); err != nil {
		t.Fatalf("seed write: %v", err)
	}
	broken := filepath.Join(dir, "broken.md")
	if err := os.Symlink(filepath.Join(dir, "gone"), broken); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := os.Mkdir(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := backend.Write(t.Context(), "/sub/inner.md", "x"); err != nil {
		t.Fatalf("seed nested write: %v", err)
	}

	// A broken symlink resolves to itself (EvalSymlinks fails, the path is
	// kept), so Delete's Lstat sees the link and removes just the link.
	if _, err := backend.Delete(t.Context(), "/broken.md"); err != nil {
		t.Fatalf("delete broken symlink: %v", err)
	}
	if _, err := os.Lstat(broken); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("broken link must be gone, lstat err = %v", err)
	}
	// A directory removes recursively.
	if _, err := backend.Delete(t.Context(), "/sub"); err != nil {
		t.Fatalf("delete directory: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sub")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("directory tree must be gone, stat err = %v", err)
	}
}

func TestFilesystemBackendGlobGrepEdges(t *testing.T) {
	dir := t.TempDir()
	backend, err := NewFilesystemBackend(dir)
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	if _, err := backend.Write(t.Context(), "/a.go", "needle"); err != nil {
		t.Fatalf("seed write: %v", err)
	}

	// Traversal patterns are refused outright.
	if _, err := backend.Glob(t.Context(), "/", "../../*.go"); err == nil ||
		!strings.Contains(err.Error(), "Path traversal not allowed in glob pattern") {
		t.Fatalf("glob traversal = %v, want traversal literal", err)
	}
	if _, err := backend.Grep(t.Context(), "x", "/", "../../*.go", 0); err == nil ||
		!strings.Contains(err.Error(), "Path traversal not allowed in glob pattern") {
		t.Fatalf("grep traversal glob = %v, want traversal literal", err)
	}
	// A search path outside the root errors for glob...
	if _, err := backend.Glob(t.Context(), "../../etc", "*"); err == nil {
		t.Fatal("glob escaping the root must error")
	}
	// ...and is swallowed to empty matches for grep.
	if res, err := backend.Grep(t.Context(), "x", "../../etc", "", 0); err != nil || len(res.Matches) != 0 {
		t.Fatalf("grep escaping the root = (%v, %+v), want empty matches", err, res)
	}
	// A non-directory search path globs to nothing.
	if res, err := backend.Glob(t.Context(), "/a.go", "*"); err != nil || len(res.Files) != 0 {
		t.Fatalf("glob on a file = (%v, %+v), want empty", err, res)
	}

	// Canceled contexts propagate.
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := backend.Glob(canceled, "/", "**"); !errors.Is(err, context.Canceled) {
		t.Fatalf("glob with canceled ctx = %v, want context.Canceled", err)
	}
	if _, err := backend.Grep(canceled, "needle", "/", "", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("grep with canceled ctx = %v, want context.Canceled", err)
	}
}

func TestSplitLinesKeepEndsCRLineEndings(t *testing.T) {
	got := splitLinesKeepEnds("a\rb\nc")
	want := []string{"a\r", "b\n", "c"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("splitLinesKeepEnds = %q, want %q", got, want)
	}
	got = splitLinesKeepEnds("a\r\nb")
	if strings.Join(got, "|") != "a\r\n|b" {
		t.Fatalf("CRLF split = %q", got)
	}
}

func TestMustRelFallsBackToBase(t *testing.T) {
	if got := mustRel("/base", "elsewhere/x"); got != "x" {
		t.Fatalf("mustRel fallback = %q, want base name", got)
	}
	if got := mustRel("/base", "/base/sub/f"); got != "sub/f" {
		t.Fatalf("mustRel = %q, want sub/f", got)
	}
}

func TestMapMergeErrorText(t *testing.T) {
	err := &MapMergeError{Arg: "update", Got: "string"}
	want := "backends.MapMergeReducer: update must be nil, map[string]backends.FileMeta, or map[string]*backends.FileMeta; got string"
	if err.Error() != want {
		t.Fatalf("Error() = %q, want %q", err.Error(), want)
	}
}

func TestNormalizeStatePathEmptyIsRoot(t *testing.T) {
	if got, err := normalizeStatePath(""); err != nil || got != "/" {
		t.Fatalf(`normalizeStatePath("") = (%q, %v), want ("/", nil)`, got, err)
	}
}

func TestFilterStateFilesRootFileKey(t *testing.T) {
	got := filterStateFiles(map[string]FileMeta{"/": {Content: "root file"}}, "/")
	if len(got) != 1 || got[0].key != "/" || got[0].rel != "" {
		t.Fatalf("filterStateFiles(/) = %#v, want the / key with empty rel", got)
	}
}

func TestStateBackendEditBinaryAndGrepFilters(t *testing.T) {
	backend := NewStateBackend()
	ctx := WithStateView(t.Context(), NewStateView(map[string]FileMeta{
		"/bin.dat":   {Content: "eA==", Encoding: "base64"},
		"/text.md":   {Content: "needle"},
		"/other.txt": {Content: "needle"},
	}))
	if _, err := backend.Edit(ctx, "/bin.dat", "a", "b", true); err == nil ||
		!strings.Contains(err.Error(), "Cannot edit binary files") {
		t.Fatalf("edit binary = %v, want binary literal", err)
	}

	// The include glob scopes the search: /other.txt is filtered out and the
	// base64 entry is skipped as binary.
	res, err := backend.Grep(ctx, "needle", "/", "*.md", 0)
	if err != nil {
		t.Fatalf("state grep: %v", err)
	}
	if len(res.Matches) != 1 || res.Matches[0].Path != "/text.md" {
		t.Fatalf("state grep matches = %#v, want only /text.md", res.Matches)
	}
}
