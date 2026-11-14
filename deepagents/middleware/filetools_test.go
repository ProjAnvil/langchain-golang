package middleware

import (
	"context"
	"sort"
	"testing"
	"time"

	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// --- Test helpers -----------------------------------------------------

// meta builds a utf-8 FileMeta for seeding state views.
func meta(content string) backends.FileMeta {
	return backends.FileMeta{Content: content, Encoding: "utf-8"}
}

// newTempFilesystemBackend builds a FilesystemBackend rooted at a fresh temp
// directory (a Deleter, so the delete tool registers).
func newTempFilesystemBackend(t *testing.T) *backends.FilesystemBackend {
	t.Helper()
	backend, err := backends.NewFilesystemBackend(t.TempDir())
	if err != nil {
		t.Fatalf("NewFilesystemBackend: %v", err)
	}
	return backend
}

// stateCtx builds a context carrying a StateView over files, standing in for
// the injection FilesystemMiddleware.WrapToolCall performs around real calls.
func stateCtx(files map[string]backends.FileMeta) context.Context {
	return backends.WithStateView(context.Background(), backends.NewStateView(files))
}

func mustNewFilesystemMiddleware(t *testing.T, backend backends.Backend, rules ...backends.PermissionRule) *FilesystemMiddleware {
	t.Helper()
	m, err := NewFilesystemMiddleware(backend, rules...)
	if err != nil {
		t.Fatalf("NewFilesystemMiddleware: %v", err)
	}
	return m
}

func fileToolByName(t *testing.T, m *FilesystemMiddleware, name string) coretools.Tool {
	t.Helper()
	for _, tool := range m.Tools {
		if tool.Name() == name {
			return tool
		}
	}
	t.Fatalf("tool %q not registered; have %d tools", name, len(m.Tools))
	return nil
}

// invokeFileTool runs a tool and enforces the port-wide contract that a
// filesystem tool never fails the run: every outcome, including backend
// errors, arrives as (Result, nil).
func invokeFileTool(t *testing.T, ctx context.Context, m *FilesystemMiddleware, name string, args map[string]any) coretools.Result {
	t.Helper()
	tool := fileToolByName(t, m, name)
	res, err := tool.Invoke(ctx, args)
	if err != nil {
		t.Fatalf("tool %s returned error %v; want (Result, nil) with error text in Content", name, err)
	}
	return res
}

// recordingBackend wraps a Backend and counts calls per operation so tests
// can assert the deny gate short-circuits before the backend is touched.
// It deliberately does not implement Deleter.
type recordingBackend struct {
	inner     backends.Backend
	calls     map[string]int
	globPath  string
	grepPath  string
	grepMax   int
	grepGlobs string
}

func newRecordingBackend(inner backends.Backend) *recordingBackend {
	return &recordingBackend{inner: inner, calls: map[string]int{}}
}

func (r *recordingBackend) Ls(ctx context.Context, path string) (backends.LsResult, error) {
	r.calls["ls"]++
	return r.inner.Ls(ctx, path)
}

func (r *recordingBackend) Read(ctx context.Context, path string, offset, limit int) (backends.ReadResult, error) {
	r.calls["read"]++
	return r.inner.Read(ctx, path, offset, limit)
}

func (r *recordingBackend) Write(ctx context.Context, path string, content string) (backends.WriteResult, error) {
	r.calls["write"]++
	return r.inner.Write(ctx, path, content)
}

func (r *recordingBackend) Edit(ctx context.Context, path string, oldString, newString string, replaceAll bool) (backends.EditResult, error) {
	r.calls["edit"]++
	return r.inner.Edit(ctx, path, oldString, newString, replaceAll)
}

func (r *recordingBackend) Glob(ctx context.Context, path string, pattern string) (backends.GlobResult, error) {
	r.calls["glob"]++
	r.globPath = path
	return r.inner.Glob(ctx, path, pattern)
}

func (r *recordingBackend) Grep(ctx context.Context, pattern string, path string, glob string, maxCount int) (backends.GrepResult, error) {
	r.calls["grep"]++
	r.grepPath = path
	r.grepMax = maxCount
	r.grepGlobs = glob
	return r.inner.Grep(ctx, pattern, path, glob, maxCount)
}

// Delete delegates to the inner backend's Deleter implementation so the
// recordingBackend satisfies backends.Deleter and the delete tool registers.
func (r *recordingBackend) Delete(ctx context.Context, path string) (backends.DeleteResult, error) {
	r.calls["delete"]++
	return r.inner.(backends.Deleter).Delete(ctx, path)
}

// blockingBackend blocks every operation until the context deadline fires,
// standing in for a runaway backend in the timeout tests.
type blockingBackend struct{}

func (b *blockingBackend) block(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

func (b *blockingBackend) Ls(ctx context.Context, path string) (backends.LsResult, error) {
	return backends.LsResult{}, b.block(ctx)
}

func (b *blockingBackend) Read(ctx context.Context, path string, offset, limit int) (backends.ReadResult, error) {
	return backends.ReadResult{}, b.block(ctx)
}

func (b *blockingBackend) Write(ctx context.Context, path string, content string) (backends.WriteResult, error) {
	return backends.WriteResult{}, b.block(ctx)
}

func (b *blockingBackend) Edit(ctx context.Context, path string, oldString, newString string, replaceAll bool) (backends.EditResult, error) {
	return backends.EditResult{}, b.block(ctx)
}

func (b *blockingBackend) Glob(ctx context.Context, path string, pattern string) (backends.GlobResult, error) {
	return backends.GlobResult{}, b.block(ctx)
}

func (b *blockingBackend) Grep(ctx context.Context, pattern string, path string, glob string, maxCount int) (backends.GrepResult, error) {
	return backends.GrepResult{}, b.block(ctx)
}

// overrideFileToolTimeouts swaps the package-level glob/grep deadlines for the
// duration of one test and restores them on cleanup.
func overrideFileToolTimeouts(t *testing.T, glob, grep time.Duration) {
	t.Helper()
	oldGlob, oldGrep := fsGlobTimeout, fsGrepTimeout
	fsGlobTimeout, fsGrepTimeout = glob, grep
	t.Cleanup(func() { fsGlobTimeout, fsGrepTimeout = oldGlob, oldGrep })
}

// filesArtifact extracts the Command artifact's files delta, failing when the
// artifact is missing or malformed.
func filesArtifact(t *testing.T, res coretools.Result) map[string]*backends.FileMeta {
	t.Helper()
	cmd, ok := res.Artifact.(*types.Command)
	if !ok {
		t.Fatalf("Artifact = %#v; want *types.Command", res.Artifact)
	}
	files, ok := cmd.Update["files"].(map[string]*backends.FileMeta)
	if !ok {
		t.Fatalf("Command.Update[files] = %#v; want map[string]*backends.FileMeta", cmd.Update["files"])
	}
	return files
}

var denyLockedWrites = backends.PermissionRule{
	Operations: []string{"write"},
	Paths:      []string{"/locked/**"},
	Mode:       backends.PermissionDeny,
}

var denySecretsRead = backends.PermissionRule{
	Operations: []string{"read"},
	Paths:      []string{"/secret/**"},
	Mode:       backends.PermissionDeny,
}

// --- ls ---------------------------------------------------------------

func TestLsTool(t *testing.T) {
	seed := map[string]backends.FileMeta{
		"/a.txt":     meta("x"),
		"/sub/c.txt": meta("y"),
	}
	tests := []struct {
		name       string
		backend    backends.Backend
		ctxFiles   map[string]backends.FileMeta
		args       map[string]any
		want       string
		wantNoCall bool // deny rows must not reach the backend
	}{
		{
			name:     "lists root entries as python list",
			backend:  backends.NewStateBackend(),
			ctxFiles: seed,
			args:     map[string]any{"path": "/"},
			want:     "['/a.txt', '/sub']",
		},
		{
			name:     "empty directory reports no files found",
			backend:  backends.NewStateBackend(),
			ctxFiles: seed,
			args:     map[string]any{"path": "/empty"},
			want:     "No files found",
		},
		{
			name:     "relative path is normalized before listing",
			backend:  backends.NewStateBackend(),
			ctxFiles: map[string]backends.FileMeta{"/dir/a.txt": meta("x")},
			args:     map[string]any{"path": "dir"},
			want:     "['/dir/a.txt']",
		},
		{
			name:     "missing directory on filesystem backend",
			backend:  newTempFilesystemBackend(t),
			ctxFiles: nil,
			args:     map[string]any{"path": "/nope"},
			want:     "Error: Path '/nope': path_not_found",
		},
		{
			name:       "deny gate blocks the backend call",
			backend:    newRecordingBackend(backends.NewStateBackend()),
			ctxFiles:   map[string]backends.FileMeta{"/secret/a.txt": meta("x")},
			args:       map[string]any{"path": "/secret"},
			want:       "Error: permission denied for read on /secret",
			wantNoCall: true,
		},
		{
			name:     "path traversal is rejected",
			backend:  backends.NewStateBackend(),
			ctxFiles: seed,
			args:     map[string]any{"path": "../etc"},
			want:     "Error: Path traversal not allowed: ../etc",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec, isRec := tt.backend.(*recordingBackend)
			m := mustNewFilesystemMiddleware(t, tt.backend, denySecretsRead)
			res := invokeFileTool(t, stateCtx(tt.ctxFiles), m, "ls", tt.args)
			if res.Content != tt.want {
				t.Fatalf("Content = %q, want %q", res.Content, tt.want)
			}
			if res.Artifact != nil {
				t.Fatalf("ls is read-only; Artifact = %#v, want nil", res.Artifact)
			}
			if isRec && tt.wantNoCall && rec.calls["ls"] != 0 {
				t.Fatalf("backend Ls called %d times under deny; want 0", rec.calls["ls"])
			}
		})
	}
}

// --- read_file ----------------------------------------------------------

func TestReadFileTool(t *testing.T) {
	files := map[string]backends.FileMeta{"/a.txt": meta("alpha\nbeta\n")}
	tests := []struct {
		name     string
		ctxFiles map[string]backends.FileMeta
		args     map[string]any
		want     string
	}{
		{
			name:     "whole file with header",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/a.txt"},
			want:     "@@ lines 1-2 of 2 @@\nalpha\nbeta",
		},
		{
			name:     "first page carries next offset",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/a.txt", "offset": 0, "limit": 1},
			want:     "@@ lines 1-1 of 2 | next offset 1 @@\nalpha",
		},
		{
			name:     "resuming at offset 1 reaches EOF",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/a.txt", "offset": 1},
			want:     "@@ lines 2-2 of 2 @@\nbeta",
		},
		{
			name:     "negative offset reads from line 1 with a notice",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/a.txt", "offset": -5},
			want:     "[Requested offset -5 is before the start of the file; read from line 1 instead.]\n@@ lines 1-2 of 2 @@\nalpha\nbeta",
		},
		{
			name:     "missing file",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/missing.txt"},
			want:     "Error: File '/missing.txt' not found",
		},
		{
			name:     "empty file returns the system reminder",
			ctxFiles: map[string]backends.FileMeta{"/empty.txt": meta("")},
			args:     map[string]any{"file_path": "/empty.txt"},
			want:     "System reminder: File exists but has empty contents",
		},
		{
			name:     "offset beyond EOF",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/a.txt", "offset": 9},
			want:     "Error: Line offset 9 exceeds file length (2 lines)",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
			res := invokeFileTool(t, stateCtx(tt.ctxFiles), m, "read_file", tt.args)
			if res.Content != tt.want {
				t.Fatalf("Content = %q, want %q", res.Content, tt.want)
			}
			if res.Artifact != nil {
				t.Fatalf("read_file is read-only; Artifact = %#v, want nil", res.Artifact)
			}
		})
	}
}

func TestReadFileToolDenyGate(t *testing.T) {
	rec := newRecordingBackend(backends.NewStateBackend())
	m := mustNewFilesystemMiddleware(t, rec, denySecretsRead)
	res := invokeFileTool(t, stateCtx(map[string]backends.FileMeta{"/secret/a.txt": meta("x")}), m,
		"read_file", map[string]any{"file_path": "/secret/a.txt"})
	want := "Error: permission denied for read on /secret/a.txt"
	if res.Content != want {
		t.Fatalf("Content = %q, want %q", res.Content, want)
	}
	if rec.calls["read"] != 0 {
		t.Fatalf("backend Read called %d times under deny; want 0", rec.calls["read"])
	}
}

// --- write_file ---------------------------------------------------------

func TestWriteFileTool(t *testing.T) {
	t.Run("writes on state backend return updated file and files artifact", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
		res := invokeFileTool(t, stateCtx(nil), m, "write_file",
			map[string]any{"file_path": "/a.txt", "content": "hi"})
		if want := "Updated file /a.txt"; res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		files := filesArtifact(t, res)
		if len(files) != 1 || files["/a.txt"] == nil || files["/a.txt"].Content != "hi" {
			t.Fatalf("files delta = %#v, want one entry /a.txt with content \"hi\"", files)
		}
	})

	t.Run("writes on filesystem backend return no artifact", func(t *testing.T) {
		backend := newTempFilesystemBackend(t)
		m := mustNewFilesystemMiddleware(t, backend)
		res := invokeFileTool(t, context.Background(), m, "write_file",
			map[string]any{"file_path": "/a.txt", "content": "hi"})
		if want := "Updated file /a.txt"; res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		if res.Artifact != nil {
			t.Fatalf("Artifact = %#v; want nil without a state view in context", res.Artifact)
		}
	})

	t.Run("deny gate blocks the backend call", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec, denyLockedWrites)
		res := invokeFileTool(t, stateCtx(nil), m, "write_file",
			map[string]any{"file_path": "/locked/a.txt", "content": "x"})
		want := "Error: permission denied for write on /locked/a.txt"
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		if rec.calls["write"] != 0 {
			t.Fatalf("backend Write called %d times under deny; want 0", rec.calls["write"])
		}
		if res.Artifact != nil {
			t.Fatalf("Artifact = %#v under deny; want nil", res.Artifact)
		}
	})

	t.Run("deny rule matches the normalized path when the model sends a relative one", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec, denyLockedWrites)
		res := invokeFileTool(t, stateCtx(nil), m, "write_file",
			map[string]any{"file_path": "locked/a.txt", "content": "x"})
		want := "Error: permission denied for write on /locked/a.txt"
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		if rec.calls["write"] != 0 {
			t.Fatalf("backend Write called %d times under deny; want 0", rec.calls["write"])
		}
	})

	t.Run("path traversal is rejected", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
		res := invokeFileTool(t, stateCtx(nil), m, "write_file",
			map[string]any{"file_path": "../etc/passwd", "content": "x"})
		want := "Error: Path traversal not allowed: ../etc/passwd"
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
	})

	t.Run("interrupt mode behaves as allow for now", func(t *testing.T) {
		interrupt := backends.PermissionRule{
			Operations: []string{"write"},
			Paths:      []string{"/**"},
			Mode:       backends.PermissionInterrupt,
		}
		m := mustNewFilesystemMiddleware(t, backends.NewStateBackend(), interrupt)
		res := invokeFileTool(t, stateCtx(nil), m, "write_file",
			map[string]any{"file_path": "/a.txt", "content": "hi"})
		if want := "Updated file /a.txt"; res.Content != want {
			t.Fatalf("Content = %q, want %q (interrupt placeholder must behave as allow until Task 6)", res.Content, want)
		}
	})
}

// --- edit_file ------------------------------------------------------------

func TestEditFileTool(t *testing.T) {
	files := map[string]backends.FileMeta{"/a.txt": meta("foo bar\n")}
	tests := []struct {
		name         string
		ctxFiles     map[string]backends.FileMeta
		args         map[string]any
		want         string
		wantArtifact bool
	}{
		{
			name:         "single replacement",
			ctxFiles:     files,
			args:         map[string]any{"file_path": "/a.txt", "old_string": "foo", "new_string": "baz"},
			want:         "Successfully replaced 1 instance(s) of the string in '/a.txt'",
			wantArtifact: true,
		},
		{
			name:         "replace_all",
			ctxFiles:     map[string]backends.FileMeta{"/a.txt": meta("foo foo\n")},
			args:         map[string]any{"file_path": "/a.txt", "old_string": "foo", "new_string": "bar", "replace_all": true},
			want:         "Successfully replaced 2 instance(s) of the string in '/a.txt'",
			wantArtifact: true,
		},
		{
			name:     "string not found",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/a.txt", "old_string": "zzz", "new_string": "y"},
			want:     "Error: String not found in file: 'zzz'",
		},
		{
			name:     "ambiguous without replace_all",
			ctxFiles: map[string]backends.FileMeta{"/a.txt": meta("foo foo\n")},
			args:     map[string]any{"file_path": "/a.txt", "old_string": "foo", "new_string": "bar"},
			want:     "Error: String 'foo' appears 2 times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context.",
		},
		{
			name:     "missing file",
			ctxFiles: files,
			args:     map[string]any{"file_path": "/missing.txt", "old_string": "a", "new_string": "b"},
			want:     "Error: File '/missing.txt' not found",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
			res := invokeFileTool(t, stateCtx(tt.ctxFiles), m, "edit_file", tt.args)
			if res.Content != tt.want {
				t.Fatalf("Content = %q, want %q", res.Content, tt.want)
			}
			if tt.wantArtifact && res.Artifact == nil {
				t.Fatalf("Artifact = nil; want files Command for a mutating tool")
			}
			if !tt.wantArtifact && res.Artifact != nil {
				t.Fatalf("Artifact = %#v; want nil on error path", res.Artifact)
			}
		})
	}

	t.Run("deny gate blocks the backend call", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec, denyLockedWrites)
		res := invokeFileTool(t, stateCtx(nil), m, "edit_file",
			map[string]any{"file_path": "/locked/a.txt", "old_string": "a", "new_string": "b"})
		want := "Error: permission denied for write on /locked/a.txt"
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		if rec.calls["edit"] != 0 {
			t.Fatalf("backend Edit called %d times under deny; want 0", rec.calls["edit"])
		}
	})
}

// --- delete -----------------------------------------------------------------

func TestDeleteTool(t *testing.T) {
	files := map[string]backends.FileMeta{"/a.txt": meta("x"), "/b/c.txt": meta("y")}

	t.Run("deletes by file_path argument and returns a deletion artifact", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
		res := invokeFileTool(t, stateCtx(files), m, "delete", map[string]any{"file_path": "/a.txt"})
		if want := "Deleted /a.txt"; res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		delta := filesArtifact(t, res)
		entry, ok := delta["/a.txt"]
		if !ok || entry != nil {
			t.Fatalf("files delta = %#v; want /a.txt mapped to a nil *FileMeta deletion sentinel", delta)
		}
		if len(delta) != 1 {
			t.Fatalf("files delta = %#v; want exactly the deleted key", delta)
		}
	})

	t.Run("missing file surfaces the backend error text", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
		res := invokeFileTool(t, stateCtx(files), m, "delete", map[string]any{"file_path": "/nope.txt"})
		if want := "Error: File '/nope.txt' not found"; res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
		if res.Artifact != nil {
			t.Fatalf("Artifact = %#v on error; want nil", res.Artifact)
		}
	})

	t.Run("deny gate checks file_path against write rules", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec, denyLockedWrites)
		res := invokeFileTool(t, stateCtx(files), m, "delete", map[string]any{"file_path": "/locked/a.txt"})
		want := "Error: permission denied for write on /locked/a.txt"
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
	})

	t.Run("not registered when the backend lacks Deleter", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, &blockingBackend{})
		for _, tool := range m.Tools {
			if tool.Name() == "delete" {
				t.Fatalf("delete tool registered for a non-Deleter backend: %#v", tool)
			}
		}
	})
}

// --- glob ---------------------------------------------------------------------

func TestGlobTool(t *testing.T) {
	seed := map[string]backends.FileMeta{
		"/a.py":     meta("x"),
		"/sub/b.py": meta("y"),
		"/c.md":     meta("z"),
	}
	tests := []struct {
		name  string
		args  map[string]any
		rules []backends.PermissionRule
		want  string
	}{
		{
			name: "matching pattern under root",
			args: map[string]any{"pattern": "*.py", "path": "/"},
			want: "['/a.py', '/sub/b.py']",
		},
		{
			name: "no matches",
			args: map[string]any{"pattern": "*.rs", "path": "/"},
			want: "No files found",
		},
		{
			name: "pathless glob gates and searches at root",
			args: map[string]any{"pattern": "*.md"},
			want: "['/c.md']",
		},
		{
			name: "deny gate on the search path",
			args: map[string]any{"pattern": "*.py", "path": "/secret"},
			rules: []backends.PermissionRule{{
				Operations: []string{"read"},
				Paths:      []string{"/secret/**"},
				Mode:       backends.PermissionDeny,
			}},
			want: "Error: permission denied for read on /secret",
		},
		{
			name: "pathless glob conservatively denied by a root-wide deny rule",
			args: map[string]any{"pattern": "*.py"},
			rules: []backends.PermissionRule{{
				Operations: []string{"read"},
				Paths:      []string{"/**"},
				Mode:       backends.PermissionDeny,
			}},
			want: "Error: permission denied for read on /",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rules := tt.rules
			if rules == nil {
				rules = []backends.PermissionRule{denySecretsRead}
			}
			rec := newRecordingBackend(backends.NewStateBackend())
			m := mustNewFilesystemMiddleware(t, rec, rules...)
			res := invokeFileTool(t, stateCtx(seed), m, "glob", tt.args)
			if res.Content != tt.want {
				t.Fatalf("Content = %q, want %q", res.Content, tt.want)
			}
			if res.Artifact != nil {
				t.Fatalf("glob is read-only; Artifact = %#v, want nil", res.Artifact)
			}
		})
	}

	t.Run("pathless glob passes root to the backend", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec)
		invokeFileTool(t, stateCtx(seed), m, "glob", map[string]any{"pattern": "*.md"})
		if rec.globPath != "/" {
			t.Fatalf("backend Glob path = %q; want \"/\" for a pathless call", rec.globPath)
		}
	})

	t.Run("times out with the upstream error text", func(t *testing.T) {
		overrideFileToolTimeouts(t, 50*time.Millisecond, 50*time.Millisecond)
		m := mustNewFilesystemMiddleware(t, &blockingBackend{})
		res := invokeFileTool(t, context.Background(), m, "glob", map[string]any{"pattern": "*.py", "path": "/"})
		want := "Error: glob timed out after 0.1s. Try a more specific pattern or a narrower path."
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
	})
}

// --- grep ------------------------------------------------------------------------

func TestGrepTool(t *testing.T) {
	seed := map[string]backends.FileMeta{
		"/a.go":  meta("needle here\nplain\nneedle again\n"),
		"/b.txt": meta("nothing"),
	}
	tests := []struct {
		name  string
		args  map[string]any
		rules []backends.PermissionRule
		want  string
	}{
		{
			name: "default output mode lists matching files",
			args: map[string]any{"pattern": "needle"},
			want: "/a.go",
		},
		{
			name: "content mode groups matched lines under the file",
			args: map[string]any{"pattern": "needle", "output_mode": "content"},
			want: "/a.go:\n  1: needle here\n  3: needle again",
		},
		{
			name: "count mode reports per-file counts",
			args: map[string]any{"pattern": "needle", "output_mode": "count"},
			want: "/a.go: 2",
		},
		{
			name: "no matches",
			args: map[string]any{"pattern": "zzz"},
			want: "No matches found",
		},
		{
			name: "include glob filters the search",
			args: map[string]any{"pattern": "needle", "glob": "*.txt"},
			want: "No matches found",
		},
		{
			name: "deny gate on the search path",
			args: map[string]any{"pattern": "needle", "path": "/secret"},
			rules: []backends.PermissionRule{{
				Operations: []string{"read"},
				Paths:      []string{"/secret/**"},
				Mode:       backends.PermissionDeny,
			}},
			want: "Error: permission denied for read on /secret",
		},
		{
			name: "pathless grep conservatively denied by a root-wide deny rule",
			args: map[string]any{"pattern": "needle"},
			rules: []backends.PermissionRule{{
				Operations: []string{"read"},
				Paths:      []string{"/**"},
				Mode:       backends.PermissionDeny,
			}},
			want: "Error: permission denied for read on /",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rules := tt.rules
			if rules == nil {
				rules = []backends.PermissionRule{denySecretsRead}
			}
			rec := newRecordingBackend(backends.NewStateBackend())
			m := mustNewFilesystemMiddleware(t, rec, rules...)
			res := invokeFileTool(t, stateCtx(seed), m, "grep", tt.args)
			if res.Content != tt.want {
				t.Fatalf("Content = %q, want %q", res.Content, tt.want)
			}
			if res.Artifact != nil {
				t.Fatalf("grep is read-only; Artifact = %#v, want nil", res.Artifact)
			}
		})
	}

	t.Run("default max_count of 1000 is applied when omitted", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec)
		invokeFileTool(t, stateCtx(seed), m, "grep", map[string]any{"pattern": "needle"})
		if rec.grepMax != 1000 {
			t.Fatalf("backend Grep maxCount = %d; want the upstream default 1000", rec.grepMax)
		}
	})

	t.Run("explicit max_count overrides the default", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec)
		invokeFileTool(t, stateCtx(seed), m, "grep", map[string]any{"pattern": "needle", "max_count": 5})
		if rec.grepMax != 5 {
			t.Fatalf("backend Grep maxCount = %d; want 5", rec.grepMax)
		}
	})

	t.Run("pathless grep passes root to the backend", func(t *testing.T) {
		rec := newRecordingBackend(backends.NewStateBackend())
		m := mustNewFilesystemMiddleware(t, rec)
		invokeFileTool(t, stateCtx(seed), m, "grep", map[string]any{"pattern": "needle"})
		if rec.grepPath != "/" {
			t.Fatalf("backend Grep path = %q; want \"/\" for a pathless call", rec.grepPath)
		}
	})

	t.Run("times out with the upstream error text", func(t *testing.T) {
		overrideFileToolTimeouts(t, 50*time.Millisecond, 50*time.Millisecond)
		m := mustNewFilesystemMiddleware(t, &blockingBackend{})
		res := invokeFileTool(t, context.Background(), m, "grep", map[string]any{"pattern": "needle", "path": "/"})
		want := "Error: grep timed out after 0.05s. Try a more specific pattern or a narrower path."
		if res.Content != want {
			t.Fatalf("Content = %q, want %q", res.Content, want)
		}
	})
}

// --- constructor: permission-rule validation + tool registration ------------------

func TestNewFilesystemMiddlewareValidatesPermissionRules(t *testing.T) {
	tests := []struct {
		name    string
		rule    backends.PermissionRule
		wantErr string
	}{
		{
			name:    "valid rule passes",
			rule:    backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"/docs/**"}},
			wantErr: "",
		},
		{
			name:    "relative path is rejected",
			rule:    backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"rel/path"}},
			wantErr: "Permission path must start with '/': 'rel/path'",
		},
		{
			name:    "dot-dot component is rejected",
			rule:    backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"/a/../b"}},
			wantErr: "Permission path must not contain '..': '/a/../b'",
		},
		{
			// A "~" prefix trips the '/'-prefix check first (the reference
			// checks startswith("/") before tilde components); only a tilde
			// component after the leading slash reaches the tilde check.
			name:    "tilde component is rejected",
			rule:    backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"/~/secrets"}},
			wantErr: "Permission path must not contain '~': '/~/secrets'",
		},
		{
			name:    "tilde prefix is rejected by the slash-prefix check",
			rule:    backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"~/secrets"}},
			wantErr: "Permission path must start with '/': '~/secrets'",
		},
		{
			name:    "backslash traversal is rejected",
			rule:    backends.PermissionRule{Operations: []string{"read"}, Paths: []string{"/a\\..\\b"}},
			wantErr: "Permission path must not contain '..': '/a\\..\\b'",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewFilesystemMiddleware(backends.NewStateBackend(), tt.rule)
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("NewFilesystemMiddleware: unexpected error %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("NewFilesystemMiddleware error = nil; want %q", tt.wantErr)
			}
			if err.Error() != tt.wantErr {
				t.Fatalf("error = %q; want %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestFilesystemMiddlewareRegistersTools(t *testing.T) {
	t.Run("state backend gets all seven tools", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, backends.NewStateBackend())
		want := []string{"delete", "edit_file", "glob", "grep", "ls", "read_file", "write_file"}
		got := make([]string, 0, len(m.Tools))
		for _, tool := range m.Tools {
			got = append(got, tool.Name())
		}
		sort.Strings(got)
		if len(got) != len(want) {
			t.Fatalf("tools = %v; want %v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("tools = %v; want %v", got, want)
			}
		}
		if provided := m.ProvidedTools(); len(provided) != len(want) {
			t.Fatalf("ProvidedTools() = %d tools; want %d", len(provided), len(want))
		}
	})

	t.Run("non-Deleter backend omits delete", func(t *testing.T) {
		m := mustNewFilesystemMiddleware(t, &blockingBackend{})
		if len(m.Tools) != 6 {
			t.Fatalf("len(Tools) = %d; want 6 without delete", len(m.Tools))
		}
		for _, tool := range m.Tools {
			if tool.Name() == "delete" {
				t.Fatalf("delete registered for a non-Deleter backend")
			}
		}
	})
}
