package backends

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Compile-time checks: FilesystemBackend implements Backend and Deleter
// (upstream FilesystemBackend satisfies BackendProtocol and supports delete).
var (
	_ Backend = (*FilesystemBackend)(nil)
	_ Deleter = (*FilesystemBackend)(nil)
)

// newTestBackend returns a FilesystemBackend over a fresh temp directory.
func newTestBackend(t *testing.T) (*FilesystemBackend, string) {
	t.Helper()
	root := t.TempDir()
	b, err := NewFilesystemBackend(root)
	if err != nil {
		t.Fatalf("NewFilesystemBackend(%q) error: %v", root, err)
	}
	return b, root
}

// mustWriteFile writes content at rel (slash-separated, relative to root),
// creating parent directories, failing the test on error.
func mustWriteFile(t *testing.T, root, rel, content string) {
	t.Helper()
	full := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", full, err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%q): %v", full, err)
	}
}

// wantCode asserts err is a *Error with the given code.
func wantCode(t *testing.T, err error, code string) {
	t.Helper()
	if err == nil {
		t.Fatalf("want error with code %q, got nil", code)
	}
	var opErr *Error
	if !errors.As(err, &opErr) {
		t.Fatalf("error %v (%T) is not *backends.Error", err, err)
	}
	if opErr.Code != code {
		t.Errorf("error code = %q, want %q (full: %v)", opErr.Code, code, err)
	}
}

func TestNewFilesystemBackend(t *testing.T) {
	root := t.TempDir()
	b, err := NewFilesystemBackend(root)
	if err != nil {
		t.Fatalf("NewFilesystemBackend(%q) error: %v", root, err)
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		t.Fatalf("Abs: %v", err)
	}
	abs, err = filepath.EvalSymlinks(abs)
	if err != nil {
		t.Fatalf("EvalSymlinks: %v", err)
	}
	if b.root != abs {
		t.Errorf("backend root = %q, want resolved %q", b.root, abs)
	}

	if _, err := NewFilesystemBackend(filepath.Join(root, "missing")); err == nil {
		t.Error("NewFilesystemBackend with missing root: want error, got nil")
	}
}

func TestFilesystemBackendWriteReadPagination(t *testing.T) {
	b, _ := newTestBackend(t)

	const five = "l1\nl2\nl3\nl4\nl5\n"
	res, err := b.Write(context.Background(), "/five.txt", five)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if res.Path != "/five.txt" {
		t.Errorf("WriteResult.Path = %q, want %q", res.Path, "/five.txt")
	}

	tests := []struct {
		name       string
		path       string
		offset     int
		limit      int
		content    string
		startLine  int
		endLine    int
		totalLines int
		nextOffset int
	}{
		{"first window", "/five.txt", 0, 2, "l1\nl2\n", 1, 2, 5, 2},
		{"middle window", "/five.txt", 2, 2, "l3\nl4\n", 3, 4, 5, 4},
		{"final window exact", "/five.txt", 4, 2, "l5\n", 5, 5, 5, 0},
		{"limit past eof clamps", "/five.txt", 3, 99, "l4\nl5\n", 4, 5, 5, 0},
		{"negative offset clamps to start", "/five.txt", -3, 1, "l1\n", 1, 1, 5, 1},
		{"no trailing newline", "/noeol.txt", 0, 2000, "x\ny", 1, 2, 2, 0},
		{"crlf window normalized to lf", "/crlf.txt", 0, 2000, "c1\nc2\n", 1, 2, 2, 0},
		{"crlf partial window keeps terminators", "/crlf.txt", 0, 1, "c1\n", 1, 1, 2, 1},
		{"zero limit is empty window", "/five.txt", 1, 0, "", 0, 0, 0, 0},
		{"negative limit is empty window", "/five.txt", 0, -5, "", 0, 0, 0, 0},
		{"empty file returns reminder content", "/empty.txt", 0, 10, "System reminder: File exists but has empty contents", 0, 0, 0, 0},
		{"whitespace-only file returns reminder", "/blank.txt", 0, 10, "System reminder: File exists but has empty contents", 0, 0, 0, 0},
	}
	mustWriteFile(t, b.root, "noeol.txt", "x\ny")
	mustWriteFile(t, b.root, "crlf.txt", "c1\r\nc2\r\n")
	mustWriteFile(t, b.root, "empty.txt", "")
	mustWriteFile(t, b.root, "blank.txt", " \n\t\n")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := b.Read(context.Background(), tt.path, tt.offset, tt.limit)
			if err != nil {
				t.Fatalf("Read(%q, %d, %d) error: %v", tt.path, tt.offset, tt.limit, err)
			}
			if got.Content != tt.content {
				t.Errorf("Content = %q, want %q", got.Content, tt.content)
			}
			if got.StartLine != tt.startLine || got.EndLine != tt.endLine ||
				got.TotalLines != tt.totalLines || got.NextOffset != tt.nextOffset {
				t.Errorf("pagination = (start %d, end %d, total %d, next %d), want (%d, %d, %d, %d)",
					got.StartLine, got.EndLine, got.TotalLines, got.NextOffset,
					tt.startLine, tt.endLine, tt.totalLines, tt.nextOffset)
			}
		})
	}
}

func TestFilesystemBackendReadErrors(t *testing.T) {
	b, root := newTestBackend(t)
	mustWriteFile(t, root, "dir/nested.txt", "x")

	tests := []struct {
		name    string
		path    string
		code    string
		message string
	}{
		{"missing file", "/nope.txt", ErrFileNotFound, "File '/nope.txt' not found"},
		{"directory is not a file", "/dir", ErrFileNotFound, "File '/dir' not found"},
		{"path under a file", "/dir/nested.txt/deeper", ErrFileNotFound, "File '/dir/nested.txt/deeper' not found"},
		{"dotdot traversal", "/dir/../../etc/passwd", ErrInvalidPath, "Path traversal not allowed: /dir/../../etc/passwd"},
		{"tilde rejected", "~/.ssh/id_rsa", ErrInvalidPath, "Path traversal not allowed: ~/.ssh/id_rsa"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := b.Read(context.Background(), tt.path, 0, 10)
			wantCode(t, err, tt.code)
			if err.Error() != tt.code+": "+tt.message {
				t.Errorf("error = %q, want %q", err, tt.code+": "+tt.message)
			}
		})
	}
}

func TestFilesystemBackendReadOffsetExceedsFile(t *testing.T) {
	b, root := newTestBackend(t)
	mustWriteFile(t, root, "five.txt", "l1\nl2\nl3\nl4\nl5\n")

	_, err := b.Read(context.Background(), "/five.txt", 5, 2)
	if err == nil {
		t.Fatal("Read past EOF: want error, got nil")
	}
	want := "Line offset 5 exceeds file length (5 lines)"
	if err.Error() != want {
		t.Errorf("error = %q, want %q", err, want)
	}
}

func TestFilesystemBackendWrite(t *testing.T) {
	b, root := newTestBackend(t)

	t.Run("creates parent directories", func(t *testing.T) {
		res, err := b.Write(context.Background(), "/d1/d2/f.txt", "nested")
		if err != nil {
			t.Fatalf("Write: %v", err)
		}
		if res.Path != "/d1/d2/f.txt" {
			t.Errorf("WriteResult.Path = %q, want /d1/d2/f.txt", res.Path)
		}
		got, err := os.ReadFile(filepath.Join(root, "d1", "d2", "f.txt"))
		if err != nil || string(got) != "nested" {
			t.Errorf("on-disk content = %q, %v; want %q", got, err, "nested")
		}
	})

	t.Run("overwrites existing file", func(t *testing.T) {
		if _, err := b.Write(context.Background(), "/f.txt", "old"); err != nil {
			t.Fatalf("Write: %v", err)
		}
		if _, err := b.Write(context.Background(), "/f.txt", "new"); err != nil {
			t.Fatalf("Write: %v", err)
		}
		got, _ := os.ReadFile(filepath.Join(root, "f.txt"))
		if string(got) != "new" {
			t.Errorf("content = %q, want %q", got, "new")
		}
	})

	t.Run("utf-8 content round-trips", func(t *testing.T) {
		const s = "héllo 世界 🚀\nsecond line\n"
		if _, err := b.Write(context.Background(), "/u.txt", s); err != nil {
			t.Fatalf("Write: %v", err)
		}
		res, err := b.Read(context.Background(), "/u.txt", 0, 100)
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		if res.Content != s {
			t.Errorf("round-trip = %q, want %q", res.Content, s)
		}
		if res.StartLine != 1 || res.EndLine != 2 || res.TotalLines != 2 || res.NextOffset != 0 {
			t.Errorf("pagination = %+v, want start 1 end 2 total 2 next 0", res)
		}
	})

	t.Run("traversal rejected", func(t *testing.T) {
		_, err := b.Write(context.Background(), "/../escape.txt", "x")
		wantCode(t, err, ErrInvalidPath)
	})
}

func TestFilesystemBackendLs(t *testing.T) {
	b, root := newTestBackend(t)
	mustWriteFile(t, root, "b.txt", "b")
	mustWriteFile(t, root, "a.txt", "a")
	mustWriteFile(t, root, "mdir/m1.txt", "m")
	mustWriteFile(t, root, "zdir/z1.txt", "z")

	want := time.Date(2020, 5, 1, 12, 0, 0, 0, time.UTC)
	if err := os.Chtimes(filepath.Join(root, "a.txt"), want, want); err != nil {
		t.Fatalf("Chtimes: %v", err)
	}

	t.Run("root entries sorted by name", func(t *testing.T) {
		res, err := b.Ls(context.Background(), "/")
		if err != nil {
			t.Fatalf("Ls: %v", err)
		}
		if len(res.Entries) != 4 {
			t.Fatalf("len(Entries) = %d, want 4 (%+v)", len(res.Entries), res.Entries)
		}
		wantNames := []string{"a.txt", "b.txt", "mdir", "zdir"}
		wantTypes := []string{"file", "file", "directory", "directory"}
		for i, e := range res.Entries {
			if e.Name != wantNames[i] {
				t.Errorf("Entries[%d].Name = %q, want %q", i, e.Name, wantNames[i])
			}
			if e.Type != wantTypes[i] {
				t.Errorf("Entries[%d].Type = %q, want %q", i, e.Type, wantTypes[i])
			}
		}
		if diff := res.Entries[0].ModifiedAt.Sub(want); diff < -time.Second || diff > time.Second {
			t.Errorf("a.txt ModifiedAt = %v, want ~%v", res.Entries[0].ModifiedAt, want)
		}
	})

	t.Run("nested listing non-recursive", func(t *testing.T) {
		res, err := b.Ls(context.Background(), "/mdir")
		if err != nil {
			t.Fatalf("Ls: %v", err)
		}
		if len(res.Entries) != 1 || res.Entries[0].Name != "m1.txt" || res.Entries[0].Type != "file" {
			t.Errorf("entries = %+v, want single file m1.txt", res.Entries)
		}
	})

	t.Run("empty directory", func(t *testing.T) {
		if err := os.Mkdir(filepath.Join(root, "emptydir"), 0o755); err != nil {
			t.Fatal(err)
		}
		res, err := b.Ls(context.Background(), "/emptydir")
		if err != nil {
			t.Fatalf("Ls: %v", err)
		}
		if len(res.Entries) != 0 {
			t.Errorf("entries = %+v, want empty", res.Entries)
		}
	})

	t.Run("missing path", func(t *testing.T) {
		_, err := b.Ls(context.Background(), "/missing")
		wantCode(t, err, ErrFileNotFound)
		if err.Error() != ErrFileNotFound+": Path '/missing': path_not_found" {
			t.Errorf("error = %q, want path_not_found literal", err)
		}
	})

	t.Run("file is not a directory", func(t *testing.T) {
		_, err := b.Ls(context.Background(), "/a.txt")
		wantCode(t, err, ErrInvalidPath)
		if err.Error() != ErrInvalidPath+": Path '/a.txt': not_a_directory" {
			t.Errorf("error = %q, want not_a_directory literal", err)
		}
	})
}

func TestFilesystemBackendEdit(t *testing.T) {
	const body = "alpha beta gamma\nbeta delta\n"

	t.Run("single occurrence replaced", func(t *testing.T) {
		b, _ := newTestBackend(t)
		if _, err := b.Write(context.Background(), "/f.txt", body); err != nil {
			t.Fatal(err)
		}
		res, err := b.Edit(context.Background(), "/f.txt", "gamma", "G", false)
		if err != nil {
			t.Fatalf("Edit: %v", err)
		}
		if res.Replaced != 1 {
			t.Errorf("Replaced = %d, want 1", res.Replaced)
		}
		got, _ := os.ReadFile(filepath.Join(b.root, "f.txt"))
		if string(got) != "alpha beta G\nbeta delta\n" {
			t.Errorf("content = %q", got)
		}
	})

	t.Run("replaceAll replaces every occurrence", func(t *testing.T) {
		b, _ := newTestBackend(t)
		if _, err := b.Write(context.Background(), "/f.txt", body); err != nil {
			t.Fatal(err)
		}
		res, err := b.Edit(context.Background(), "/f.txt", "beta", "BETA", true)
		if err != nil {
			t.Fatalf("Edit: %v", err)
		}
		if res.Replaced != 2 {
			t.Errorf("Replaced = %d, want 2", res.Replaced)
		}
		got, _ := os.ReadFile(filepath.Join(b.root, "f.txt"))
		if string(got) != "alpha BETA gamma\nBETA delta\n" {
			t.Errorf("content = %q", got)
		}
	})

	errorTests := []struct {
		name       string
		content    string
		oldString  string
		newString  string
		replaceAll bool
		want       string
	}{
		{
			name:      "absent string",
			content:   body,
			oldString: "zzz",
			want:      "Error: String not found in file: 'zzz'",
		},
		{
			name:      "multiple without replaceAll",
			content:   body,
			oldString: "beta",
			want:      "Error: String 'beta' appears 2 times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context.",
		},
		{
			name:      "empty old_string",
			content:   body,
			oldString: "",
			newString: "x",
			want:      "Error: old_string cannot be empty. Provide the exact text to replace.",
		},
		{
			name:      "eof newline mismatch unique",
			content:   "one\ntwo",
			oldString: "two\n",
			newString: "TWO\n",
			want: "Error: old_string ends with a newline, but the file does not end with a newline. " +
				"Retry with the trailing newline removed from old_string (and from new_string if it also ends with a newline).",
		},
		{
			name:      "eof newline mismatch ambiguous",
			content:   "two two",
			oldString: "two\n",
			want: "Error: old_string ends with a newline, but the file does not end with a newline. " +
				"With the trailing newline removed, old_string would appear 2 times in the file. " +
				"Retry with the trailing newline removed and add surrounding context so the match is unique.",
		},
	}
	for _, tt := range errorTests {
		t.Run(tt.name, func(t *testing.T) {
			b, _ := newTestBackend(t)
			if _, err := b.Write(context.Background(), "/f.txt", tt.content); err != nil {
				t.Fatal(err)
			}
			_, err := b.Edit(context.Background(), "/f.txt", tt.oldString, tt.newString, tt.replaceAll)
			if err == nil {
				t.Fatalf("Edit: want error %q, got nil", tt.want)
			}
			if err.Error() != tt.want {
				t.Errorf("error =\n%q\nwant\n%q", err, tt.want)
			}
		})
	}

	t.Run("missing file", func(t *testing.T) {
		b, _ := newTestBackend(t)
		_, err := b.Edit(context.Background(), "/nope.txt", "a", "b", false)
		wantCode(t, err, ErrFileNotFound)
		if err.Error() != ErrFileNotFound+": Error: File '/nope.txt' not found" {
			t.Errorf("error = %q", err)
		}
	})

	t.Run("traversal rejected", func(t *testing.T) {
		b, _ := newTestBackend(t)
		_, err := b.Edit(context.Background(), "/../x", "a", "b", false)
		wantCode(t, err, ErrInvalidPath)
	})
}

func TestFilesystemBackendDelete(t *testing.T) {
	t.Run("file", func(t *testing.T) {
		b, root := newTestBackend(t)
		mustWriteFile(t, root, "f.txt", "x")
		res, err := b.Delete(context.Background(), "/f.txt")
		if err != nil {
			t.Fatalf("Delete: %v", err)
		}
		if !res.Success {
			t.Error("DeleteResult.Success = false, want true")
		}
		if _, err := os.Stat(filepath.Join(root, "f.txt")); !os.IsNotExist(err) {
			t.Error("file still exists after delete")
		}
	})

	t.Run("directory recursive", func(t *testing.T) {
		b, root := newTestBackend(t)
		mustWriteFile(t, root, "d/inner/deep/f.txt", "x")
		mustWriteFile(t, root, "d/top.txt", "x")
		res, err := b.Delete(context.Background(), "/d")
		if err != nil {
			t.Fatalf("Delete: %v", err)
		}
		if !res.Success {
			t.Error("Success = false, want true")
		}
		if _, err := os.Stat(filepath.Join(root, "d")); !os.IsNotExist(err) {
			t.Error("directory tree still exists after delete")
		}
	})

	t.Run("missing path", func(t *testing.T) {
		b, _ := newTestBackend(t)
		_, err := b.Delete(context.Background(), "/nope")
		wantCode(t, err, ErrFileNotFound)
		if err.Error() != ErrFileNotFound+": Error: '/nope' not found" {
			t.Errorf("error = %q", err)
		}
	})
}

func TestFilesystemBackendGlob(t *testing.T) {
	b, root := newTestBackend(t)
	mustWriteFile(t, root, "a.go", "x")
	mustWriteFile(t, root, "a.txt", "x")
	mustWriteFile(t, root, "sub/b.go", "x")
	mustWriteFile(t, root, "sub/deep/c.go", "x")
	mustWriteFile(t, root, "visible.py", "x")
	mustWriteFile(t, root, ".hidden.py", "x")
	mustWriteFile(t, root, ".git/config", "x")

	tests := []struct {
		name    string
		path    string
		pattern string
		want    []string
		wantErr string // error code, empty means no error
	}{
		{"doublestar everything skips dotfiles", "/", "**",
			[]string{"/a.go", "/a.txt", "/sub/b.go", "/sub/deep/c.go", "/visible.py"}, ""},
		{"slash-less pattern matches basenames at any depth", "/", "*.go",
			[]string{"/a.go", "/sub/b.go", "/sub/deep/c.go"}, ""},
		{"slash pattern is relative and depth-locked", "/", "sub/*.go",
			[]string{"/sub/b.go"}, ""},
		{"doublestar between segments", "/", "sub/**/*.go", []string{"/sub/b.go", "/sub/deep/c.go"}, ""},
		{"leading slash anchors to search root", "/", "/sub/**/*.go", []string{"/sub/b.go", "/sub/deep/c.go"}, ""},
		{"search under subdir", "/sub", "**", []string{"/sub/b.go", "/sub/deep/c.go"}, ""},
		{"slash-less under subdir matches any depth", "/sub", "*.go",
			[]string{"/sub/b.go", "/sub/deep/c.go"}, ""},
		{"dotfiles excluded without explicit dot", "/", "*.py", []string{"/visible.py"}, ""},
		{"explicit dot pattern reaches dotfiles", "/", ".git/**", []string{"/.git/config"}, ""},
		{"no matches is empty not error", "/", "*.md", []string{}, ""},
		{"missing search path is empty not error", "/missing", "**", []string{}, ""},
		{"dotdot pattern rejected", "/", "..", nil, ErrInvalidPath},
		{"dotdot path rejected", "/../x", "*", nil, ErrInvalidPath},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := b.Glob(context.Background(), tt.path, tt.pattern)
			if tt.wantErr != "" {
				wantCode(t, err, tt.wantErr)
				return
			}
			if err != nil {
				t.Fatalf("Glob(%q, %q): %v", tt.path, tt.pattern, err)
			}
			if fmt.Sprint(res.Files) != fmt.Sprint(tt.want) {
				t.Errorf("Files = %v, want %v", res.Files, tt.want)
			}
		})
	}
}

func TestFilesystemBackendCRLFNormalization(t *testing.T) {
	ctx := context.Background()

	t.Run("read normalizes crlf and lone cr like universal newlines", func(t *testing.T) {
		b, root := newTestBackend(t)
		mustWriteFile(t, root, "mix.txt", "a\r\nb\rc\n")
		res, err := b.Read(ctx, "/mix.txt", 0, 100)
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		if res.Content != "a\nb\nc\n" {
			t.Errorf("Content = %q, want %q", res.Content, "a\nb\nc\n")
		}
		if res.StartLine != 1 || res.EndLine != 3 || res.TotalLines != 3 || res.NextOffset != 0 {
			t.Errorf("pagination = start %d end %d total %d next %d, want 1/3/3/0",
				res.StartLine, res.EndLine, res.TotalLines, res.NextOffset)
		}
	})

	t.Run("edit matches lf pattern in crlf file and stores lf", func(t *testing.T) {
		b, root := newTestBackend(t)
		mustWriteFile(t, root, "f.txt", "one\r\ntwo\r\n")
		res, err := b.Edit(ctx, "/f.txt", "one\ntwo", "ONE\nTWO", false)
		if err != nil {
			t.Fatalf("Edit: %v", err)
		}
		if res.Replaced != 1 {
			t.Errorf("Replaced = %d, want 1", res.Replaced)
		}
		got, _ := os.ReadFile(filepath.Join(root, "f.txt"))
		if string(got) != "ONE\nTWO\n" {
			t.Errorf("on-disk content = %q, want %q", got, "ONE\nTWO\n")
		}
	})

	t.Run("edit rewrites whole file as lf", func(t *testing.T) {
		b, root := newTestBackend(t)
		mustWriteFile(t, root, "f.txt", "keep\r\nswap\r\n")
		if _, err := b.Edit(ctx, "/f.txt", "swap", "X", false); err != nil {
			t.Fatalf("Edit: %v", err)
		}
		got, _ := os.ReadFile(filepath.Join(root, "f.txt"))
		if string(got) != "keep\nX\n" {
			t.Errorf("on-disk content = %q, want %q", got, "keep\nX\n")
		}
	})

	t.Run("write stores lf", func(t *testing.T) {
		b, root := newTestBackend(t)
		if _, err := b.Write(ctx, "/c.txt", "a\r\nb\r\n"); err != nil {
			t.Fatalf("Write: %v", err)
		}
		got, _ := os.ReadFile(filepath.Join(root, "c.txt"))
		if string(got) != "a\nb\n" {
			t.Errorf("on-disk content = %q, want %q", got, "a\nb\n")
		}
		res, err := b.Read(ctx, "/c.txt", 0, 10)
		if err != nil || res.Content != "a\nb\n" {
			t.Errorf("read back = %q, %v; want %q", res.Content, err, "a\nb\n")
		}
	})

	t.Run("grep line text has crlf stripped", func(t *testing.T) {
		b, root := newTestBackend(t)
		mustWriteFile(t, root, "g.txt", "x\r\nneedle here\r\n")
		res, err := b.Grep(ctx, "needle here", "/", "", 0)
		if err != nil {
			t.Fatalf("Grep: %v", err)
		}
		if len(res.Matches) != 1 || len(res.Matches[0].Lines) != 1 {
			t.Fatalf("matches = %+v, want one line in /g.txt", res.Matches)
		}
		if res.Matches[0].Lines[0].Text != "needle here" {
			t.Errorf("text = %q, want %q (no trailing CR)", res.Matches[0].Lines[0].Text, "needle here")
		}
	})
}

func TestFilesystemBackendGrep(t *testing.T) {
	b, root := newTestBackend(t)
	mustWriteFile(t, root, ".hid.txt", "needle hidden\n")
	mustWriteFile(t, root, "alpha.txt", "needle here\nplain line\nneedle again\n")
	mustWriteFile(t, root, "beta.txt", "other\nneedle two\n")
	mustWriteFile(t, root, "gam.go", "needle in go\n")
	mustWriteFile(t, root, "lit.txt", "abc\nxa.cy\n")
	mustWriteFile(t, root, "sub/deep/d.txt", "needle deep\n")

	paths := func(m []GrepFileMatches) string {
		parts := make([]string, 0, len(m))
		for _, fm := range m {
			nums := make([]string, 0, len(fm.Lines))
			for _, l := range fm.Lines {
				nums = append(nums, fmt.Sprintf("%d:%s", l.LineNumber, l.Text))
			}
			parts = append(parts, fm.Path+"["+strings.Join(nums, " ")+"]")
		}
		return strings.Join(parts, " ")
	}

	tests := []struct {
		name     string
		pattern  string
		path     string
		glob     string
		maxCount int
		want     string
	}{
		{"all matches grouped by file incl dotfiles", "needle", "/", "", 0,
			"/.hid.txt[1:needle hidden] /alpha.txt[1:needle here 3:needle again] " +
				"/beta.txt[2:needle two] /gam.go[1:needle in go] /sub/deep/d.txt[1:needle deep]"},
		{"maxCount caps total across files", "needle", "/", "", 2,
			"/.hid.txt[1:needle hidden] /alpha.txt[1:needle here]"},
		{"maxCount exactly at cap", "needle", "/", "", 3,
			"/.hid.txt[1:needle hidden] /alpha.txt[1:needle here 3:needle again]"},
		{"include glob basenames any depth, dotfiles excluded", "needle", "/", "*.txt", 0,
			"/alpha.txt[1:needle here 3:needle again] /beta.txt[2:needle two] /sub/deep/d.txt[1:needle deep]"},
		{"anchored include glob is depth-locked", "needle", "/", "/*.txt", 0,
			"/alpha.txt[1:needle here 3:needle again] /beta.txt[2:needle two]"},
		{"slash include glob is path-relative", "needle", "/", "sub/*.txt", 0, ""},
		{"doublestar include glob", "needle", "/", "sub/**/*.txt", 0, "/sub/deep/d.txt[1:needle deep]"},
		{"bare doublestar include glob skips dotfiles", "needle", "/", "**", 0,
			"/alpha.txt[1:needle here 3:needle again] /beta.txt[2:needle two] " +
				"/gam.go[1:needle in go] /sub/deep/d.txt[1:needle deep]"},
		{"include glob with slash is path-relative", "needle", "/", "sub/**/*.go", 0, ""},
		{"literal match is not regex", "a.c", "/", "", 0, "/lit.txt[2:xa.cy]"},
		{"single file base", "needle", "/alpha.txt", "", 0,
			"/alpha.txt[1:needle here 3:needle again]"},
		{"missing base is empty not error", "needle", "/missing", "", 0, ""},
		{"traversal base is empty not error", "needle", "/../etc", "", 0, ""},
		{"no matches", "absent-pattern", "/", "", 0, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := b.Grep(context.Background(), tt.pattern, tt.path, tt.glob, tt.maxCount)
			if err != nil {
				t.Fatalf("Grep(%q, %q, %q, %d): %v", tt.pattern, tt.path, tt.glob, tt.maxCount, err)
			}
			if got := paths(res.Matches); got != tt.want {
				t.Errorf("matches =\n%q\nwant\n%q", got, tt.want)
			}
		})
	}

	t.Run("dotdot include glob rejected", func(t *testing.T) {
		_, err := b.Grep(context.Background(), "x", "/", "../*", 0)
		wantCode(t, err, ErrInvalidPath)
		if err.Error() != ErrInvalidPath+": Path traversal not allowed in glob pattern '../*'" {
			t.Errorf("error = %q", err)
		}
	})
}

func TestFilesystemBackendSymlinkEscapeRejected(t *testing.T) {
	b, root := newTestBackend(t)
	outside := t.TempDir()
	mustWriteFile(t, outside, "secret.txt", "secret")
	if err := os.Symlink(filepath.Join(outside, "secret.txt"), filepath.Join(root, "leak.txt")); err != nil {
		t.Fatalf("Symlink: %v", err)
	}

	_, err := b.Read(context.Background(), "/leak.txt", 0, 10)
	wantCode(t, err, ErrInvalidPath)
	if !strings.Contains(err.Error(), "outside root directory") {
		t.Errorf("error = %q, want outside-root-directory message", err)
	}
}

func TestFilesystemBackendPermissionDenied(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: mode bits are not enforced")
	}
	b, root := newTestBackend(t)
	lockDir := filepath.Join(root, "locked")
	mustWriteFile(t, lockDir, "inner/secret.txt", "s")
	if err := os.Chmod(lockDir, 0o0000); err != nil {
		t.Fatalf("Chmod: %v", err)
	}
	t.Cleanup(func() { _ = os.Chmod(lockDir, 0o755) })

	_, err := b.Read(context.Background(), "/locked/inner/secret.txt", 0, 10)
	wantCode(t, err, ErrPermissionDenied)

	_, err = b.Write(context.Background(), "/locked/new.txt", "x")
	wantCode(t, err, ErrPermissionDenied)
}

func TestPermissionModesAndFileMeta(t *testing.T) {
	if PermissionAllow != "allow" || PermissionDeny != "deny" || PermissionInterrupt != "interrupt" {
		t.Errorf("permission mode literals = %q/%q/%q, want allow/deny/interrupt",
			PermissionAllow, PermissionDeny, PermissionInterrupt)
	}
	var mode PermissionMode = PermissionInterrupt
	var rule PermissionRule
	rule.Operations = []string{"read"}
	rule.Paths = []string{"/logs/**"}
	rule.Mode = mode
	if len(rule.Operations) != 1 || len(rule.Paths) != 1 || rule.Mode != "interrupt" {
		t.Errorf("rule = %+v", rule)
	}

	meta := FileMeta{Content: "c", Encoding: "utf-8"}
	if meta.Encoding != "utf-8" {
		t.Errorf("FileMeta.Encoding = %q", meta.Encoding)
	}
}
