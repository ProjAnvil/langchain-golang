package backends

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// Compile-time checks: StateBackend implements Backend and Deleter (upstream
// StateBackend satisfies BackendProtocol and supports delete). The
// channels.Reducer assertion on MapMergeReducer lives in state.go next to the
// reducer itself.
var (
	_ Backend = (*StateBackend)(nil)
	_ Deleter = (*StateBackend)(nil)
)

func meta(content string) FileMeta {
	now := time.Now()
	return FileMeta{Content: content, Encoding: "utf-8", CreatedAt: now, ModifiedAt: now}
}

func TestMapMergeReducer(t *testing.T) {
	a1, a2 := meta("one"), meta("two")
	b1, b2 := meta("ONE"), meta("TWO")

	tests := []struct {
		name     string
		existing any
		update   any
		want     map[string]FileMeta
		wantErr  bool
		errArg   string // MapMergeError.Arg when wantErr
	}{
		{
			name:     "merge disjoint keys",
			existing: map[string]FileMeta{"/a": a1},
			update:   map[string]*FileMeta{"/b": &b1},
			want:     map[string]FileMeta{"/a": a1, "/b": b1},
		},
		{
			name:     "update overwrites existing value",
			existing: map[string]FileMeta{"/a": a1},
			update:   map[string]*FileMeta{"/a": &b1},
			want:     map[string]FileMeta{"/a": b1},
		},
		{
			name:     "nil pointer deletes key",
			existing: map[string]FileMeta{"/a": a1, "/b": b2},
			update:   map[string]*FileMeta{"/a": nil},
			want:     map[string]FileMeta{"/b": b2},
		},
		{
			name:     "delete absent key is a no-op",
			existing: map[string]FileMeta{"/a": a1},
			update:   map[string]*FileMeta{"/gone": nil},
			want:     map[string]FileMeta{"/a": a1},
		},
		{
			name:     "nil existing returns filtered update",
			existing: nil,
			update:   map[string]*FileMeta{"/a": &a1, "/b": nil},
			want:     map[string]FileMeta{"/a": a1},
		},
		{
			name:     "nil update returns copy of existing",
			existing: map[string]FileMeta{"/a": a1},
			update:   nil,
			want:     map[string]FileMeta{"/a": a1},
		},
		{
			name:     "both nil yields empty map",
			existing: nil,
			update:   nil,
			want:     map[string]FileMeta{},
		},
		{
			name:     "delete and write disjoint keys both applied",
			existing: map[string]FileMeta{"/keep": a1, "/drop": a2},
			update:   map[string]*FileMeta{"/drop": nil, "/new": &b1},
			want:     map[string]FileMeta{"/keep": a1, "/new": b1},
		},
		{
			name:     "typed nil maps behave as empty",
			existing: map[string]FileMeta(nil),
			update:   map[string]*FileMeta(nil),
			want:     map[string]FileMeta{},
		},
		{
			name:     "non-map existing is rejected",
			existing: 42,
			update:   nil,
			wantErr:  true,
			errArg:   "existing",
		},
		{
			name:     "non-map update is rejected",
			existing: nil,
			update:   "files",
			wantErr:  true,
			errArg:   "update",
		},
		{
			name:     "wrong map value type update is rejected",
			existing: nil,
			update:   map[string]string{"/a": "content"},
			wantErr:  true,
			errArg:   "update",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MapMergeReducer(tt.existing, tt.update)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("MapMergeReducer(%v, %v) = %v, want error", tt.existing, tt.update, got)
				}
				var me *MapMergeError
				if !errors.As(err, &me) {
					t.Fatalf("error %v (%T) is not *MapMergeError", err, err)
				}
				if me.Arg != tt.errArg {
					t.Errorf("MapMergeError.Arg = %q, want %q (full: %v)", me.Arg, tt.errArg, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("MapMergeReducer error: %v", err)
			}
			m, ok := got.(map[string]FileMeta)
			if !ok {
				t.Fatalf("result type %T, want map[string]FileMeta", got)
			}
			if m == nil {
				t.Fatal("result map is nil, want non-nil copy")
			}
			if !reflect.DeepEqual(m, tt.want) {
				t.Errorf("result = %v, want %v", m, tt.want)
			}
		})
	}
}

func TestMapMergeReducerReturnsCopy(t *testing.T) {
	a1 := meta("one")
	existing := map[string]FileMeta{"/a": a1}
	got, err := MapMergeReducer(existing, nil)
	if err != nil {
		t.Fatalf("MapMergeReducer error: %v", err)
	}
	m := got.(map[string]FileMeta)
	m["/b"] = meta("injected")
	delete(m, "/a")
	if len(existing) != 1 || existing["/a"] != a1 {
		t.Errorf("mutating the result changed the existing map: %v", existing)
	}
}

func TestStateView(t *testing.T) {
	t.Run("empty view", func(t *testing.T) {
		v := NewStateView(nil)
		if len(v.Files()) != 0 {
			t.Errorf("Files() = %v, want empty", v.Files())
		}
		if v.Files() == nil {
			t.Error("Files() is nil, want non-nil map")
		}
		if ch := v.Changed(); ch != nil {
			t.Errorf("Changed() = %v, want nil before any mutation", ch)
		}
	})

	t.Run("seed map is copied", func(t *testing.T) {
		seed := map[string]FileMeta{"/a": meta("one")}
		v := NewStateView(seed)
		v.Set("/b", meta("two"))
		v.Delete("/a")
		if len(seed) != 1 || seed["/a"].Content != "one" {
			t.Errorf("mutations escaped into the seed map: %v", seed)
		}
	})

	t.Run("set and delete record the delta and the live map", func(t *testing.T) {
		v := NewStateView(map[string]FileMeta{"/keep": meta("kept")})
		m := meta("written")
		v.Set("/new", m)
		if got := v.Files()["/new"]; got != m {
			t.Errorf("Files()[\"/new\"] = %v, want %v (read-your-writes)", got, m)
		}
		delta := v.Changed()
		if len(delta) != 1 || delta["/new"] == nil || *delta["/new"] != m {
			t.Errorf("Changed() = %v, want {\"/new\": %v}", delta, m)
		}

		v.Delete("/new")
		if _, ok := v.Files()["/new"]; ok {
			t.Error("Delete did not remove /new from Files()")
		}
		delta = v.Changed()
		if delta["/new"] != nil {
			t.Errorf("Changed()[\"/new\"] = %v, want nil delete marker", delta["/new"])
		}
		// /keep was never touched, so the delta stays one key.
		if len(delta) != 1 {
			t.Errorf("Changed() = %v, want exactly the /new key", delta)
		}
	})

	t.Run("set after delete wins", func(t *testing.T) {
		v := NewStateView(map[string]FileMeta{"/a": meta("one")})
		v.Delete("/a")
		m := meta("resurrected")
		v.Set("/a", m)
		delta := v.Changed()
		if delta["/a"] == nil || *delta["/a"] != m {
			t.Errorf("Changed()[\"/a\"] = %v, want write after delete", delta["/a"])
		}
	})
}

func TestStateViewContext(t *testing.T) {
	v := NewStateView(nil)
	ctx := WithStateView(context.Background(), v)
	got, ok := StateViewFromContext(ctx)
	if !ok || got != v {
		t.Fatalf("StateViewFromContext = (%p, %v), want (%p, true)", got, ok, v)
	}
	if _, ok := StateViewFromContext(context.Background()); ok {
		t.Error("StateViewFromContext on bare context: want ok=false")
	}
	if _, ok := StateViewFromContext(WithStateView(context.Background(), nil)); ok {
		t.Error("StateViewFromContext with nil view: want ok=false")
	}
}

// TestStateViewSnapshotSemantics pins the concurrency contract's snapshot
// rule: maps returned by Files and Changed never observe later mutations.
func TestStateViewSnapshotSemantics(t *testing.T) {
	v := NewStateView(map[string]FileMeta{"/a": meta("one")})

	filesSnap := v.Files()
	deltaSnap := v.Changed() // nil: no mutations yet
	if deltaSnap != nil {
		t.Fatalf("Changed() before mutation = %v, want nil", deltaSnap)
	}
	v.Set("/b", meta("two"))
	v.Delete("/a")

	if len(filesSnap) != 1 || filesSnap["/a"].Content != "one" {
		t.Errorf("Files snapshot changed after later mutations: %v", filesSnap)
	}
	if _, ok := filesSnap["/b"]; ok {
		t.Error("Files snapshot gained /b after a later Set")
	}

	v.Delete("/b")
	deltaSnap = v.Changed() // {/a: nil, /b: nil}
	v.Set("/c", meta("three"))
	if _, ok := deltaSnap["/c"]; ok {
		t.Error("Changed snapshot gained /c after a later Set")
	}
	if len(deltaSnap) != 2 {
		t.Errorf("Changed snapshot = %v, want the two earlier keys only", deltaSnap)
	}

	fresh := v.Files()
	if len(fresh) != 1 || fresh["/c"].Content != "three" {
		t.Errorf("fresh Files() = %v, want only /c", fresh)
	}
}

// TestStateViewConcurrentUse exercises the parallel-tool-batches contract
// under the race detector: N goroutines mutate disjoint and shared keys
// concurrently while reader goroutines snapshot Files().
func TestStateViewConcurrentUse(t *testing.T) {
	const goroutines = 16
	const iterations = 50

	tests := []struct {
		name string
		run  func(t *testing.T, v *StateView)
	}{
		{
			// Disjoint keys: each goroutine owns /g<i>/k<j>.txt, so the
			// final delta per key is deterministic — a delete marker for
			// k==3 (its last operation), the final write otherwise.
			name: "disjoint keys interleaved set and delete",
			run: func(t *testing.T, v *StateView) {
				var wg sync.WaitGroup
				for g := 0; g < goroutines; g++ {
					wg.Add(1)
					go func(g int) {
						defer wg.Done()
						for k := 0; k < 5; k++ {
							key := fmt.Sprintf("/g%d/k%d.txt", g, k)
							for i := 0; i < iterations; i++ {
								v.Set(key, meta(fmt.Sprintf("g%d-k%d-v%d", g, k, i)))
								if i%2 == 1 {
									v.Delete(key)
								}
							}
							if k == 3 {
								v.Delete(key)
							} else {
								v.Set(key, meta(fmt.Sprintf("g%d-k%d-final", g, k)))
							}
						}
					}(g)
				}
				wg.Wait()
				delta := v.Changed()
				for g := 0; g < goroutines; g++ {
					for k := 0; k < 5; k++ {
						key := fmt.Sprintf("/g%d/k%d.txt", g, k)
						entry, ok := delta[key]
						if !ok {
							t.Errorf("delta[%s] missing", key)
							continue
						}
						if k == 3 {
							if entry != nil {
								t.Errorf("delta[%s] = %v, want nil delete marker", key, entry)
							}
							continue
						}
						if entry == nil || entry.Content != fmt.Sprintf("g%d-k%d-final", g, k) {
							t.Errorf("delta[%s] = %v, want final write", key, entry)
						}
					}
				}
			},
		},
		{
			// Shared key: the final value is nondeterministic under
			// contention, but the delta must carry the key and stay
			// consistent with the final file map.
			name: "same key hammered by all goroutines",
			run: func(t *testing.T, v *StateView) {
				var wg sync.WaitGroup
				for g := 0; g < goroutines; g++ {
					wg.Add(1)
					go func(g int) {
						defer wg.Done()
						for i := 0; i < iterations; i++ {
							if i%2 == 0 {
								v.Set("/shared.txt", meta(fmt.Sprintf("g%d-i%d", g, i)))
							} else {
								v.Delete("/shared.txt")
							}
						}
					}(g)
				}
				wg.Wait()
				delta := v.Changed()
				entry, ok := delta["/shared.txt"]
				if !ok {
					t.Fatal("delta[/shared.txt] missing")
				}
				file, present := v.Files()["/shared.txt"]
				if entry == nil && present {
					t.Error("delta records a delete but the file is present")
				}
				if entry != nil {
					if !present {
						t.Error("delta records a write but the file is absent")
					} else if file.Content != entry.Content {
						t.Errorf("file content %q != delta content %q", file.Content, entry.Content)
					}
				}
			},
		},
		{
			// Concurrent Files() readers must observe stable snapshots while
			// writers mutate disjoint keys.
			name: "concurrent Files snapshots during writes",
			run: func(t *testing.T, v *StateView) {
				var writers, readers sync.WaitGroup
				stop := make(chan struct{})
				for r := 0; r < 4; r++ {
					readers.Add(1)
					go func() {
						defer readers.Done()
						for {
							select {
							case <-stop:
								return
							default:
							}
							for key, m := range v.Files() {
								if m.Content == "" {
									t.Errorf("snapshot entry %s has empty content", key)
								}
							}
						}
					}()
				}
				for g := 0; g < goroutines; g++ {
					writers.Add(1)
					go func(g int) {
						defer writers.Done()
						key := fmt.Sprintf("/w%d.txt", g)
						for i := 0; i < iterations; i++ {
							v.Set(key, meta(fmt.Sprintf("w%d-i%d", g, i)))
						}
					}(g)
				}
				writers.Wait()
				close(stop)
				readers.Wait()
				files := v.Files()
				for g := 0; g < goroutines; g++ {
					key := fmt.Sprintf("/w%d.txt", g)
					if got, ok := files[key]; !ok || got.Content != fmt.Sprintf("w%d-i%d", g, iterations-1) {
						t.Errorf("files[%s] = (%v, %v), want final write", key, got, ok)
					}
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.run(t, NewStateView(nil))
		})
	}
}

// newStateCtx returns a context carrying a fresh StateView seeded with files.
func newStateCtx(files map[string]FileMeta) (context.Context, *StateView, *StateBackend) {
	v := NewStateView(files)
	return WithStateView(context.Background(), v), v, NewStateBackend()
}

func TestStateBackendNoView(t *testing.T) {
	b := NewStateBackend()
	ctx := context.Background()
	calls := map[string]func() error{
		"Ls":    func() error { _, err := b.Ls(ctx, "/"); return err },
		"Read":  func() error { _, err := b.Read(ctx, "/a.txt", 0, 10); return err },
		"Write": func() error { _, err := b.Write(ctx, "/a.txt", "x"); return err },
		"Edit":  func() error { _, err := b.Edit(ctx, "/a.txt", "x", "y", false); return err },
		"Glob":  func() error { _, err := b.Glob(ctx, "/", "*"); return err },
		"Grep":  func() error { _, err := b.Grep(ctx, "x", "/", "", 0); return err },
		"Delete": func() error {
			_, err := b.Delete(ctx, "/a.txt")
			return err
		},
	}
	for name, call := range calls {
		err := call()
		if err == nil {
			t.Errorf("%s without state view: want error, got nil", name)
			continue
		}
		if err.Error() != "no state view in context" {
			t.Errorf("%s error = %q, want %q", name, err.Error(), "no state view in context")
		}
	}
}

func TestStateBackendEmptyMapIsEmptyResults(t *testing.T) {
	ctx, _, b := newStateCtx(nil)
	ls, err := b.Ls(ctx, "/")
	if err != nil {
		t.Fatalf("Ls on empty state: %v", err)
	}
	if len(ls.Entries) != 0 {
		t.Errorf("Ls entries = %v, want empty", ls.Entries)
	}
	glob, err := b.Glob(ctx, "/", "**")
	if err != nil {
		t.Fatalf("Glob on empty state: %v", err)
	}
	if !slices.Equal(glob.Files, []string{}) {
		t.Errorf("Glob files = %v, want empty non-nil", glob.Files)
	}
	grep, err := b.Grep(ctx, "x", "/", "", 0)
	if err != nil {
		t.Fatalf("Grep on empty state: %v", err)
	}
	if len(grep.Matches) != 0 {
		t.Errorf("Grep matches = %v, want empty", grep.Matches)
	}
}

func TestStateBackendRoundtrip(t *testing.T) {
	ctx, view, b := newStateCtx(nil)

	if res, err := b.Write(ctx, "/a.txt", "hello world\nsecond hello\n"); err != nil || res.Path != "/a.txt" {
		t.Fatalf("Write(/a.txt) = (%v, %v)", res, err)
	}
	if _, err := b.Write(ctx, "/sub/b.py", "print('hello')\n"); err != nil {
		t.Fatalf("Write(/sub/b.py): %v", err)
	}

	t.Run("read window", func(t *testing.T) {
		res, err := b.Read(ctx, "/a.txt", 0, 1)
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		want := ReadResult{Content: "hello world\n", StartLine: 1, EndLine: 1, TotalLines: 2, NextOffset: 1}
		if res != want {
			t.Errorf("Read = %+v, want %+v", res, want)
		}
	})

	t.Run("read missing file", func(t *testing.T) {
		_, err := b.Read(ctx, "/missing.txt", 0, 10)
		wantCode(t, err, ErrFileNotFound)
		if err.Error() != "file_not_found: File '/missing.txt' not found" {
			t.Errorf("Read error = %q", err.Error())
		}
	})

	t.Run("ls lists files and subdirectories sorted", func(t *testing.T) {
		res, err := b.Ls(ctx, "/")
		if err != nil {
			t.Fatalf("Ls(/): %v", err)
		}
		want := []LsEntry{
			{Name: "a.txt", Type: "file", ModifiedAt: view.Files()["/a.txt"].ModifiedAt},
			{Name: "sub", Type: "directory"},
		}
		if !reflect.DeepEqual(res.Entries, want) {
			t.Errorf("Ls(/) = %+v, want %+v", res.Entries, want)
		}

		sub, err := b.Ls(ctx, "/sub")
		if err != nil {
			t.Fatalf("Ls(/sub): %v", err)
		}
		if len(sub.Entries) != 1 || sub.Entries[0].Name != "b.py" || sub.Entries[0].Type != "file" {
			t.Errorf("Ls(/sub) = %+v, want single file b.py", sub.Entries)
		}

		missing, err := b.Ls(ctx, "/missing")
		if err != nil || len(missing.Entries) != 0 {
			t.Errorf("Ls(/missing) = (%v, %v), want empty entries, nil error", missing.Entries, err)
		}
	})

	t.Run("grep groups matches per file in sorted order", func(t *testing.T) {
		res, err := b.Grep(ctx, "hello", "/", "", 0)
		if err != nil {
			t.Fatalf("Grep: %v", err)
		}
		want := []GrepFileMatches{
			{Path: "/a.txt", Lines: []GrepLine{
				{LineNumber: 1, Text: "hello world"},
				{LineNumber: 2, Text: "second hello"},
			}},
			{Path: "/sub/b.py", Lines: []GrepLine{
				{LineNumber: 1, Text: "print('hello')"},
			}},
		}
		if !reflect.DeepEqual(res.Matches, want) {
			t.Errorf("Grep = %+v, want %+v", res.Matches, want)
		}
	})

	t.Run("grep maxCount caps total across files", func(t *testing.T) {
		res, err := b.Grep(ctx, "hello", "/", "", 2)
		if err != nil {
			t.Fatalf("Grep: %v", err)
		}
		if len(res.Matches) != 1 || len(res.Matches[0].Lines) != 2 {
			t.Errorf("Grep maxCount=2 = %+v, want the two /a.txt lines only", res.Matches)
		}
	})

	t.Run("edit replaces and updates timestamps", func(t *testing.T) {
		before := view.Files()["/a.txt"]
		res, err := b.Edit(ctx, "/a.txt", "hello", "goodbye", true)
		if err != nil {
			t.Fatalf("Edit: %v", err)
		}
		if res.Replaced != 2 {
			t.Errorf("Edit replaced = %d, want 2", res.Replaced)
		}
		after := view.Files()["/a.txt"]
		if after.Content != "goodbye world\nsecond goodbye\n" {
			t.Errorf("content after edit = %q", after.Content)
		}
		if !after.CreatedAt.Equal(before.CreatedAt) {
			t.Errorf("CreatedAt changed on edit: %v -> %v", before.CreatedAt, after.CreatedAt)
		}
		if after.ModifiedAt.Before(before.ModifiedAt) {
			t.Errorf("ModifiedAt went backwards on edit: %v -> %v", before.ModifiedAt, after.ModifiedAt)
		}
	})

	t.Run("edit missing file", func(t *testing.T) {
		_, err := b.Edit(ctx, "/missing.txt", "a", "b", false)
		wantCode(t, err, ErrFileNotFound)
		if err.Error() != "file_not_found: Error: File '/missing.txt' not found" {
			t.Errorf("Edit error = %q", err.Error())
		}
	})

	t.Run("delete removes exact and nested keys", func(t *testing.T) {
		res, err := b.Delete(ctx, "/sub")
		if err != nil {
			t.Fatalf("Delete(/sub): %v", err)
		}
		if !res.Success {
			t.Error("Delete success = false")
		}
		if _, ok := view.Files()["/sub/b.py"]; ok {
			t.Error("nested key /sub/b.py survived directory delete")
		}
		if delta := view.Changed(); delta["/sub/b.py"] != nil {
			t.Errorf("Changed delta for /sub/b.py = %v, want nil delete marker", delta["/sub/b.py"])
		}

		if _, err := b.Delete(ctx, "/nope"); err == nil {
			t.Error("Delete of missing path: want error")
		} else if err.Error() != "file_not_found: Error: File '/nope' not found" {
			t.Errorf("Delete error = %q", err.Error())
		}
	})
}

func TestStateBackendWritePreservesCreatedAt(t *testing.T) {
	ctx, view, b := newStateCtx(nil)
	if _, err := b.Write(ctx, "/a.txt", "v1"); err != nil {
		t.Fatalf("first Write: %v", err)
	}
	first := view.Files()["/a.txt"]
	if first.Encoding != "utf-8" {
		t.Errorf("encoding = %q, want utf-8", first.Encoding)
	}
	if first.ModifiedAt.Before(first.CreatedAt) {
		t.Errorf("ModifiedAt %v before CreatedAt %v on create", first.ModifiedAt, first.CreatedAt)
	}

	if _, err := b.Write(ctx, "/a.txt", "v2"); err != nil {
		t.Fatalf("second Write: %v", err)
	}
	second := view.Files()["/a.txt"]
	if second.Content != "v2" {
		t.Errorf("content = %q, want v2", second.Content)
	}
	if !second.CreatedAt.Equal(first.CreatedAt) {
		t.Errorf("CreatedAt = %v, want preserved %v", second.CreatedAt, first.CreatedAt)
	}
}

func TestStateBackendEditBinaryRejected(t *testing.T) {
	ctx, _, b := newStateCtx(map[string]FileMeta{
		"/img.png": {Content: "aGVsbG8=", Encoding: "base64"},
	})
	_, err := b.Edit(ctx, "/img.png", "a", "b", false)
	if err == nil || err.Error() != "Error: Cannot edit binary files with text replacements." {
		t.Errorf("Edit base64 error = %v, want binary-rejection message", err)
	}
}

func TestStateBackendGrep(t *testing.T) {
	files := map[string]FileMeta{
		"/a.txt":     meta("needle here\nplain\nneedle again"),
		"/sub/b.txt": meta("sub needle"),
		"/.hidden":   meta("needle hidden"),
		"/bin.dat":   {Content: "needle", Encoding: "base64"},
	}
	ctx, _, b := newStateCtx(files)

	tests := []struct {
		name     string
		pattern  string
		path     string
		glob     string
		maxCount int
		want     []string // matched paths, in order
		wantErr  string   // non-empty expects an error message substring
	}{
		{name: "all files under root", pattern: "needle", path: "/", want: []string{"/.hidden", "/a.txt", "/sub/b.txt"}},
		{name: "path narrows to subtree", pattern: "needle", path: "/sub", want: []string{"/sub/b.txt"}},
		{name: "exact file path", pattern: "needle", path: "/a.txt", want: []string{"/a.txt"}},
		{name: "base64 files skipped", pattern: "needle", path: "/bin.dat", want: []string{}},
		{name: "glob filter keeps dot basenames out", pattern: "needle", path: "/", glob: "*.txt", want: []string{"/a.txt", "/sub/b.txt"}},
		{name: "glob with slash anchors relative path", pattern: "needle", path: "/", glob: "sub/*.txt", want: []string{"/sub/b.txt"}},
		{name: "dot basename glob admits dotfile", pattern: "needle", path: "/", glob: ".hidden", want: []string{"/.hidden"}},
		{name: "maxCount caps total", pattern: "needle", path: "/", maxCount: 1, want: []string{"/.hidden"}},
		{name: "traversal path swallowed to empty", pattern: "needle", path: "/../..", want: []string{}},
		{name: "refused glob errors", pattern: "needle", path: "/", glob: "../x", wantErr: "Path traversal not allowed in glob pattern"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := b.Grep(ctx, tt.pattern, tt.path, tt.glob, tt.maxCount)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Grep error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Grep: %v", err)
			}
			var got []string
			for _, m := range res.Matches {
				got = append(got, m.Path)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("matched paths = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestStateBackendGlob(t *testing.T) {
	files := map[string]FileMeta{
		"/a.txt":         meta("one"),
		"/sub/b.py":      meta("two"),
		"/sub/deep/c.py": meta("three"),
		"/.hidden.py":    meta("dot"),
	}
	ctx, _, b := newStateCtx(files)

	tests := []struct {
		name    string
		path    string
		pattern string
		want    []string
		wantErr string
	}{
		{name: "slash-less pattern matches basenames at any depth", path: "/", pattern: "*.py", want: []string{"/sub/b.py", "/sub/deep/c.py"}},
		{name: "doublestar traverses non-dot directories", path: "/", pattern: "**", want: []string{"/a.txt", "/sub/b.py", "/sub/deep/c.py"}},
		{name: "dot pattern admits dot basename", path: "/", pattern: ".hidden.py", want: []string{"/.hidden.py"}},
		{name: "slash pattern is root-relative", path: "/", pattern: "sub/*.py", want: []string{"/sub/b.py"}},
		{name: "doublestar inside slash pattern", path: "/", pattern: "sub/**/*.py", want: []string{"/sub/b.py", "/sub/deep/c.py"}},
		{name: "path narrows the search root", path: "/sub", pattern: "**", want: []string{"/sub/b.py", "/sub/deep/c.py"}},
		{name: "exact file path matches basename", path: "/a.txt", pattern: "a.*", want: []string{"/a.txt"}},
		{name: "exact file path non-matching basename", path: "/a.txt", pattern: "b.*", want: []string{}},
		{name: "missing path is empty result", path: "/missing", pattern: "**", want: []string{}},
		{name: "traversal path swallowed to empty", path: "/..", pattern: "**", want: []string{}},
		{name: "refused pattern errors", path: "/", pattern: "../x", wantErr: "Path traversal not allowed in glob pattern"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := b.Glob(ctx, tt.path, tt.pattern)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("Glob error = %v, want containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Glob: %v", err)
			}
			if !slices.Equal(res.Files, tt.want) {
				t.Errorf("Glob(%q, %q) = %v, want %v", tt.path, tt.pattern, res.Files, tt.want)
			}
		})
	}
}

func TestStateBackendPathValidation(t *testing.T) {
	ctx, _, b := newStateCtx(nil)
	calls := map[string]func() error{
		"Ls":    func() error { _, err := b.Ls(ctx, "/../etc"); return err },
		"Read":  func() error { _, err := b.Read(ctx, "/a/../b.txt", 0, 10); return err },
		"Write": func() error { _, err := b.Write(ctx, "/a/../b.txt", "x"); return err },
		"Edit":  func() error { _, err := b.Edit(ctx, "/a/../b.txt", "x", "y", false); return err },
		"Delete": func() error {
			_, err := b.Delete(ctx, "/a/..")
			return err
		},
	}
	for _, call := range calls {
		wantCode(t, call(), ErrInvalidPath)
	}
}

func TestStateBackendDeleteRootWipesAll(t *testing.T) {
	ctx, view, b := newStateCtx(map[string]FileMeta{
		"/a.txt":     meta("one"),
		"/sub/b.txt": meta("two"),
	})
	if _, err := b.Delete(ctx, "/"); err != nil {
		t.Fatalf("Delete(/): %v", err)
	}
	if len(view.Files()) != 0 {
		t.Errorf("files after root delete = %v, want empty", view.Files())
	}
	delta := view.Changed()
	if delta["/a.txt"] != nil || delta["/sub/b.txt"] != nil {
		t.Errorf("Changed() = %v, want nil markers for both files", delta)
	}
}

// TestStateViewDeltaFeedsReducer ties the two halves together: the middleware
// applies the view's Changed() delta to the seeded state via MapMergeReducer,
// exactly as upstream queues partial files updates through CONFIG_KEY_SEND.
func TestStateViewDeltaFeedsReducer(t *testing.T) {
	seed := map[string]FileMeta{"/drop": meta("old"), "/keep": meta("kept")}
	ctx, view, b := newStateCtx(seed)

	if _, err := b.Write(ctx, "/new", "written"); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := b.Delete(ctx, "/drop"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	merged, err := MapMergeReducer(seed, view.Changed())
	if err != nil {
		t.Fatalf("MapMergeReducer(seed, delta): %v", err)
	}
	want := map[string]FileMeta{"/keep": seed["/keep"], "/new": view.Files()["/new"]}
	if !reflect.DeepEqual(merged, want) {
		t.Errorf("merged = %v, want %v", merged, want)
	}
}
