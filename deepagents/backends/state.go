package backends

import (
	"context"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	channels "github.com/projanvil/langchain-golang/langgraph/channels"
)

// FilesStateKey is the agent-state key holding the virtual filesystem's file
// map. It mirrors the `files` field of FilesystemState in the Python
// reference, deepagents/middleware/filesystem.py, and the channel name the
// reference's StateBackend reads and writes through CONFIG_KEY_READ /
// CONFIG_KEY_SEND (deepagents/backends/state.py).
const FilesStateKey = "files"

// MapMergeError reports that MapMergeReducer received a value whose type is
// neither nil nor one of the accepted map forms for that argument. It has no
// upstream counterpart: the Python reducer is typed by annotation and cannot
// be called with a non-mapping, while the Go reducer is invoked through
// `any`.
type MapMergeError struct {
	// Arg is the parameter that carried the offending value: "existing" or
	// "update".
	Arg string
	// Got is the Go type of the offending value.
	Got string
}

func (e *MapMergeError) Error() string {
	return fmt.Sprintf(
		"backends.MapMergeReducer: %s must be nil, map[string]backends.FileMeta, or map[string]*backends.FileMeta; got %s",
		e.Arg, e.Got,
	)
}

// MapMergeReducer is typed as a channels.Reducer so it can be registered on
// the `files` state field directly.
var _ channels.Reducer = MapMergeReducer

// MapMergeReducer merges `files` state updates, mirroring the dict-merge
// reducer on FilesystemState.files in the Python reference
// (deepagents/middleware/filesystem.py:1324-1377). Both sides are the
// channel's stored form, map[string]FileMeta: the update overlays the
// existing map key-by-key (same-key last-write-wins), keys carried only by
// the existing map survive, and the result is always a fresh
// map[string]FileMeta copy, so callers may mutate it without aliasing
// channel state.
//
// Update form == stored form is load-bearing: langgraph's binaryOperator
// seeds an unset channel's first write raw, without applying the reducer
// (langgraph/channels/binop.go — the set-flag path; Go's Reducer is an
// untyped func, so there is no Python-style typ() zero factory, and the
// set-flag path corresponds to Python's MISSING path, which is reachable
// there only when typ() is not instantiable — Python's instantiable types
// seed typ() at construction and run the operator on every update). A first
// write in any other shape would be stored as-is and crash every later
// merge's type assertion, so the canonical update here —
// backends.StateView.Changed's full post-mutation snapshot — is exactly the
// stored form. Deletion is NOT expressed through this reducer: the port's
// own tools send the snapshot wrapped in channels.NewOverwrite (D15), which
// replaces the whole value — deletions are simply absent from the snapshot.
// Unmarked writes reaching this reducer overlay (and would resurrect deleted
// keys); no port code sends that shape, it is the documented defense path.
//
// Tolerated as defense in depth: nil on either side (unset channel / no
// changes), typed-nil maps, and the delta form map[string]*FileMeta, in
// which a nil entry deletes the key (the Go spelling of the reference's
// None deletion markers, for callers that queue partial updates explicitly).
// Anything else returns a *MapMergeError.
//
// The signature matches langgraph/channels.Reducer so the filesystem
// middleware (Task 4) can register it on the FilesStateKey state field; the
// middleware package constructs that field itself, because backends must not
// import langchain/agents/middleware.
func MapMergeReducer(existing, update any) (any, error) {
	result := make(map[string]FileMeta)

	switch e := existing.(type) {
	case nil:
		// Unset channel / first write.
	case map[string]FileMeta:
		for k, v := range e {
			result[k] = v
		}
	case map[string]*FileMeta:
		// Tolerated delta form: nil entries carry nothing to merge.
		for k, v := range e {
			if v != nil {
				result[k] = *v
			}
		}
	default:
		return nil, &MapMergeError{Arg: "existing", Got: fmt.Sprintf("%T", existing)}
	}

	switch u := update.(type) {
	case nil:
		// No changes in this update.
	case map[string]FileMeta:
		for k, v := range u {
			result[k] = v
		}
	case map[string]*FileMeta:
		// Tolerated delta form: nil entries delete (UP None markers).
		for k, v := range u {
			if v == nil {
				delete(result, k)
			} else {
				result[k] = *v
			}
		}
	default:
		return nil, &MapMergeError{Arg: "update", Got: fmt.Sprintf("%T", update)}
	}
	return result, nil
}

// stateViewCtxKey is the context key type for the StateView, avoiding
// collisions with other packages' context values.
type stateViewCtxKey struct{}

// WithStateView returns a context carrying v as its StateView. A nil view is
// ignored (the context keeps its previous, likely absent, view), so
// StateViewFromContext reports ok=false rather than handing out a nil view.
// It is the Go replacement for the Pregel config plumbing
// (CONFIG_KEY_READ/CONFIG_KEY_SEND) that the Python reference's StateBackend
// requires, deepagents/backends/state.py:57-79.
func WithStateView(ctx context.Context, v *StateView) context.Context {
	if v == nil {
		return ctx
	}
	return context.WithValue(ctx, stateViewCtxKey{}, v)
}

// StateViewFromContext returns the StateView carried by ctx, reporting
// whether one is present.
func StateViewFromContext(ctx context.Context) (*StateView, bool) {
	v, _ := ctx.Value(stateViewCtxKey{}).(*StateView)
	if v == nil {
		return nil, false
	}
	return v, true
}

// StateView is the working set of files a StateBackend reads and writes: an
// in-memory map plus a changed flag. It stands in for the LangGraph state
// channel access the Python reference wires through CONFIG_KEY_READ /
// CONFIG_KEY_SEND (deepagents/backends/state.py:81-119): reads see the
// mutations completed before them (the reference's fresh=True
// read-your-writes), and Changed produces the artifact the middleware sends
// to the `files` channel at the node boundary.
//
// Channel artifact contract: Changed returns the FULL post-mutation snapshot
// as map[string]FileMeta — the update form equals the channel's stored form.
// langgraph's binaryOperator (like Python's) seeds a channel's first write
// raw, without applying the reducer (langgraph/channels/binop.go — the
// set-flag path), so a delta-shaped first write (such as a
// map[string]*FileMeta with nil delete markers) would be stored as-is and
// crash every subsequent merge's type assertion. Deletions therefore live
// INSIDE the view — a deleted key is simply absent from the snapshot — and
// the middleware commits the snapshot as a REPLACEMENT: filesCommand wraps
// it in channels.Overwrite and the channel replaces its stored map wholesale
// (DIVERGENCES D15), so an absent key stays deleted. MapMergeReducer, the
// channel's reducer, is defense for UNMARKED writes only: it overlays
// snapshots key-by-key with the same key's last write winning (each Changed
// snapshot is taken under the view's lock, so it is self-consistent even
// when parallel tools in one batch mutate concurrently).
//
// Concurrency contract: a StateView is safe for concurrent use. The runtime
// may execute the tools of one batch in parallel, so multiple goroutines may
// call Set, Delete, Files, and Changed at the same time. A mutex guards the
// file map and the changed flag; Files and Changed return snapshot copies,
// and per key the last completed operation wins. A map returned by Files or
// Changed reflects exactly the mutations completed before the call and never
// changes afterward, so no caller can observe a concurrent mutation through
// a previously returned snapshot; NewStateView likewise copies the seed map,
// so mutations never alias the caller's snapshot.
type StateView struct {
	mu    sync.Mutex
	files map[string]FileMeta
	dirty bool
}

// NewStateView returns a StateView seeded with a copy of files (nil seeds an
// empty view).
func NewStateView(files map[string]FileMeta) *StateView {
	seeded := make(map[string]FileMeta, len(files))
	for k, v := range files {
		seeded[k] = v
	}
	return &StateView{files: seeded}
}

// snapshotLocked returns a copy of the file map; the view's lock must be
// held.
func (v *StateView) snapshotLocked() map[string]FileMeta {
	snapshot := make(map[string]FileMeta, len(v.files))
	for k, meta := range v.files {
		snapshot[k] = meta
	}
	return snapshot
}

// Files returns a snapshot copy of the files map. The copy reflects every
// mutation completed before the call (read-your-writes for a sequential
// caller) and never changes afterward; concurrent mutations that complete
// later are visible only through subsequent calls.
func (v *StateView) Files() map[string]FileMeta {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.snapshotLocked()
}

// Set records path as written with meta, updating the file map under the
// view's lock and marking the view changed.
func (v *StateView) Set(path string, meta FileMeta) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.files[path] = meta
	v.dirty = true
}

// Delete removes path from the file map and marks the view changed. Unlike
// the pointer-form delta (see MapMergeReducer's tolerated forms), deletion is
// expressed by the key's absence from the Changed snapshot, not by a marker.
func (v *StateView) Delete(path string) {
	v.mu.Lock()
	defer v.mu.Unlock()
	delete(v.files, path)
	v.dirty = true
}

// Changed returns the channel-update artifact: the full post-mutation
// snapshot of the files map, in the channel's stored form
// (map[string]FileMeta — see the StateView contract for why the update form
// must equal the stored form). It returns nil when no Set or Delete has
// occurred since the view was created. Like Files, the returned snapshot
// reflects every mutation completed before the call and never changes
// afterward.
func (v *StateView) Changed() map[string]FileMeta {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.dirty {
		return nil
	}
	return v.snapshotLocked()
}

// errNoStateView is returned by every StateBackend method when the context
// carries no StateView. It stands in for the reference's RuntimeError
// explaining that StateBackend must be used inside a graph execution
// (deepagents/backends/state.py:62-68); the Go wording is pinned by the port
// plan so tooling can match on it.
var errNoStateView = errors.New("no state view in context")

// StateBackend is the ephemeral Backend that stores files in agent state. It
// mirrors StateBackend in the Python reference, deepagents/backends/state.py:
// files live in a map keyed by virtual path, persist within a conversation
// thread (via the state channel, in the reference), and never touch disk.
// Reads and writes go through the StateView in the context (the reference's
// CONFIG_KEY_READ/CONFIG_KEY_SEND plumbing), so the same backend instance is
// safe to share across steps; every method fails with "no state view in
// context" when the context carries no view. Each method snapshots the view
// at entry (see StateView's concurrency contract), so parallel tools sharing
// one backend over one view are race-free, with the last write per key
// winning.
//
// Divergences from the reference, intentional:
//   - the reference keys files by the path exactly as passed and never
//     normalizes; this port routes every path through NormalizePath first
//     (rejecting `..` traversal outright, like FilesystemBackend) so both
//     backends enforce the same virtual-path discipline, and writes land on
//     the canonical key
//   - the reference's read() returns whole (unsliced) content for
//     binary-classified extensions; this port's FileMeta only stores utf-8
//     text, so read always slices, like FilesystemBackend.Read
//   - the reference's glob sorts by modification time (most recent first);
//     this port sorts matching paths lexicographically, matching the
//     GlobResult contract shared with FilesystemBackend.Glob
//   - upload_files/download_files are not ported (binary transfer helpers,
//     out of scope for this MVP)
type StateBackend struct{}

// NewStateBackend returns a StateBackend. The backend is stateless; all file
// state is reached through the StateView in the context passed to each call.
func NewStateBackend() *StateBackend {
	return &StateBackend{}
}

// view returns the context's StateView, or the no-view error.
func (b *StateBackend) view(ctx context.Context) (*StateView, error) {
	v, ok := StateViewFromContext(ctx)
	if !ok {
		return nil, errNoStateView
	}
	return v, nil
}

// normalizeStatePath normalizes a state path, mapping the empty string to "/"
// like the reference's `path or "/"` in _normalize_path,
// deepagents/backends/utils.py.
func normalizeStatePath(p string) (string, error) {
	if p == "" {
		return "/", nil
	}
	return NormalizePath(p)
}

// stateScopedFile pairs a stored key with its path relative to a search root.
type stateScopedFile struct {
	key string
	rel string
}

// filterStateFiles scopes files to a normalized search root, mirroring
// _filter_files_by_path and _relative_to_root in the Python reference,
// deepagents/backends/utils.py: an exact file key scopes to that single file
// (matched by its basename), a directory key scopes to the subtree under
// `root + "/"`, and the root "/" scopes to every key (relative path with the
// leading slash dropped). The result is sorted by key so map iteration order
// never leaks into results.
func filterStateFiles(files map[string]FileMeta, root string) []stateScopedFile {
	if _, ok := files[root]; ok {
		// Upstream _relative_to_root returns file_path[1:] when the root is
		// "/" — the empty string for a file stored at "/" itself — and the
		// basename otherwise.
		rel := path.Base(root)
		if root == "/" {
			rel = ""
		}
		return []stateScopedFile{{key: root, rel: rel}}
	}
	prefix := "/"
	if root != "/" {
		prefix = root + "/"
	}
	scoped := make([]stateScopedFile, 0, len(files))
	for key := range files {
		if strings.HasPrefix(key, prefix) {
			scoped = append(scoped, stateScopedFile{key: key, rel: strings.TrimPrefix(key, prefix)})
		}
	}
	sort.Slice(scoped, func(i, j int) bool { return scoped[i].key < scoped[j].key })
	return scoped
}

// Ls lists the direct children of the directory at path, mirroring ls() in
// the Python reference, deepagents/backends/state.py:125-173: files directly
// under the directory are listed as file entries with their stored
// modification time, deeper keys surface as directory entries whose Name
// carries a trailing "/" (the reference's FileInfo.path for a directory is
// normalized_path + subdir + "/", state.py:158; upstream directories carry
// no timestamp, hence the zero ModifiedAt), and the result is sorted by
// name — with the directory marker included, the same path-with-slash key
// the reference sorts by. A path with no stored files yields empty entries,
// not an error — there is no directory to be missing in an in-memory map.
// Unlike the reference's raw prefix matching, the path is normalized first
// (see the StateBackend divergence notes).
func (b *StateBackend) Ls(ctx context.Context, vpath string) (LsResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return LsResult{}, err
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return LsResult{}, err
	}
	prefix := "/"
	if norm != "/" {
		prefix = norm + "/"
	}
	entries := make([]LsEntry, 0)
	seenDirs := make(map[string]bool)
	for key, meta := range v.Files() {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		rel := strings.TrimPrefix(key, prefix)
		if idx := strings.Index(rel, "/"); idx >= 0 {
			subdir := rel[:idx]
			if !seenDirs[subdir] {
				seenDirs[subdir] = true
				entries = append(entries, LsEntry{Name: subdir + "/", Type: "directory"})
			}
			continue
		}
		entries = append(entries, LsEntry{Name: rel, Type: "file", ModifiedAt: meta.ModifiedAt})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	return LsResult{Entries: entries}, nil
}

// Read reads the requested line window of the file at path, mirroring read()
// in the Python reference, deepagents/backends/state.py:175-204: a missing
// key reports "File '<path>' not found", and stored content is routed through
// the shared slice_read_response pagination (deepagents/backends/utils.py).
// This port always slices — see the StateBackend divergence notes for the
// dropped binary-extension branch.
func (b *StateBackend) Read(ctx context.Context, vpath string, offset, limit int) (ReadResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return ReadResult{}, err
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return ReadResult{}, err
	}
	meta, ok := v.Files()[norm]
	if !ok {
		return ReadResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("File '%s' not found", vpath),
		}
	}
	return sliceReadResponse(meta.Content, offset, limit)
}

// Write writes content to the file at path, creating or overwriting it,
// mirroring write() in the Python reference, deepagents/backends/state.py:206-
// 221: a new file gets fresh created/modified timestamps, an existing file
// keeps its creation timestamp (update_file_data, deepagents/backends/
// utils.py), the encoding is forced to "utf-8", and the update is queued on
// the state view (the reference's CONFIG_KEY_SEND) rather than returned.
func (b *StateBackend) Write(ctx context.Context, vpath string, content string) (WriteResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return WriteResult{}, err
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return WriteResult{}, err
	}
	now := time.Now()
	meta := FileMeta{Content: content, Encoding: "utf-8", CreatedAt: now, ModifiedAt: now}
	if existing, ok := v.Files()[norm]; ok {
		meta.CreatedAt = existing.CreatedAt
	}
	v.Set(norm, meta)
	return WriteResult{Path: vpath}, nil
}

// Edit replaces occurrences of oldString with newString in the file at path,
// mirroring edit() in the Python reference, deepagents/backends/state.py:223-
// 252: a missing key reports "Error: File '<path>' not found", base64-encoded
// files refuse text edits, replacement validation and error messages come
// from perform_string_replacement (deepagents/backends/utils.py), and a
// successful edit preserves the creation timestamp while restamping the
// modification time. Unlike FilesystemBackend.Edit, content and search
// strings are matched verbatim without newline normalization — the reference
// edits the in-memory string as stored, and there is no disk read to
// universal-newline.
func (b *StateBackend) Edit(ctx context.Context, vpath string, oldString, newString string, replaceAll bool) (EditResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return EditResult{}, err
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return EditResult{}, err
	}
	meta, ok := v.Files()[norm]
	if !ok {
		return EditResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("Error: File '%s' not found", vpath),
		}
	}
	if meta.Encoding == "base64" {
		return EditResult{}, errors.New("Error: Cannot edit binary files with text replacements.") //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
	}
	newContent, occurrences, err := performStringReplacement(meta.Content, oldString, newString, replaceAll)
	if err != nil {
		return EditResult{}, err
	}
	edited := meta
	edited.Content = newContent
	edited.ModifiedAt = time.Now()
	v.Set(norm, edited)
	return EditResult{Replaced: occurrences}, nil
}

// Delete removes a file or directory tree from state, mirroring delete() in
// the Python reference, deepagents/backends/state.py:254-278: the exact key
// plus every nested key under `path + "/"` is removed (a directory is removed
// recursively), each removal is queued as a nil-valued delta entry (the
// reference sends None values the files reducer interprets as deletions),
// and deleting "/" removes everything (upstream's rstrip("/") yields the
// all-matching "/" prefix). When nothing is stored at or under the path, the
// upstream "Error: File '<path>' not found" wording is returned.
func (b *StateBackend) Delete(ctx context.Context, vpath string) (DeleteResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return DeleteResult{}, err
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return DeleteResult{}, err
	}
	var toDelete []string
	if norm == "/" {
		for key := range v.Files() {
			toDelete = append(toDelete, key)
		}
	} else {
		prefix := norm + "/"
		for key := range v.Files() {
			if key == norm || strings.HasPrefix(key, prefix) {
				toDelete = append(toDelete, key)
			}
		}
	}
	if len(toDelete) == 0 {
		return DeleteResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("Error: File '%s' not found", vpath),
		}
	}
	sort.Strings(toDelete)
	for _, key := range toDelete {
		v.Delete(key)
	}
	return DeleteResult{Success: true}, nil
}

// Glob finds stored files under path matching pattern, mirroring glob() in
// the Python reference, deepagents/backends/state.py:292-324, which routes
// through the shared include-glob contract (_glob_search_files and
// compile_grep_include_glob, deepagents/backends/utils.py). This port applies
// the same includeGlobMatches helper as FilesystemBackend.Glob (filesystem.go
// in this package): a pattern without '/' — including a bare '**' — matches
// basenames at any depth under the search root (dot directories traversed,
// dot basenames excluded), while a pattern containing '/' (a leading '/'
// anchors) matches the path relative to the search root. An exact file key
// scopes the search to that file matched by basename; a missing or
// unnormalizable path yields an empty result (the reference's swallowed
// ValueError returning "No files found"), while a pattern containing a `..`
// segment is refused with the upstream "Path traversal not allowed in glob
// pattern '<p>'" wording (the reference's InvalidGlobPatternError). Matching
// paths are returned sorted lexicographically — see the StateBackend
// divergence notes for the dropped modification-time ordering.
func (b *StateBackend) Glob(ctx context.Context, vpath string, pattern string) (GlobResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return GlobResult{}, err
	}
	if hasDotDotComponent(strings.ReplaceAll(pattern, "\\", "/")) {
		return GlobResult{}, &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path traversal not allowed in glob pattern '%s'", pattern),
		}
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return GlobResult{Files: []string{}}, nil
	}
	var files []string
	for _, scoped := range filterStateFiles(v.Files(), norm) {
		if !includeGlobMatches(pattern, scoped.rel) {
			continue
		}
		files = append(files, scoped.key)
	}
	sort.Strings(files)
	if files == nil {
		files = []string{}
	}
	return GlobResult{Files: files}, nil
}

// Grep searches for pattern as a literal substring in stored files under
// path, mirroring grep() in the Python reference, deepagents/backends/
// state.py:280-290, which routes through grep_matches_from_files
// (deepagents/backends/utils.py): the search root scopes by subtree or exact
// file key, the include glob filters by the shared include-glob contract (the
// same includeGlobMatches helper as FilesystemBackend.Grep — see Glob),
// base64-encoded entries are skipped, and matches are grouped per file.
// maxCount caps the total number of matches across files; a non-positive
// value means unlimited (the middleware applies the upstream default cap).
// Files are visited in sorted key order, so grouping is deterministic, and a
// refused include glob (a `..` segment) is reported with the upstream
// "Path traversal not allowed in glob pattern '<p>'" wording while an
// unnormalizable search path is swallowed to empty matches, matching the
// reference's swallowed ValueError.
func (b *StateBackend) Grep(ctx context.Context, pattern string, vpath string, glob string, maxCount int) (GrepResult, error) {
	v, err := b.view(ctx)
	if err != nil {
		return GrepResult{}, err
	}
	if glob != "" && hasDotDotComponent(strings.ReplaceAll(glob, "\\", "/")) {
		return GrepResult{}, &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path traversal not allowed in glob pattern '%s'", glob),
		}
	}
	norm, err := normalizeStatePath(vpath)
	if err != nil {
		return GrepResult{Matches: []GrepFileMatches{}}, nil
	}
	matches := make([]GrepFileMatches, 0)
	total := 0
	files := v.Files()
	for _, scoped := range filterStateFiles(files, norm) {
		if glob != "" && !includeGlobMatches(glob, scoped.rel) {
			continue
		}
		meta := files[scoped.key]
		if meta.Encoding == "base64" {
			continue
		}
		remaining := 0
		if maxCount > 0 {
			remaining = maxCount - total
			if remaining <= 0 {
				break
			}
		}
		lines := GrepLines(meta.Content, pattern, remaining)
		if len(lines) == 0 {
			continue
		}
		total += len(lines)
		matches = append(matches, GrepFileMatches{Path: scoped.key, Lines: lines})
	}
	return GrepResult{Matches: matches}, nil
}
