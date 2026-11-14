package backends

import (
	"context"
	"errors"
	"fmt"
	"path"
	"sort"
	"strings"
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
// neither nil nor the accepted map type for that argument. It has no upstream
// counterpart: the Python reducer is typed by annotation and cannot be called
// with a non-mapping, while the Go reducer is invoked through `any`.
type MapMergeError struct {
	// Arg is the parameter that carried the offending value: "existing" or
	// "update".
	Arg string
	// Got is the Go type of the offending value.
	Got string
}

func (e *MapMergeError) Error() string {
	return fmt.Sprintf(
		"backends.MapMergeReducer: %s must be nil or %s; got %s",
		e.Arg, e.accepted(), e.Got,
	)
}

func (e *MapMergeError) accepted() string {
	if e.Arg == "existing" {
		return "map[string]backends.FileMeta"
	}
	return "map[string]*backends.FileMeta"
}

// MapMergeReducer is typed as a channels.Reducer so it can be registered on
// the `files` state field directly.
var _ channels.Reducer = MapMergeReducer

// MapMergeReducer merges partial updates into the `files` state map, with nil
// pointers as deletion markers. It mirrors _file_data_reducer in the Python
// reference, deepagents/middleware/filesystem.py:1324-1377: the existing value
// may be nil (channel initialization), the update merges key-by-key with
// update values overwriting existing ones, and a nil value in the update
// removes the key. The result is always a fresh map[string]FileMeta copy, so
// callers may mutate it without aliasing channel state.
//
// Accepted types: existing is nil or map[string]FileMeta (the channel's
// stored form); update is nil or map[string]*FileMeta (a partial update whose
// nil entries delete, the Go form of the reference's
// dict[str, FileData | None]). A nil update yields a copy of existing.
// Anything else returns a *MapMergeError.
//
// The signature matches langgraph/channels.Reducer so the filesystem
// middleware (Task 4) can register it on the FilesStateKey state field; the
// middleware package constructs that field itself, because backends must not
// import langchain/agents/middleware.
func MapMergeReducer(existing, update any) (any, error) {
	var base map[string]FileMeta
	switch e := existing.(type) {
	case nil:
		base = nil
	case map[string]FileMeta:
		base = e
	default:
		return nil, &MapMergeError{Arg: "existing", Got: fmt.Sprintf("%T", existing)}
	}

	result := make(map[string]FileMeta, len(base))
	for k, v := range base {
		result[k] = v
	}

	switch u := update.(type) {
	case nil:
		// No writes: the result is just the copy of existing.
	case map[string]*FileMeta:
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

// StateView is the working set of files a StateBackend reads and writes: a
// live in-memory map plus the delta of the mutations performed through it. It
// stands in for the LangGraph state channel access the Python reference wires
// through CONFIG_KEY_READ / CONFIG_KEY_SEND (deepagents/backends/state.py:81-
// 119): reads see the live map, so writes are immediately visible to later
// reads within the same step (the reference's fresh=True read-your-writes),
// while the recorded delta plays the role of the queued channel writes and is
// meant to be applied to the real state map via MapMergeReducer at the node
// boundary (unchanged keys need not be re-sent — the reducer preserves them).
//
// NewStateView copies the seed map, so mutations never alias the caller's
// snapshot; Files hands out the live internal map (zero-copy, read-your-
// writes), which callers must treat as read-only. A StateView is not safe for
// concurrent use; like the reference's per-node state access, it is used from
// one goroutine at a time.
type StateView struct {
	files   map[string]FileMeta
	changed map[string]*FileMeta
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

// Files returns the live files map. It is the working copy owned by the view
// (the seed passed to NewStateView is not aliased), reflects every Set and
// Delete immediately, and must be treated as read-only by callers.
func (v *StateView) Files() map[string]FileMeta {
	return v.files
}

// Set records path as written with meta, updating the live map and the
// change delta. The delta entry is a pointer to a copy of meta.
func (v *StateView) Set(path string, meta FileMeta) {
	v.files[path] = meta
	if v.changed == nil {
		v.changed = make(map[string]*FileMeta)
	}
	stored := meta
	v.changed[path] = &stored
}

// Delete records path as deleted: it is removed from the live map, and the
// delta carries a nil marker for it, which MapMergeReducer interprets as a
// deletion (upstream _file_data_reducer's None handling).
func (v *StateView) Delete(path string) {
	delete(v.files, path)
	if v.changed == nil {
		v.changed = make(map[string]*FileMeta)
	}
	v.changed[path] = nil
}

// Changed returns the mutation delta since the view was created, keyed by
// path with nil meaning delete. It returns nil when nothing has been mutated,
// and otherwise the live delta map, which callers must treat as read-only.
// Later operations on the same path overwrite earlier delta entries, so the
// delta always reflects the net effect.
func (v *StateView) Changed() map[string]*FileMeta {
	if len(v.changed) == 0 {
		return nil
	}
	return v.changed
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
// context" when the context carries no view.
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
		return []stateScopedFile{{key: root, rel: path.Base(root)}}
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
// modification time, deeper keys surface as directory entries (upstream
// directories carry no timestamp, hence the zero ModifiedAt), and the result
// is sorted by name. A path with no stored files yields empty entries, not an
// error — there is no directory to be missing in an in-memory map. Unlike the
// reference's raw prefix matching, the path is normalized first (see the
// StateBackend divergence notes).
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
	for key, meta := range v.files {
		if !strings.HasPrefix(key, prefix) {
			continue
		}
		rel := strings.TrimPrefix(key, prefix)
		if idx := strings.Index(rel, "/"); idx >= 0 {
			subdir := rel[:idx]
			if !seenDirs[subdir] {
				seenDirs[subdir] = true
				entries = append(entries, LsEntry{Name: subdir, Type: "directory"})
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
	meta, ok := v.files[norm]
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
	if existing, ok := v.files[norm]; ok {
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
	meta, ok := v.files[norm]
	if !ok {
		return EditResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("Error: File '%s' not found", vpath),
		}
	}
	if meta.Encoding == "base64" {
		return EditResult{}, errors.New("Error: Cannot edit binary files with text replacements.")
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
		for key := range v.files {
			toDelete = append(toDelete, key)
		}
	} else {
		prefix := norm + "/"
		for key := range v.files {
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
	for _, scoped := range filterStateFiles(v.files, norm) {
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
	for _, scoped := range filterStateFiles(v.files, norm) {
		if glob != "" && !includeGlobMatches(glob, scoped.rel) {
			continue
		}
		meta := v.files[scoped.key]
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
