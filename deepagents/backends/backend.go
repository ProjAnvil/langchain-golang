package backends

import (
	"context"
	"time"
)

// Standardized error codes for file operations. They mirror the
// FileOperationError literals and named constants in the Python reference,
// deepagents/backends/protocol.py (FILE_NOT_FOUND, PERMISSION_DENIED,
// IS_DIRECTORY, INVALID_PATH), so error values stay byte-identical across
// the Python and Go implementations.
const (
	ErrFileNotFound     = "file_not_found"
	ErrPermissionDenied = "permission_denied"
	ErrIsDirectory      = "is_directory"
	ErrInvalidPath      = "invalid_path"
)

// Error is a file-operation error carrying one of the standardized error
// codes above, mirroring FileOperationError in the Python reference,
// deepagents/backends/protocol.py.
type Error struct {
	// Code is one of the Err* code constants in this package.
	Code string
	// Message is the human-readable detail, typically phrased exactly like
	// the matching ValueError or result-error message in the Python
	// reference, deepagents/backends/filesystem.py.
	Message string
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// FileMeta is a file's stored content plus metadata. It mirrors the FileData
// TypedDict in the Python reference, deepagents/backends/protocol.py, and is
// the value type of the public `files` state key. Encoding is "utf-8" for
// text or "base64" for binary content; this port's MVP always writes "utf-8".
//
// CreatedAt: the Python reference carries an optional ISO-8601 creation
// timestamp, but portable creation-time (birth time) stats are not available
// on every filesystem Go targets, so CreatedAt is populated with the file's
// modification time when the birth time is unavailable (see FilesystemBackend
// users and the StateBackend in state.go).
type FileMeta struct {
	Content    string
	Encoding   string
	CreatedAt  time.Time
	ModifiedAt time.Time
}

// PermissionMode is the disposition of a permission rule. It mirrors the
// Literal["allow", "deny", "interrupt"] on FilesystemPermission.mode in the
// Python reference, deepagents/middleware/filesystem.py.
type PermissionMode string

const (
	// PermissionAllow lets the call proceed (upstream default).
	PermissionAllow PermissionMode = "allow"
	// PermissionDeny makes the tool return a permission-denied error.
	PermissionDeny PermissionMode = "deny"
	// PermissionInterrupt pauses the call for human approval via HITL.
	PermissionInterrupt PermissionMode = "interrupt"
)

// PermissionRule grants a permission mode to operations on path patterns.
// It mirrors FilesystemPermission in the Python reference,
// deepagents/middleware/filesystem.py (operations: "read"/"write",
// paths: virtual-path glob patterns, mode: allow/deny/interrupt). The
// evaluator itself lives in the middleware package; the types live here
// because they govern backend operations.
type PermissionRule struct {
	Operations []string
	Paths      []string
	Mode       PermissionMode
}

// LsResult mirrors LsResult/FileInfo in the Python reference,
// deepagents/backends/protocol.py. Entries lists the direct children of the
// listed directory, sorted by name.
type LsResult struct {
	Entries []LsEntry
}

// LsEntry is one directory entry. Type is "file" or "directory" (the Python
// reference's FileInfo uses path/is_dir/size; the Go port keeps the name,
// kind, and modification time). ModifiedAt comes from the filesystem stat
// mtime, like the modified_at field in the reference's ls().
type LsEntry struct {
	Name       string
	Type       string
	ModifiedAt time.Time
}

// ReadResult mirrors ReadResult in the Python reference,
// deepagents/backends/protocol.py, carrying the raw (unformatted) content
// window plus pagination metadata. FilesystemBackend normalizes CRLF/CR
// content to LF on read (universal newlines), so Content is LF text;
// slice_read_response in the reference additionally normalizes the returned
// window, which state-backed stores rely on (deepagents/backends/utils.py).
//
// The pagination fields are 0 (unset) when no window was returned. NextOffset
// is the 0-indexed resume offset for the next unread line; it is 0 exactly
// when the window reached EOF (the Python reference uses None here), because
// a valid resume offset is always >= 1.
type ReadResult struct {
	Content    string
	StartLine  int
	EndLine    int
	TotalLines int
	NextOffset int
}

// WriteResult mirrors WriteResult in the Python reference,
// deepagents/backends/protocol.py. Path is the path as passed to Write (the
// reference returns file_path unchanged).
type WriteResult struct {
	Path string
}

// EditResult mirrors EditResult in the Python reference,
// deepagents/backends/protocol.py. Replaced is the occurrences count
// (upstream field name: occurrences).
type EditResult struct {
	Replaced int
}

// DeleteResult mirrors DeleteResult in the Python reference,
// deepagents/backends/protocol.py. Success is true when the path (file,
// symlink, or directory tree) was removed.
type DeleteResult struct {
	Success bool
}

// GlobResult mirrors GlobResult in the Python reference,
// deepagents/backends/protocol.py. Files holds the matching virtual paths,
// sorted. The reference's truncated/truncation_reason fields are not carried
// by this MVP struct.
type GlobResult struct {
	Files []string
}

// GrepResult mirrors GrepResult in the Python reference,
// deepagents/backends/protocol.py. Matches groups the matched lines by file.
// The reference's truncated field is not carried by this MVP struct; the
// maxCount cap is still enforced.
type GrepResult struct {
	Matches []GrepFileMatches
}

// GrepFileMatches is one file's grep matches: the virtual path plus every
// matched line with its 1-based line number, mirroring the per-path grouping
// of the reference's GrepResult.matches.
type GrepFileMatches struct {
	Path  string
	Lines []GrepLine
}

// Backend is the pluggable virtual-filesystem backend contract. It mirrors
// BackendProtocol in the Python reference, deepagents/backends/protocol.py:
// every method takes a context (the middleware layer enforces the upstream
// grep/glob timeouts via a ctx deadline) and returns a result struct plus a
// Go error in place of the reference's per-result error strings. Tool errors
// render as tool result text, never aborting the run.
type Backend interface {
	// Ls lists the direct children (non-recursive) of the directory at
	// path, sorted by name, mirroring the reference's ls().
	Ls(ctx context.Context, path string) (LsResult, error)

	// Read reads the requested line window of the file at path. offset is
	// 0-indexed and clamped to the start of the file when negative; limit
	// is the maximum number of lines, where a non-positive value yields an
	// empty window. Pagination metadata and trailing-newline handling
	// mirror slice_read_response in deepagents/backends/utils.py, and
	// empty/whitespace-only files return the empty-contents reminder
	// string as Content, mirroring read() in
	// deepagents/backends/filesystem.py.
	Read(ctx context.Context, path string, offset, limit int) (ReadResult, error)

	// Write writes content to the file at path, creating parent
	// directories as needed and creating or overwriting the file,
	// mirroring write() in deepagents/backends/filesystem.py.
	Write(ctx context.Context, path string, content string) (WriteResult, error)

	// Edit replaces occurrences of oldString with newString in the file at
	// path. With replaceAll false the edit fails unless oldString occurs
	// exactly once; error messages are the upstream literals from
	// perform_string_replacement, deepagents/backends/utils.py.
	Edit(ctx context.Context, path string, oldString, newString string, replaceAll bool) (EditResult, error)

	// Glob finds files under path matching pattern, mirroring glob() in
	// deepagents/backends/filesystem.py: a pattern without '/' (including
	// a bare '**') matches basenames at any depth under the search path,
	// traversing dot directories and excluding only dot basenames; a
	// pattern containing '/' (a leading '/' anchors) matches the full
	// path relative to the search path, where the dot rule also keeps
	// '**' out of dot directories (compile_grep_include_glob,
	// deepagents/backends/utils.py). Missing or non-directory search
	// paths return an empty result.
	Glob(ctx context.Context, path string, pattern string) (GlobResult, error)

	// Grep searches for pattern as a literal substring (not a regex) in
	// files under path, optionally filtered by the include glob (a glob
	// without '/' — including a bare '**' — matches basenames at any
	// depth, traversing dot directories and excluding only dot basenames;
	// one with '/' matches the path relative to the search root),
	// mirroring grep() in deepagents/backends/filesystem.py. maxCount
	// caps the total number
	// of matches across all files; a non-positive value means unlimited.
	// The middleware applies the upstream default cap (1000); this
	// backend method itself treats 0 as "no cap", matching the reference.
	//
	// Divergence from the reference, intentional: a search path that fails
	// normalization (e.g. `..` traversal) returns an empty result rather
	// than an error, matching the reference's swallowed ValueError, but
	// unreadable files are skipped silently instead of being surfaced in a
	// partial-error field.
	Grep(ctx context.Context, pattern string, path string, glob string, maxCount int) (GrepResult, error)
}

// Deleter is the optional delete capability, probed by interface assertion
// (upstream _supports_delete). Delete removes a file, a symlink (without
// following it into its target), or a directory tree recursively, mirroring
// delete() in deepagents/backends/filesystem.py.
type Deleter interface {
	Delete(ctx context.Context, path string) (DeleteResult, error)
}
