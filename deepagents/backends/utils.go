package backends

import (
	"fmt"
	"path"
	"strings"
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
	// the matching ValueError message in the Python reference.
	Message string
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// NormalizePath validates and normalizes a virtual filesystem path and
// returns a canonical path starting with '/' and using forward slashes. It
// mirrors validate_path in the Python reference, deepagents/backends/utils.py:
//
//   - backslashes are rewritten to forward slashes (to_posix_path)
//   - a leading '~' is rejected as path traversal
//   - any path containing '..' as a whole path component is rejected as path
//     traversal, even when it would resolve inside the root (so "/a/../b" is
//     rejected, not resolved to "/b") — matching the upstream security
//     semantics; substrings like the ".." in "foo..bar.txt" are fine
//   - Windows absolute paths such as "C:/..." or "C:\\..." are rejected
//   - '.' and duplicate-slash segments are resolved, and relative results
//     are rooted with a leading '/' (so "." and "" normalize to "/.")
//
// Divergence from the Python reference, intentional: the reference's
// os.path.normpath preserves exactly two leading slashes ("//a" stays "//a"),
// while this port collapses duplicate leading slashes ("//a" -> "/a"), which
// is more robust against malformed tool input. Errors carry the upstream
// invalid_path code and the upstream ValueError wording.
func NormalizePath(p string) (string, error) {
	posix := strings.ReplaceAll(p, "\\", "/")
	if strings.HasPrefix(p, "~") || hasDotDotComponent(posix) {
		return "", &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path traversal not allowed: %s", p),
		}
	}
	if len(p) >= 2 && p[1] == ':' && isAlpha(p[0]) {
		return "", &Error{
			Code: ErrInvalidPath,
			Message: fmt.Sprintf(
				"Windows absolute paths are not supported: %s. Please use virtual paths starting with / (e.g., /workspace/file.txt)",
				p,
			),
		}
	}
	cleaned := path.Clean(posix)
	if !strings.HasPrefix(cleaned, "/") {
		cleaned = "/" + cleaned
	}
	return cleaned, nil
}

// hasDotDotComponent reports whether any whole path component of the
// slash-separated path is "..", mirroring the PurePosixPath(...).parts check
// in validate_path, deepagents/backends/utils.py. Only traversal as a whole
// component is rejected, so filenames like "foo..bar.txt" pass.
func hasDotDotComponent(posix string) bool {
	for _, part := range strings.Split(posix, "/") {
		if part == ".." {
			return true
		}
	}
	return false
}

// isAlpha reports whether c is an ASCII letter, matching the [a-zA-Z] class
// of the Windows-drive regex in validate_path, deepagents/backends/utils.py.
func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// NormalizeReadBounds clamps a read window for file reads. A negative offset
// is clamped to 0, and a limit of zero or less becomes an empty window
// (count 0). Non-negative offsets and positive limits pass through
// unchanged.
func NormalizeReadBounds(offset, limit int) (start, count int) {
	if offset < 0 {
		offset = 0
	}
	if limit <= 0 {
		limit = 0
	}
	return offset, limit
}
