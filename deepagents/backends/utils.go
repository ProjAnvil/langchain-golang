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
//   - Windows absolute paths such as "C:/..." or "C:\\..." are rejected
//   - '.' and duplicate-slash segments are resolved, and relative results
//     are rooted with a leading '/' (so "." and "" normalize to "/.")
//   - a path whose lexical resolution escapes the root (a leading '..' after
//     cleaning, e.g. "a/../../x") is rejected as path traversal
//
// Divergences from the Python reference, per the porting contract: the
// reference rejects any path containing a '..' component outright (so
// "/a/../b" raises) and preserves POSIX double leading slashes ("//a"), while
// this port resolves '..' segments lexically ("/a/../b" -> "/b") and collapses
// duplicate leading slashes ("//a" -> "/a"), rejecting only paths that escape
// the root. Errors carry the upstream invalid_path code and the upstream
// ValueError wording.
func NormalizePath(p string) (string, error) {
	if strings.HasPrefix(p, "~") {
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
	cleaned := path.Clean(strings.ReplaceAll(p, "\\", "/"))
	if cleaned == ".." || strings.HasPrefix(cleaned, "../") {
		return "", &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path traversal not allowed: %s", p),
		}
	}
	if !strings.HasPrefix(cleaned, "/") {
		cleaned = "/" + cleaned
	}
	return cleaned, nil
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
