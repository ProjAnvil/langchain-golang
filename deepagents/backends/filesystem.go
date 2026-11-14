package backends

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
)

// emptyContentWarning mirrors EMPTY_CONTENT_WARNING in the Python reference,
// deepagents/backends/utils.py (duplicated there and in
// deepagents/middleware/filesystem.py): read() returns it as the content of
// an empty or whitespace-only file.
const emptyContentWarning = "System reminder: File exists but has empty contents"

// maxGrepFileSizeBytes mirrors the default max_file_size_mb=10 constructor
// bound of FilesystemBackend in the Python reference,
// deepagents/backends/filesystem.py: larger files are skipped by grep.
const maxGrepFileSizeBytes = 10 * 1024 * 1024

// FilesystemBackend reads and writes files directly on the filesystem, rooted
// at a real directory exposed as a virtual filesystem. It mirrors
// FilesystemBackend (virtual_mode=True) in the Python reference,
// deepagents/backends/filesystem.py: paths are virtual absolute paths
// anchored at the root, traversal (`..`, `~`) is rejected by NormalizePath,
// and every resolved path is checked to remain inside the root even through
// symlinks (the reference's resolve()+relative_to check).
type FilesystemBackend struct {
	root string
}

// NewFilesystemBackend returns a FilesystemBackend rooted at rootDir, which
// must be an existing directory. The root is made absolute and resolved
// through symlinks once, mirroring `Path(root_dir).resolve()` in the
// reference constructor. Content is always handled as utf-8 text in this
// port's MVP.
func NewFilesystemBackend(rootDir string) (*FilesystemBackend, error) {
	abs, err := filepath.Abs(rootDir)
	if err != nil {
		return nil, fmt.Errorf("resolving root directory: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return nil, fmt.Errorf("root directory does not exist: %s", rootDir)
	}
	info, err := os.Stat(resolved)
	if err != nil || !info.IsDir() {
		return nil, fmt.Errorf("root is not a directory: %s", rootDir)
	}
	return &FilesystemBackend{root: resolved}, nil
}

// resolvePath maps a virtual path to a real filesystem path under the root,
// mirroring _resolve_path (virtual_mode) in the Python reference,
// deepagents/backends/filesystem.py: NormalizePath rejects traversal, then
// the joined path is resolved through symlinks and required to stay inside
// the root. Missing path components resolve through their deepest existing
// ancestor (the reference's Path.resolve(strict=False)).
func (b *FilesystemBackend) resolvePath(key string) (string, error) {
	norm, err := NormalizePath(key)
	if err != nil {
		return "", err
	}
	full := filepath.Join(b.root, filepath.FromSlash(strings.TrimPrefix(norm, "/")))
	ev := resolveExisting(full)
	if !withinDir(b.root, ev) {
		return "", &Error{
			Code: ErrInvalidPath,
			Message: fmt.Sprintf(
				"Path:%s outside root directory: %s", ev, b.root,
			),
		}
	}
	return ev, nil
}

// resolveExisting resolves path through symlinks as far as its deepest
// existing ancestor, approximating pathlib's resolve(strict=False) for the
// containment check in resolvePath. A path whose final component is missing
// keeps that tail unresolved so write() can create it.
func resolveExisting(p string) string {
	if _, err := os.Lstat(p); err == nil {
		if ev, err := filepath.EvalSymlinks(p); err == nil {
			return ev
		}
		return p
	}
	parent := filepath.Dir(p)
	if parent == p {
		return p
	}
	return filepath.Join(resolveExisting(parent), filepath.Base(p))
}

// withinDir reports whether p is dir or lies underneath it.
func withinDir(dir, p string) bool {
	rel, err := filepath.Rel(dir, p)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// osError wraps an OS failure with the upstream per-operation error prefix
// from the Python reference, deepagents/backends/filesystem.py (prefix is a
// complete message head such as "Error reading file '/a.txt'"). Permission
// failures carry the permission_denied code (the reference maps
// PermissionError to PERMISSION_DENIED in _map_exception_to_standard_error).
func osError(prefix string, err error) error {
	if errors.Is(err, fs.ErrPermission) {
		return &Error{
			Code:    ErrPermissionDenied,
			Message: fmt.Sprintf("%s: %v", prefix, err),
		}
	}
	return fmt.Errorf("%s: %v", prefix, err)
}

// statNotExist reports whether err is a not-found or not-a-directory stat
// failure, both of which upstream's exists() treats as a missing path.
func statNotExist(err error) bool {
	return errors.Is(err, fs.ErrNotExist) || errors.Is(err, syscall.ENOTDIR)
}

// Ls lists the direct children of the directory at path, sorted by name,
// mirroring ls() in the Python reference, deepagents/backends/filesystem.py.
// A missing path reports the upstream "Path '<path>': path_not_found"
// wording with the file_not_found code, and a non-directory reports
// "Path '<path>': not_a_directory" (the reference maps NotADirectoryError to
// INVALID_PATH in _map_exception_to_standard_error).
func (b *FilesystemBackend) Ls(ctx context.Context, vpath string) (LsResult, error) {
	dir, err := b.resolvePath(vpath)
	if err != nil {
		return LsResult{}, err
	}
	info, err := os.Stat(dir)
	if err != nil {
		if statNotExist(err) {
			return LsResult{}, &Error{
				Code:    ErrFileNotFound,
				Message: fmt.Sprintf("Path '%s': path_not_found", vpath),
			}
		}
		return LsResult{}, osError(fmt.Sprintf("Cannot list '%s'", vpath), err)
	}
	if !info.IsDir() {
		return LsResult{}, &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path '%s': not_a_directory", vpath),
		}
	}
	dirEntries, err := os.ReadDir(dir)
	if err != nil {
		return LsResult{}, osError(fmt.Sprintf("Cannot list '%s'", vpath), err)
	}
	entries := make([]LsEntry, 0, len(dirEntries))
	for _, de := range dirEntries {
		child := filepath.Join(dir, de.Name())
		// Follow symlinks like the reference's child_path.is_file()/
		// is_dir(); entries that are neither (or whose resolution escapes
		// the root) are skipped, as in the reference's virtual-mode ls().
		ci, err := os.Stat(child)
		if err != nil || (!ci.IsDir() && !ci.Mode().IsRegular()) {
			continue
		}
		if !withinDir(b.root, resolveExisting(child)) {
			continue
		}
		entry := LsEntry{Name: de.Name(), ModifiedAt: ci.ModTime()}
		if ci.IsDir() {
			entry.Type = "directory"
		} else {
			entry.Type = "file"
		}
		entries = append(entries, entry)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name < entries[j].Name })
	return LsResult{Entries: entries}, nil
}

// Read reads the requested line window of the file at path, mirroring read()
// in the Python reference, deepagents/backends/filesystem.py, which reads in
// text mode (universal newlines: CRLF and CR become LF on input) and routes
// the content through slice_read_response (deepagents/backends/utils.py). A
// missing path or directory reports "File '<path>' not found" (the
// reference's exists()+is_file() check); an offset past EOF reports the
// upstream "Line offset N exceeds file length (M lines)" wording.
func (b *FilesystemBackend) Read(ctx context.Context, vpath string, offset, limit int) (ReadResult, error) {
	full, err := b.resolvePath(vpath)
	if err != nil {
		return ReadResult{}, err
	}
	info, err := os.Stat(full)
	if err != nil {
		if statNotExist(err) {
			return ReadResult{}, &Error{
				Code:    ErrFileNotFound,
				Message: fmt.Sprintf("File '%s' not found", vpath),
			}
		}
		return ReadResult{}, osError(fmt.Sprintf("Error reading file '%s'", vpath), err)
	}
	if info.IsDir() {
		return ReadResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("File '%s' not found", vpath),
		}
	}
	data, err := os.ReadFile(full)
	if err != nil {
		return ReadResult{}, osError(fmt.Sprintf("Error reading file '%s'", vpath), err)
	}
	// Universal newlines, like the reference's text-mode read: the whole
	// content is LF before slicing, so pagination and the EOF-newline
	// mismatch hints see normalized lines.
	return sliceReadResponse(normalizeLineEndings(string(data)), offset, limit)
}

// sliceReadResponse slices content to the requested line window with
// pagination metadata, mirroring slice_read_response in the Python
// reference, deepagents/backends/utils.py: bounds are clamped through
// NormalizeReadBounds, blank content is reported before the zero-limit
// short-circuit, lines keep their terminators so trailing-newline state
// round-trips, and only the returned window has CRLF/CR normalized to LF.
func sliceReadResponse(content string, offset, limit int) (ReadResult, error) {
	offset, limit = NormalizeReadBounds(offset, limit)

	// Ordering note: blank content is reported before the zero-limit
	// check, so a whitespace-only file returns its content regardless of
	// limit, exactly as upstream.
	if content == "" || strings.TrimSpace(content) == "" {
		return ReadResult{Content: emptyContentWarning}, nil
	}
	if limit == 0 {
		return ReadResult{}, nil
	}

	lines := splitLinesKeepEnds(content)
	total := len(lines)
	if offset >= total {
		return ReadResult{}, fmt.Errorf( //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
			"Line offset %d exceeds file length (%d lines)", offset, total,
		)
	}
	end := offset + limit
	if end > total {
		end = total
	}
	window := strings.ReplaceAll(
		strings.ReplaceAll(strings.Join(lines[offset:end], ""), "\r\n", "\n"),
		"\r", "\n",
	)
	res := ReadResult{
		Content:    window,
		StartLine:  offset + 1,
		EndLine:    end,
		TotalLines: total,
	}
	if end < total {
		res.NextOffset = end
	}
	return res, nil
}

// splitLinesKeepEnds splits content into lines, each keeping its terminator,
// like Python's splitlines(keepends=True) for the LF, CRLF, and CR cases
// (the exotic Unicode line boundaries are not split, matching what files
// written by this port can contain). A trailing fragment without a
// terminator is the final line; a terminator-terminated string has no empty
// trailing line.
func splitLinesKeepEnds(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '\n':
			lines = append(lines, s[start:i+1])
			start = i + 1
		case '\r':
			if i+1 < len(s) && s[i+1] == '\n' {
				continue // the \n iteration closes the CRLF pair
			}
			lines = append(lines, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

// Write writes content to the file at path, creating parent directories and
// creating or overwriting the file, mirroring write() in the Python
// reference, deepagents/backends/filesystem.py (mode 0644, UTF-8). Content
// is stored verbatim — the reference opens with newline="", which converts
// in neither direction, so CRLF input is preserved on disk and Read
// normalizes on the way in.
func (b *FilesystemBackend) Write(ctx context.Context, vpath string, content string) (WriteResult, error) {
	full, err := b.resolvePath(vpath)
	if err != nil {
		return WriteResult{}, err
	}
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return WriteResult{}, osError(fmt.Sprintf("Error writing file '%s'", vpath), err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		return WriteResult{}, osError(fmt.Sprintf("Error writing file '%s'", vpath), err)
	}
	return WriteResult{Path: vpath}, nil
}

// Edit replaces occurrences of oldString with newString in the file at path,
// mirroring edit() in the Python reference, deepagents/backends/filesystem.py:
// the file content and the search strings are LF-normalized before matching
// (the reference reads with universal newlines), replacement validation and
// error messages come from perform_string_replacement
// (deepagents/backends/utils.py), and the file is rewritten in place — as
// upstream, an edit stores the whole file LF-normalized.
func (b *FilesystemBackend) Edit(ctx context.Context, vpath string, oldString, newString string, replaceAll bool) (EditResult, error) {
	full, err := b.resolvePath(vpath)
	if err != nil {
		return EditResult{}, err
	}
	info, err := os.Stat(full)
	if err != nil {
		if !statNotExist(err) {
			return EditResult{}, osError(fmt.Sprintf("Error editing file '%s'", vpath), err)
		}
		return EditResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("Error: File '%s' not found", vpath),
		}
	}
	if info.IsDir() {
		return EditResult{}, &Error{
			Code:    ErrFileNotFound,
			Message: fmt.Sprintf("Error: File '%s' not found", vpath),
		}
	}
	data, err := os.ReadFile(full)
	if err != nil {
		return EditResult{}, osError(fmt.Sprintf("Error editing file '%s'", vpath), err)
	}
	newContent, occurrences, err := performStringReplacement(
		normalizeLineEndings(string(data)),
		normalizeLineEndings(oldString), normalizeLineEndings(newString), replaceAll,
	)
	if err != nil {
		return EditResult{}, err
	}
	if err := os.WriteFile(full, []byte(newContent), 0o644); err != nil {
		return EditResult{}, osError(fmt.Sprintf("Error editing file '%s'", vpath), err)
	}
	return EditResult{Replaced: occurrences}, nil
}

// normalizeLineEndings rewrites CRLF and bare CR to LF, matching Python's
// universal-newline text reads that the Python reference performs
// everywhere it ingests file content (read, edit, and the grep fallback in
// deepagents/backends/filesystem.py) as well as edit()'s normalization of
// the search strings.
func normalizeLineEndings(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\r\n", "\n"), "\r", "\n")
}

// performStringReplacement replaces oldString in content with validation and
// the exact upstream error messages, mirroring perform_string_replacement in
// the Python reference, deepagents/backends/utils.py. On success it returns
// the new content and the number of occurrences replaced.
func performStringReplacement(content, oldString, newString string, replaceAll bool) (string, int, error) {
	if oldString == "" {
		return "", 0, errors.New( //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
			"Error: old_string cannot be empty. Provide the exact text to replace.",
		)
	}

	occurrences := strings.Count(content, oldString)
	if occurrences == 0 {
		// Detect the common EOF mismatch: old_string carries a trailing
		// newline the file lacks, per upstream.
		if strings.HasSuffix(oldString, "\n") && len(oldString) > 1 &&
			strings.HasSuffix(content, strings.TrimSuffix(oldString, "\n")) {
			stripped := strings.TrimSuffix(oldString, "\n")
			strippedCount := strings.Count(content, stripped)
			if strippedCount == 1 {
				return "", 0, errors.New( //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
					"Error: old_string ends with a newline, but the file does " +
						"not end with a newline. Retry with the trailing newline " +
						"removed from old_string (and from new_string if it also " +
						"ends with a newline).",
				)
			}
			return "", 0, fmt.Errorf( //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
				"Error: old_string ends with a newline, but the file does "+
					"not end with a newline. With the trailing newline removed, "+
					"old_string would appear %d times in the file. "+
					"Retry with the trailing newline removed and add surrounding "+
					"context so the match is unique.", strippedCount,
			)
		}
		return "", 0, fmt.Errorf("Error: String not found in file: '%s'", oldString)
	}

	if occurrences > 1 && !replaceAll {
		return "", 0, fmt.Errorf( //nolint:staticcheck // ST1005: upstream-verbatim error literal (parity pinned by tests)
			"Error: String '%s' appears %d times in file. "+
				"Use replace_all=True to replace all instances, or provide a more specific string with surrounding context.",
			oldString, occurrences,
		)
	}

	return strings.ReplaceAll(content, oldString, newString), occurrences, nil
}

// Delete removes a file, a symlink (without following it), or a directory
// tree recursively, mirroring delete() in the Python reference,
// deepagents/backends/filesystem.py, including the "Error: '<path>' not
// found" wording for missing paths.
func (b *FilesystemBackend) Delete(ctx context.Context, vpath string) (DeleteResult, error) {
	full, err := b.resolvePath(vpath)
	if err != nil {
		return DeleteResult{}, err
	}
	info, err := os.Lstat(full)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return DeleteResult{}, &Error{
				Code:    ErrFileNotFound,
				Message: fmt.Sprintf("Error: '%s' not found", vpath),
			}
		}
		return DeleteResult{}, osError(fmt.Sprintf("Error deleting '%s'", vpath), err)
	}
	switch {
	case info.Mode()&os.ModeSymlink != 0:
		err = os.Remove(full)
	case info.IsDir():
		err = os.RemoveAll(full)
	default:
		err = os.Remove(full)
	}
	if err != nil {
		return DeleteResult{}, osError(fmt.Sprintf("Error deleting '%s'", vpath), err)
	}
	return DeleteResult{Success: true}, nil
}

// Glob finds files under path matching pattern, mirroring glob() in the
// Python reference, deepagents/backends/filesystem.py, which routes through
// the shared include-glob contract (compile_grep_include_glob,
// deepagents/backends/utils.py): a pattern without '/' — including a bare
// '**' — matches basenames at any depth under the search path ('*.py'
// matches 'a/b.py' and '.github/x.py'), a pattern containing '/' matches
// the full path relative to the search path ('src/*.py' is top of src
// only; a leading '/' anchors to the search root). Dot handling: a
// basename starting with '.' needs a leading '.' in the pattern segment,
// while slash-less patterns traverse dot directories freely; in '/'
// patterns the dot rule also keeps '**' out of dot directories.
// Only regular files match; results are virtual paths sorted
// lexicographically. A pattern or path containing a `..` segment is refused
// with the upstream "Path traversal not allowed in glob pattern '<p>'"
// wording (the reference raises InvalidGlobPatternError). A missing or
// non-directory search path returns an empty result, not an error.
func (b *FilesystemBackend) Glob(ctx context.Context, vpath string, pattern string) (GlobResult, error) {
	// Compile before touching the filesystem, as upstream: a refused
	// pattern is a property of the pattern alone.
	if hasDotDotComponent(strings.ReplaceAll(pattern, "\\", "/")) {
		return GlobResult{}, &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path traversal not allowed in glob pattern '%s'", pattern),
		}
	}
	search, err := b.resolvePath(vpath)
	if err != nil {
		return GlobResult{}, err
	}
	normSearch, err := NormalizePath(vpath)
	if err != nil {
		return GlobResult{}, err
	}
	info, err := os.Stat(search)
	if err != nil || !info.IsDir() {
		return GlobResult{Files: []string{}}, nil
	}

	var files []string
	err = filepath.WalkDir(search, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if d.IsDir() || !d.Type().IsRegular() {
			return nil
		}
		rel, err := filepath.Rel(search, p)
		if err != nil {
			return nil
		}
		relSlash := filepath.ToSlash(rel)
		if !includeGlobMatches(pattern, relSlash) {
			return nil
		}
		files = append(files, path.Join(normSearch, relSlash))
		return nil
	})
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return GlobResult{}, err
		}
		return GlobResult{}, fmt.Errorf("Error globbing path '%s': %v", vpath, err)
	}
	sort.Strings(files)
	if files == nil {
		files = []string{}
	}
	return GlobResult{Files: files}, nil
}

// Grep searches for pattern as a literal substring in files under path,
// mirroring grep() (Python fallback) in the Python reference,
// deepagents/backends/filesystem.py. Matches are grouped per file in
// deterministic walk order; an include glob without '/' matches basenames
// at any depth while one with '/' matches the search-root-relative path
// (compile_grep_include_glob in deepagents/backends/utils.py); maxCount
// caps the total across files (non-positive means unlimited). Files larger
// than 10 MB are skipped, as upstream.
func (b *FilesystemBackend) Grep(ctx context.Context, pattern string, vpath string, glob string, maxCount int) (GrepResult, error) {
	// Validate the include glob before resolving the search path, as
	// upstream, so a refused pattern is reported rather than silently
	// matching nothing.
	if glob != "" && hasDotDotComponent(strings.ReplaceAll(glob, "\\", "/")) {
		return GrepResult{}, &Error{
			Code:    ErrInvalidPath,
			Message: fmt.Sprintf("Path traversal not allowed in glob pattern '%s'", glob),
		}
	}

	// The reference swallows the path-normalization ValueError and the
	// missing-path case, returning empty matches.
	base, err := b.resolvePath(vpath)
	if err != nil {
		return GrepResult{Matches: []GrepFileMatches{}}, nil
	}
	info, err := os.Stat(base)
	if err != nil {
		return GrepResult{Matches: []GrepFileMatches{}}, nil
	}
	normBase, err := NormalizePath(vpath)
	if err != nil {
		return GrepResult{Matches: []GrepFileMatches{}}, nil
	}

	var files []string
	isDir := info.IsDir()
	if isDir {
		err = filepath.WalkDir(base, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if d.IsDir() || !d.Type().IsRegular() {
				return nil
			}
			files = append(files, p)
			return nil
		})
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return GrepResult{}, err
			}
			return GrepResult{}, osError(fmt.Sprintf("Error searching path '%s'", vpath), err)
		}
	} else {
		files = []string{base}
	}

	var matches []GrepFileMatches
	total := 0
	for _, file := range files {
		if err := ctx.Err(); err != nil {
			return GrepResult{}, err
		}
		// For a directory base, files are keyed by their search-root-
		// relative path; for a single-file base the file is keyed by its
		// own name, as the reference's fallback search does.
		var relSlash, virt string
		if isDir {
			relSlash = filepath.ToSlash(mustRel(base, file))
			virt = path.Join(normBase, relSlash)
		} else {
			relSlash = filepath.Base(file)
			virt = normBase
		}
		if glob != "" && !includeGlobMatches(glob, relSlash) {
			continue
		}
		fi, err := os.Stat(file)
		if err != nil || fi.Size() > maxGrepFileSizeBytes {
			continue
		}
		data, err := os.ReadFile(file)
		if err != nil {
			// Unreadable files are skipped rather than failing the
			// search; see the divergence note on Backend.Grep.
			continue
		}
		remaining := 0
		if maxCount > 0 {
			remaining = maxCount - total
			if remaining <= 0 {
				break
			}
		}
		// Universal newlines, like the reference's Python-fallback search:
		// matched line text never carries a trailing CR.
		lines := GrepLines(normalizeLineEndings(string(data)), pattern, remaining)
		if len(lines) == 0 {
			continue
		}
		total += len(lines)
		matches = append(matches, GrepFileMatches{
			Path:  virt,
			Lines: lines,
		})
	}
	if matches == nil {
		matches = []GrepFileMatches{}
	}
	return GrepResult{Matches: matches}, nil
}

// includeGlobMatches applies the shared include-glob contract from
// compile_grep_include_glob in the Python reference,
// deepagents/backends/utils.py — the single source of truth for both
// grep(..., glob=...) and backend glob() (filesystem.py:1357): a pattern
// without '/' (including a bare '**') matches the basename at any depth,
// so dot directories are traversed and only dot basenames are excluded
// ('**' returns '.git/config' and '*.py' matches '.github/x.py', while
// neither matches '.hidden.py'); a pattern containing '/' (a leading '/'
// anchors) matches the full path relative to the search root, where the
// dot rule also keeps '**' out of dot directories.
func includeGlobMatches(glob, relSlash string) bool {
	pattern := strings.TrimPrefix(glob, "/")
	if !strings.Contains(glob, "/") {
		return MatchGlob(pattern, path.Base(relSlash))
	}
	return MatchGlob(pattern, relSlash)
}

// mustRel returns filepath.Rel(base, p) with a slash-separator fallback to
// the base name on error (which cannot happen for WalkDir-produced paths).
func mustRel(base, p string) string {
	rel, err := filepath.Rel(base, p)
	if err != nil {
		return filepath.Base(p)
	}
	return rel
}
