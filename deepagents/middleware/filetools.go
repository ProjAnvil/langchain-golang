package middleware

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/schema"
	coretools "github.com/projanvil/langchain-golang/core/tools"
	"github.com/projanvil/langchain-golang/deepagents/backends"
)

// The seven filesystem tools, mirroring the tool factories in the Python
// reference, deepagents/middleware/filesystem.py:2115ff (_create_ls_tool ...
// _create_grep_tool). Every tool follows the reference's per-call flow:
// normalize the path argument (validate_path), evaluate the permission rules
// on the normalized path (validate-then-check, filesystem.py:2707-2716),
// call the backend, and render the result with the reference's formatter
// literals. Tool outcomes never fail the run: errors and denies arrive as
// (Result{Content: error text}, nil).

// Tool descriptions, verbatim from the Python reference constants in
// deepagents/middleware/filesystem.py:1569-1670. The read_file description is
// the text-only variant of _READ_FILE_TOOL_DESCRIPTION_TEMPLATE (first_line
// "By default, it reads up to 100 lines..."); the port has no video support.
// The grep description is the without-execute variant
// (_GREP_TOOL_DESCRIPTION_WITHOUT_EXECUTE): the port has no execute tool to
// point regex users at, so the fallback bullet is dropped, exactly as the
// reference drops it when execute is unavailable.
const (
	lsToolDescription = "Lists all files in a directory.\n\nThis is useful for exploring the filesystem and finding the right file to read or edit.\nYou should almost ALWAYS use this tool before using the read_file or edit_file tools."

	readFileToolDescription = "Reads a file from the filesystem. Assume any path the user provides is valid; reading a missing file returns an error.\n\nUsage:\n- By default, it reads up to 100 lines starting from the beginning of the file. Use `offset`/`limit` to page through large files instead of reading them whole.\n- A status header, `@@ field | field | ... @@`, sits above the file content, and every line after it is verbatim file content. When content is truncated, there may be an explanation before the header. Never include the header when editing.\n- Speculatively batch multiple `read_file` calls in one response when several files may be useful.\n- An empty file returns a system-reminder warning in place of contents.\n- Large tool results may be offloaded to a file; the tool message gives the path. Read that path here, paging with `offset`/`limit`.\n- Images (`.png`, `.jpg`, etc.), audio, video, and PDFs return multimodal content blocks (https://docs.langchain.com/oss/python/langchain/messages#multimodal).\n- For images and PDFs, pagination via `offset`/`limit` is text-only - supply `file_path` only\n- Always read a file before editing it."

	writeFileToolDescription = "Writes content to a file. Creates the file if it does not exist; replaces it entirely if it does.\n\nUsage:\n- Use this tool when you intend to create a new file or replace the whole file. You do not need to read the file first.\n- Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.\n"

	editFileToolDescription = "Performs exact string replacements in files.\n\nUsage:\n- You must read the file before editing; this tool errors otherwise.\n- Preserve the exact source indentation from the read output, and never include the read status header in old_string or new_string.\n- Prefer editing an existing file over creating a new one.\n- Only use emojis if the user explicitly requests it."

	deleteToolDescription = "Deletes a file or directory from the filesystem.\n\nUsage:\n- Permanently removes the file or directory at the given absolute path.\n- Deleting a directory removes it and everything inside it, recursively. Prefer\n  deleting a directory in one call over deleting each file individually.\n- This cannot be undone, so only delete paths you are sure are no longer needed.\n"

	globToolDescription = "Find files matching a glob pattern, returning absolute paths.\n\nSupports `*` (any characters within a path segment), `**` (any directories), `?` (single character), `[abc]` (one character from a set), and `{a,b}` (alternatives), e.g. `*.py`, `src/**/*.py`, `*.{yml,yaml}`.\n\nA pattern without `/` matches the file name at any depth under the search root (`*.py` matches `src/app/main.py`). A pattern containing `/` matches the search-root-relative path (`src/**/*.py`). A leading `/` anchors to search root (`/*.py` matches only top-level Python files).\n\nLeading-dot names are only matched when the pattern segment itself starts with `.` (use `.env`, or `.github/**/*.yml`). Because `**` will not descend into dot-directories, the bare form `*.yml` is *broader* than `**/*.yml` and is usually what you want."

	grepToolDescription = "Search for a LITERAL text pattern across files (NOT regex).\n\nThe pattern is matched verbatim: regex metacharacters are ordinary characters, not operators. To match any of several strings, run a separate grep for each; `grep(pattern=\"foo|bar\")` searches for the literal text \"foo|bar\", and `.*` or `\\.` match those characters literally.\n\nReturns matching files or content per `output_mode`. Offloaded large tool results live under the artifacts root (`/large_tool_results/` by default); grep that directory to search them when you do not know the exact path."
)

// Schema field descriptions, verbatim from the Field(description=...) values
// on the reference's input models, deepagents/middleware/filesystem.py:
// 1447-1556 (LsSchema ... GrepSchema) and 1429-1446 (GREP_GLOB_DESCRIPTION,
// GREP_OUTPUT_MODE_DESCRIPTION).
const (
	lsPathArgDescription = "Absolute path to the directory to list. Must be absolute, not relative."

	readFilePathArgDescription = "Absolute path to the file to read. Must be absolute, not relative."
	readOffsetArgDescription   = "Line number to start reading from (0-indexed). Use for pagination of large files."
	readLimitArgDescription    = "Maximum number of lines to read. Use for pagination of large files."

	writeFilePathArgDescription = "Absolute path where the file should be written. Must be absolute, not relative."
	writeContentArgDescription  = "The text content to write to the file. This parameter is required."

	editFilePathArgDescription   = "Absolute path to the file to edit. Must be absolute, not relative."
	editOldStringArgDescription  = "The exact text to find and replace. Must be unique in the file unless replace_all is True."
	editNewStringArgDescription  = "The text to replace old_string with. Must be different from old_string."
	editReplaceAllArgDescription = "If True, replace all occurrences of old_string. If False (default), old_string must be unique."

	deleteFilePathArgDescription = "Absolute path to the file to delete. Must be absolute, not relative."

	globPatternArgDescription = "Glob pattern to match files (e.g., '*.py', '**/*.py', '/subdir/**/*.md'). A pattern without '/' matches the file name at any depth; a pattern containing '/' matches the search-root-relative path; a leading '/' anchors to the search root ('/*.py' matches only top-level files). Leading-dot names are excluded unless the pattern segment starts with '.', so prefer the bare form '*.py' over '**/*.py' -- '**' will not descend into dot-directories like '.github'."
	globPathArgDescription    = "Base directory to search from. Defaults to the backend's default root."

	grepPatternArgDescription    = "Text pattern to search for (literal string, not regex)."
	grepPathArgDescription       = "Directory to search in. Defaults to current working directory."
	grepGlobArgDescription       = "Glob pattern (NOT regex) limiting which files are searched (e.g. '*.py', '*.ts'). A pattern without '/' matches the file name at any depth; a pattern containing '/' matches the search-root-relative path (e.g. 'src/**/*.py'). This is an in-tool file filter, not a call to the separate glob tool. Brace expansion (e.g. '*.{ts,tsx}') is not supported on all backends; run a separate search per extension for reliable results."
	grepOutputModeArgDescription = "Shape of the returned text. 'files_with_matches' (default): newline-separated matching file paths. 'content': matching lines grouped by file under a '<path>:' header, each line indented and formatted '<line_number>: <line text>' (only the matched line, no surrounding context). 'count': one '<path>: <match_count>' line per file."
	grepMaxCountArgDescription   = "Optional cap on the total number of matches returned across all files. Leave unset to use the configured default. When the cap is hit, results are truncated and a note says so; narrow the pattern or path to see the rest."
)

// Read-window defaults, mirroring DEFAULT_READ_OFFSET/DEFAULT_READ_LIMIT,
// deepagents/middleware/filesystem.py:1161-1162.
const (
	fsDefaultReadOffset = 0
	fsDefaultReadLimit  = 100
)

// fsGrepDefaultMaxCount is the total match cap applied when the grep call
// omits max_count, mirroring FilesystemMiddleware(grep_max_count=1000) in the
// reference constructor, filesystem.py:1984.
const fsGrepDefaultMaxCount = 1000

// Result-text literals, mirroring the reference formatters: ls/glob empty
// output (_format_file_paths, filesystem.py:966-970), the read empty-file
// and zero-line reminders (EMPTY_CONTENT_WARNING / NO_LINES_REQUESTED_WARNING,
// filesystem.py:1113-1119, also backends/filesystem.py:20 in this port), and
// grep's empty output (format_grep_matches, backends/utils.py:1150).
const (
	fsNoFilesFound         = "No files found"
	fsNoMatchesFound       = "No matches found"
	fsEmptyContentWarning  = "System reminder: File exists but has empty contents"
	fsNoLinesWarningFormat = "System reminder: no lines were read because `limit` was %d. The file was not inspected and may have contents; retry with `limit` >= 1 to read it."
)

// fsGlobTimeout and fsGrepTimeout bound the bulk search tools, mirroring
// GLOB_TIMEOUT (filesystem.py:1120) and DEFAULT_GREP_TIMEOUT
// (backends/protocol.py:20). They are package vars, read at call time like
// the reference reads GLOB_TIMEOUT, so tests can override the deadlines
// (filetools_test.go, overrideFileToolTimeouts). On expiry the tool returns
// the reference's timeout error text; the backend call itself is abandoned
// when its context fires.
var (
	fsGlobTimeout = 10 * time.Second
	fsGrepTimeout = 15 * time.Second
)

// globTimeoutMessage mirrors _glob_timeout_message, filesystem.py:1146-1152
// ("Error: glob timed out after 10.0s. ..." for the default — the reference's
// float constant renders with a decimal).
func globTimeoutMessage(d time.Duration) string {
	return fmt.Sprintf("Error: glob timed out after %.1fs. Try a more specific pattern or a narrower path.", d.Seconds())
}

// grepTimeoutMessage mirrors the async grep timeout error,
// backends/filesystem.py:912 ("Error: grep timed out after 15s. ..." for the
// DEFAULT_GREP_TIMEOUT integer constant).
func grepTimeoutMessage(d time.Duration) string {
	return fmt.Sprintf("Error: grep timed out after %gs. Try a more specific pattern or a narrower path.", d.Seconds())
}

// filesystemTools builds the seven filesystem tools bound to m. The delete
// tool is registered only when the backend implements backends.Deleter,
// probed by interface assertion like the reference's _supports_delete.
func (m *FilesystemMiddleware) filesystemTools() ([]coretools.Tool, error) {
	builders := []struct {
		name  string
		build func() (coretools.Func, error)
	}{
		{"ls", m.newLsTool},
		{"read_file", m.newReadFileTool},
		{"write_file", m.newWriteFileTool},
		{"edit_file", m.newEditFileTool},
		{"glob", m.newGlobTool},
		{"grep", m.newGrepTool},
	}
	if _, ok := m.Backend.(backends.Deleter); ok {
		builders = append(builders, struct {
			name  string
			build func() (coretools.Func, error)
		}{"delete", m.newDeleteTool})
	}
	tools := make([]coretools.Tool, 0, len(builders))
	for _, b := range builders {
		tool, err := b.build()
		if err != nil {
			return nil, fmt.Errorf("filesystem tool %s: %w", b.name, err)
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

// --- Shared plumbing ----------------------------------------------------

// normalizeToolPath normalizes a tool's path argument before the permission
// gate, mirroring the reference's validate_path-then-check ordering. A path
// that is absent or empty maps to "/" — the conservative gate/search root
// the pathless glob and grep calls use (the reference passes path=None and
// lets each backend pick its default root; both backends in this port treat
// "/" as that root, so the tools gate and search at "/" — a documented
// conservative simplification of the reference's post-filtering,
// filesystem.py:818-835). Tools whose path argument is required (ls,
// read_file, write_file, edit_file, delete) reject a missing or empty
// argument with fsRequiredArg before reaching this function, so the "/"
// fallback applies only to the optional glob/grep path.
func normalizeToolPath(raw string) (string, error) {
	if raw == "" {
		return "/", nil
	}
	return backends.NormalizePath(raw)
}

// denyGate evaluates the middleware's permission rules for a filesystem tool
// call on the normalized path, mirroring _check_fs_permission +
// _permission_error, filesystem.py:417-452. It returns the deny error text
// when the call is denied; the backend is never consulted in that case.
// PermissionInterrupt deliberately falls through as allow: interrupt rules
// gate through the HITL middleware (InterruptConfigsFromPermissions →
// NewInterruptHumanInTheLoopMiddleware), which reviews and pauses the call
// BEFORE the tools node runs it. A wired agent's tools therefore never see an
// interrupt disposition; when the HITL middleware is absent, interrupt rules
// simply do not gate (full contract on InterruptConfigsFromPermissions,
// permissions.go).
func (m *FilesystemMiddleware) denyGate(tool, norm string) (string, bool) {
	mode, _ := m.Check(tool, map[string]any{fsToolPathArgs[tool]: norm})
	if mode == backends.PermissionDeny {
		return fmt.Sprintf("Error: permission denied for %s on %s", fsToolOperations[tool], norm), true
	}
	return "", false
}

// readDenied reports whether the read-class permission rules deny reads of
// path — the post-search filter behind _filter_grep_matches_by_permission
// and _apply_permissions_to_glob_results (filesystem.py:818-833, 942-964,
// via _denied_lookup/_filter_paths_by_permission without backend/scope: no
// symlink resolution and no interrupt-scope removal in this port). The
// search tools drop denied paths from their results after the backend call,
// so a read-denied subtree never leaks through a broader allowed search.
func (m *FilesystemMiddleware) readDenied(path string) bool {
	mode, _ := m.checkPath(fsOperationRead, path)
	return mode == backends.PermissionDeny
}

// deleteDenyGate evaluates the write-class deny rules for a delete of norm,
// mirroring _delete_permission_error + _find_delete_deny_patterns
// (filesystem.py:474-495, 662-717): a recursive delete removes norm and its
// whole subtree, so EVERY deny-write rule that could match norm or anything
// in its subtree blocks the operation regardless of rule order — an earlier
// allow rule cannot guarantee every descendant is safe. On denial it returns
// the reference's error text naming the conflicting pattern(s).
//
// Divergences from the reference, intentional and conservative:
//   - the reference first probes the backend to confirm a plain file and
//     then resolves it first-match-wins (_find_delete_deny_patterns_for_
//     leaf); this port always applies the subtree analysis, so an allow rule
//     never overrides a path-overlapping deny even for confirmed files
//   - interrupt-mode rules do not block: they gate through the HITL
//     middleware before the tool runs (permissions.go), so the subtree
//     analysis accounts for deny rules only
//   - no symlink resolution of the delete target (parked).
func (m *FilesystemMiddleware) deleteDenyGate(norm string) (string, bool) {
	denying := m.deleteDenyPatterns(norm)
	if len(denying) == 0 {
		return "", false
	}
	return fmt.Sprintf("Error: permission denied for write on %s (matches deny rule(s): %s)",
		norm, strings.Join(denying, ", ")), true
}

// deleteDenyPatterns returns the deny-write rule patterns whose matches
// intersect norm's delete subtree, in rule order, mirroring
// _find_delete_deny_patterns (filesystem.py:678-717).
func (m *FilesystemMiddleware) deleteDenyPatterns(norm string) []string {
	var denying []string
	seen := make(map[string]bool)
	for _, rule := range m.Permissions {
		if rule.Mode != backends.PermissionDeny || !slices.Contains(rule.Operations, fsOperationWrite) {
			continue
		}
		for _, pattern := range rule.Paths {
			if seen[pattern] {
				continue
			}
			anchor := globAnchor(pattern)
			var overlaps bool
			if strings.ContainsAny(pattern, "*?[") {
				overlaps = wildcardDeleteOverlap(pattern, anchor, norm)
			} else {
				// Literal pattern: subtree containment either way — a deny
				// on a directory blocks deleting anything inside it and
				// blocks deleting an ancestor that contains it.
				overlaps = pathsOverlap(norm, anchor)
			}
			if overlaps {
				seen[pattern] = true
				denying = append(denying, pattern)
			}
		}
	}
	return denying
}

// globAnchor returns the longest wildcard-free leading directory of a rule
// pattern, mirroring _glob_anchor, deepagents/backends/utils.py:789-806:
// "/work/secrets/**" → "/work/secrets", "/a/*/b" → "/a", and a wildcard at
// or near the root ("/**/x") falls back to "/", which the overlap check
// treats as overlapping any subtree (conservative over-gating).
func globAnchor(pattern string) string {
	var safe []string
	for _, seg := range strings.Split(strings.ReplaceAll(pattern, "\\", "/"), "/") {
		if seg == "" {
			continue
		}
		if strings.ContainsAny(seg, "*?[") {
			break
		}
		safe = append(safe, seg)
	}
	if len(safe) == 0 {
		return "/"
	}
	return "/" + strings.Join(safe, "/")
}

// pathWithin reports whether p is dir itself or nested under it.
func pathWithin(p, dir string) bool {
	if dir == "/" {
		return true
	}
	return p == dir || strings.HasPrefix(p, dir+"/")
}

// pathsOverlap reports whether two literal paths share a subtree either way
// (one equals or is nested under the other).
func pathsOverlap(a, b string) bool {
	return pathWithin(a, b) || pathWithin(b, a)
}

// wildcardDeleteOverlap reports whether a wildcard deny pattern's matches
// intersect a recursive delete target, mirroring _wildcard_delete_overlap
// (filesystem.py:553-601): a root anchor blocks everything; a direct glob
// match blocks; an anchor inside the deleted subtree blocks (the delete
// would remove matching descendants); a target below the anchor blocks
// unless the pattern suffix is a single non-** component that cannot match
// the target's ancestors — so deny /work/*.log does not block deleting
// /work/notes.txt, but deny /work/* blocks deleting /work/app/child because
// it mutates the denied /work/app.
func wildcardDeleteOverlap(pattern, anchor, target string) bool {
	if anchor == "/" {
		return true
	}
	if backends.MatchGlob(pattern, target) {
		return true
	}
	if pathWithin(anchor, target) {
		return true
	}
	if !pathWithin(target, anchor) {
		return false
	}
	anchorSegs := strings.Split(anchor, "/")
	patternSegs := strings.Split(pattern, "/")
	suffix := patternSegs[len(anchorSegs):]
	if len(suffix) != 1 || strings.Contains(suffix[0], "**") {
		return true
	}
	// Block when any ancestor of the target between anchor and target
	// matches the glob (the delete mutates a denied directory's contents).
	targetSegs := strings.Split(target, "/")
	for depth := len(anchorSegs); depth < len(targetSegs); depth++ {
		if backends.MatchGlob(pattern, "/"+strings.Join(targetSegs[1:depth], "/")) {
			return true
		}
	}
	return false
}

// fsErrorResult renders a tool error as a successful invocation carrying
// error text, mirroring the reference's error ToolMessages: the run never
// aborts on a filesystem error.
func fsErrorResult(text string) (coretools.Result, error) {
	return coretools.Result{Content: text}, nil
}

// fsErrText extracts the human-readable message from a backend error. A
// *backends.Error carries the reference's wording in Message (sometimes
// already prefixed "Error: " by the backend, as in StateBackend.Edit/Delete);
// any other error renders via its Error text.
func fsErrText(err error) string {
	var fsErr *backends.Error
	if errors.As(err, &fsErr) {
		return fsErr.Message
	}
	return err.Error()
}

// filesCommandFromContext pulls the StateView the middleware injected around
// the call (FilesystemMiddleware.WrapToolCall) and renders its snapshot as
// the Overwrite-wrapped files Command artifact (a channel replacement; see
// filesCommand); nil when the context carries no view (a non-state backend)
// or the view recorded no changes. Mutating tools call it after a successful
// backend call; read-only tools never do.
func filesCommandFromContext(ctx context.Context) any {
	view, ok := backends.StateViewFromContext(ctx)
	if !ok {
		return nil
	}
	return filesCommand(view)
}

// joinVirtualPath joins a normalized directory and an entry name.
func joinVirtualPath(dir, name string) string {
	if dir == "/" {
		return "/" + name
	}
	return dir + "/" + name
}

// formatPathList renders paths the way the reference's _format_file_paths
// does — Python's str() of a list of strings, e.g. "['/a', '/b']". The
// reference additionally size-truncates through truncate_if_too_long; the
// port has no token-budget helper at this layer, so long lists pass through
// untruncated (a documented MVP simplification).
func formatPathList(paths []string) string {
	quoted := make([]string, len(paths))
	for i, p := range paths {
		quoted[i] = "'" + p + "'"
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

// formatReadResult assembles a read_file result from a backend ReadResult,
// mirroring _handle_read_result → _window_fields/_read_header/
// _assemble_read/_clamped_offset_notice, filesystem.py:992-1112. The status
// header states the line window ("lines A-B of T") plus the resume offset
// ("next offset N") whenever the window stopped short of EOF; every line
// after the header is verbatim file content (one trailing newline dropped,
// like _format_source_block). An empty/whitespace-only file returns the
// backend's empty-contents reminder in place of a header, and a window that
// read zero lines returns the zero-line reminder.
func formatReadResult(res backends.ReadResult, offset, limit int) string {
	if res.Content == fsEmptyContentWarning {
		return res.Content
	}
	if res.StartLine == 0 && res.Content == "" {
		return fmt.Sprintf(fsNoLinesWarningFormat, limit)
	}
	span := fmt.Sprintf("lines %d-%d", res.StartLine, res.EndLine)
	if res.TotalLines > 0 {
		span = fmt.Sprintf("lines %d-%d of %d", res.StartLine, res.EndLine, res.TotalLines)
	}
	fields := []string{span}
	if res.NextOffset > 0 {
		fields = append(fields, fmt.Sprintf("next offset %d", res.NextOffset))
	}
	parts := make([]string, 0, 3)
	if offset < 0 {
		parts = append(parts, fmt.Sprintf(
			"[Requested offset %d is before the start of the file; read from line 1 instead.]", offset))
	}
	parts = append(parts, "@@ "+strings.Join(fields, " | ")+" @@", strings.TrimSuffix(res.Content, "\n"))
	return strings.Join(parts, "\n")
}

// formatGrepResult renders grep matches per output_mode, mirroring
// _format_grep_results, backends/utils.py:1121ff: files_with_matches lists
// one path per line, count emits "<path>: <n>" per file, and content groups
// matched lines under a "<path>:" header, each indented "<lineno>: <text>".
// Files are sorted by path. The reference clips long matched lines at
// GREP_LINE_CHAR_LIMIT before formatting (filesystem.py:874-898); the port
// passes lines through unclipped (a documented MVP simplification).
func formatGrepResult(res backends.GrepResult, outputMode string) string {
	if len(res.Matches) == 0 {
		return fsNoMatchesFound
	}
	matches := append([]backends.GrepFileMatches(nil), res.Matches...)
	sort.Slice(matches, func(i, j int) bool { return matches[i].Path < matches[j].Path })
	switch outputMode {
	case "content":
		var b strings.Builder
		for _, fm := range matches {
			b.WriteString(fm.Path)
			b.WriteString(":\n")
			for _, line := range fm.Lines {
				fmt.Fprintf(&b, "  %d: %s\n", line.LineNumber, line.Text)
			}
		}
		return strings.TrimSuffix(b.String(), "\n")
	case "count":
		lines := make([]string, len(matches))
		for i, fm := range matches {
			lines[i] = fmt.Sprintf("%s: %d", fm.Path, len(fm.Lines))
		}
		return strings.Join(lines, "\n")
	default: // files_with_matches
		paths := make([]string, len(matches))
		for i, fm := range matches {
			paths[i] = fm.Path
		}
		return strings.Join(paths, "\n")
	}
}

// --- Argument coercion ---------------------------------------------------

// argString reads a string argument, defaulting to "".
func argString(input map[string]any, key string) string {
	s, _ := input[key].(string)
	return s
}

// argInt reads an integer argument, accepting the int (direct invocation)
// and float64 (JSON-decoded) shapes, defaulting to def when absent or of
// another type.
func argInt(input map[string]any, key string, def int) int {
	switch v := input[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return def
	}
}

// argBool reads a boolean argument, defaulting to def when absent.
func argBool(input map[string]any, key string, def bool) bool {
	if v, ok := input[key].(bool); ok {
		return v
	}
	return def
}

// argIntPresent reads an integer argument and reports whether it was
// present at all (distinguishing "omitted" from "zero").
func argIntPresent(input map[string]any, key string) (int, bool) {
	switch v := input[key].(type) {
	case int:
		return v, true
	case int64:
		return int(v), true
	case float64:
		return int(v), true
	default:
		return 0, false
	}
}

// fsRequiredArg renders the validation error for a missing or empty
// required path argument. Upstream these arguments are enforced by pydantic
// before the tool body runs (LsSchema.path, ReadFileSchema.file_path, ...);
// the Go tool performs the equivalent check at the function boundary so an
// empty path can never fall through to normalization (and from there to
// "/"), which would have wiped the root on write-class tools.
func fsRequiredArg(name string) (coretools.Result, error) {
	return fsErrorResult("Error: " + name + " is required")
}

// --- Tool builders -------------------------------------------------------

// newLsTool builds the ls tool, mirroring _create_ls_tool,
// filesystem.py:2148ff. Entries are rendered as full virtual paths in a
// Python list literal (the reference's _format_file_paths over the entries'
// path field; this port's LsEntry carries only the name, so the tool joins it
// onto the normalized directory). After the backend call, entries under
// read-denied rules are dropped (_apply_permissions_to_ls_results,
// filesystem.py:2183-2185) — the readDenied post-filter family glob/grep
// already use.
func (m *FilesystemMiddleware) newLsTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"ls",
		lsToolDescription,
		schema.Object(map[string]schema.Schema{
			"path": schema.String(lsPathArgDescription),
		}, "path"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			if argString(input, "path") == "" {
				return fsRequiredArg("path")
			}
			norm, err := normalizeToolPath(argString(input, "path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.denyGate("ls", norm); denied {
				return fsErrorResult(deny)
			}
			lsRes, err := m.Backend.Ls(ctx, norm)
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if len(lsRes.Entries) == 0 {
				return coretools.Result{Content: fsNoFilesFound}, nil
			}
			paths := make([]string, 0, len(lsRes.Entries))
			for _, entry := range lsRes.Entries {
				// Drop entries under read-denied rules after the backend
				// call, mirroring _apply_permissions_to_ls_results (UP
				// filesystem.py:2183-2185, 942-951) — the same readDenied
				// post-filter family glob/grep use, so a read-denied subtree
				// never leaks through a listing of an allowed parent.
				p := joinVirtualPath(norm, entry.Name)
				if m.readDenied(p) {
					continue
				}
				paths = append(paths, p)
			}
			if len(paths) == 0 {
				return coretools.Result{Content: fsNoFilesFound}, nil
			}
			return coretools.Result{Content: formatPathList(paths)}, nil
		},
	)
}

// newReadFileTool builds the read_file tool, mirroring
// _create_read_file_tool, filesystem.py:2243ff. Read errors are prefixed
// "Error: " like the reference's read error ToolMessages.
func (m *FilesystemMiddleware) newReadFileTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"read_file",
		readFileToolDescription,
		schema.Object(map[string]schema.Schema{
			"file_path": schema.String(readFilePathArgDescription),
			"offset":    schema.Integer(readOffsetArgDescription),
			"limit":     schema.Integer(readLimitArgDescription),
		}, "file_path"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			if argString(input, "file_path") == "" {
				return fsRequiredArg("file_path")
			}
			norm, err := normalizeToolPath(argString(input, "file_path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.denyGate("read_file", norm); denied {
				return fsErrorResult(deny)
			}
			offset := argInt(input, "offset", fsDefaultReadOffset)
			limit := argInt(input, "limit", fsDefaultReadLimit)
			res, err := m.Backend.Read(ctx, norm, offset, limit)
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			return coretools.Result{Content: formatReadResult(res, offset, limit)}, nil
		},
	)
}

// newWriteFileTool builds the write_file tool, mirroring
// _create_write_file_tool, filesystem.py:2421ff. On a state backend the
// write lands in the injected StateView and the tool returns the view's
// snapshot as an Overwrite-wrapped files Command artifact (the ToolNode
// commits it as a channel replacement; see filesCommand), the port's
// stand-in for the reference sending through its ToolRuntime. Backend
// errors render as-is, like the reference passing WriteResult.error
// through.
func (m *FilesystemMiddleware) newWriteFileTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"write_file",
		writeFileToolDescription,
		schema.Object(map[string]schema.Schema{
			"file_path": schema.String(writeFilePathArgDescription),
			"content":   schema.String(writeContentArgDescription),
		}, "file_path", "content"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			if argString(input, "file_path") == "" {
				return fsRequiredArg("file_path")
			}
			norm, err := normalizeToolPath(argString(input, "file_path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.denyGate("write_file", norm); denied {
				return fsErrorResult(deny)
			}
			res, err := m.Backend.Write(ctx, norm, argString(input, "content"))
			if err != nil {
				return fsErrorResult(fsErrText(err))
			}
			return coretools.Result{
				Content:  fmt.Sprintf("Updated file %s", res.Path),
				Artifact: filesCommandFromContext(ctx),
			}, nil
		},
	)
}

// newEditFileTool builds the edit_file tool, mirroring
// _create_edit_file_tool, filesystem.py:2512ff. The success message names
// the edited path; the reference reads it off EditResult.path, which this
// port's EditResult does not carry, so the tool substitutes the normalized
// path it passed to the backend (identical for both port backends, which key
// by normalized path). Replacement error literals come verbatim from the
// backend's performStringReplacement.
func (m *FilesystemMiddleware) newEditFileTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"edit_file",
		editFileToolDescription,
		schema.Object(map[string]schema.Schema{
			"file_path":   schema.String(editFilePathArgDescription),
			"old_string":  schema.String(editOldStringArgDescription),
			"new_string":  schema.String(editNewStringArgDescription),
			"replace_all": schema.Boolean(editReplaceAllArgDescription),
		}, "file_path", "old_string", "new_string"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			if argString(input, "file_path") == "" {
				return fsRequiredArg("file_path")
			}
			norm, err := normalizeToolPath(argString(input, "file_path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.denyGate("edit_file", norm); denied {
				return fsErrorResult(deny)
			}
			res, err := m.Backend.Edit(ctx, norm,
				argString(input, "old_string"), argString(input, "new_string"),
				argBool(input, "replace_all", false))
			if err != nil {
				return fsErrorResult(fsErrText(err))
			}
			return coretools.Result{
				Content:  fmt.Sprintf("Successfully replaced %d instance(s) of the string in '%s'", res.Replaced, norm),
				Artifact: filesCommandFromContext(ctx),
			}, nil
		},
	)
}

// newDeleteTool builds the delete tool, mirroring _create_delete_tool,
// filesystem.py:2609ff. Its path argument is `file_path`, matching the
// reference's DeleteSchema (filesystem.py:1510-1513). The success message
// names the deleted path; the reference reads it off DeleteResult.path,
// which this port's DeleteResult does not carry, so the tool substitutes the
// normalized path it passed to the backend. Permission gating goes through
// deleteDenyGate's subtree-overlap analysis (see its doc comment) instead of
// the plain per-path gate.
func (m *FilesystemMiddleware) newDeleteTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"delete",
		deleteToolDescription,
		schema.Object(map[string]schema.Schema{
			"file_path": schema.String(deleteFilePathArgDescription),
		}, "file_path"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			if argString(input, "file_path") == "" {
				return fsRequiredArg("file_path")
			}
			norm, err := normalizeToolPath(argString(input, "file_path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.deleteDenyGate(norm); denied {
				return fsErrorResult(deny)
			}
			if _, err := m.Backend.(backends.Deleter).Delete(ctx, norm); err != nil {
				return fsErrorResult(fsErrText(err))
			}
			return coretools.Result{
				Content:  fmt.Sprintf("Deleted %s", norm),
				Artifact: filesCommandFromContext(ctx),
			}, nil
		},
	)
}

// newGlobTool builds the glob tool, mirroring _create_glob_tool,
// filesystem.py:2703ff: normalize the path (absent → "/", the conservative
// gate/search root for a pathless call), gate read permissions on it, run
// the backend glob under a GLOB_TIMEOUT context deadline, then drop
// read-denied matches from the result (the reference's
// _apply_permissions_to_glob_results, filesystem.py:954-964). A pathless
// call denied by a root-wide rule is the documented conservative divergence:
// the reference keeps pathless globs allowed unless the default root itself
// is denied (and appends GLOB_PATHLESS_DENIED_HINT); the port gates at "/"
// for every call. Backend failures render as
// "Error: glob failed: {e}" (filesystem.py:2788).
func (m *FilesystemMiddleware) newGlobTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"glob",
		globToolDescription,
		schema.Object(map[string]schema.Schema{
			"pattern": schema.String(globPatternArgDescription),
			"path":    schema.String(globPathArgDescription),
		}, "pattern"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			norm, err := normalizeToolPath(argString(input, "path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.denyGate("glob", norm); denied {
				return fsErrorResult(deny)
			}
			searchCtx, cancel := context.WithTimeout(ctx, fsGlobTimeout)
			defer cancel()
			res, err := m.Backend.Glob(searchCtx, norm, argString(input, "pattern"))
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					return fsErrorResult(globTimeoutMessage(fsGlobTimeout))
				}
				return fsErrorResult("Error: glob failed: " + fsErrText(err))
			}
			files := make([]string, 0, len(res.Files))
			for _, f := range res.Files {
				if !m.readDenied(f) {
					files = append(files, f)
				}
			}
			if len(files) == 0 {
				return coretools.Result{Content: fsNoFilesFound}, nil
			}
			return coretools.Result{Content: formatPathList(files)}, nil
		},
	)
}

// newGrepTool builds the grep tool, mirroring _create_grep_tool,
// filesystem.py:2899ff: normalize the path when present (absent → "/", the
// conservative gate/search root — the reference performs no pathless gate at
// all and instead filters matches after the search, filesystem.py:818-835;
// gating the whole call at "/" is the port's documented conservative
// simplification), validate max_count like the reference's GrepSchema
// constraint (Field(gt=0): zero or negative is a validation error, never an
// unlimited pass-through; only an omitted max_count yields the configured
// default of 1000), run the backend grep under a DEFAULT_GREP_TIMEOUT
// context deadline, drop read-denied files' matches from the result in every
// output mode (_filter_grep_matches_by_permission, filesystem.py:818-833),
// and format what remains per output_mode. An unrecognized output_mode falls
// back to the default files_with_matches.
func (m *FilesystemMiddleware) newGrepTool() (coretools.Func, error) {
	return coretools.NewFunc(
		"grep",
		grepToolDescription,
		schema.Object(map[string]schema.Schema{
			"pattern":     schema.String(grepPatternArgDescription),
			"path":        schema.String(grepPathArgDescription),
			"glob":        schema.String(grepGlobArgDescription),
			"output_mode": schema.Schema{"type": "string", "enum": []string{"files_with_matches", "content", "count"}, "description": grepOutputModeArgDescription},
			"max_count":   schema.Integer(grepMaxCountArgDescription),
		}, "pattern"),
		func(ctx context.Context, input map[string]any) (coretools.Result, error) {
			norm, err := normalizeToolPath(argString(input, "path"))
			if err != nil {
				return fsErrorResult("Error: " + fsErrText(err))
			}
			if deny, denied := m.denyGate("grep", norm); denied {
				return fsErrorResult(deny)
			}
			maxCount := fsGrepDefaultMaxCount
			if v, present := argIntPresent(input, "max_count"); present {
				if v <= 0 {
					return fsErrorResult("Error: max_count must be greater than 0")
				}
				maxCount = v
			}
			outputMode := argString(input, "output_mode")
			if outputMode != "content" && outputMode != "count" && outputMode != "files_with_matches" {
				outputMode = "files_with_matches"
			}
			searchCtx, cancel := context.WithTimeout(ctx, fsGrepTimeout)
			defer cancel()
			res, err := m.Backend.Grep(searchCtx,
				argString(input, "pattern"), norm,
				argString(input, "glob"),
				maxCount)
			if err != nil {
				if errors.Is(err, context.DeadlineExceeded) {
					return fsErrorResult(grepTimeoutMessage(fsGrepTimeout))
				}
				return fsErrorResult(fsErrText(err))
			}
			matches := make([]backends.GrepFileMatches, 0, len(res.Matches))
			for _, fm := range res.Matches {
				if !m.readDenied(fm.Path) {
					matches = append(matches, fm)
				}
			}
			res.Matches = matches
			return coretools.Result{Content: formatGrepResult(res, outputMode)}, nil
		},
	)
}
