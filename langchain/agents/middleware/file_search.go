package middleware

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

type GrepOutputMode string

const (
	GrepFilesWithMatches GrepOutputMode = "files_with_matches"
	GrepContent          GrepOutputMode = "content"
	GrepCount            GrepOutputMode = "count"
)

type FilesystemFileSearchMiddleware struct {
	RootPath         string
	MaxFileSizeBytes int64
	Tools            []tools.Tool
	// UseRipgrep enables shelling out to the `rg` binary (if present on PATH)
	// for GrepSearch, mirroring Python's `use_ripgrep` fast path. It falls
	// back to the pure-Go walk-based search when `rg` is unavailable or
	// errors, so this is safe to leave enabled by default.
	UseRipgrep bool
}

func NewFilesystemFileSearchMiddleware(rootPath string, maxFileSizeMB int) (*FilesystemFileSearchMiddleware, error) {
	if maxFileSizeMB <= 0 {
		maxFileSizeMB = 10
	}
	root, err := filepath.Abs(rootPath)
	if err != nil {
		return nil, err
	}
	root, err = filepath.EvalSymlinks(root)
	if err != nil {
		return nil, err
	}
	m := &FilesystemFileSearchMiddleware{
		RootPath:         root,
		MaxFileSizeBytes: int64(maxFileSizeMB) * 1024 * 1024,
		UseRipgrep:       true,
	}
	globTool, err := tools.NewFunc(
		"glob_search",
		"Fast file pattern matching tool that works with any codebase size.",
		schema.Object(map[string]schema.Schema{
			"pattern": schema.String("The glob pattern to match files against."),
			"path":    schema.String("The directory to search in."),
		}, "pattern"),
		func(ctx context.Context, input map[string]any) (tools.Result, error) {
			pattern, _ := input["pattern"].(string)
			path, _ := input["path"].(string)
			if path == "" {
				path = "/"
			}
			return tools.Result{Content: m.GlobSearch(pattern, path)}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	grepTool, err := tools.NewFunc(
		"grep_search",
		"Fast content search tool that works with any codebase size.",
		schema.Object(map[string]schema.Schema{
			"pattern":     schema.String("The regular expression pattern to search for in file contents."),
			"path":        schema.String("The directory to search in."),
			"include":     schema.String("File pattern to filter."),
			"output_mode": schema.String("Output format."),
		}, "pattern"),
		func(ctx context.Context, input map[string]any) (tools.Result, error) {
			pattern, _ := input["pattern"].(string)
			path, _ := input["path"].(string)
			if path == "" {
				path = "/"
			}
			include, _ := input["include"].(string)
			mode, _ := input["output_mode"].(string)
			if mode == "" {
				mode = string(GrepFilesWithMatches)
			}
			return tools.Result{Content: m.GrepSearch(pattern, path, include, GrepOutputMode(mode))}, nil
		},
	)
	if err != nil {
		return nil, err
	}
	m.Tools = []tools.Tool{globTool, grepTool}
	return m, nil
}

func (m *FilesystemFileSearchMiddleware) GlobSearch(pattern string, path string) string {
	base, err := m.validateAndResolvePath(path)
	if err != nil {
		return "No files found"
	}
	info, err := os.Stat(base)
	if err != nil || !info.IsDir() {
		return "No files found"
	}
	if invalidGlobPattern(pattern) {
		return "No files found"
	}
	matcher, err := globMatcher(pattern)
	if err != nil {
		return "No files found"
	}
	type match struct {
		path    string
		modTime time.Time
	}
	matches := []match{}
	_ = filepath.WalkDir(base, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		if !m.isWithinRoot(path) {
			return nil
		}
		rel, err := filepath.Rel(base, path)
		if err != nil {
			return nil
		}
		relSlash := filepath.ToSlash(rel)
		if !matcher(relSlash) {
			return nil
		}
		rootRel, err := filepath.Rel(m.RootPath, path)
		if err != nil {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return nil
		}
		matches = append(matches, match{path: "/" + filepath.ToSlash(rootRel), modTime: info.ModTime()})
		return nil
	})
	if len(matches) == 0 {
		return "No files found"
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].modTime.After(matches[j].modTime)
	})
	out := make([]string, len(matches))
	for i, match := range matches {
		out[i] = match.path
	}
	return strings.Join(out, "\n")
}

func (m *FilesystemFileSearchMiddleware) GrepSearch(pattern string, path string, include string, outputMode GrepOutputMode) string {
	regex, err := regexp.Compile(pattern)
	if err != nil {
		return "Invalid regex pattern: " + err.Error()
	}
	if include != "" && !isValidIncludePattern(include) {
		return "Invalid include pattern"
	}
	var results map[string][]grepMatch
	if m.UseRipgrep {
		if rgResults, ok := m.ripgrepSearch(pattern, path, include); ok {
			results = rgResults
		}
	}
	if results == nil {
		results = m.pythonSearch(regex, path, include)
	}
	if len(results) == 0 {
		return "No matches found"
	}
	return formatGrepResults(results, outputMode)
}

// ripgrepSearch runs `rg` against the resolved, root-confined search
// directory and parses its `path:line:content` output. It returns ok=false
// (rather than an error) whenever ripgrep can't be used for any reason —
// binary missing, non-zero/non-"no matches" exit code, or a path that fails
// root validation — so callers can transparently fall back to pythonSearch,
// matching Python's `_ripgrep_search()`/`_python_search()` fallback pair.
func (m *FilesystemFileSearchMiddleware) ripgrepSearch(pattern string, basePath string, include string) (map[string][]grepMatch, bool) {
	base, err := m.validateAndResolvePath(basePath)
	if err != nil {
		return nil, false
	}
	if _, err := os.Stat(base); err != nil {
		return nil, false
	}
	rgPath, err := exec.LookPath("rg")
	if err != nil {
		return nil, false
	}

	args := []string{"--line-number", "--no-heading", "--with-filename", "--color=never"}
	if include != "" {
		args = append(args, "-g", include)
	}
	args = append(args, "-e", pattern, "--", base)

	cmd := exec.Command(rgPath, args...)
	output, err := cmd.Output()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 {
			// Exit code 1 means "ran fine, no matches" for ripgrep.
			return map[string][]grepMatch{}, true
		}
		return nil, false
	}

	results := map[string][]grepMatch{}
	for _, line := range strings.Split(string(output), "\n") {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 3)
		if len(parts) != 3 {
			continue
		}
		if !m.isWithinRoot(parts[0]) {
			continue
		}
		lineNumber, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		rel, err := filepath.Rel(m.RootPath, parts[0])
		if err != nil {
			continue
		}
		virtual := "/" + filepath.ToSlash(rel)
		results[virtual] = append(results[virtual], grepMatch{lineNumber: lineNumber, line: strings.TrimSuffix(parts[2], "\r")})
	}
	return results, true
}

func (m *FilesystemFileSearchMiddleware) validateAndResolvePath(path string) (string, error) {
	if path == "" {
		path = "/"
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	if strings.Contains(path, "..") || strings.Contains(path, "~") || strings.ContainsRune(path, '\x00') {
		return "", fmt.Errorf("path traversal not allowed")
	}
	full := filepath.Join(m.RootPath, strings.TrimPrefix(path, "/"))
	resolved, err := filepath.Abs(full)
	if err != nil {
		return "", err
	}
	if !m.isWithinRoot(resolved) {
		return "", fmt.Errorf("path outside root")
	}
	return resolved, nil
}

func (m *FilesystemFileSearchMiddleware) isWithinRoot(path string) bool {
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		resolved, err = filepath.Abs(path)
		if err != nil {
			return false
		}
	}
	rel, err := filepath.Rel(m.RootPath, resolved)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func (m *FilesystemFileSearchMiddleware) pythonSearch(regex *regexp.Regexp, basePath string, include string) map[string][]grepMatch {
	base, err := m.validateAndResolvePath(basePath)
	if err != nil {
		return map[string][]grepMatch{}
	}
	if _, err := os.Stat(base); err != nil {
		return map[string][]grepMatch{}
	}
	results := map[string][]grepMatch{}
	_ = filepath.WalkDir(base, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if entry.IsDir() {
			if !m.isWithinRoot(path) {
				return filepath.SkipDir
			}
			return nil
		}
		if !m.isWithinRoot(path) {
			return nil
		}
		if include != "" && !matchIncludePattern(filepath.Base(path), include) {
			return nil
		}
		info, err := entry.Info()
		if err != nil || info.Size() > m.MaxFileSizeBytes {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		lines := strings.Split(string(data), "\n")
		for i, line := range lines {
			if regex.MatchString(line) {
				rel, err := filepath.Rel(m.RootPath, path)
				if err != nil {
					continue
				}
				virtual := "/" + filepath.ToSlash(rel)
				results[virtual] = append(results[virtual], grepMatch{lineNumber: i + 1, line: strings.TrimSuffix(line, "\r")})
			}
		}
		return nil
	})
	return results
}

type grepMatch struct {
	lineNumber int
	line       string
}

func formatGrepResults(results map[string][]grepMatch, outputMode GrepOutputMode) string {
	paths := make([]string, 0, len(results))
	for path := range results {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	lines := []string{}
	switch outputMode {
	case GrepContent:
		for _, path := range paths {
			for _, match := range results[path] {
				lines = append(lines, fmt.Sprintf("%s:%d:%s", path, match.lineNumber, match.line))
			}
		}
	case GrepCount:
		for _, path := range paths {
			lines = append(lines, fmt.Sprintf("%s:%d", path, len(results[path])))
		}
	default:
		lines = paths
	}
	return strings.Join(lines, "\n")
}

func invalidGlobPattern(pattern string) bool {
	return pattern == "" || strings.HasPrefix(pattern, "/") || strings.Contains(pattern, "\x00") || pathHasParentSegment(pattern)
}

func pathHasParentSegment(path string) bool {
	for _, part := range strings.Split(filepath.ToSlash(path), "/") {
		if part == ".." {
			return true
		}
	}
	return false
}

func isValidIncludePattern(pattern string) bool {
	if pattern == "" || strings.ContainsAny(pattern, "\x00\n\r") {
		return false
	}
	expanded := expandIncludePatterns(pattern)
	if len(expanded) == 0 {
		return false
	}
	for _, candidate := range expanded {
		if _, err := globMatcher(candidate); err != nil {
			return false
		}
	}
	return true
}

func matchIncludePattern(basename string, pattern string) bool {
	for _, candidate := range expandIncludePatterns(pattern) {
		matcher, err := globMatcher(candidate)
		if err == nil && matcher(basename) {
			return true
		}
	}
	return false
}

func expandIncludePatterns(pattern string) []string {
	start := strings.Index(pattern, "{")
	if start == -1 {
		if strings.Contains(pattern, "}") {
			return nil
		}
		return []string{pattern}
	}
	end := strings.Index(pattern[start:], "}")
	if end == -1 {
		return nil
	}
	end += start
	inner := pattern[start+1 : end]
	if inner == "" {
		return nil
	}
	out := []string{}
	for _, option := range strings.Split(inner, ",") {
		for _, expanded := range expandIncludePatterns(pattern[:start] + option + pattern[end+1:]) {
			out = append(out, expanded)
		}
	}
	return out
}

func globMatcher(pattern string) (func(string) bool, error) {
	regexPattern := globToRegex(filepath.ToSlash(pattern))
	regex, err := regexp.Compile(regexPattern)
	if err != nil {
		return nil, err
	}
	return regex.MatchString, nil
}

func globToRegex(pattern string) string {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				b.WriteString(".*")
				i++
			} else {
				b.WriteString(`[^/]*`)
			}
		case '?':
			b.WriteString(`[^/]`)
		default:
			b.WriteString(regexp.QuoteMeta(string(ch)))
		}
	}
	b.WriteString("$")
	return b.String()
}
