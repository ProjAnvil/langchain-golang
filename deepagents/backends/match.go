// Package backends ports the deepagents backends subsystem to Go. It mirrors
// the Python reference implementation under deepagents/backends/ in the
// upstream deepagents repository (libs/deepagents/deepagents/backends/),
// starting with glob matching, line grep helpers, and virtual filesystem
// path normalization.
package backends

import (
	"path"
	"strings"
)

// MatchGlob reports whether the full path name matches the glob pattern.
//
// Supported syntax:
//   - '*' matches any sequence of non-separator characters within one path
//     segment
//   - '?' matches exactly one non-separator character
//   - '[...]' matches a character class, as in path.Match
//   - '**' as a complete segment matches zero or more whole segments,
//     so "a/**/*.go" matches both "a/c.go" and "a/b/c.go"
//   - braces have no special meaning; "a.{x,y}" matches only the literal
//     string "a.{x,y}"
//
// Pattern and name are both split on '/', so wildcards never span a slash
// unless they are a standalone "**" segment. A malformed character class in
// the pattern never matches. This mirrors the glob contract implemented by
// _glob_search_files in the Python reference, deepagents/backends/utils.py.
func MatchGlob(pattern, name string) bool {
	return matchSegments(strings.Split(pattern, "/"), strings.Split(name, "/"))
}

// matchSegments matches name segments against pattern segments recursively.
// A "**" pattern segment consumes zero or more name segments; every other
// segment is matched with path.Match, which keeps '*', '?', and '[...]'
// confined to a single segment.
func matchSegments(pattern, name []string) bool {
	if len(pattern) == 0 {
		return len(name) == 0
	}
	if pattern[0] == "**" {
		for i := 0; i <= len(name); i++ {
			if matchSegments(pattern[1:], name[i:]) {
				return true
			}
		}
		return false
	}
	if len(name) == 0 {
		return false
	}
	ok, err := path.Match(pattern[0], name[0])
	if err != nil || !ok {
		return false
	}
	return matchSegments(pattern[1:], name[1:])
}

// GrepLine is a single grep match: a 1-based line number and the full text
// of the matched line.
type GrepLine struct {
	LineNumber int
	Text       string
}

// GrepLines returns the lines of content that contain pattern as a literal
// substring, in order, with 1-based line numbers. The match is plain
// substring containment, not a regular expression, so a pattern like "a.c"
// does not match the line "abc". A maxCount of zero or less is treated as
// unlimited; a positive maxCount stops the scan after that many matches.
func GrepLines(content, pattern string, maxCount int) []GrepLine {
	var matches []GrepLine
	for i, line := range strings.Split(content, "\n") {
		if !strings.Contains(line, pattern) {
			continue
		}
		matches = append(matches, GrepLine{LineNumber: i + 1, Text: line})
		if maxCount > 0 && len(matches) >= maxCount {
			break
		}
	}
	return matches
}
