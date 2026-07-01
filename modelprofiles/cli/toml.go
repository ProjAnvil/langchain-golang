package cli

// Package cli implements the "refresh" workflow of Python's
// langchain_model_profiles.cli module: downloading model capability data from
// models.dev, merging it with local overrides, and writing a canonical
// profiles data file.
//
// This file implements a minimal TOML subset parser sufficient for
// `profile_augmentations.toml` files, which only use top-level scalar
// assignments and one level of (optionally dotted/quoted) table headers, for
// example:
//
//	provider = "anthropic"
//
//	[overrides]
//	tool_call_streaming = true
//
//	[overrides."claude-haiku-4-5"]
//	structured_output = true
//
// It intentionally does not implement full TOML (arrays, multiline strings,
// inline tables, dates, etc.) since the augmentations files never use those
// features. Anything outside this subset produces an explicit parse error
// rather than silently misbehaving.

import (
	"fmt"
	"strconv"
	"strings"
)

// parseTOMLSubset parses the restricted TOML subset described above into a
// tree of nested maps keyed by table path segment.
func parseTOMLSubset(data []byte) (map[string]any, error) {
	root := map[string]any{}
	var current map[string]any = root

	lines := strings.Split(string(data), "\n")
	for lineNo, rawLine := range lines {
		line := stripTOMLComment(rawLine)
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "[") {
			if !strings.HasSuffix(line, "]") {
				return nil, fmt.Errorf("line %d: malformed table header %q", lineNo+1, rawLine)
			}
			header := line[1 : len(line)-1]
			segments, err := splitTOMLTablePath(header)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNo+1, err)
			}
			table, err := getOrCreateTable(root, segments)
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", lineNo+1, err)
			}
			current = table
			continue
		}

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("line %d: expected key = value, got %q", lineNo+1, rawLine)
		}
		key = strings.TrimSpace(unquoteTOMLKey(strings.TrimSpace(key)))
		parsed, err := parseTOMLValue(strings.TrimSpace(value))
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo+1, err)
		}
		current[key] = parsed
	}

	return root, nil
}

func stripTOMLComment(line string) string {
	inQuotes := false
	for i, r := range line {
		switch r {
		case '"':
			inQuotes = !inQuotes
		case '#':
			if !inQuotes {
				return line[:i]
			}
		}
	}
	return line
}

// splitTOMLTablePath splits a table header body like `overrides."claude-x".y`
// into its path segments, honoring quoted segments that may contain dots.
func splitTOMLTablePath(header string) ([]string, error) {
	var segments []string
	var current strings.Builder
	inQuotes := false

	flush := func() {
		segments = append(segments, current.String())
		current.Reset()
	}

	for i := 0; i < len(header); i++ {
		ch := header[i]
		switch {
		case ch == '"':
			inQuotes = !inQuotes
		case ch == '.' && !inQuotes:
			flush()
		default:
			current.WriteByte(ch)
		}
	}
	if inQuotes {
		return nil, fmt.Errorf("unterminated quoted segment in table header %q", header)
	}
	flush()

	out := make([]string, 0, len(segments))
	for _, seg := range segments {
		trimmed := strings.TrimSpace(seg)
		if trimmed == "" {
			return nil, fmt.Errorf("empty segment in table header %q", header)
		}
		out = append(out, trimmed)
	}
	return out, nil
}

func unquoteTOMLKey(key string) string {
	if len(key) >= 2 && strings.HasPrefix(key, "\"") && strings.HasSuffix(key, "\"") {
		return key[1 : len(key)-1]
	}
	return key
}

func getOrCreateTable(root map[string]any, path []string) (map[string]any, error) {
	table := root
	for _, segment := range path {
		next, ok := table[segment]
		if !ok {
			created := map[string]any{}
			table[segment] = created
			table = created
			continue
		}
		nextTable, ok := next.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("path segment %q is not a table", segment)
		}
		table = nextTable
	}
	return table, nil
}

func parseTOMLValue(value string) (any, error) {
	switch value {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "":
		return nil, fmt.Errorf("empty value")
	}
	if strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"") && len(value) >= 2 {
		return value[1 : len(value)-1], nil
	}
	if i, err := strconv.ParseInt(value, 10, 64); err == nil {
		return i, nil
	}
	if f, err := strconv.ParseFloat(value, 64); err == nil {
		return f, nil
	}
	return nil, fmt.Errorf("unsupported TOML value %q", value)
}
