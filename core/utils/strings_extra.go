package utils

import (
	"fmt"
	"slices"
	"strings"
)

// StringifyDict renders a map as "key: value\n" lines. It is the Go port of
// LangChain's stringify_dict: nested maps are indented by a leading newline
// and lists are joined with newlines, mirroring stringify_value. Keys are
// emitted in sorted order for deterministic output because Go maps have no
// defined iteration order. A nil or empty map produces an empty string.
func StringifyDict(data map[string]any) string {
	keys := make([]string, 0, len(data))
	for k := range data {
		keys = append(keys, k)
	}
	slices.Sort(keys)

	var b strings.Builder
	for _, k := range keys {
		b.WriteString(k)
		b.WriteString(": ")
		b.WriteString(smallAStringifyValue(data[k]))
		b.WriteByte('\n')
	}
	return b.String()
}

// smallAStringifyValue renders a value the way LangChain's stringify_value
// does: strings pass through, maps gain a leading newline plus StringifyDict,
// slices are joined with newlines, and everything else uses fmt.Sprint.
func smallAStringifyValue(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case map[string]any:
		return "\n" + StringifyDict(v)
	case []any:
		parts := make([]string, len(v))
		for i, item := range v {
			parts[i] = smallAStringifyValue(item)
		}
		return strings.Join(parts, "\n")
	case []string:
		return strings.Join(v, "\n")
	default:
		return fmt.Sprint(v)
	}
}

// CommaList renders a slice as a comma-and-space separated string. It is the
// Go port of LangChain's comma_list; each item is stringified with fmt.Sprint.
// A nil or empty slice produces an empty string.
func CommaList(items []any) string {
	parts := make([]string, len(items))
	for i, item := range items {
		parts[i] = fmt.Sprint(item)
	}
	return strings.Join(parts, ", ")
}

// SanitizeForPostgres removes NUL (0x00) bytes from text. PostgreSQL text
// fields cannot store NUL bytes, so they are stripped to keep documents
// insertable. It is the Go port of LangChain's sanitize_for_postgres with the
// default empty replacement. Text without NUL bytes is returned unchanged.
func SanitizeForPostgres(text string) string {
	return strings.ReplaceAll(text, "\x00", "")
}
