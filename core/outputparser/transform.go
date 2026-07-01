package outputparser

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
)

// JSONPatchOperation is the JSON Patch-shaped diff operation emitted by
// cumulative JSON streaming.
type JSONPatchOperation struct {
	Op    string `json:"op"`
	Path  string `json:"path"`
	Value any    `json:"value,omitempty"`
}

// Transform parses independent text chunks with a parser.
func Transform[T any](ctx context.Context, parser Parser[T], chunks []string) ([]T, error) {
	out := make([]T, 0, len(chunks))
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		parsed, err := parser.Parse(ctx, chunk)
		if err != nil {
			return nil, err
		}
		out = append(out, parsed)
	}
	return out, nil
}

// CumulativeJSONParser parses accumulated JSON chunks. It emits a new value
// only when the partial JSON parses and differs from the previous value.
type CumulativeJSONParser struct {
	Diff bool
}

// Transform parses text chunks cumulatively. When Diff is false it returns the
// parsed values; when Diff is true it returns []JSONPatchOperation values.
func (p CumulativeJSONParser) Transform(ctx context.Context, chunks []string) ([]any, error) {
	var previous any
	var acc strings.Builder
	out := []any{}
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		acc.WriteString(chunk)
		parsed, ok, err := ParsePartialJSON(acc.String())
		if err != nil {
			return nil, err
		}
		if !ok || reflect.DeepEqual(parsed, previous) {
			continue
		}
		if p.Diff {
			out = append(out, MakeJSONPatch(previous, parsed))
		} else {
			out = append(out, parsed)
		}
		previous = cloneJSONValue(parsed)
	}
	return out, nil
}

// ParsePartialJSON parses valid JSON or a best-effort partial JSON object/array
// by closing unfinished strings and delimiters.
func ParsePartialJSON(text string) (any, bool, error) {
	candidate := strings.TrimSpace(stripJSONFence(text))
	if candidate == "" {
		return nil, false, nil
	}
	if value, err := decodeJSON(candidate); err == nil {
		return value, true, nil
	}
	completed, ok := completePartialJSON(candidate)
	if !ok {
		return nil, false, nil
	}
	value, err := decodeJSON(completed)
	if err != nil {
		return nil, false, nil
	}
	return value, true, nil
}

func stripJSONFence(text string) string {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, "```") {
		return text
	}
	lines := strings.Split(trimmed, "\n")
	if len(lines) < 2 {
		return text
	}
	if strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
		lines = lines[:len(lines)-1]
	}
	return strings.Join(lines, "\n")
}

func decodeJSON(text string) (any, error) {
	var out any
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	if err := decoder.Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

func completePartialJSON(text string) (string, bool) {
	var stack []rune
	inString := false
	escaped := false
	for _, r := range text {
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if r == '\\' {
				escaped = true
				continue
			}
			if r == '"' {
				inString = false
			}
			continue
		}
		switch r {
		case '"':
			inString = true
		case '{':
			stack = append(stack, '}')
		case '[':
			stack = append(stack, ']')
		case '}', ']':
			if len(stack) == 0 || stack[len(stack)-1] != r {
				return "", false
			}
			stack = stack[:len(stack)-1]
		}
	}
	builder := strings.Builder{}
	builder.WriteString(strings.TrimRight(text, " \t\r\n,"))
	if inString {
		builder.WriteRune('"')
	}
	for i := len(stack) - 1; i >= 0; i-- {
		builder.WriteRune(stack[i])
	}
	return builder.String(), true
}

// MakeJSONPatch builds JSON Patch-shaped add/replace/remove operations between
// two JSON-like values.
func MakeJSONPatch(previous any, next any) []JSONPatchOperation {
	ops := []JSONPatchOperation{}
	buildPatch(&ops, "", previous, next)
	return ops
}

func buildPatch(ops *[]JSONPatchOperation, path string, previous any, next any) {
	if previous == nil && next != nil {
		*ops = append(*ops, JSONPatchOperation{Op: "add", Path: patchPath(path), Value: next})
		return
	}
	if next == nil && previous != nil {
		*ops = append(*ops, JSONPatchOperation{Op: "remove", Path: patchPath(path)})
		return
	}
	prevMap, prevOK := previous.(map[string]any)
	nextMap, nextOK := next.(map[string]any)
	if prevOK && nextOK {
		for key, prevValue := range prevMap {
			if _, ok := nextMap[key]; !ok {
				*ops = append(*ops, JSONPatchOperation{Op: "remove", Path: patchPath(joinPatchPath(path, key))})
				continue
			}
			buildPatch(ops, joinPatchPath(path, key), prevValue, nextMap[key])
		}
		for key, nextValue := range nextMap {
			if _, ok := prevMap[key]; !ok {
				*ops = append(*ops, JSONPatchOperation{Op: "add", Path: patchPath(joinPatchPath(path, key)), Value: nextValue})
			}
		}
		return
	}
	if !reflect.DeepEqual(previous, next) {
		*ops = append(*ops, JSONPatchOperation{Op: "replace", Path: patchPath(path), Value: next})
	}
}

func patchPath(path string) string {
	if path == "" {
		return ""
	}
	return path
}

func joinPatchPath(base string, token string) string {
	escaped := strings.ReplaceAll(strings.ReplaceAll(token, "~", "~0"), "/", "~1")
	return base + "/" + escaped
}

func cloneJSONValue(value any) any {
	data, err := json.Marshal(value)
	if err != nil {
		return value
	}
	var out any
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.UseNumber()
	if err := decoder.Decode(&out); err != nil {
		return value
	}
	return out
}
