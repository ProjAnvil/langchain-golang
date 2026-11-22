package utils

import (
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

var (
	smallBErrInvalidJSON  = errors.New("invalid JSON")
	smallBErrNonObject    = errors.New("expected JSON object")
	smallBErrMissingFence = errors.New("missing fences")
)

// smallBStripChars is the set of characters stripped from the start and end of
// a JSON payload, mirroring LangChain's _json_strip_chars.
const smallBStripChars = " \n\r\t`"

// smallBInfoLineRe matches a fenced-code info string (a language tag such as
// "json", or an empty line). It lets smallBExtractJSON skip the line right
// after an opening fence without ever swallowing a line that is part of the
// JSON payload.
var smallBInfoLineRe = regexp.MustCompile(`^[A-Za-z0-9_+.-]*$`)

// smallBActionInputRe captures an "action_input" string value so that raw
// control characters inside it can be escaped, mirroring LangChain's
// _custom_parser. The value is matched non-greedily up to the next quote.
var smallBActionInputRe = regexp.MustCompile(`(?s)("action_input"\s*:\s*")(.*?)(")`)

// ParseAndCheckJSONMarkdown extracts and parses a JSON object from text. It
// accepts either a fenced code block delimited by triple backticks (with an
// optional "json" language tag) or a bare JSON object. The top-level parsed
// value is required to be a JSON object; arrays and scalar values are
// rejected. Raw control characters inside an "action_input" string value
// (newlines, carriage returns, tabs, NUL, and other bytes below 0x20) are
// escaped before parsing so that common LLM tool-call output is accepted.
// Distinct errors are returned for invalid JSON, non-object results, and a
// fenced block that is missing its closing fence.
func ParseAndCheckJSONMarkdown(text string) (map[string]any, error) {
	content, err := smallBExtractJSON(text)
	if err != nil {
		return nil, err
	}
	stripped := strings.Trim(content, smallBStripChars)
	repaired := smallBCustomParser(stripped)
	var parsed any
	if err := json.Unmarshal([]byte(repaired), &parsed); err != nil {
		return nil, fmt.Errorf("%w: %v", smallBErrInvalidJSON, err)
	}
	object, ok := parsed.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: got %s", smallBErrNonObject, smallBJSONTypeName(parsed))
	}
	return object, nil
}

// smallBExtractJSON returns the JSON payload to parse. A bare value (a trimmed
// string starting with '{' or '[') is used as-is. Otherwise the first fenced
// block is used; a fenced block without a closing fence yields a
// missing-fences error.
func smallBExtractJSON(text string) (string, error) {
	trimmed := strings.TrimSpace(text)
	if strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[") {
		return text, nil
	}
	open := strings.Index(text, "```")
	if open < 0 {
		return text, nil
	}
	body := text[open+3:]
	if nl := strings.IndexByte(body, '\n'); nl >= 0 && smallBInfoLineRe.MatchString(strings.TrimSpace(body[:nl])) {
		body = body[nl+1:]
	}
	end := strings.Index(body, "```")
	if end < 0 {
		return "", fmt.Errorf("%w: no closing fence found for the json block", smallBErrMissingFence)
	}
	return body[:end], nil
}

// smallBCustomParser escapes raw control characters inside "action_input"
// string values. It mirrors LangChain's _custom_parser and additionally
// escapes every byte below 0x20 (including NUL) as a JSON-safe sequence.
func smallBCustomParser(s string) string {
	return smallBActionInputRe.ReplaceAllStringFunc(s, func(match string) string {
		parts := smallBActionInputRe.FindStringSubmatch(match)
		if len(parts) != 4 {
			return match
		}
		return parts[1] + smallBEscapeActionInputValue(parts[2]) + parts[3]
	})
}

// smallBEscapeActionInputValue converts raw control characters in an
// "action_input" value into their JSON escape sequences.
func smallBEscapeActionInputValue(value string) string {
	var b strings.Builder
	for i := 0; i < len(value); i++ {
		c := value[i]
		switch c {
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '"':
			b.WriteString(`\"`)
		default:
			if c < 0x20 {
				b.WriteString(fmt.Sprintf(`\u%04x`, c))
			} else {
				b.WriteByte(c)
			}
		}
	}
	return b.String()
}

// smallBJSONTypeName returns a readable name for a decoded JSON value, used in
// the non-object error message.
func smallBJSONTypeName(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case float64, json.Number:
		return "number"
	case bool:
		return "boolean"
	default:
		return fmt.Sprintf("%T", value)
	}
}
