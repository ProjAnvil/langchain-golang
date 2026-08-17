package ollama

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestExtractImageVariants(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		block map[string]any
		want  string
	}{
		{"base64 data URL", map[string]any{"type": "image", "base64": "data:image/png;base64,aGVsbG8="}, "aGVsbG8="},
		{"base64 plain payload", map[string]any{"type": "image", "base64": "aGVsbG8="}, "aGVsbG8="},
		{"v0 data field", map[string]any{"type": "image", "data": "c3Rhcg=="}, "c3Rhcg=="},
		{"v0 data field data URL", map[string]any{"type": "image", "data": "data:image/png;base64,c3Rhcg=="}, "c3Rhcg=="},
		{"v0 source object", map[string]any{"type": "image", "source": map[string]any{"data": "c291cmNl"}}, "c291cmNl"},
		{"image_url object", map[string]any{"type": "image_url", "image_url": map[string]any{"url": "b2Jq"}}, "b2Jq"},
		{"image_url string", map[string]any{"type": "image_url", "image_url": "c3RyaW5n"}, "c3RyaW5n"},
		{"bare url field", map[string]any{"type": "image_url", "url": "dXJs"}, "dXJs"},
		{"no usable payload", map[string]any{"type": "image"}, ""},
		{"empty base64 falls through", map[string]any{"type": "image", "base64": "", "data": "ZmFsbA=="}, "ZmFsbA=="},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := extractImage(tc.block); got != tc.want {
				t.Fatalf("extractImage: got %q want %q", got, tc.want)
			}
		})
	}
}

func TestStripDataURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		value string
		want  string
	}{
		{"data URI stripped", "data:image/png;base64,QUJD", "QUJD"},
		{"plain value unchanged", "QUJD", "QUJD"},
		{"data prefix without comma unchanged", "data:novalue", "data:novalue"},
		{"http URL unchanged", "https://example.com/x.png", "https://example.com/x.png"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := stripDataURL(tc.value); got != tc.want {
				t.Fatalf("stripDataURL(%q): got %q want %q", tc.value, got, tc.want)
			}
		})
	}
}

func TestExtractContentJoinsTextAndSkipsToolUse(t *testing.T) {
	t.Parallel()
	message := messages.Human("leading content")
	message.ContentBlocks = []messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "text", "text": "block text"}),
		messages.ParseContentBlock(map[string]any{"type": "tool_use", "name": "add", "input": map[string]any{"a": 1}}),
		messages.ParseContentBlock(map[string]any{"type": "text", "text": ""}),
	}
	content, images := extractContent(message)
	if content != "leading content\nblock text" {
		t.Fatalf("content: got %q", content)
	}
	if len(images) != 0 {
		t.Fatalf("images: got %v want none", images)
	}
}

func TestToOllamaMessageMapsReasoningContent(t *testing.T) {
	t.Parallel()

	thinking := messages.AI("answer")
	thinking.AdditionalKwargs = map[string]any{"reasoning_content": "let me think"}
	out := toOllamaMessage(thinking)
	if out.Thinking != "let me think" {
		t.Fatalf("thinking: got %q", out.Thinking)
	}

	nonString := messages.AI("answer")
	nonString.AdditionalKwargs = map[string]any{"reasoning_content": 42}
	if out := toOllamaMessage(nonString); out.Thinking != "" {
		t.Fatalf("non-string reasoning_content should be ignored: got %q", out.Thinking)
	}

	empty := messages.AI("answer")
	empty.AdditionalKwargs = map[string]any{"reasoning_content": ""}
	if out := toOllamaMessage(empty); out.Thinking != "" {
		t.Fatalf("empty reasoning_content should be ignored: got %q", out.Thinking)
	}
}

func TestParseToolCallArguments(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		arguments any
		want      map[string]any
		wantOK    bool
	}{
		{"nil arguments", nil, map[string]any{}, true},
		{"empty string", "", map[string]any{}, true},
		{"whitespace string", "   ", map[string]any{}, true},
		{"JSON object string", `{"a":1}`, map[string]any{"a": float64(1)}, true},
		{"invalid JSON string", "{bad json}", nil, false},
		{"non-string non-map", float64(7), nil, false},
		{"plain map", map[string]any{"x": "y"}, map[string]any{"x": "y"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := parseToolCallArguments(tc.arguments, "tool")
			if ok != tc.wantOK {
				t.Fatalf("ok: got %v want %v", ok, tc.wantOK)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("args: got %#v want %#v", got, tc.want)
			}
		})
	}
}

func TestNormalizeArgumentMap(t *testing.T) {
	t.Parallel()

	got := normalizeArgumentMap(map[string]any{
		"functionName": "add",
		"nested":       `{"k":2}`,
		"list":         "[1,2]",
		"plain":        "text",
		"broken":       "{nope}",
		"number":       float64(3),
	}, "add")

	if _, present := got["functionName"]; present {
		t.Fatalf("functionName echo should be filtered: %#v", got)
	}
	if !reflect.DeepEqual(got["nested"], map[string]any{"k": float64(2)}) {
		t.Fatalf("nested: got %#v", got["nested"])
	}
	if !reflect.DeepEqual(got["list"], []any{float64(1), float64(2)}) {
		t.Fatalf("list: got %#v", got["list"])
	}
	if got["plain"] != "text" {
		t.Fatalf("plain: got %#v", got["plain"])
	}
	if got["broken"] != "{nope}" {
		t.Fatalf("broken: got %#v", got["broken"])
	}
	if got["number"] != float64(3) {
		t.Fatalf("number: got %#v", got["number"])
	}
}

func TestNormalizeArgumentMapKeepsMismatchedFunctionName(t *testing.T) {
	t.Parallel()
	got := normalizeArgumentMap(map[string]any{"functionName": "other"}, "add")
	if got["functionName"] != "other" {
		t.Fatalf("functionName not matching the tool name must be kept: %#v", got)
	}
}

func TestTryParseJSONValue(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		value string
		want  any
		wantOK bool
	}{
		{"empty", "", "", false},
		{"whitespace", "  ", "  ", false},
		{"scalar", "42", "42", false},
		{"plain text", "hello", "hello", false},
		{"object", `{"a":1}`, map[string]any{"a": float64(1)}, true},
		{"array with padding", " [1,2] ", []any{float64(1), float64(2)}, true},
		{"malformed object", "{bad", "{bad", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tryParseJSONValue(tc.value)
			if ok != tc.wantOK {
				t.Fatalf("ok: got %v want %v", ok, tc.wantOK)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("value: got %#v want %#v", got, tc.want)
			}
		})
	}
}
