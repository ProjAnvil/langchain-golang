package anthropic

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestImageSourceVariants(t *testing.T) {
	for _, tc := range []struct {
		name string
		block map[string]any
		want map[string]any
	}{
		{
			name:  "source_type id",
			block: map[string]any{"type": "image", "source_type": "id", "id": "file_1"},
			want:  map[string]any{"type": "file", "file_id": "file_1"},
		},
		{
			name:  "file_id fallback",
			block: map[string]any{"type": "image", "file_id": "file_2"},
			want:  map[string]any{"type": "file", "file_id": "file_2"},
		},
		{
			name:  "source_type id falls back to file_id key",
			block: map[string]any{"type": "image", "source_type": "id", "file_id": "file_3"},
			want:  map[string]any{"type": "file", "file_id": "file_3"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := imageSource(tc.block)
			if err != nil {
				t.Fatalf("imageSource: %v", err)
			}
			for key, want := range tc.want {
				if source[key] != want {
					t.Fatalf("source[%q]: got %v want %v (%v)", key, source[key], want, source)
				}
			}
		})
	}
}

func TestImageSourceMissingFails(t *testing.T) {
	for _, block := range []map[string]any{
		{"type": "image"},
		// source_type=base64 without the base64 payload has nothing to send.
		{"type": "image", "source_type": "base64", "mime_type": "image/png"},
	} {
		if _, err := imageSource(block); err == nil {
			t.Fatalf("imageSource(%v) should fail", block)
		}
	}
}

func TestDocumentSourceVariants(t *testing.T) {
	for _, tc := range []struct {
		name  string
		block map[string]any
		want  map[string]any
	}{
		{
			name:  "url",
			block: map[string]any{"type": "file", "source_type": "url", "url": "https://example.com/doc.pdf"},
			want:  map[string]any{"type": "url", "url": "https://example.com/doc.pdf"},
		},
		{
			name:  "data uri url",
			block: map[string]any{"type": "file", "url": "data:application/pdf;base64,QUJD"},
			want:  map[string]any{"type": "base64", "media_type": "application/pdf", "data": "QUJD"},
		},
		{
			name:  "source_type id",
			block: map[string]any{"type": "file", "source_type": "id", "id": "file_1"},
			want:  map[string]any{"type": "file", "file_id": "file_1"},
		},
		{
			name:  "source_type base64 default mime",
			block: map[string]any{"type": "file", "source_type": "base64", "base64": "QUJD"},
			want:  map[string]any{"type": "base64", "media_type": "application/pdf", "data": "QUJD"},
		},
		{
			name:  "base64 without source_type",
			block: map[string]any{"type": "file", "base64": "Rkc=", "mime_type": "image/png"},
			want:  map[string]any{"type": "base64", "media_type": "image/png", "data": "Rkc="},
		},
		{
			name:  "file_id fallback",
			block: map[string]any{"type": "file", "file_id": "file_2"},
			want:  map[string]any{"type": "file", "file_id": "file_2"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := documentSource(tc.block)
			if err != nil {
				t.Fatalf("documentSource: %v", err)
			}
			for key, want := range tc.want {
				if source[key] != want {
					t.Fatalf("source[%q]: got %v want %v (%v)", key, source[key], want, source)
				}
			}
		})
	}
}

func TestDocumentSourceMissingFails(t *testing.T) {
	if _, err := documentSource(map[string]any{"type": "file"}); err == nil {
		t.Fatal("documentSource without any payload should fail")
	}
}

func TestFormatContentBlockPlainText(t *testing.T) {
	block, err := formatContentBlock(messages.PlainTextBlock{Text: "hello doc"})
	if err != nil {
		t.Fatalf("formatContentBlock: %v", err)
	}
	if block.Type != "document" {
		t.Fatalf("type: %v", block.Type)
	}
	if block.Source["type"] != "text" || block.Source["media_type"] != "text/plain" || block.Source["data"] != "hello doc" {
		t.Fatalf("source: %v", block.Source)
	}
}

func TestFormatContentBlockErrorPropagation(t *testing.T) {
	_, err := formatContentBlocks([]messages.ContentBlock{
		messages.ParseContentBlock(map[string]any{"type": "text", "text": "ok"}),
		messages.ParseContentBlock(map[string]any{"type": "image"}),
	})
	if err == nil || !strings.Contains(err.Error(), "image content block") {
		t.Fatalf("formatContentBlocks should surface the image error: %v", err)
	}
}

func TestParseDataURI(t *testing.T) {
	for _, tc := range []struct {
		uri       string
		mediaType string
		data      string
		ok        bool
	}{
		{uri: "data:image/png;base64,Zm9v", mediaType: "image/png", data: "Zm9v", ok: true},
		{uri: "data:;base64,Zm9v", mediaType: "text/plain", data: "Zm9v", ok: true},
		{uri: "data:text/plain,hello", ok: false},
		{uri: "data:nocomma", ok: false},
		{uri: "https://example.com/cat.png", ok: false},
	} {
		t.Run(tc.uri, func(t *testing.T) {
			mediaType, data, ok := parseDataURI(tc.uri)
			if ok != tc.ok {
				t.Fatalf("ok: got %v want %v", ok, tc.ok)
			}
			if !ok {
				return
			}
			if mediaType != tc.mediaType || data != tc.data {
				t.Fatalf("got (%q, %q) want (%q, %q)", mediaType, data, tc.mediaType, tc.data)
			}
		})
	}
}

func TestStringValue(t *testing.T) {
	if got := stringValue("value", "fallback"); got != "value" {
		t.Fatalf("stringValue: %q", got)
	}
	if got := stringValue("", "fallback"); got != "fallback" {
		t.Fatalf("stringValue empty: %q", got)
	}
	if got := stringValue(nil, "fallback"); got != "fallback" {
		t.Fatalf("stringValue nil: %q", got)
	}
	if got := stringValue(nil, 42); got != "" {
		t.Fatalf("stringValue non-string fallback: %q", got)
	}
}

func TestPassthroughBlock(t *testing.T) {
	block := passthroughBlock(map[string]any{
		"type":          "thinking",
		"text":          "some text",
		"id":            "blk_1",
		"name":          "web_search",
		"input":         map[string]any{"q": "news"},
		"tool_use_id":   "toolu_1",
		"thinking":      "reasoning",
		"signature":     "sig_1",
		"data":          "ZW5j",
		"cache_control": map[string]any{"type": "ephemeral"},
		"unknown_field": "ignored",
	})
	if block.Type != "thinking" || block.Text != "some text" || block.ID != "blk_1" ||
		block.Name != "web_search" || block.ToolUseID != "toolu_1" ||
		block.Thinking != "reasoning" || block.Signature != "sig_1" || block.Data != "ZW5j" {
		t.Fatalf("passthrough block: %+v", block)
	}
	if block.Input["q"] != "news" {
		t.Fatalf("passthrough input: %v", block.Input)
	}
	if block.CacheControl["type"] != "ephemeral" {
		t.Fatalf("passthrough cache_control: %v", block.CacheControl)
	}
}

func TestPassthroughBlockIgnoresMismatchedTypes(t *testing.T) {
	block := passthroughBlock(map[string]any{
		"type":          5,
		"text":          true,
		"id":            1.5,
		"name":          nil,
		"input":         "not-a-map",
		"tool_use_id":   3,
		"thinking":      []string{"x"},
		"signature":     9,
		"data":          struct{}{},
		"cache_control": "not-a-map",
	})
	if block.Type != "" || block.Text != "" || block.ID != "" || block.Name != "" ||
		block.ToolUseID != "" || block.Thinking != "" || block.Signature != "" || block.Data != "" ||
		block.Input != nil || block.CacheControl != nil {
		t.Fatalf("mismatched field types should be dropped: %+v", block)
	}
}

func TestPassthroughBlockViaRequest(t *testing.T) {
	// Provider-native blocks on an assistant message round-trip untouched.
	blocks, err := formatContentBlocks([]messages.ContentBlock{
		messages.NonStandardContentBlock{
			Type: "server_tool_use",
			Value: map[string]any{
				"type":  "server_tool_use",
				"id":    "srv_1",
				"name":  "web_search",
				"input": map[string]any{"q": "news"},
			},
		},
	})
	if err != nil {
		t.Fatalf("formatContentBlocks: %v", err)
	}
	if len(blocks) != 1 || blocks[0].Type != "server_tool_use" || blocks[0].ID != "srv_1" {
		t.Fatalf("passthrough blocks: %+v", blocks)
	}
}

func TestFormatContentBlockFileErrorPropagation(t *testing.T) {
	_, err := formatContentBlock(messages.ParseContentBlock(map[string]any{"type": "file"}))
	if err == nil || !strings.Contains(err.Error(), "file content block") {
		t.Fatalf("formatContentBlock should surface the file error: %v", err)
	}
}
