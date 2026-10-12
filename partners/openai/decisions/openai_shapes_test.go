package decisions

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/messages"
)

func TestMessageToInputPartsText(t *testing.T) {
	msg := messages.Message{
		Role:          messages.RoleHuman,
		ContentBlocks: []messages.ContentBlock{messages.TextBlock{Text: "look at this"}},
	}
	got, err := messageToInputParts(msg)
	if err != nil {
		t.Fatalf("messageToInputParts() error = %v", err)
	}
	want := []map[string]any{{"type": "input_text", "text": "look at this"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messageToInputParts() = %#v, want %#v", got, want)
	}
}

func TestMessageToInputPartsPlainStringContent(t *testing.T) {
	msg := messages.Human("hello")
	got, err := messageToInputParts(msg)
	if err != nil {
		t.Fatalf("messageToInputParts() error = %v", err)
	}
	want := []map[string]any{{"type": "input_text", "text": "hello"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("messageToInputParts() = %#v, want %#v", got, want)
	}
}

func TestMessageToInputPartsImages(t *testing.T) {
	tests := []struct {
		name  string
		block messages.ContentBlock
		url   string
	}{
		{
			name:  "base64 with mime type",
			block: messages.ImageBlock{Base64: "abcd", MimeType: "image/png"},
			url:   "data:image/png;base64,abcd",
		},
		{
			name:  "base64 without mime type defaults to jpeg",
			block: messages.ImageBlock{Base64: "abcd"},
			url:   "data:image/jpeg;base64,abcd",
		},
		{
			name:  "hosted url passes through",
			block: messages.ImageBlock{URL: "https://example.com/cat.png"},
			url:   "https://example.com/cat.png",
		},
		{
			name:  "data url passes through",
			block: messages.ImageBlock{URL: "data:image/webp;base64,xyz"},
			url:   "data:image/webp;base64,xyz",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := messages.Message{
				Role:          messages.RoleHuman,
				ContentBlocks: []messages.ContentBlock{tt.block},
			}
			got, err := messageToInputParts(msg)
			if err != nil {
				t.Fatalf("messageToInputParts() error = %v", err)
			}
			want := []map[string]any{{"type": "input_image", "image_url": tt.url}}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("messageToInputParts() = %#v, want %#v", got, want)
			}
		})
	}
}

func TestMessageToInputPartsUnsupportedBlocks(t *testing.T) {
	tests := []struct {
		name     string
		block    messages.ContentBlock
		contains string
	}{
		{"audio", messages.AudioBlock{Base64: "x"}, "audio"},
		{"video", messages.VideoBlock{URL: "https://example.com/v.mp4"}, "video"},
		{"file", messages.FileBlock{FileID: "file-1"}, "file"},
		{"image with only file id", messages.ImageBlock{FileID: "file-2"}, "image"},
		{"reasoning", messages.ReasoningBlock{Reasoning: "hmm"}, "reasoning"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := messages.Message{
				Role:          messages.RoleHuman,
				ContentBlocks: []messages.ContentBlock{tt.block},
			}
			_, err := messageToInputParts(msg)
			if err == nil {
				t.Fatalf("messageToInputParts() expected error for %s", tt.name)
			}
			if !errorsIsInvalidInput(err) {
				t.Fatalf("messageToInputParts() error = %v, want ErrInvalidInput", err)
			}
			if !strings.Contains(err.Error(), tt.contains) {
				t.Fatalf("messageToInputParts() error = %v, want it to mention %q", err, tt.contains)
			}
		})
	}
}

func TestIsBase64ImageBlock(t *testing.T) {
	tests := []struct {
		name  string
		block messages.ContentBlock
		want  bool
	}{
		{"data url image", messages.ImageBlock{URL: "data:image/png;base64,x"}, true},
		{"base64 image", messages.ImageBlock{Base64: "x", MimeType: "image/png"}, true},
		{"hosted url image", messages.ImageBlock{URL: "https://example.com/i.png"}, false},
		{"file id image", messages.ImageBlock{FileID: "f"}, false},
		{"text block", messages.TextBlock{Text: "hi"}, false},
		{
			"nonstandard image_url with data url",
			messages.NonStandardContentBlock{Type: "image_url", Value: map[string]any{
				"image_url": map[string]any{"url": "data:image/png;base64,x"},
			}},
			true,
		},
		{
			"nonstandard image_url with hosted url",
			messages.NonStandardContentBlock{Type: "image_url", Value: map[string]any{
				"image_url": map[string]any{"url": "https://example.com/i.png"},
			}},
			false,
		},
		{"nonstandard other type", messages.NonStandardContentBlock{Type: "audio", Value: map[string]any{}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsBase64ImageBlock(tt.block); got != tt.want {
				t.Fatalf("IsBase64ImageBlock(%v) = %v, want %v", tt.block, got, tt.want)
			}
		})
	}
}

func errorsIsInvalidInput(err error) bool {
	return err != nil && errors.Is(err, lcerrors.ErrInvalidInput)
}
