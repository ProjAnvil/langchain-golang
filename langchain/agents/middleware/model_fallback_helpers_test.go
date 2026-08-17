package middleware

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
)

func TestModelFallbackMiddlewareUnresolvableStringSpec(t *testing.T) {
	// The spec parses as provider:model but no such provider is registered,
	// so resolution fails and the error is returned immediately.
	fallback := NewModelFallbackMiddleware("middleware-no-such-provider:model-x")
	request, err := NewModelRequest(ModelRequest{Model: "primary"})
	if err != nil {
		t.Fatalf("new model request: %v", err)
	}
	_, err = fallback.WrapModelCall(context.Background(), request, func(context.Context, ModelRequest) (ModelResponse, error) {
		return ModelResponse{}, errors.New("primary failed")
	})
	if err == nil || strings.Contains(err.Error(), "primary failed") {
		t.Fatalf("expected resolution error (not the primary error), got %v", err)
	}
}

func TestStripCacheControlFromSettings(t *testing.T) {
	// Without the marker the map is returned as-is.
	settings := map[string]any{"temperature": 0.5}
	if got := stripCacheControlFromSettings(settings); got["temperature"] != 0.5 {
		t.Fatalf("settings mismatch: %#v", got)
	}

	// With the marker only the marker is removed; other keys are preserved.
	withMarker := map[string]any{"cache_control": cacheControlMarker(), "temperature": 0.5}
	got := stripCacheControlFromSettings(withMarker)
	if _, ok := got["cache_control"]; ok {
		t.Fatalf("cache_control should be stripped: %#v", got)
	}
	if got["temperature"] != 0.5 {
		t.Fatalf("other settings must be preserved: %#v", got)
	}
	if _, ok := withMarker["cache_control"]; !ok {
		t.Fatal("input map must not be mutated")
	}
}

func TestSanitizeContentBlocks(t *testing.T) {
	if got := sanitizeContentBlocks(nil); got != nil {
		t.Fatalf("nil blocks mismatch: %#v", got)
	}

	blocks := []messages.ContentBlock{
		contentBlockWithCacheControl("first"),
		messages.TextBlock{Text: "plain"},
	}
	got := sanitizeContentBlocks(blocks)
	if len(got) != 2 {
		t.Fatalf("sanitized blocks mismatch: %#v", got)
	}
	for _, block := range got {
		if blockHasCacheControl(block) {
			t.Fatalf("cache_control should be stripped: %#v", block)
		}
	}
	if messages.BlockToMap(got[0])["text"] != "first" {
		t.Fatalf("block content mismatch: %#v", got[0])
	}
}
