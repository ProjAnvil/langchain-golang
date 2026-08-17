package messages

import (
	"testing"
)

func TestRegisterAndGetTranslator(t *testing.T) {
	provider := "test-provider-" + t.Name()
	called := false
	translator := func(msg Message) []ContentBlock {
		called = true
		return []ContentBlock{TextBlock{Text: "translated"}}
	}
	RegisterTranslator(provider, translator)

	got, ok := GetTranslator(provider)
	if !ok {
		t.Fatalf("GetTranslator returned ok=false for %q", provider)
	}
	result := got(Message{})
	if !called {
		t.Fatal("translator not called")
	}
	m := BlockToMap(result[0])
	if m["text"] != "translated" {
		t.Fatalf("unexpected result: %v", result)
	}
}

func TestGetTranslatorMissing(t *testing.T) {
	_, ok := GetTranslator("no-such-provider")
	if ok {
		t.Fatal("expected ok=false for unregistered provider")
	}
}

func TestContentBlocksUsesTranslator(t *testing.T) {
	provider := "test-prov-" + t.Name()
	RegisterTranslator(provider, func(msg Message) []ContentBlock {
		return []ContentBlock{TextBlock{Text: "from-translator"}}
	})

	msg := Message{
		Role:             RoleAI,
		Content:          "raw content",
		ResponseMetadata: map[string]any{"model_provider": provider},
	}
	blocks := ContentBlocks(msg)
	if len(blocks) != 1 || BlockToMap(blocks[0])["text"] != "from-translator" {
		t.Fatalf("unexpected blocks: %v", blocks)
	}
}

func TestContentBlocksFallbackToExistingBlocks(t *testing.T) {
	msg := Message{
		Role: RoleAI,
		ContentBlocks: []ContentBlock{
			TextBlock{Text: "block1"},
		},
	}
	blocks := ContentBlocks(msg)
	if len(blocks) != 1 || BlockToMap(blocks[0])["text"] != "block1" {
		t.Fatalf("unexpected blocks: %v", blocks)
	}
}

func TestContentBlocksFallbackToContent(t *testing.T) {
	msg := Message{Role: RoleAI, Content: "hello"}
	blocks := ContentBlocks(msg)
	if len(blocks) != 1 || BlockToMap(blocks[0])["type"] != "text" || BlockToMap(blocks[0])["text"] != "hello" {
		t.Fatalf("unexpected blocks: %v", blocks)
	}
}

func TestContentBlocksEmpty(t *testing.T) {
	msg := Message{Role: RoleAI}
	blocks := ContentBlocks(msg)
	if blocks != nil {
		t.Fatalf("expected nil, got %v", blocks)
	}
}

func TestContentBlocksUnknownProviderFallsBack(t *testing.T) {
	msg := Message{
		Role:             RoleAI,
		Content:          "fallback",
		ResponseMetadata: map[string]any{"model_provider": "no-translator-for-this"},
	}
	blocks := ContentBlocks(msg)
	if len(blocks) != 1 || BlockToMap(blocks[0])["text"] != "fallback" {
		t.Fatalf("unexpected blocks: %v", blocks)
	}
}

func TestRegisteredProviders(t *testing.T) {
	p1 := "test-prov-list-b-" + t.Name()
	p2 := "test-prov-list-a-" + t.Name()
	RegisterTranslator(p1, func(Message) []ContentBlock { return nil })
	RegisterTranslator(p2, func(Message) []ContentBlock { return nil })

	providers := RegisteredProviders()
	// Check both are present (there may be others from other tests).
	foundP1, foundP2 := false, false
	for _, p := range providers {
		if p == p1 {
			foundP1 = true
		}
		if p == p2 {
			foundP2 = true
		}
	}
	if !foundP1 || !foundP2 {
		t.Fatalf("expected both providers in list, got %v", providers)
	}
	// Verify sorted order within our subset.
	if !isSortedSubset(providers, p2, p1) {
		t.Fatalf("providers not sorted: %v", providers)
	}
}

func TestRegisterTranslatorOverwrite(t *testing.T) {
	provider := "test-overwrite-" + t.Name()
	RegisterTranslator(provider, func(Message) []ContentBlock {
		return []ContentBlock{TextBlock{Text: "first"}}
	})
	RegisterTranslator(provider, func(Message) []ContentBlock {
		return []ContentBlock{TextBlock{Text: "second"}}
	})

	tr, _ := GetTranslator(provider)
	result := tr(Message{})
	if len(result) != 1 || BlockToMap(result[0])["text"] != "second" {
		t.Fatalf("expected second translator to win, got %v", result)
	}
}

// isSortedSubset checks that p1 appears before p2 in sorted slice.
func isSortedSubset(sorted []string, first, second string) bool {
	i1, i2 := -1, -1
	for i, s := range sorted {
		if s == first {
			i1 = i
		}
		if s == second {
			i2 = i
		}
	}
	return i1 != -1 && i2 != -1 && i1 < i2
}

func TestContentBlocksInvalidProviderMetadataFallsBack(t *testing.T) {
	// Non-string model_provider values are ignored and best-effort parsing applies.
	nonString := Message{
		Role:             RoleAI,
		Content:          "fallback",
		ResponseMetadata: map[string]any{"model_provider": 123},
	}
	blocks := ContentBlocks(nonString)
	if len(blocks) != 1 || BlockToMap(blocks[0])["text"] != "fallback" {
		t.Fatalf("unexpected blocks for non-string provider: %v", blocks)
	}

	// Empty provider strings are ignored too.
	empty := Message{
		Role:             RoleAI,
		Content:          "fallback",
		ResponseMetadata: map[string]any{"model_provider": ""},
	}
	blocks = ContentBlocks(empty)
	if len(blocks) != 1 || BlockToMap(blocks[0])["text"] != "fallback" {
		t.Fatalf("unexpected blocks for empty provider: %v", blocks)
	}
}
