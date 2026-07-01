package messages

import "sync"

// ContentBlockTranslator translates provider-specific AIMessage content
// (stored as raw ContentBlock slices or strings) into normalized
// ContentBlocks. The input is the full message so the translator can inspect
// ResponseMetadata and AdditionalKwargs in addition to Content and ContentBlocks.
type ContentBlockTranslator func(message Message) []ContentBlock

// translatorRegistry is the internal map of provider name → translator.
type translatorRegistry struct {
	mu          sync.RWMutex
	translators map[string]ContentBlockTranslator
}

var registry = &translatorRegistry{
	translators: make(map[string]ContentBlockTranslator),
}

// RegisterTranslator registers content block translator functions for a
// provider. It is the Go equivalent of Python's register_translator. Partner
// packages call this from their init() or package-level var blocks so that
// AIMessage.ContentBlocks returns normalized blocks for their provider.
//
// Calling RegisterTranslator a second time for the same provider replaces
// the existing translator — the last registration wins.
func RegisterTranslator(provider string, translator ContentBlockTranslator) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	registry.translators[provider] = translator
}

// GetTranslator returns the translator registered for provider, and a bool
// indicating whether one was found.  When no translator is registered for a
// provider, the caller should fall back to best-effort parsing (see
// ContentBlocks).
func GetTranslator(provider string) (ContentBlockTranslator, bool) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	t, ok := registry.translators[provider]
	return t, ok
}

// RegisteredProviders returns the sorted list of provider names for which a
// translator has been registered. Useful for diagnostics and tests.
func RegisteredProviders() []string {
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	out := make([]string, 0, len(registry.translators))
	for provider := range registry.translators {
		out = append(out, provider)
	}
	// Deterministic order for tests.
	sortStrings(out)
	return out
}

// ContentBlocks returns a normalized []ContentBlock for the message using
// the following resolution order:
//
//  1. If ResponseMetadata["model_provider"] identifies a registered translator,
//     that translator is used.
//  2. Otherwise, best-effort parsing is applied: if ContentBlocks is already
//     populated it is returned as-is; if not, a single text block is derived
//     from the Content field.
//
// This mirrors Python's AIMessage.content_blocks property including the
// translator lookup via model_provider.
func ContentBlocks(message Message) []ContentBlock {
	// Resolve provider from response_metadata.
	if provider, ok := providerFromMetadata(message); ok {
		if translator, ok := GetTranslator(provider); ok {
			return translator(message)
		}
	}
	// Best-effort fallback.
	if len(message.ContentBlocks) > 0 {
		return cloneBlocks(message.ContentBlocks)
	}
	if message.Content != "" {
		return []ContentBlock{{"type": "text", "text": message.Content}}
	}
	return nil
}

// providerFromMetadata extracts the "model_provider" key from ResponseMetadata.
func providerFromMetadata(message Message) (string, bool) {
	if message.ResponseMetadata == nil {
		return "", false
	}
	value, ok := message.ResponseMetadata["model_provider"]
	if !ok {
		return "", false
	}
	provider, ok := value.(string)
	return provider, ok && provider != ""
}

// sortStrings sorts a string slice in-place.
func sortStrings(ss []string) {
	// Avoid importing sort just for this helper — use a simple insertion sort
	// since this is only called for small provider lists.
	for i := 1; i < len(ss); i++ {
		key := ss[i]
		j := i - 1
		for j >= 0 && ss[j] > key {
			ss[j+1] = ss[j]
			j--
		}
		ss[j+1] = key
	}
}
