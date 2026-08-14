package standardtests

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/caches"
)

// CacheFactory creates a fresh, empty cache for standard tests.
type CacheFactory func(t testing.TB) caches.Cache

// RunCacheBasics verifies behavior expected from every cache. It mirrors
// Python's SyncCacheTestSuite: empty cache miss, update-then-hit, clear, and
// multiple generations round-trip.
func RunCacheBasics(t *testing.T, factory CacheFactory) {
	t.Helper()

	const prompt = "Sample prompt for testing."
	const llmString = "Sample LLM string configuration."

	t.Run("empty cache miss", func(t *testing.T) {
		cache := factory(t)
		_, found, err := cache.Lookup(context.Background(), prompt, llmString)
		if err != nil {
			t.Fatalf("lookup: %v", err)
		}
		if found {
			t.Fatal("expected cache miss on empty cache")
		}
	})

	t.Run("update then hit", func(t *testing.T) {
		cache := factory(t)
		want := []caches.Generation{
			{Text: "Sample generated text.", GenerationInfo: map[string]any{"reason": "test"}},
		}
		if err := cache.Update(context.Background(), prompt, llmString, want); err != nil {
			t.Fatalf("update: %v", err)
		}
		got, found, err := cache.Lookup(context.Background(), prompt, llmString)
		if err != nil {
			t.Fatalf("lookup: %v", err)
		}
		if !found {
			t.Fatal("expected cache hit after update")
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("generations: got %#v want %#v", got, want)
		}
	})

	t.Run("multiple generations round trip", func(t *testing.T) {
		cache := factory(t)
		want := []caches.Generation{
			{Text: "First generated text.", GenerationInfo: map[string]any{"reason": "test"}},
			{Text: "Second generated text."},
		}
		if err := cache.Update(context.Background(), prompt, llmString, want); err != nil {
			t.Fatalf("update: %v", err)
		}
		got, found, err := cache.Lookup(context.Background(), prompt, llmString)
		if err != nil {
			t.Fatalf("lookup: %v", err)
		}
		if !found {
			t.Fatal("expected cache hit after update")
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("generations: got %#v want %#v", got, want)
		}
	})

	t.Run("clear empties", func(t *testing.T) {
		cache := factory(t)
		if err := cache.Update(context.Background(), prompt, llmString, []caches.Generation{
			{Text: "Sample generated text."},
		}); err != nil {
			t.Fatalf("update: %v", err)
		}
		if err := cache.Clear(context.Background()); err != nil {
			t.Fatalf("clear: %v", err)
		}
		_, found, err := cache.Lookup(context.Background(), prompt, llmString)
		if err != nil {
			t.Fatalf("lookup: %v", err)
		}
		if found {
			t.Fatal("expected cache miss after clear")
		}
	})
}
