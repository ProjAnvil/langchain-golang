package caches

import (
	"context"
	"reflect"
	"testing"
)

func cacheItem(itemID string) (string, string, []Generation) {
	return "prompt" + itemID, "llm_string" + itemID, []Generation{{Text: "text" + itemID}}
}

func TestInMemoryCacheInitialization(t *testing.T) {
	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}
	if cache.MaxSize() != 0 {
		t.Fatalf("expected unlimited max size, got %d", cache.MaxSize())
	}
	if cache.Len() != 0 {
		t.Fatalf("expected empty cache, got %d entries", cache.Len())
	}

	cacheWithMaxSize, err := NewInMemoryCache(WithMaxSize(2))
	if err != nil {
		t.Fatalf("new cache with max size: %v", err)
	}
	if cacheWithMaxSize.MaxSize() != 2 {
		t.Fatalf("max size mismatch: got %d", cacheWithMaxSize.MaxSize())
	}

	if _, err := NewInMemoryCache(WithMaxSize(0)); err == nil {
		t.Fatal("expected max size validation error")
	}
}

func TestInMemoryCacheLookup(t *testing.T) {
	ctx := context.Background()
	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}

	prompt, llmString, generations := cacheItem("1")
	if err := cache.Update(ctx, prompt, llmString, generations); err != nil {
		t.Fatalf("update: %v", err)
	}

	got, ok, err := cache.Lookup(ctx, prompt, llmString)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if !ok {
		t.Fatal("expected cache hit")
	}
	if !reflect.DeepEqual(got, generations) {
		t.Fatalf("generations mismatch:\n got %#v\nwant %#v", got, generations)
	}

	_, ok, err = cache.Lookup(ctx, "prompt2", "llm_string2")
	if err != nil {
		t.Fatalf("lookup missing: %v", err)
	}
	if ok {
		t.Fatal("expected cache miss")
	}
}

func TestInMemoryCacheUpdateWithNoMaxSize(t *testing.T) {
	ctx := context.Background()
	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}

	prompt, llmString, generations := cacheItem("1")
	if err := cache.Update(ctx, prompt, llmString, generations); err != nil {
		t.Fatalf("update: %v", err)
	}

	got, ok, err := cache.Lookup(ctx, prompt, llmString)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if !ok || !reflect.DeepEqual(got, generations) {
		t.Fatalf("lookup mismatch: ok=%v got=%#v want=%#v", ok, got, generations)
	}
}

func TestInMemoryCacheUpdateWithMaxSizeEvictsOldest(t *testing.T) {
	ctx := context.Background()
	cache, err := NewInMemoryCache(WithMaxSize(2))
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}

	prompt1, llmString1, generations1 := cacheItem("1")
	prompt2, llmString2, generations2 := cacheItem("2")
	prompt3, llmString3, generations3 := cacheItem("3")

	if err := cache.Update(ctx, prompt1, llmString1, generations1); err != nil {
		t.Fatalf("update 1: %v", err)
	}
	if err := cache.Update(ctx, prompt2, llmString2, generations2); err != nil {
		t.Fatalf("update 2: %v", err)
	}
	if err := cache.Update(ctx, prompt3, llmString3, generations3); err != nil {
		t.Fatalf("update 3: %v", err)
	}

	if _, ok, err := cache.Lookup(ctx, prompt1, llmString1); err != nil {
		t.Fatalf("lookup evicted: %v", err)
	} else if ok {
		t.Fatal("expected oldest cache entry to be evicted")
	}
	assertCacheHit(t, cache, prompt2, llmString2, generations2)
	assertCacheHit(t, cache, prompt3, llmString3, generations3)
}

func TestInMemoryCacheClear(t *testing.T) {
	ctx := context.Background()
	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}

	prompt, llmString, generations := cacheItem("1")
	if err := cache.Update(ctx, prompt, llmString, generations); err != nil {
		t.Fatalf("update: %v", err)
	}
	if err := cache.Clear(ctx); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, ok, err := cache.Lookup(ctx, prompt, llmString); err != nil {
		t.Fatalf("lookup after clear: %v", err)
	} else if ok {
		t.Fatal("expected cache miss after clear")
	}
}

func TestInMemoryCacheLookupReturnsCopy(t *testing.T) {
	ctx := context.Background()
	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("new cache: %v", err)
	}

	prompt, llmString, generations := cacheItem("1")
	if err := cache.Update(ctx, prompt, llmString, generations); err != nil {
		t.Fatalf("update: %v", err)
	}
	got, ok, err := cache.Lookup(ctx, prompt, llmString)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if !ok {
		t.Fatal("expected cache hit")
	}
	got[0].Text = "mutated"

	again, ok, err := cache.Lookup(ctx, prompt, llmString)
	if err != nil {
		t.Fatalf("lookup again: %v", err)
	}
	if !ok {
		t.Fatal("expected cache hit")
	}
	if again[0].Text != "text1" {
		t.Fatalf("cache exposed internal generation slice: got %q", again[0].Text)
	}
}

func assertCacheHit(t *testing.T, cache *InMemoryCache, prompt string, llmString string, want []Generation) {
	t.Helper()
	got, ok, err := cache.Lookup(context.Background(), prompt, llmString)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if !ok {
		t.Fatalf("expected cache hit for %q/%q", prompt, llmString)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("generations mismatch:\n got %#v\nwant %#v", got, want)
	}
}
