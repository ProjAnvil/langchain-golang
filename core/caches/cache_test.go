package caches

import (
	"context"
	"sync"
	"testing"
)

// TestLLMCacheGlobal pins the get_llm_cache/set_llm_cache port: unset reads
// as nil, SetLLMCache installs a cache readable through GetLLMCache, and
// SetLLMCache(nil) clears it.
func TestLLMCacheGlobal(t *testing.T) {
	t.Cleanup(func() { SetLLMCache(nil) })
	SetLLMCache(nil)
	if got := GetLLMCache(); got != nil {
		t.Fatalf("GetLLMCache() = %#v, want nil by default", got)
	}

	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("NewInMemoryCache: %v", err)
	}
	SetLLMCache(cache)
	if got := GetLLMCache(); got != cache {
		t.Fatalf("GetLLMCache() = %#v, want the installed cache", got)
	}

	// The global is usable end to end.
	if err := GetLLMCache().Update(context.Background(), "p", "llm", []Generation{{Text: "answer"}}); err != nil {
		t.Fatalf("Update via global: %v", err)
	}
	generations, found, err := GetLLMCache().Lookup(context.Background(), "p", "llm")
	if err != nil || !found || len(generations) != 1 || generations[0].Text != "answer" {
		t.Fatalf("Lookup via global = (%#v, %v, %v)", generations, found, err)
	}

	SetLLMCache(nil)
	if got := GetLLMCache(); got != nil {
		t.Fatalf("GetLLMCache() after clear = %#v, want nil", got)
	}
}

// TestLLMCacheGlobalConcurrent exercises the global under concurrent readers
// and writers (meaningful under -race: Python's module global needs no lock;
// the Go port must be safe).
func TestLLMCacheGlobalConcurrent(t *testing.T) {
	t.Cleanup(func() { SetLLMCache(nil) })
	cache, err := NewInMemoryCache()
	if err != nil {
		t.Fatalf("NewInMemoryCache: %v", err)
	}
	var group sync.WaitGroup
	for i := range 32 {
		group.Add(1)
		go func(i int) {
			defer group.Done()
			if i%2 == 0 {
				SetLLMCache(cache)
			} else {
				SetLLMCache(nil)
			}
			_ = GetLLMCache()
		}(i)
	}
	group.Wait()
}
