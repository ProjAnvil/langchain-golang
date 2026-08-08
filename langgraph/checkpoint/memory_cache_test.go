package checkpoint

import (
	"context"
	"slices"
	"testing"
	"time"
)

func TestInMemoryCacheSetGetRoundtrip(t *testing.T) {
	c := NewInMemoryCache()
	ctx := context.Background()

	if _, ok, err := c.Get(ctx, "ns", "k"); err != nil || ok {
		t.Fatalf("Get() on empty cache = (ok=%v, err=%v), want a clean miss", ok, err)
	}

	writes := []Write{
		{Channel: "a", Value: 1},
		{Channel: ReservedTasks, Value: "routed"},
	}
	if err := c.Set(ctx, "ns", "k", writes, 0); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	got, ok, err := c.Get(ctx, "ns", "k")
	if err != nil || !ok {
		t.Fatalf("Get() after Set = (ok=%v, err=%v), want a hit", ok, err)
	}
	if !slices.EqualFunc(got, writes, func(a, b Write) bool {
		return a.Channel == b.Channel && a.Value == b.Value
	}) {
		t.Fatalf("Get() writes = %+v, want %+v", got, writes)
	}

	// Mutating the returned slice must not corrupt the stored entry.
	got[0].Value = 999
	again, ok, err := c.Get(ctx, "ns", "k")
	if err != nil || !ok {
		t.Fatalf("Get() after mutation = (ok=%v, err=%v), want a hit", ok, err)
	}
	if again[0].Value != 1 {
		t.Fatalf("stored write mutated through Get's return: %+v", again)
	}
}

func TestInMemoryCacheZeroValueReady(t *testing.T) {
	var c InMemoryCache
	ctx := context.Background()
	if err := c.Set(ctx, "ns", "k", []Write{{Channel: "a", Value: 1}}, 0); err != nil {
		t.Fatalf("zero-value Set() error = %v", err)
	}
	if _, ok, err := c.Get(ctx, "ns", "k"); err != nil || !ok {
		t.Fatalf("zero-value Get() = (ok=%v, err=%v), want a hit", ok, err)
	}
}

func TestInMemoryCacheTTLExpiry(t *testing.T) {
	c := NewInMemoryCache()
	ctx := context.Background()

	if err := c.Set(ctx, "ns", "short", []Write{{Channel: "a", Value: 1}}, 20*time.Millisecond); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	if _, ok, _ := c.Get(ctx, "ns", "short"); !ok {
		t.Fatalf("Get() before expiry = miss, want a hit")
	}
	time.Sleep(40 * time.Millisecond)
	if _, ok, _ := c.Get(ctx, "ns", "short"); ok {
		t.Fatalf("Get() after expiry = hit, want a miss (absolute expiry checked on read)")
	}

	// A zero TTL never expires.
	if err := c.Set(ctx, "ns", "forever", []Write{{Channel: "a", Value: 1}}, 0); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	time.Sleep(40 * time.Millisecond)
	if _, ok, _ := c.Get(ctx, "ns", "forever"); !ok {
		t.Fatalf("Get() of TTL-0 entry = miss, want a hit (0 means never expires)")
	}
}

func TestInMemoryCacheClearNamespace(t *testing.T) {
	c := NewInMemoryCache()
	ctx := context.Background()
	if err := c.Set(ctx, "ns1", "k", []Write{{Channel: "a", Value: 1}}, 0); err != nil {
		t.Fatalf("Set() error = %v", err)
	}
	if err := c.Set(ctx, "ns2", "k", []Write{{Channel: "a", Value: 2}}, 0); err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	if err := c.Clear(ctx, "ns1"); err != nil {
		t.Fatalf("Clear() error = %v", err)
	}
	if _, ok, _ := c.Get(ctx, "ns1", "k"); ok {
		t.Fatalf("Get() of cleared namespace = hit, want a miss")
	}
	if got, ok, _ := c.Get(ctx, "ns2", "k"); !ok || got[0].Value != 2 {
		t.Fatalf("Clear(ns1) disturbed ns2: (writes=%+v, ok=%v)", got, ok)
	}
}
