package stores

import (
	"context"
	"reflect"
	"sort"
	"testing"
)

func TestInMemoryStoreMGet(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[string]()

	if err := store.MSet(ctx, []KeyValue[string]{
		{Key: "key1", Value: "value1"},
		{Key: "key2", Value: "value2"},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}

	values, err := store.MGet(ctx, []string{"key1", "key2"})
	if err != nil {
		t.Fatalf("mget: %v", err)
	}
	assertFoundValues(t, values, []string{"value1", "value2"})

	missing, err := store.MGet(ctx, []string{"key3"})
	if err != nil {
		t.Fatalf("mget missing: %v", err)
	}
	if len(missing) != 1 || missing[0].Found {
		t.Fatalf("expected one missing value, got %#v", missing)
	}
}

func TestInMemoryStoreMSet(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[string]()

	if err := store.MSet(ctx, []KeyValue[string]{
		{Key: "key1", Value: "value1"},
		{Key: "key2", Value: "value2"},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}

	values, err := store.MGet(ctx, []string{"key1", "key2"})
	if err != nil {
		t.Fatalf("mget: %v", err)
	}
	assertFoundValues(t, values, []string{"value1", "value2"})
}

func TestInMemoryStoreMDelete(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[string]()

	if err := store.MSet(ctx, []KeyValue[string]{
		{Key: "key1", Value: "value1"},
		{Key: "key2", Value: "value2"},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}
	if err := store.MDelete(ctx, []string{"key1"}); err != nil {
		t.Fatalf("mdelete: %v", err)
	}

	values, err := store.MGet(ctx, []string{"key1", "key2"})
	if err != nil {
		t.Fatalf("mget: %v", err)
	}
	if values[0].Found {
		t.Fatalf("expected key1 to be deleted, got %#v", values[0])
	}
	if !values[1].Found || values[1].Value != "value2" {
		t.Fatalf("expected key2 to remain, got %#v", values[1])
	}

	if err := store.MDelete(ctx, []string{"key3"}); err != nil {
		t.Fatalf("deleting non-existent key should not fail: %v", err)
	}
}

func TestInMemoryStoreYieldKeys(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[string]()

	if err := store.MSet(ctx, []KeyValue[string]{
		{Key: "key1", Value: "value1"},
		{Key: "key2", Value: "value2"},
		{Key: "key3", Value: "value3"},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}

	keys, err := store.YieldKeys(ctx, "")
	if err != nil {
		t.Fatalf("yield keys: %v", err)
	}
	sort.Strings(keys)
	if !reflect.DeepEqual(keys, []string{"key1", "key2", "key3"}) {
		t.Fatalf("keys mismatch: got %#v", keys)
	}

	keysWithPrefix, err := store.YieldKeys(ctx, "key")
	if err != nil {
		t.Fatalf("yield keys with prefix: %v", err)
	}
	sort.Strings(keysWithPrefix)
	if !reflect.DeepEqual(keysWithPrefix, []string{"key1", "key2", "key3"}) {
		t.Fatalf("keys with prefix mismatch: got %#v", keysWithPrefix)
	}

	keysWithInvalidPrefix, err := store.YieldKeys(ctx, "x")
	if err != nil {
		t.Fatalf("yield keys with invalid prefix: %v", err)
	}
	if len(keysWithInvalidPrefix) != 0 {
		t.Fatalf("expected no keys, got %#v", keysWithInvalidPrefix)
	}
}

func TestInMemoryByteStore(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryByteStore()

	if err := store.MSet(ctx, []KeyValue[[]byte]{
		{Key: "key1", Value: []byte("value1")},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}
	values, err := store.MGet(ctx, []string{"key1"})
	if err != nil {
		t.Fatalf("mget: %v", err)
	}
	if len(values) != 1 || !values[0].Found || string(values[0].Value) != "value1" {
		t.Fatalf("byte value mismatch: %#v", values)
	}

	values[0].Value[0] = 'V'
	again, err := store.MGet(ctx, []string{"key1"})
	if err != nil {
		t.Fatalf("mget again: %v", err)
	}
	if string(again[0].Value) != "value1" {
		t.Fatalf("byte store exposed internal slice: got %q", string(again[0].Value))
	}
}

func assertFoundValues(t *testing.T, values []MaybeValue[string], want []string) {
	t.Helper()
	if len(values) != len(want) {
		t.Fatalf("value count mismatch: got %d want %d", len(values), len(want))
	}
	for i := range want {
		if !values[i].Found {
			t.Fatalf("value %d missing: %#v", i, values[i])
		}
		if values[i].Value != want[i] {
			t.Fatalf("value %d mismatch: got %q want %q", i, values[i].Value, want[i])
		}
	}
}
