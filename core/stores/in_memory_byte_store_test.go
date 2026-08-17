package stores

import (
	"context"
	"reflect"
	"sort"
	"testing"
)

func TestInMemoryByteStoreMDelete(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryByteStore()

	if err := store.MSet(ctx, []KeyValue[[]byte]{
		{Key: "key1", Value: []byte("value1")},
		{Key: "key2", Value: []byte("value2")},
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
	if !values[1].Found || string(values[1].Value) != "value2" {
		t.Fatalf("expected key2 to remain, got %#v", values[1])
	}

	if err := store.MDelete(ctx, []string{"key3"}); err != nil {
		t.Fatalf("deleting non-existent key should not fail: %v", err)
	}
}

func TestInMemoryByteStoreYieldKeys(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryByteStore()

	if err := store.MSet(ctx, []KeyValue[[]byte]{
		{Key: "key1", Value: []byte("value1")},
		{Key: "key2", Value: []byte("value2")},
		{Key: "other", Value: []byte("value3")},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}

	keys, err := store.YieldKeys(ctx, "")
	if err != nil {
		t.Fatalf("yield keys: %v", err)
	}
	sort.Strings(keys)
	if !reflect.DeepEqual(keys, []string{"key1", "key2", "other"}) {
		t.Fatalf("keys mismatch: got %#v", keys)
	}

	keysWithPrefix, err := store.YieldKeys(ctx, "key")
	if err != nil {
		t.Fatalf("yield keys with prefix: %v", err)
	}
	sort.Strings(keysWithPrefix)
	if !reflect.DeepEqual(keysWithPrefix, []string{"key1", "key2"}) {
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

func TestInMemoryByteStoreEmptyValues(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryByteStore()

	if err := store.MSet(ctx, []KeyValue[[]byte]{
		{Key: "empty", Value: []byte{}},
		{Key: "nil", Value: nil},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}

	values, err := store.MGet(ctx, []string{"empty", "nil"})
	if err != nil {
		t.Fatalf("mget: %v", err)
	}
	for i, v := range values {
		if !v.Found {
			t.Fatalf("value %d missing: %#v", i, v)
		}
		if len(v.Value) != 0 {
			t.Fatalf("expected empty value %d, got %#v", i, v.Value)
		}
	}

	missing, err := store.MGet(ctx, []string{"missing"})
	if err != nil {
		t.Fatalf("mget missing: %v", err)
	}
	if len(missing) != 1 || missing[0].Found {
		t.Fatalf("expected one missing value, got %#v", missing)
	}
	if missing[0].Value != nil {
		t.Fatalf("expected nil value for missing key, got %#v", missing[0].Value)
	}
}

func TestInMemoryByteStoreSetCopiesInput(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryByteStore()

	input := []byte("value1")
	if err := store.MSet(ctx, []KeyValue[[]byte]{
		{Key: "key1", Value: input},
	}); err != nil {
		t.Fatalf("mset: %v", err)
	}
	input[0] = 'V'

	values, err := store.MGet(ctx, []string{"key1"})
	if err != nil {
		t.Fatalf("mget: %v", err)
	}
	if string(values[0].Value) != "value1" {
		t.Fatalf("byte store retained caller slice: got %q", string(values[0].Value))
	}
}
