package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/stores"
)

// StoreFactory creates a fresh, empty key-value store for standard tests.
type StoreFactory func(t testing.TB) stores.BaseStore[string]

// RunStoreBasics verifies behavior expected from every key-value store. It
// mirrors Python's BaseStoreSyncTests: empty store miss, set-then-get, missing
// key, delete, and idempotent set.
func RunStoreBasics(t *testing.T, factory StoreFactory) {
	t.Helper()

	t.Run("empty store miss", func(t *testing.T) {
		store := factory(t)
		values, err := store.MGet(context.Background(), []string{"foo", "bar", "buzz"})
		if err != nil {
			t.Fatalf("mget: %v", err)
		}
		if len(values) != 3 {
			t.Fatalf("values: got %d want 3", len(values))
		}
		for i, value := range values {
			if value.Found {
				t.Fatalf("value[%d]: expected miss on empty store", i)
			}
		}
	})

	t.Run("set then get", func(t *testing.T) {
		store := factory(t)
		if err := store.MSet(context.Background(), []stores.KeyValue[string]{
			{Key: "foo", Value: "alpha"},
			{Key: "bar", Value: "beta"},
		}); err != nil {
			t.Fatalf("mset: %v", err)
		}
		values, err := store.MGet(context.Background(), []string{"foo", "bar"})
		if err != nil {
			t.Fatalf("mget: %v", err)
		}
		if len(values) != 2 {
			t.Fatalf("values: got %d want 2", len(values))
		}
		if !values[0].Found || values[0].Value != "alpha" {
			t.Fatalf("values[0]: got %#v", values[0])
		}
		if !values[1].Found || values[1].Value != "beta" {
			t.Fatalf("values[1]: got %#v", values[1])
		}
	})

	t.Run("get missing key", func(t *testing.T) {
		store := factory(t)
		if err := store.MSet(context.Background(), []stores.KeyValue[string]{
			{Key: "foo", Value: "alpha"},
		}); err != nil {
			t.Fatalf("mset: %v", err)
		}
		values, err := store.MGet(context.Background(), []string{"foo", "missing"})
		if err != nil {
			t.Fatalf("mget: %v", err)
		}
		if len(values) != 2 {
			t.Fatalf("values: got %d want 2", len(values))
		}
		if !values[0].Found || values[0].Value != "alpha" {
			t.Fatalf("values[0]: got %#v", values[0])
		}
		if values[1].Found {
			t.Fatal("values[1]: expected miss for missing key")
		}
		if values[1].Value != "" {
			t.Fatalf("values[1].value: got %q want zero value", values[1].Value)
		}
	})

	t.Run("delete removes", func(t *testing.T) {
		store := factory(t)
		if err := store.MSet(context.Background(), []stores.KeyValue[string]{
			{Key: "foo", Value: "alpha"},
			{Key: "bar", Value: "beta"},
		}); err != nil {
			t.Fatalf("mset: %v", err)
		}
		if err := store.MDelete(context.Background(), []string{"foo"}); err != nil {
			t.Fatalf("mdelete: %v", err)
		}
		values, err := store.MGet(context.Background(), []string{"foo", "bar"})
		if err != nil {
			t.Fatalf("mget: %v", err)
		}
		if values[0].Found {
			t.Fatal("values[0]: expected miss after delete")
		}
		if !values[1].Found || values[1].Value != "beta" {
			t.Fatalf("values[1]: got %#v", values[1])
		}
	})

	t.Run("idempotent set", func(t *testing.T) {
		store := factory(t)
		pairs := []stores.KeyValue[string]{
			{Key: "foo", Value: "alpha"},
			{Key: "bar", Value: "beta"},
		}
		if err := store.MSet(context.Background(), pairs); err != nil {
			t.Fatalf("mset: %v", err)
		}
		if err := store.MSet(context.Background(), pairs); err != nil {
			t.Fatalf("mset again: %v", err)
		}
		values, err := store.MGet(context.Background(), []string{"foo", "bar"})
		if err != nil {
			t.Fatalf("mget: %v", err)
		}
		if !values[0].Found || values[0].Value != "alpha" {
			t.Fatalf("values[0]: got %#v", values[0])
		}
		if !values[1].Found || values[1].Value != "beta" {
			t.Fatalf("values[1]: got %#v", values[1])
		}
	})
}
