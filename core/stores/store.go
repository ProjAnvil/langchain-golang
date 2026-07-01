package stores

import (
	"context"
	"strings"
	"sync"
)

type KeyValue[V any] struct {
	Key   string
	Value V
}

type MaybeValue[V any] struct {
	Value V
	Found bool
}

type BaseStore[V any] interface {
	MGet(ctx context.Context, keys []string) ([]MaybeValue[V], error)
	MSet(ctx context.Context, keyValuePairs []KeyValue[V]) error
	MDelete(ctx context.Context, keys []string) error
	YieldKeys(ctx context.Context, prefix string) ([]string, error)
}

type InMemoryStore[V any] struct {
	mu    sync.RWMutex
	store map[string]V
}

func NewInMemoryStore[V any]() *InMemoryStore[V] {
	return &InMemoryStore[V]{
		store: make(map[string]V),
	}
}

func (s *InMemoryStore[V]) MGet(_ context.Context, keys []string) ([]MaybeValue[V], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	values := make([]MaybeValue[V], len(keys))
	for i, key := range keys {
		value, ok := s.store[key]
		values[i] = MaybeValue[V]{Value: value, Found: ok}
	}
	return values, nil
}

func (s *InMemoryStore[V]) MSet(_ context.Context, keyValuePairs []KeyValue[V]) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, pair := range keyValuePairs {
		s.store[pair.Key] = pair.Value
	}
	return nil
}

func (s *InMemoryStore[V]) MDelete(_ context.Context, keys []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, key := range keys {
		delete(s.store, key)
	}
	return nil
}

func (s *InMemoryStore[V]) YieldKeys(_ context.Context, prefix string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keys := make([]string, 0, len(s.store))
	for key := range s.store {
		if prefix == "" || strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
	}
	return keys, nil
}

type InMemoryByteStore struct {
	mu    sync.RWMutex
	store map[string][]byte
}

func NewInMemoryByteStore() *InMemoryByteStore {
	return &InMemoryByteStore{
		store: make(map[string][]byte),
	}
}

func (s *InMemoryByteStore) MGet(_ context.Context, keys []string) ([]MaybeValue[[]byte], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	values := make([]MaybeValue[[]byte], len(keys))
	for i, key := range keys {
		value, ok := s.store[key]
		values[i] = MaybeValue[[]byte]{Value: cloneBytes(value), Found: ok}
	}
	return values, nil
}

func (s *InMemoryByteStore) MSet(_ context.Context, keyValuePairs []KeyValue[[]byte]) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, pair := range keyValuePairs {
		s.store[pair.Key] = cloneBytes(pair.Value)
	}
	return nil
}

func (s *InMemoryByteStore) MDelete(_ context.Context, keys []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, key := range keys {
		delete(s.store, key)
	}
	return nil
}

func (s *InMemoryByteStore) YieldKeys(_ context.Context, prefix string) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keys := make([]string, 0, len(s.store))
	for key := range s.store {
		if prefix == "" || strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
	}
	return keys, nil
}

func cloneBytes(input []byte) []byte {
	if len(input) == 0 {
		return nil
	}
	cloned := make([]byte, len(input))
	copy(cloned, input)
	return cloned
}
