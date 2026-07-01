package caches

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

type Generation struct {
	Text           string         `json:"text"`
	GenerationInfo map[string]any `json:"generation_info,omitempty"`
}

type Cache interface {
	Lookup(ctx context.Context, prompt string, llmString string) ([]Generation, bool, error)
	Update(ctx context.Context, prompt string, llmString string, returnVal []Generation) error
	Clear(ctx context.Context) error
}

type InMemoryCacheOption func(*InMemoryCache) error

func WithMaxSize(maxSize int) InMemoryCacheOption {
	return func(cache *InMemoryCache) error {
		if maxSize <= 0 {
			return errors.New("maxsize must be greater than 0")
		}
		cache.maxSize = maxSize
		return nil
	}
}

type InMemoryCache struct {
	mu      sync.RWMutex
	entries map[cacheKey][]Generation
	order   []cacheKey
	maxSize int
}

type cacheKey struct {
	prompt    string
	llmString string
}

func NewInMemoryCache(opts ...InMemoryCacheOption) (*InMemoryCache, error) {
	cache := &InMemoryCache{
		entries: make(map[cacheKey][]Generation),
	}
	for _, opt := range opts {
		if err := opt(cache); err != nil {
			return nil, err
		}
	}
	return cache, nil
}

func (c *InMemoryCache) Lookup(_ context.Context, prompt string, llmString string) ([]Generation, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	generations, ok := c.entries[cacheKey{prompt: prompt, llmString: llmString}]
	if !ok {
		return nil, false, nil
	}
	return cloneGenerations(generations), true, nil
}

func (c *InMemoryCache) Update(_ context.Context, prompt string, llmString string, returnVal []Generation) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := cacheKey{prompt: prompt, llmString: llmString}
	if _, exists := c.entries[key]; !exists {
		if c.maxSize > 0 && len(c.entries) == c.maxSize {
			c.evictOldest()
		}
		c.order = append(c.order, key)
	}
	c.entries[key] = cloneGenerations(returnVal)
	return nil
}

func (c *InMemoryCache) Clear(context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[cacheKey][]Generation)
	c.order = nil
	return nil
}

func (c *InMemoryCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

func (c *InMemoryCache) MaxSize() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.maxSize
}

func (c *InMemoryCache) evictOldest() {
	if len(c.order) == 0 {
		return
	}
	oldest := c.order[0]
	c.order = c.order[1:]
	delete(c.entries, oldest)
}

func cloneGenerations(generations []Generation) []Generation {
	if len(generations) == 0 {
		return nil
	}
	cloned := make([]Generation, len(generations))
	for i, generation := range generations {
		cloned[i] = generation
		if generation.GenerationInfo != nil {
			cloned[i].GenerationInfo = cloneMap(generation.GenerationInfo)
		}
	}
	return cloned
}

func cloneMap(input map[string]any) map[string]any {
	if len(input) == 0 {
		return nil
	}
	cloned := make(map[string]any, len(input))
	for key, value := range input {
		cloned[key] = value
	}
	return cloned
}

func (k cacheKey) String() string {
	return fmt.Sprintf("%s/%s", k.prompt, k.llmString)
}
