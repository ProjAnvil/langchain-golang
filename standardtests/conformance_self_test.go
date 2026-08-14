package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/tools"
)

func TestRunToolBasicsWithInMemoryTool(t *testing.T) {
	RunToolBasics(t, func(t testing.TB) tools.Tool {
		t.Helper()
		tool, err := tools.NewFunc(
			"echo",
			"echoes the provided text",
			schema.Object(map[string]schema.Schema{
				"text": schema.String("text to echo"),
			}, "text"),
			func(ctx context.Context, input map[string]any) (tools.Result, error) {
				text, _ := input["text"].(string)
				return tools.Result{Content: "echo: " + text}, nil
			},
		)
		if err != nil {
			t.Fatalf("NewFunc: %v", err)
		}
		return tool
	})
}

func TestRunCacheBasicsWithInMemoryCache(t *testing.T) {
	RunCacheBasics(t, func(t testing.TB) caches.Cache {
		t.Helper()
		cache, err := caches.NewInMemoryCache()
		if err != nil {
			t.Fatalf("NewInMemoryCache: %v", err)
		}
		return cache
	})
}

func TestRunStoreBasicsWithInMemoryStore(t *testing.T) {
	RunStoreBasics(t, func(t testing.TB) stores.BaseStore[string] {
		t.Helper()
		return stores.NewInMemoryStore[string]()
	})
}
