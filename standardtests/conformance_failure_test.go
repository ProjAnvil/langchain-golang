package standardtests

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/caches"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/tools"
)

// errConformanceStub is the error broken stub implementations return.
var errConformanceStub = errors.New("conformance stub failure")

// expectConformanceFailure runs body under an isolated testing harness and
// requires that it fails. Conformance helpers report contract violations via
// t.Fatalf; driving them with deliberately broken implementations exercises
// those failure branches without failing the parent test run.
func expectConformanceFailure(t *testing.T, name string, body func(t *testing.T)) {
	t.Helper()
	t.Run(name, func(t *testing.T) {
		ok := testing.RunTests(
			func(_, _ string) (bool, error) { return true, nil },
			[]testing.InternalTest{{Name: name, F: body}},
		)
		if ok {
			t.Fatalf("expected %s to fail, but every check passed", name)
		}
	})
}

// stubCache is a cache whose behavior is configured per scenario.
type stubCache struct {
	lookupErr error
	updateErr error
	clearErr  error
	alwaysHit bool
}

func (c stubCache) Lookup(context.Context, string, string) ([]caches.Generation, bool, error) {
	if c.lookupErr != nil {
		return nil, false, c.lookupErr
	}
	if c.alwaysHit {
		return []caches.Generation{{Text: "stale cached text."}}, true, nil
	}
	return nil, false, nil
}

func (c stubCache) Update(context.Context, string, string, []caches.Generation) error {
	return c.updateErr
}

func (c stubCache) Clear(context.Context) error {
	return c.clearErr
}

func TestRunCacheBasicsFailures(t *testing.T) {
	factory := func(cache stubCache) CacheFactory {
		return func(t testing.TB) caches.Cache {
			t.Helper()
			return cache
		}
	}

	expectConformanceFailure(t, "lookup errors", func(t *testing.T) {
		RunCacheBasics(t, factory(stubCache{lookupErr: errConformanceStub}))
	})
	expectConformanceFailure(t, "hits on empty and after clear", func(t *testing.T) {
		RunCacheBasics(t, factory(stubCache{alwaysHit: true}))
	})
	expectConformanceFailure(t, "update errors", func(t *testing.T) {
		RunCacheBasics(t, factory(stubCache{updateErr: errConformanceStub}))
	})
	expectConformanceFailure(t, "updates are never found", func(t *testing.T) {
		RunCacheBasics(t, factory(stubCache{}))
	})
	expectConformanceFailure(t, "clear errors", func(t *testing.T) {
		RunCacheBasics(t, factory(stubCache{clearErr: errConformanceStub}))
	})
}

// stubStore wraps a working in-memory store with error injection and canned
// MGet results.
type stubStore struct {
	inner      *stores.InMemoryStore[string]
	mgetErr    error
	mgetResult []stores.MaybeValue[string]
	msetErr    error
	msetErrOn  int
	msetCalls  int
	mdeleteErr error
}

func newStubStore() *stubStore {
	return &stubStore{inner: stores.NewInMemoryStore[string]()}
}

func (s *stubStore) MGet(ctx context.Context, keys []string) ([]stores.MaybeValue[string], error) {
	if s.mgetErr != nil {
		return nil, s.mgetErr
	}
	if s.mgetResult != nil {
		return s.mgetResult, nil
	}
	return s.inner.MGet(ctx, keys)
}

func (s *stubStore) MSet(ctx context.Context, pairs []stores.KeyValue[string]) error {
	s.msetCalls++
	if s.msetErr != nil && (s.msetErrOn == 0 || s.msetCalls == s.msetErrOn) {
		return s.msetErr
	}
	return s.inner.MSet(ctx, pairs)
}

func (s *stubStore) MDelete(ctx context.Context, keys []string) error {
	if s.mdeleteErr != nil {
		return s.mdeleteErr
	}
	return s.inner.MDelete(ctx, keys)
}

func (s *stubStore) YieldKeys(ctx context.Context, prefix string) ([]string, error) {
	return s.inner.YieldKeys(ctx, prefix)
}

func TestRunStoreBasicsFailures(t *testing.T) {
	factory := func(configure func(*stubStore)) StoreFactory {
		return func(t testing.TB) stores.BaseStore[string] {
			t.Helper()
			store := newStubStore()
			configure(store)
			return store
		}
	}
	found := func(value string) stores.MaybeValue[string] {
		return stores.MaybeValue[string]{Value: value, Found: true}
	}

	expectConformanceFailure(t, "mget errors", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) { s.mgetErr = errConformanceStub }))
	})
	expectConformanceFailure(t, "mset errors", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) { s.msetErr = errConformanceStub }))
	})
	expectConformanceFailure(t, "second mset errors", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.msetErr = errConformanceStub
			s.msetErrOn = 2
		}))
	})
	expectConformanceFailure(t, "mget returns too few values", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = make([]stores.MaybeValue[string], 2)
		}))
	})
	expectConformanceFailure(t, "mget reports hits on empty store", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{found("a"), found("b"), found("c")}
		}))
	})
	expectConformanceFailure(t, "mget loses the first value", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{{}, found("beta")}
		}))
	})
	expectConformanceFailure(t, "mget invents missing keys and undeleted values", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{found("alpha"), found("beta")}
		}))
	})
	expectConformanceFailure(t, "mget returns value for missing key", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{found("alpha"), {Value: "ghost"}}
		}))
	})
	expectConformanceFailure(t, "mdelete errors", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) { s.mdeleteErr = errConformanceStub }))
	})
	expectConformanceFailure(t, "mget loses all values", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{{}, {}}
		}))
	})
	expectConformanceFailure(t, "mget corrupts the first value", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{found("wrong"), found("beta")}
		}))
	})
	expectConformanceFailure(t, "mget corrupts the second value", func(t *testing.T) {
		RunStoreBasics(t, factory(func(s *stubStore) {
			s.mgetResult = []stores.MaybeValue[string]{found("alpha"), found("wrong")}
		}))
	})
}

// stubTool is a tool whose behavior is configured per scenario.
type stubTool struct {
	name   string
	args   schema.Schema
	result tools.Result
	err    error
}

func (s stubTool) Name() string        { return s.name }
func (s stubTool) Description() string { return "stub tool" }
func (s stubTool) ArgsSchema() schema.Schema {
	return s.args
}

func (s stubTool) Invoke(context.Context, map[string]any) (tools.Result, error) {
	if s.err != nil {
		return tools.Result{}, s.err
	}
	return s.result, nil
}

func TestRunToolBasicsFailures(t *testing.T) {
	factory := func(tool stubTool) ToolFactory {
		return func(t testing.TB) tools.Tool {
			t.Helper()
			return tool
		}
	}
	args := schema.Object(map[string]schema.Schema{
		"text": schema.String("text to echo"),
	}, "text")

	expectConformanceFailure(t, "missing name and schema", func(t *testing.T) {
		RunToolBasics(t, factory(stubTool{result: tools.Result{Content: "ok"}}))
	})
	expectConformanceFailure(t, "invoke errors", func(t *testing.T) {
		RunToolBasics(t, factory(stubTool{name: "stub", args: args, err: errConformanceStub}))
	})
	expectConformanceFailure(t, "invoke returns empty content", func(t *testing.T) {
		RunToolBasics(t, factory(stubTool{name: "stub", args: args}))
	})
}

func TestMinimalToolInput(t *testing.T) {
	t.Run("non-map properties yield empty input", func(t *testing.T) {
		input := minimalToolInput(schema.Schema{"properties": "not-a-map"})
		if len(input) != 0 {
			t.Fatalf("expected empty input, got %#v", input)
		}
	})

	t.Run("property forms are coerced or skipped", func(t *testing.T) {
		input := minimalToolInput(schema.Schema{"properties": map[string]any{
			"typed": schema.String("typed value"),
			"plain": map[string]any{"type": "integer"},
			"bogus": 42,
		}})
		want := map[string]any{"typed": "", "plain": 0}
		if !reflect.DeepEqual(input, want) {
			t.Fatalf("input: got %#v want %#v", input, want)
		}
	})
}

func TestZeroValueForSchema(t *testing.T) {
	cases := []struct {
		name string
		prop schema.Schema
		want any
	}{
		{"string", schema.Schema{"type": "string"}, ""},
		{"integer", schema.Schema{"type": "integer"}, 0},
		{"number", schema.Schema{"type": "number"}, 0},
		{"boolean", schema.Schema{"type": "boolean"}, false},
		{"array", schema.Schema{"type": "array"}, []any{}},
		{"object", schema.Schema{"type": "object"}, map[string]any{}},
		{"unknown type", schema.Schema{"type": "null"}, nil},
		{"missing type", schema.Schema{}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := zeroValueForSchema(tc.prop); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("zero value: got %#v want %#v", got, tc.want)
			}
		})
	}
}
