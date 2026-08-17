package storetest_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/langgraph/store"
	"github.com/projanvil/langchain-golang/langgraph/store/storetest"
)

// This file meta-tests the conformance suite: storetest.Run must FAIL when
// the Store under test violates the contract. Each case wraps a conformant
// InMemoryStore in a fakeStore injecting one kind of violation, runs the full
// suite against it inside an isolated testing harness, and requires the run
// to fail. A passing nested run would mean the suite has a blind spot.
//
// The nested runs fail by design; their failures are contained by
// testing.RunTests (mirroring standardtests/conformance_failure_test.go), so
// the parent test run stays green.

// errFakeStore is the error faulty operations return. It deliberately does
// NOT contain "invalid namespace" so the put_rejects_invalid_namespace suite
// step rejects it as the wrong error kind.
var errFakeStore = errors.New("fake store failure")

// expectConformanceFailure runs body under an isolated testing harness and
// requires that it fails. Conformance helpers report contract violations via
// t.Fatalf/t.Errorf; driving them with deliberately broken stores exercises
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

// fakeStore delegates to a conformant InMemoryStore; each operation can be
// overridden with a hook that injects a contract violation. Hooks receive the
// same arguments as the operation and may delegate to the embedded inner
// store for seeding/partial behavior.
type fakeStore struct {
	inner    store.Store
	getCalls atomic.Int32 // counts Get calls, for call-dependent hooks
	putCalls atomic.Int32 // counts Put calls, for call-dependent hooks

	putFn    func(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error
	getFn    func(ctx context.Context, namespace []string, key string) (*store.Item, error)
	deleteFn func(ctx context.Context, namespace []string, key string) error
	searchFn func(ctx context.Context, namespacePrefix []string, opts store.SearchOptions) ([]store.SearchItem, error)
	listFn   func(ctx context.Context, opts store.ListNamespacesOptions) ([][]string, error)
}

func (f *fakeStore) Get(ctx context.Context, namespace []string, key string) (*store.Item, error) {
	if f.getFn != nil {
		return f.getFn(ctx, namespace, key)
	}
	return f.inner.Get(ctx, namespace, key)
}

func (f *fakeStore) Put(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error {
	if f.putFn != nil {
		return f.putFn(ctx, namespace, key, value, index)
	}
	return f.inner.Put(ctx, namespace, key, value, index)
}

func (f *fakeStore) Delete(ctx context.Context, namespace []string, key string) error {
	if f.deleteFn != nil {
		return f.deleteFn(ctx, namespace, key)
	}
	return f.inner.Delete(ctx, namespace, key)
}

func (f *fakeStore) Search(ctx context.Context, namespacePrefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
	if f.searchFn != nil {
		return f.searchFn(ctx, namespacePrefix, opts)
	}
	return f.inner.Search(ctx, namespacePrefix, opts)
}

func (f *fakeStore) ListNamespaces(ctx context.Context, opts store.ListNamespacesOptions) ([][]string, error) {
	if f.listFn != nil {
		return f.listFn(ctx, opts)
	}
	return f.inner.ListNamespaces(ctx, opts)
}

// fakeStoreFactory returns a storetest newStore factory producing a fresh
// fakeStore (backed by empty storage) per suite subtest, configured by
// configure.
func fakeStoreFactory(configure func(f *fakeStore)) func(t *testing.T) store.Store {
	return func(t *testing.T) store.Store {
		t.Helper()
		f := &fakeStore{inner: store.NewInMemoryStore()}
		configure(f)
		return f
	}
}

// runAgainstSuite runs the full conformance suite against stores built by
// newStore; used as the body of expectConformanceFailure cases.
func runAgainstSuite(newStore func(t *testing.T) store.Store) func(t *testing.T) {
	return func(t *testing.T) {
		storetest.Run(t, newStore)
	}
}

// TestPutGetFailures exercises the suite's put/get contract checks with
// stores whose Put or Get misbehaves.
func TestPutGetFailures(t *testing.T) {
	expectConformanceFailure(t, "put errors", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.putFn = func(context.Context, []string, string, map[string]any, []string) error {
			return errFakeStore
		}
	})))

	expectConformanceFailure(t, "get errors", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.getFn = func(context.Context, []string, string) (*store.Item, error) {
			return nil, errFakeStore
		}
	})))

	expectConformanceFailure(t, "get returns nil for stored keys", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.getFn = func(context.Context, []string, string) (*store.Item, error) {
			return nil, nil
		}
	})))

	expectConformanceFailure(t, "get corrupts the stored item", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		// Wrong key, namespace, and value; timestamps left zero.
		f.getFn = func(context.Context, []string, string) (*store.Item, error) {
			return &store.Item{
				Namespace: []string{"wrong"},
				Key:       "wrong",
				Value:     map[string]any{"wrong": true},
			}, nil
		}
	})))

	expectConformanceFailure(t, "get invents items for missing keys", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.getFn = func(_ context.Context, namespace []string, key string) (*store.Item, error) {
			return &store.Item{
				Namespace: namespace,
				Key:       key,
				Value:     map[string]any{"ghost": true},
			}, nil
		}
	})))

	expectConformanceFailure(t, "get invents items for missing namespaces", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.getFn = func(ctx context.Context, namespace []string, key string) (*store.Item, error) {
			if len(namespace) > 0 && namespace[0] == "nope" {
				return &store.Item{
					Namespace: namespace,
					Key:       key,
					Value:     map[string]any{"ghost": true},
				}, nil
			}
			return f.inner.Get(ctx, namespace, key)
		}
	})))

	expectConformanceFailure(t, "put errors on the second write", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.putFn = func(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error {
			if n := f.putCalls.Add(1); n > 1 {
				return errFakeStore
			}
			return f.inner.Put(ctx, namespace, key, value, index)
		}
	})))

	expectConformanceFailure(t, "get errors after an update", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.putFn = func(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error {
			f.putCalls.Add(1)
			return f.inner.Put(ctx, namespace, key, value, index)
		}
		f.getFn = func(ctx context.Context, namespace []string, key string) (*store.Item, error) {
			if f.putCalls.Load() >= 2 {
				return nil, errFakeStore
			}
			return f.inner.Get(ctx, namespace, key)
		}
	})))

	expectConformanceFailure(t, "updated_at goes backwards on re-put", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.getFn = func(ctx context.Context, namespace []string, key string) (*store.Item, error) {
			n := f.getCalls.Add(1)
			item, err := f.inner.Get(ctx, namespace, key)
			if err != nil || item == nil || n == 1 {
				return item, err
			}
			// Every Get after the first reports a timestamp one hour stale.
			cp := *item
			cp.UpdatedAt = cp.UpdatedAt.Add(-time.Hour)
			return &cp, nil
		}
	})))
}

// TestDeleteFailures exercises the suite's delete contract checks.
func TestDeleteFailures(t *testing.T) {
	expectConformanceFailure(t, "delete errors", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.deleteFn = func(context.Context, []string, string) error {
			return errFakeStore
		}
	})))

	expectConformanceFailure(t, "delete is a no-op", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.deleteFn = func(context.Context, []string, string) error {
			return nil // claims success but removes nothing
		}
	})))

	expectConformanceFailure(t, "delete wipes sibling keys", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.deleteFn = func(ctx context.Context, namespace []string, _ string) error {
			items, err := f.inner.Search(ctx, namespace, store.SearchOptions{Limit: 100})
			if err != nil {
				return err
			}
			for _, item := range items {
				if err := f.inner.Delete(ctx, namespace, item.Key); err != nil {
					return err
				}
			}
			return nil
		}
	})))
}

// TestSearchFailures exercises the suite's search contract checks: prefix
// matching, filtering, and limit/offset pagination.
func TestSearchFailures(t *testing.T) {
	expectConformanceFailure(t, "search errors", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(context.Context, []string, store.SearchOptions) ([]store.SearchItem, error) {
			return nil, errFakeStore
		}
	})))

	expectConformanceFailure(t, "search ignores the namespace prefix", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, _ []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			return f.inner.Search(ctx, nil, opts)
		}
	})))

	expectConformanceFailure(t, "search mixes in foreign namespaces", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			results, err := f.inner.Search(ctx, prefix, opts)
			if err == nil && len(results) == 2 && len(prefix) == 1 && prefix[0] == "docs" {
				// Right count, but one result lies about its namespace.
				results[1].Namespace = []string{"notes"}
			}
			return results, err
		}
	})))

	expectConformanceFailure(t, "search ignores the filter", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			opts.Filter = nil
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search ignores the limit", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			opts.Limit = 0 // default limit returns everything seeded here
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search ignores the offset", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			opts.Offset = 0
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search ignores the offset when a limit is set", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			if opts.Offset > 0 && opts.Limit > 0 {
				opts.Offset = 0
			}
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search returns results past the end", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			if opts.Offset >= 100 {
				opts.Offset = 0
			}
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "empty prefix search drops namespaces", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			if len(prefix) == 0 {
				prefix = []string{"a"} // silently narrows the match-all search
			}
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search truncates results", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			results, err := f.inner.Search(ctx, prefix, opts)
			if len(results) > 1 {
				results = results[:1]
			}
			return results, err
		}
	})))

	expectConformanceFailure(t, "search errors on a deeper prefix", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			if len(prefix) == 2 && prefix[0] == "docs" {
				return nil, errFakeStore
			}
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search drops items at a deeper prefix", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			if len(prefix) == 2 && prefix[0] == "docs" {
				return nil, nil
			}
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	expectConformanceFailure(t, "search errors on a narrowed hierarchical prefix", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
			if len(prefix) == 2 && prefix[0] == "users" {
				return nil, errFakeStore
			}
			return f.inner.Search(ctx, prefix, opts)
		}
	})))

	// One case per paginated search in search_limit_offset: each errors on
	// exactly one Limit/Offset combination so the preceding checks pass.
	paginationErrors := []struct {
		name   string
		offset int
		limit  int
	}{
		{"search errors with an explicit limit", 0, 2},
		{"search errors with an offset", 2, 0},
		{"search errors with offset and limit", 1, 2},
		{"search errors with an offset past the end", 100, 0},
	}
	for _, c := range paginationErrors {
		expectConformanceFailure(t, c.name, runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
			f.searchFn = func(ctx context.Context, prefix []string, opts store.SearchOptions) ([]store.SearchItem, error) {
				if opts.Offset == c.offset && opts.Limit == c.limit {
					return nil, errFakeStore
				}
				return f.inner.Search(ctx, prefix, opts)
			}
		})))
	}
}

// TestNamespaceValidationFailures exercises the suite's namespace-validation
// contract checks.
func TestNamespaceValidationFailures(t *testing.T) {
	expectConformanceFailure(t, "put accepts invalid namespaces", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.putFn = func(ctx context.Context, namespace []string, key string, value map[string]any, index []string) error {
			_ = f.inner.Put(ctx, namespace, key, value, index)
			return nil // swallows the validation error, claiming success
		}
	})))

	// A Put that fails with a non-validation error is covered by the
	// "put errors" case above: errFakeStore lacks the "invalid namespace"
	// marker, tripping the error-kind check, and it also fails the valid
	// namespace sanity put.
}

// TestListNamespacesFailures exercises the suite's ListNamespaces checks.
func TestListNamespacesFailures(t *testing.T) {
	expectConformanceFailure(t, "list namespaces errors", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.listFn = func(context.Context, store.ListNamespacesOptions) ([][]string, error) {
			return nil, errFakeStore
		}
	})))

	expectConformanceFailure(t, "list namespaces invents an extra namespace", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.listFn = func(ctx context.Context, opts store.ListNamespacesOptions) ([][]string, error) {
			results, err := f.inner.ListNamespaces(ctx, opts)
			return append(results, []string{"extra"}), err
		}
	})))

	expectConformanceFailure(t, "list namespaces returns the wrong order", runAgainstSuite(fakeStoreFactory(func(f *fakeStore) {
		f.listFn = func(ctx context.Context, opts store.ListNamespacesOptions) ([][]string, error) {
			results, err := f.inner.ListNamespaces(ctx, opts)
			for i, j := 0, len(results)-1; i < j; i, j = i+1, j-1 {
				results[i], results[j] = results[j], results[i]
			}
			return results, err
		}
	})))
}
