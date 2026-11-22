package retrievers

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/structuredquery"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// covStore is a VectorStore that also implements relevanceScoreSearcher. Its
// behavior is fully configurable so tests can inject errors and canned
// relevance results without any network access.
type covStore struct {
	addErr     error
	simErr     error
	simDocs    []documents.Document
	relErr     error
	relResults []vectorstores.SearchResult
	relK       int
}

func (s *covStore) AddDocuments(context.Context, []documents.Document) ([]string, error) {
	return nil, s.addErr
}

func (s *covStore) Delete(context.Context, []string) error { return nil }

func (s *covStore) GetByIDs(context.Context, []string) ([]documents.Document, error) {
	return nil, nil
}

func (s *covStore) SimilaritySearch(context.Context, string, int) ([]documents.Document, error) {
	return s.simDocs, s.simErr
}

func (s *covStore) SimilaritySearchWithScore(
	context.Context, string, int,
) ([]vectorstores.SearchResult, error) {
	return nil, nil
}

func (s *covStore) SimilaritySearchWithRelevanceScores(
	_ context.Context,
	_ string,
	k int,
	_ *float64,
) ([]vectorstores.SearchResult, error) {
	s.relK = k
	return s.relResults, s.relErr
}

// covDocstore is a BaseStore whose MGet/MSet outcomes are configurable.
type covDocstore struct {
	mgetErr error
	msetErr error
	values  map[string]documents.Document
}

func (d *covDocstore) MGet(
	_ context.Context,
	keys []string,
) ([]stores.MaybeValue[documents.Document], error) {
	if d.mgetErr != nil {
		return nil, d.mgetErr
	}
	out := make([]stores.MaybeValue[documents.Document], len(keys))
	for i, key := range keys {
		if value, ok := d.values[key]; ok {
			out[i] = stores.MaybeValue[documents.Document]{Value: value, Found: true}
		}
	}
	return out, nil
}

func (d *covDocstore) MSet(
	_ context.Context,
	_ []stores.KeyValue[documents.Document],
) error {
	return d.msetErr
}

func (d *covDocstore) MDelete(context.Context, []string) error { return nil }

func (d *covDocstore) YieldKeys(context.Context, string) ([]string, error) {
	return nil, nil
}

// covTextModel is a TextLLM with a configurable response and an optional
// callback fired on Invoke (used to cancel a context between stages).
type covTextModel struct {
	out      string
	err      error
	onInvoke func()
}

func (m covTextModel) Invoke(
	_ context.Context,
	_ string,
	_ ...runnables.Option,
) (string, error) {
	if m.onInvoke != nil {
		m.onInvoke()
	}
	return m.out, m.err
}

// covCrossEncoder is a CrossEncoder returning a fixed score slice.
type covCrossEncoder struct {
	scores []float64
}

func (c covCrossEncoder) Score([][2]string) ([]float64, error) {
	return c.scores, nil
}

// covErrorVisitor is a structuredquery.Visitor whose every method fails.
type covErrorVisitor struct {
	err error
}

func (v covErrorVisitor) VisitOperation(*structuredquery.Operation) (any, error) {
	return nil, v.err
}

func (v covErrorVisitor) VisitComparison(*structuredquery.Comparison) (any, error) {
	return nil, v.err
}

func (v covErrorVisitor) VisitStructuredQuery(*structuredquery.StructuredQuery) (any, error) {
	return nil, v.err
}

// TestCovTimeWeightedBufferIndex exercises every branch of the buffer-index
// metadata reader, including the missing/nil guard, all supported numeric
// types, and the unsupported-type default.
func TestCovTimeWeightedBufferIndex(t *testing.T) {
	cases := []struct {
		name string
		doc  documents.Document
		want int
		ok   bool
	}{
		{"missing key", documents.New("x", nil), 0, false},
		{"nil value", documents.New("x", map[string]any{"buffer_idx": nil}), 0, false},
		{"int", documents.New("x", map[string]any{"buffer_idx": 7}), 7, true},
		{"int64", documents.New("x", map[string]any{"buffer_idx": int64(8)}), 8, true},
		{"float64", documents.New("x", map[string]any{"buffer_idx": 9.9}), 9, true},
		{"float32", documents.New("x", map[string]any{"buffer_idx": float32(10.5)}), 10, true},
		{"unsupported type", documents.New("x", map[string]any{"buffer_idx": "nope"}), 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := timeWeightedBufferIndex(tc.doc)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("timeWeightedBufferIndex = (%d, %v), want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestCovTimeWeightedNumericMetadata exercises every branch of the numeric
// metadata reader.
func TestCovTimeWeightedNumericMetadata(t *testing.T) {
	cases := []struct {
		name string
		meta map[string]any
		want float64
		ok   bool
	}{
		{"missing key", nil, 0, false},
		{"nil value", map[string]any{"score": nil}, 0, false},
		{"float64", map[string]any{"score": 1.5}, 1.5, true},
		{"float32", map[string]any{"score": float32(2.5)}, 2.5, true},
		{"int", map[string]any{"score": 3}, 3, true},
		{"int64", map[string]any{"score": int64(4)}, 4, true},
		{"unsupported type", map[string]any{"score": "high"}, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := timeWeightedNumericMetadata(documents.New("x", tc.meta), "score")
			if ok != tc.ok || got != tc.want {
				t.Fatalf("timeWeightedNumericMetadata = (%v, %v), want (%v, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestCovTimeWeightedDocumentTime exercises every branch of the timestamp
// metadata reader, using fixed timestamps so the test is deterministic.
func TestCovTimeWeightedDocumentTime(t *testing.T) {
	fallback := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	stamp := time.Date(2021, 6, 15, 12, 30, 0, 0, time.UTC)
	unixSec := int64(1_600_000_000)
	cases := []struct {
		name string
		meta map[string]any
		want time.Time
	}{
		{"missing key", nil, fallback},
		{"nil value", map[string]any{"at": nil}, fallback},
		{"time.Time", map[string]any{"at": stamp}, stamp},
		{"int64 unix", map[string]any{"at": unixSec}, time.Unix(unixSec, 0)},
		{"int unix", map[string]any{"at": int(unixSec)}, time.Unix(unixSec, 0)},
		{"float64 unix", map[string]any{"at": float64(3)}, time.Unix(0, int64(float64(3)*float64(time.Second)))},
		{"float32 unix", map[string]any{"at": float32(4)}, time.Unix(0, int64(float64(float32(4))*float64(time.Second)))},
		{"unsupported type", map[string]any{"at": "yesterday"}, fallback},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := timeWeightedDocumentTime(documents.New("x", tc.meta), "at", fallback)
			if !got.Equal(tc.want) {
				t.Fatalf("timeWeightedDocumentTime = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestCovTimeWeightedOptions applies the two previously unused options and
// verifies their effects, including that the search kwargs "k" is forwarded
// to the relevance-score search.
func TestCovTimeWeightedOptions(t *testing.T) {
	store := &covStore{}
	r, err := NewTimeWeightedVectorStoreRetriever(
		store,
		WithTimeWeightedSearchKwargs(map[string]any{"k": 3}),
		WithTimeWeightedDefaultSalience(0.75),
	)
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	if got := r.searchKwargs["k"]; got != 3 {
		t.Fatalf("searchKwargs k = %v, want 3", got)
	}
	if r.defaultSal == nil || *r.defaultSal != 0.75 {
		t.Fatalf("defaultSal = %v, want 0.75", r.defaultSal)
	}
	if _, err := r.GetSalientDocs(t.Context(), "q"); err != nil {
		t.Fatalf("get salient docs: %v", err)
	}
	if store.relK != 3 {
		t.Fatalf("GetSalientDocs used k=%d, want 3", store.relK)
	}
}

// TestCovTimeWeightedAddDocumentsCanceled covers the ctx.Err guard in
// AddDocuments.
func TestCovTimeWeightedAddDocumentsCanceled(t *testing.T) {
	r, err := NewTimeWeightedVectorStoreRetriever(&covStore{})
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := r.AddDocuments(ctx, []documents.Document{documents.New("x", nil)}); !errors.Is(err, context.Canceled) {
		t.Fatalf("AddDocuments error = %v, want context.Canceled", err)
	}
}

// TestCovTimeWeightedGetSalientDocsCanceled covers the ctx.Err guard in
// GetSalientDocs.
func TestCovTimeWeightedGetSalientDocsCanceled(t *testing.T) {
	r, err := NewTimeWeightedVectorStoreRetriever(&covStore{})
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := r.GetSalientDocs(ctx, "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("GetSalientDocs error = %v, want context.Canceled", err)
	}
}

// TestCovTimeWeightedGetSalientDocsSearchError covers the search-error branch
// in GetSalientDocs and its propagation through GetRelevantDocuments.
func TestCovTimeWeightedGetSalientDocsSearchError(t *testing.T) {
	wantErr := errors.New("relevance search failed")
	r, err := NewTimeWeightedVectorStoreRetriever(&covStore{relErr: wantErr})
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	if _, err := r.GetSalientDocs(t.Context(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("GetSalientDocs error = %v, want %v", err, wantErr)
	}
	if _, err := r.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("GetRelevantDocuments error = %v, want %v", err, wantErr)
	}
}

// TestCovTimeWeightedGetSalientDocsSkipsInvalidBufferIndex covers the
// continue branch for results lacking a usable buffer_idx.
func TestCovTimeWeightedGetSalientDocsSkipsInvalidBufferIndex(t *testing.T) {
	store := &covStore{relResults: []vectorstores.SearchResult{
		{Document: documents.New("no-index", nil), Score: 0.9},
		{Document: documents.New("wrong-type", map[string]any{"buffer_idx": "bad"}), Score: 0.8},
		{Document: documents.New("out-of-range", map[string]any{"buffer_idx": 999}), Score: 0.7},
	}}
	r, err := NewTimeWeightedVectorStoreRetriever(store)
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	if err := r.AddDocuments(t.Context(), []documents.Document{documents.New("real", nil)}); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	salient, err := r.GetSalientDocs(t.Context(), "q")
	if err != nil {
		t.Fatalf("get salient docs: %v", err)
	}
	if len(salient) != 0 {
		t.Fatalf("salient docs = %v, want none", salient)
	}
}

// TestCovTimeWeightedGetRelevantDocumentsSkipsDocsWithoutBufferIndex covers
// the continue branch for memory-stream docs missing a buffer_idx.
func TestCovTimeWeightedGetRelevantDocumentsSkipsDocsWithoutBufferIndex(t *testing.T) {
	r, err := NewTimeWeightedVectorStoreRetriever(&covStore{})
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	r.memoryStream = []documents.Document{documents.New("no-index", map[string]any{})}
	docs, err := r.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs = %v, want none", docs)
	}
}

// TestCovTimeWeightedGetRelevantDocumentsStableTieBreak forces many equal
// combined scores so the stable-sort tie-break branches are all exercised.
func TestCovTimeWeightedGetRelevantDocumentsStableTieBreak(t *testing.T) {
	now := time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC)
	r, err := NewTimeWeightedVectorStoreRetriever(
		&covStore{},
		WithTimeWeightedNow(func() time.Time { return now }),
		WithTimeWeightedK(24),
	)
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	docs := make([]documents.Document, 24)
	for i := range docs {
		docs[i] = documents.New(fmt.Sprintf("doc-%d", i), nil)
	}
	if err := r.AddDocuments(t.Context(), docs); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	got, err := r.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(got) != 24 {
		t.Fatalf("docs = %d, want 24", len(got))
	}
}

// TestCovTimeWeightedGetRelevantDocumentsSkipsOutOfRangeIndex covers the
// continue branch for a candidate whose buffer_idx is out of range.
func TestCovTimeWeightedGetRelevantDocumentsSkipsOutOfRangeIndex(t *testing.T) {
	r, err := NewTimeWeightedVectorStoreRetriever(&covStore{})
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	r.memoryStream = []documents.Document{
		documents.New("ghost", map[string]any{"buffer_idx": 999}),
	}
	docs, err := r.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs = %v, want none", docs)
	}
}

// TestCovParentDocumentRetrieverOptions applies WithSearchK and the empty
// id-key fallback.
func TestCovParentDocumentRetrieverOptions(t *testing.T) {
	docstore := stores.NewInMemoryStore[documents.Document]()
	if _, err := NewParentDocumentRetriever(&covStore{}, docstore, WithSearchK(2)); err != nil {
		t.Fatalf("WithSearchK: %v", err)
	}
	r, err := NewParentDocumentRetriever(&covStore{}, docstore, WithIDKey(""))
	if err != nil {
		t.Fatalf("WithIDKey(\"\"): %v", err)
	}
	if r.idKey != defaultParentIDKey {
		t.Fatalf("idKey = %q, want %q", r.idKey, defaultParentIDKey)
	}
}

// TestCovParentDocumentRetrieverAddDocumentsErrors covers the docstore MSet
// and vector store AddDocuments failure branches.
func TestCovParentDocumentRetrieverAddDocumentsErrors(t *testing.T) {
	t.Run("docstore MSet error", func(t *testing.T) {
		wantErr := errors.New("mset failed")
		r := ParentDocumentRetriever{
			store:    &covStore{},
			docstore: &covDocstore{msetErr: wantErr},
			idKey:    defaultParentIDKey,
		}
		err := r.AddDocuments(t.Context(), []documents.Document{documents.New("x", nil)})
		if !errors.Is(err, wantErr) {
			t.Fatalf("AddDocuments error = %v, want %v", err, wantErr)
		}
	})
	t.Run("vector store AddDocuments error", func(t *testing.T) {
		wantErr := errors.New("add failed")
		r := ParentDocumentRetriever{
			store:    &covStore{addErr: wantErr},
			docstore: stores.NewInMemoryStore[documents.Document](),
			idKey:    defaultParentIDKey,
		}
		err := r.AddDocuments(t.Context(), []documents.Document{documents.New("x", nil)})
		if !errors.Is(err, wantErr) {
			t.Fatalf("AddDocuments error = %v, want %v", err, wantErr)
		}
	})
}

// TestCovParentDocumentRetrieverGetRelevantDocumentsBranches covers the error
// and empty-result branches of parent-document retrieval.
func TestCovParentDocumentRetrieverGetRelevantDocumentsBranches(t *testing.T) {
	newR := func(
		child Retriever,
		docstore stores.BaseStore[documents.Document],
	) ParentDocumentRetriever {
		return ParentDocumentRetriever{
			childRetriever: child,
			docstore:       docstore,
			idKey:          defaultParentIDKey,
		}
	}
	t.Run("child retriever error", func(t *testing.T) {
		wantErr := errors.New("child search failed")
		r := newR(
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return nil, wantErr
			}),
			stores.NewInMemoryStore[documents.Document](),
		)
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
			t.Fatalf("error = %v, want %v", err, wantErr)
		}
	})
	t.Run("child missing parent id key", func(t *testing.T) {
		r := newR(
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return []documents.Document{documents.New("orphan", nil)}, nil
			}),
			stores.NewInMemoryStore[documents.Document](),
		)
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err == nil {
			t.Fatal("child without parent id key must error")
		}
	})
	t.Run("parent id is not a string", func(t *testing.T) {
		r := newR(
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return []documents.Document{
					documents.New("child", map[string]any{defaultParentIDKey: 42}),
				}, nil
			}),
			stores.NewInMemoryStore[documents.Document](),
		)
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err == nil {
			t.Fatal("non-string parent id must error")
		}
	})
	t.Run("no children returns empty", func(t *testing.T) {
		r := newR(
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return nil, nil
			}),
			stores.NewInMemoryStore[documents.Document](),
		)
		docs, err := r.GetRelevantDocuments(t.Context(), "q")
		if err != nil || len(docs) != 0 {
			t.Fatalf("docs = %v err = %v, want empty", docs, err)
		}
	})
	t.Run("docstore MGet error", func(t *testing.T) {
		wantErr := errors.New("mget failed")
		r := newR(
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return []documents.Document{
					documents.New("child", map[string]any{defaultParentIDKey: "A"}),
				}, nil
			}),
			&covDocstore{mgetErr: wantErr},
		)
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
			t.Fatalf("error = %v, want %v", err, wantErr)
		}
	})
}

// TestCovMultiVectorOptions applies WithMultiVectorK and the empty id-key and
// search-type fallbacks.
func TestCovMultiVectorOptions(t *testing.T) {
	docstore := stores.NewInMemoryStore[documents.Document]()
	if _, err := NewMultiVectorRetriever(&covStore{}, docstore, WithMultiVectorK(2)); err != nil {
		t.Fatalf("WithMultiVectorK: %v", err)
	}
	r, err := NewMultiVectorRetriever(&covStore{}, docstore, WithMultiVectorIDKey(""))
	if err != nil {
		t.Fatalf("WithMultiVectorIDKey(\"\"): %v", err)
	}
	if r.idKey != defaultMultiVectorIDKey {
		t.Fatalf("idKey = %q, want %q", r.idKey, defaultMultiVectorIDKey)
	}
	if _, err := NewMultiVectorRetriever(&covStore{}, docstore, WithMultiVectorSearchType("")); err != nil {
		t.Fatalf("WithMultiVectorSearchType(\"\"): %v", err)
	}
}

// TestCovMultiVectorGetRelevantDocumentsBranches covers the child-retriever
// error, non-string id stringification, and docstore MGet error branches.
func TestCovMultiVectorGetRelevantDocumentsBranches(t *testing.T) {
	t.Run("child retriever error", func(t *testing.T) {
		wantErr := errors.New("child search failed")
		r, err := NewMultiVectorRetriever(
			&covStore{simErr: wantErr},
			stores.NewInMemoryStore[documents.Document](),
		)
		if err != nil {
			t.Fatalf("new retriever: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
			t.Fatalf("error = %v, want %v", err, wantErr)
		}
	})
	t.Run("non-string parent id is stringified", func(t *testing.T) {
		store := &covStore{simDocs: []documents.Document{
			documents.New("child", map[string]any{defaultMultiVectorIDKey: 42}),
		}}
		r, err := NewMultiVectorRetriever(store, stores.NewInMemoryStore[documents.Document]())
		if err != nil {
			t.Fatalf("new retriever: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); err != nil {
			t.Fatalf("retrieve: %v", err)
		}
	})
	t.Run("docstore MGet error", func(t *testing.T) {
		store := &covStore{simDocs: []documents.Document{
			documents.New("child", map[string]any{defaultMultiVectorIDKey: "A"}),
		}}
		wantErr := errors.New("mget failed")
		r, err := NewMultiVectorRetriever(store, &covDocstore{mgetErr: wantErr})
		if err != nil {
			t.Fatalf("new retriever: %v", err)
		}
		if _, err := r.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
			t.Fatalf("error = %v, want %v", err, wantErr)
		}
	})
}

// TestCovNewStructuredQueryTranslatorBranches covers the canceled-context,
// nil-query, and visitor-error branches of the generic translator.
func TestCovNewStructuredQueryTranslatorBranches(t *testing.T) {
	tr := NewStructuredQueryTranslator(nil)
	t.Run("canceled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, _, err := tr.Translate(ctx, nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v, want context.Canceled", err)
		}
	})
	t.Run("nil query", func(t *testing.T) {
		if _, _, err := tr.Translate(t.Context(), nil); err == nil {
			t.Fatal("nil structured query must error")
		}
	})
	t.Run("visitor error", func(t *testing.T) {
		wantErr := errors.New("visitor failed")
		failing := NewStructuredQueryTranslator(covErrorVisitor{err: wantErr})
		sq := structuredquery.NewStructuredQuery("q", nil, 0)
		if _, _, err := failing.Translate(t.Context(), sq); !errors.Is(err, wantErr) {
			t.Fatalf("error = %v, want %v", err, wantErr)
		}
	})
}

// TestCovSelfQueryCancellationAfterTranslate cancels the context during
// Translate so the post-translate ctx.Err guard fires.
func TestCovSelfQueryCancellationAfterTranslate(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	ctx, cancel := context.WithCancel(t.Context())
	translator := StructuredQueryTranslatorFunc(
		func(_ context.Context, _ *structuredquery.StructuredQuery) (string, map[string]any, error) {
			cancel()
			return "rewritten", map[string]any{}, nil
		},
	)
	r, err := NewSelfQueryRetriever(
		fnRetriever(func(context.Context, string) ([]documents.Document, error) {
			return nil, nil
		}),
		QueryConstructorFunc(
			func(context.Context, string) (*structuredquery.StructuredQuery, error) {
				return sq, nil
			},
		),
		translator,
	)
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	if _, err := r.GetRelevantDocuments(ctx, "some query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

// TestCovSelfQueryCancellationAfterBase cancels the context during the base
// retrieval so the post-base ctx.Err guard fires.
func TestCovSelfQueryCancellationAfterBase(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	ctx, cancel := context.WithCancel(t.Context())
	base := fnRetriever(func(_ context.Context, _ string) ([]documents.Document, error) {
		cancel()
		return []documents.Document{documents.New("x", nil)}, nil
	})
	translator := StructuredQueryTranslatorFunc(
		func(_ context.Context, _ *structuredquery.StructuredQuery) (string, map[string]any, error) {
			return "rewritten", map[string]any{}, nil
		},
	)
	r, err := NewSelfQueryRetriever(
		base,
		QueryConstructorFunc(
			func(context.Context, string) (*structuredquery.StructuredQuery, error) {
				return sq, nil
			},
		),
		translator,
	)
	if err != nil {
		t.Fatalf("new retriever: %v", err)
	}
	if _, err := r.GetRelevantDocuments(ctx, "some query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
}

// TestCovEmbeddingsFilterCosineEdgeCases covers the early-return branches of
// the cosine helper.
func TestCovEmbeddingsFilterCosineEdgeCases(t *testing.T) {
	cases := []struct {
		name string
		a    []float64
		b    []float64
		want float64
	}{
		{"empty vector", nil, []float64{1}, 0},
		{"length mismatch", []float64{1, 0}, []float64{1}, 0},
		{"zero norm", []float64{0, 0}, []float64{0, 0}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := embeddingsFilterCosine(tc.a, tc.b); got != tc.want {
				t.Fatalf("embeddingsFilterCosine = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestCovEmbeddingsFilterTieBreak forces equal similarities so the sort
// comparator's index tie-break is exercised.
func TestCovEmbeddingsFilterTieBreak(t *testing.T) {
	f, err := NewEmbeddingsFilter(embeddings.Static{
		DocumentVectors: [][]float64{{1, 0}, {1, 0}},
		QueryVector:     []float64{1, 0},
	})
	if err != nil {
		t.Fatalf("new filter: %v", err)
	}
	docs := []documents.Document{
		documents.New("a", nil),
		documents.New("b", nil),
	}
	out, err := f.CompressDocuments(t.Context(), docs, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("docs = %d, want 2", len(out))
	}
}

// TestCovEnsembleDocKeyNonStringID covers the fmt.Sprint fallback for a
// non-string id metadata value.
func TestCovEnsembleDocKeyNonStringID(t *testing.T) {
	cases := []struct {
		name string
		doc  documents.Document
		want string
	}{
		{"string id", documents.New("content", map[string]any{"id": "abc"}), "abc"},
		{"non-string id", documents.New("content", map[string]any{"id": 7}), "7"},
		{"nil id", documents.New("content", map[string]any{"id": nil}), "content"},
		{"missing id", documents.New("content", nil), "content"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ensembleDocKey(tc.doc); got != tc.want {
				t.Fatalf("ensembleDocKey = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestCovCrossEncoderNegativeTopN covers the negative-top-N clamp branch.
func TestCovCrossEncoderNegativeTopN(t *testing.T) {
	r, err := NewCrossEncoderReranker(
		covCrossEncoder{scores: []float64{1, 2}},
		WithCrossEncoderTopN(-1),
	)
	if err != nil {
		t.Fatalf("new reranker: %v", err)
	}
	out, err := r.CompressDocuments(t.Context(), []documents.Document{
		documents.New("a", nil), documents.New("b", nil),
	}, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("docs = %d, want 0", len(out))
	}
}

// TestCovListwiseRerankNegativeTopN covers the negative-top-N clamp branch.
func TestCovListwiseRerankNegativeTopN(t *testing.T) {
	r, err := NewLLMListwiseRerank(
		covTextModel{out: "1 0"},
		WithListwiseRerankTopN(-1),
	)
	if err != nil {
		t.Fatalf("new reranker: %v", err)
	}
	out, err := r.CompressDocuments(t.Context(), []documents.Document{
		documents.New("a", nil), documents.New("b", nil),
	}, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("docs = %d, want 0", len(out))
	}
}

// TestCovLLMChainExtractorParseError cancels the context during the model
// call so the parser returns an error and the propagation branch fires.
func TestCovLLMChainExtractorParseError(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	model := covTextModel{out: "extracted text", onInvoke: cancel}
	r, err := NewLLMChainExtractor(model)
	if err != nil {
		t.Fatalf("new extractor: %v", err)
	}
	if _, err := r.CompressDocuments(
		ctx,
		[]documents.Document{documents.New("a", nil)},
		"q",
	); err == nil {
		t.Fatal("parser must fail after the context is canceled")
	}
}
