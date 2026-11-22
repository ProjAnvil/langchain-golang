package retrievers_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/runnables"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// stubRetriever adapts a function to retrievers.Retriever for tests.
type stubRetriever func(context.Context, string) ([]documents.Document, error)

func (f stubRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	return f(ctx, query)
}

// textModelFunc adapts a plain function to retrievers.TextLLM so tests can
// inject errors or capture prompts without a real model.
type textModelFunc func(context.Context, string) (string, error)

func (f textModelFunc) Invoke(
	ctx context.Context,
	prompt string,
	_ ...runnables.Option,
) (string, error) {
	return f(ctx, prompt)
}

// buildInMemoryStore builds a vectorstores.InMemory pre-populated with texts
// and optional per-text metadata using the supplied embedder.
func buildInMemoryStore(
	t *testing.T,
	embedder embeddings.Embeddings,
	texts []string,
	metadatas []map[string]any,
) *vectorstores.InMemory {
	t.Helper()
	store := vectorstores.NewInMemory(embedder)
	docs := make([]documents.Document, len(texts))
	for i, text := range texts {
		var md map[string]any
		if i < len(metadatas) {
			md = metadatas[i]
		}
		docs[i] = documents.New(text, md)
	}
	if _, err := store.AddDocuments(t.Context(), docs); err != nil {
		t.Fatalf("add documents to in-memory store: %v", err)
	}
	return store
}

// docContents extracts page contents from documents for concise assertions.
func docContents(docs []documents.Document) []string {
	out := make([]string, len(docs))
	for i, doc := range docs {
		out[i] = doc.PageContent
	}
	return out
}

// staticMultiVectorEmbedder produces a deterministic, strictly ordered
// similarity ranking doc0 > doc1 > doc2 for the query vector (1, 0).
func staticMultiVectorEmbedder() embeddings.Static {
	return embeddings.Static{
		DocumentVectors: [][]float64{{1, 0}, {0.9, 0.1}, {0.8, 0.2}},
		QueryVector:     []float64{1, 0},
	}
}

func newParentDocstore(t *testing.T, parents map[string]string) stores.BaseStore[documents.Document] {
	t.Helper()
	docstore := stores.NewInMemoryStore[documents.Document]()
	pairs := make([]stores.KeyValue[documents.Document], 0, len(parents))
	for id, content := range parents {
		pairs = append(pairs, stores.KeyValue[documents.Document]{
			Key:   id,
			Value: documents.New(content, nil).WithID(id),
		})
	}
	if err := docstore.MSet(t.Context(), pairs); err != nil {
		t.Fatalf("store parents: %v", err)
	}
	return docstore
}

func TestMultiVectorRetrieverReturnsParentsInHitOrderAndDedups(t *testing.T) {
	// Three children: doc0 -> A, doc1 -> B, doc2 -> A. Similarity order is
	// doc0, doc1, doc2, so the deduplicated parent order is A, B.
	store := buildInMemoryStore(
		t,
		staticMultiVectorEmbedder(),
		[]string{"child 0", "child 1", "child 2"},
		[]map[string]any{
			{"doc_id": "A"},
			{"doc_id": "B"},
			{"doc_id": "A"},
		},
	)
	docstore := newParentDocstore(t, map[string]string{"A": "parent A", "B": "parent B"})

	retriever, err := retrievers.NewMultiVectorRetriever(store, docstore)
	if err != nil {
		t.Fatalf("new multi-vector retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	got := docContents(docs)
	if len(got) != 2 || got[0] != "parent A" || got[1] != "parent B" {
		t.Fatalf("parents = %v, want [parent A parent B]", got)
	}
}

func TestMultiVectorRetrieverCustomIDKey(t *testing.T) {
	store := buildInMemoryStore(
		t,
		staticMultiVectorEmbedder(),
		[]string{"child 0", "child 1", "child 2"},
		[]map[string]any{
			{"parent": "A"},
			{"parent": "B"},
			{"parent": "A"},
		},
	)
	docstore := newParentDocstore(t, map[string]string{"A": "parent A", "B": "parent B"})

	retriever, err := retrievers.NewMultiVectorRetriever(
		store,
		docstore,
		retrievers.WithMultiVectorIDKey("parent"),
	)
	if err != nil {
		t.Fatalf("new multi-vector retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if got := docContents(docs); len(got) != 2 || got[0] != "parent A" || got[1] != "parent B" {
		t.Fatalf("parents = %v, want [parent A parent B]", got)
	}
}

func TestMultiVectorRetrieverSkipsMissingParentsAndMissingIDKey(t *testing.T) {
	store := buildInMemoryStore(
		t,
		staticMultiVectorEmbedder(),
		[]string{"child 0", "child 1", "child 2"},
		[]map[string]any{
			{"doc_id": "A"},
			nil, // child without the id key is skipped
			{"doc_id": "C"},
		},
	)
	// Only parent A exists; C is missing from the docstore.
	docstore := newParentDocstore(t, map[string]string{"A": "parent A"})

	retriever, err := retrievers.NewMultiVectorRetriever(store, docstore)
	if err != nil {
		t.Fatalf("new multi-vector retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if got := docContents(docs); len(got) != 1 || got[0] != "parent A" {
		t.Fatalf("parents = %v, want [parent A]", got)
	}
}

func TestMultiVectorRetrieverEmptyWhenNoChildrenHaveIDKey(t *testing.T) {
	store := buildInMemoryStore(
		t,
		staticMultiVectorEmbedder(),
		[]string{"child 0", "child 1"},
		[]map[string]any{nil, nil},
	)
	docstore := newParentDocstore(t, map[string]string{"A": "parent A"})

	retriever, err := retrievers.NewMultiVectorRetriever(store, docstore)
	if err != nil {
		t.Fatalf("new multi-vector retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs = %v, want empty", docContents(docs))
	}
}

func TestMultiVectorRetrieverSearchTypes(t *testing.T) {
	cases := []struct {
		name string
		opts []retrievers.MultiVectorOption
	}{
		{name: "similarity"},
		{name: "mmr", opts: []retrievers.MultiVectorOption{
			retrievers.WithMultiVectorSearchType("mmr"),
		}},
		{name: "similarity_score_threshold", opts: []retrievers.MultiVectorOption{
			retrievers.WithMultiVectorSearchType("similarity_score_threshold"),
			retrievers.WithMultiVectorSearchKwargs(map[string]any{"k": 4, "score_threshold": 0.0}),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := buildInMemoryStore(
				t,
				staticMultiVectorEmbedder(),
				[]string{"child 0", "child 1", "child 2"},
				[]map[string]any{{"doc_id": "A"}, {"doc_id": "B"}, {"doc_id": "A"}},
			)
			docstore := newParentDocstore(t, map[string]string{"A": "parent A", "B": "parent B"})
			retriever, err := retrievers.NewMultiVectorRetriever(store, docstore, tc.opts...)
			if err != nil {
				t.Fatalf("new multi-vector retriever: %v", err)
			}
			docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
			if err != nil {
				t.Fatalf("retrieve: %v", err)
			}
			if len(docs) != 2 {
				t.Fatalf("docs = %v, want both parents", docContents(docs))
			}
			seen := map[string]bool{}
			for _, doc := range docs {
				seen[doc.PageContent] = true
			}
			if !seen["parent A"] || !seen["parent B"] {
				t.Fatalf("docs = %v, want parents A and B", docContents(docs))
			}
		})
	}
}

func TestMultiVectorRetrieverConstructorErrors(t *testing.T) {
	store := buildInMemoryStore(t, staticMultiVectorEmbedder(), []string{"x"}, nil)
	docstore := newParentDocstore(t, nil)

	if _, err := retrievers.NewMultiVectorRetriever(nil, docstore); err == nil {
		t.Fatal("nil vector store must error")
	}
	if _, err := retrievers.NewMultiVectorRetriever(store, nil); err == nil {
		t.Fatal("nil docstore must error")
	}
	if _, err := retrievers.NewMultiVectorRetriever(
		store,
		docstore,
		retrievers.WithMultiVectorSearchType("bogus"),
	); err == nil {
		t.Fatal("invalid search type must error")
	}
}

func TestMultiVectorRetrieverCanceledContext(t *testing.T) {
	store := buildInMemoryStore(t, staticMultiVectorEmbedder(), []string{"x"}, []map[string]any{{"doc_id": "A"}})
	docstore := newParentDocstore(t, map[string]string{"A": "parent A"})
	retriever, err := retrievers.NewMultiVectorRetriever(store, docstore)
	if err != nil {
		t.Fatalf("new multi-vector retriever: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := retriever.GetRelevantDocuments(ctx, "query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled retrieval error = %v, want context.Canceled", err)
	}
}
