package retrievers_test

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// truncatingCompressor keeps the first keep documents and marks them, standing
// in for a hosted reranker at the end of the chain.
type truncatingCompressor struct {
	calls int
	keep  int
}

func (c *truncatingCompressor) CompressDocuments(
	_ context.Context,
	docs []documents.Document,
	_ string,
) ([]documents.Document, error) {
	c.calls++
	if len(docs) > c.keep {
		docs = docs[:c.keep]
	}
	out := make([]documents.Document, len(docs))
	for i, doc := range docs {
		out[i] = doc.Clone()
		if out[i].Metadata == nil {
			out[i].Metadata = map[string]any{}
		}
		out[i].Metadata["compressed"] = true
	}
	return out, nil
}

// TestRAGChainMultiQueryEnsembleCompression exercises the M1 RAG stack end to
// end: a chat model expands the query, an ensemble fuses dense and keyword
// retrieval with RRF, and a compressor truncates and annotates the merged
// hits — every component wired through the plain Retriever interface.
func TestRAGChainMultiQueryEnsembleCompression(t *testing.T) {
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	_, err := store.AddDocuments(t.Context(), []documents.Document{
		documents.New("alpha alpha", map[string]any{"source": "dense"}),
		documents.New("beta beta", map[string]any{"source": "dense"}),
		documents.New("gamma gamma", map[string]any{"source": "dense"}),
		documents.New("delta delta", map[string]any{"source": "dense"}),
	})
	if err != nil {
		t.Fatalf("index documents: %v", err)
	}

	// Dense side over the in-memory store; keyword side mirrors an exact-match
	// retriever over the same corpus.
	dense, err := retrievers.AsRetriever(
		store,
		retrievers.WithSearchKwargs(map[string]any{"k": 2}),
	)
	if err != nil {
		t.Fatalf("as retriever: %v", err)
	}
	keyword := funcRetriever(func(_ context.Context, query string) ([]documents.Document, error) {
		return []documents.Document{
			documents.New(query+" "+query, map[string]any{"source": "keyword"}),
		}, nil
	})
	ensemble, err := retrievers.NewEnsembleRetriever([]retrievers.Retriever{dense, keyword})
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}

	model := language.NewFakeChatModel(
		language.WithResponses(messages.AI("alpha\nalpha alpha")),
	)
	multiQuery, err := retrievers.NewMultiQueryRetriever(ensemble, retrievers.WithQueryModel(model))
	if err != nil {
		t.Fatalf("new multi-query retriever: %v", err)
	}

	compressor := &truncatingCompressor{keep: 2}
	chain, err := retrievers.NewContextualCompressionRetriever(multiQuery, compressor)
	if err != nil {
		t.Fatalf("new compression retriever: %v", err)
	}

	docs, err := chain.GetRelevantDocuments(t.Context(), "alpha")
	if err != nil {
		t.Fatalf("retrieve through chain: %v", err)
	}

	if compressor.calls != 1 {
		t.Fatalf("compressor calls: got %d want 1", compressor.calls)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: got %d want 2 (truncated by compressor)", len(docs))
	}
	for _, doc := range docs {
		if doc.Metadata["compressed"] != true {
			t.Fatalf("compressed marker missing: %v", doc.Metadata)
		}
	}
}
