package retrievers

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

// fnRetriever adapts a function to the Retriever interface for tests.
type fnRetriever func(context.Context, string) ([]documents.Document, error)

func (f fnRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	return f(ctx, query)
}

// recordingCompressor records calls and returns a fixed compressed slice so
// tests can assert whether compression ran and what it produced.
type recordingCompressor struct {
	calls   int
	lastArg []documents.Document
	out     []documents.Document
	err     error
}

func (c *recordingCompressor) CompressDocuments(
	_ context.Context,
	docs []documents.Document,
	_ string,
) ([]documents.Document, error) {
	c.calls++
	c.lastArg = docs
	if c.err != nil {
		return nil, c.err
	}
	return c.out, nil
}

func TestContextualCompressionRetrieverCompressesRetrievedDocs(t *testing.T) {
	base := Static{Documents: []documents.Document{
		documents.New("first", nil),
		documents.New("second", nil),
		documents.New("third", nil),
	}}
	// Python parity: compressors such as CohereRerank reorder the retrieved
	// documents by relevance and annotate relevance_score metadata
	// (cohere_rerank.py compress_documents).
	compressor := &recordingCompressor{
		out: []documents.Document{
			documents.New("third", map[string]any{"relevance_score": 0.9}),
			documents.New("first", map[string]any{"relevance_score": 0.4}),
		},
	}

	retriever, err := NewContextualCompressionRetriever(base, compressor)
	if err != nil {
		t.Fatalf("new compression retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 2 || docs[0].PageContent != "third" || docs[1].PageContent != "first" {
		t.Fatalf("compressed docs: got %v", docs)
	}
	if compressor.calls != 1 {
		t.Fatalf("compressor calls: got %d want 1", compressor.calls)
	}
	if len(compressor.lastArg) != 3 {
		t.Fatalf("compressor received %d docs, want the 3 retrieved docs", len(compressor.lastArg))
	}
	if got, ok := docs[0].Metadata["relevance_score"]; !ok || got != 0.9 {
		t.Fatalf("relevance score metadata missing: %v", docs[0].Metadata)
	}
}

// Python parity: ContextualCompressionRetriever returns [] without calling
// the compressor when the base retriever finds nothing
// (contextual_compression.py: `if docs: ... return []`).
func TestContextualCompressionRetrieverEmptyResultSkipsCompressor(t *testing.T) {
	base := Static{}
	compressor := &recordingCompressor{out: []documents.Document{documents.New("x", nil)}}

	retriever, err := NewContextualCompressionRetriever(base, compressor)
	if err != nil {
		t.Fatalf("new compression retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs: got %d want 0", len(docs))
	}
	if compressor.calls != 0 {
		t.Fatalf("compressor must not run for empty retrieval, ran %d times", compressor.calls)
	}
}

func TestContextualCompressionRetrieverErrors(t *testing.T) {
	t.Run("base retriever error propagates", func(t *testing.T) {
		baseErr := errors.New("boom")
		retriever, err := NewContextualCompressionRetriever(
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return nil, baseErr
			}),
			&recordingCompressor{},
		)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, baseErr) {
			t.Fatalf("want base error, got %v", err)
		}
	})

	t.Run("compressor error propagates", func(t *testing.T) {
		compressErr := errors.New("rerank failed")
		retriever, err := NewContextualCompressionRetriever(
			Static{Documents: []documents.Document{documents.New("a", nil)}},
			&recordingCompressor{err: compressErr},
		)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, compressErr) {
			t.Fatalf("want compressor error, got %v", err)
		}
	})

	t.Run("constructor requires base and compressor", func(t *testing.T) {
		if _, err := NewContextualCompressionRetriever(nil, &recordingCompressor{}); err == nil {
			t.Fatal("nil base retriever must error")
		}
		if _, err := NewContextualCompressionRetriever(Static{}, nil); err == nil {
			t.Fatal("nil compressor must error")
		}
	})

	t.Run("canceled context", func(t *testing.T) {
		retriever, err := NewContextualCompressionRetriever(Static{}, &recordingCompressor{})
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := retriever.GetRelevantDocuments(ctx, "q"); !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	})
}
