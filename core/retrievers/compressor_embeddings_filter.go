package retrievers

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
)

// EmbeddingsFilter is a DocumentCompressor that drops documents unrelated to
// the query by embedding both the query and the documents and keeping only the
// most similar ones. A cosine similarity is computed for every document, then
// the top k (by similarity) are optionally selected and an optional
// similarity threshold is applied (strictly greater than). Selected documents
// are annotated with a "query_similarity_score" metadata entry.
//
// Python parity: langchain_classic.retrievers.document_compressors.
// embeddings_filter.EmbeddingsFilter (k defaults to 20, similarity_threshold
// to None, and at least one of the two must be specified; the score is stored
// in the stateful document's state["query_similarity_score"]).
type EmbeddingsFilter struct {
	embeddings embeddings.Embeddings
	k          int
	threshold  *float64
}

// EmbeddingsFilterOption configures an EmbeddingsFilter.
type EmbeddingsFilterOption func(*EmbeddingsFilter)

// WithEmbeddingsFilterK sets the number of top documents to keep. A value <= 0
// disables the top-k cap, in which case a similarity threshold must be set
// (Python: k=None). The default is 20.
func WithEmbeddingsFilterK(k int) EmbeddingsFilterOption {
	return func(f *EmbeddingsFilter) {
		if k > 0 {
			f.k = k
		} else {
			f.k = 0
		}
	}
}

// WithEmbeddingsFilterSimilarityThreshold sets the minimum cosine similarity a
// document must exceed to be kept.
func WithEmbeddingsFilterSimilarityThreshold(threshold float64) EmbeddingsFilterOption {
	return func(f *EmbeddingsFilter) {
		t := threshold
		f.threshold = &t
	}
}

// NewEmbeddingsFilter creates an embeddings-based filter using embedder. The
// embedder is required. At least one of the top-k cap (default 20) or a
// similarity threshold must be active.
func NewEmbeddingsFilter(
	embedder embeddings.Embeddings,
	opts ...EmbeddingsFilterOption,
) (EmbeddingsFilter, error) {
	if embedder == nil {
		return EmbeddingsFilter{}, fmt.Errorf("embeddings model is required")
	}
	filter := EmbeddingsFilter{embeddings: embedder, k: 20}
	for _, opt := range opts {
		if opt != nil {
			opt(&filter)
		}
	}
	if filter.k <= 0 && filter.threshold == nil {
		return EmbeddingsFilter{}, fmt.Errorf(
			"must specify one of k or similarity_threshold",
		)
	}
	return filter, nil
}

var _ DocumentCompressor = EmbeddingsFilter{}

// CompressDocuments computes cosine similarities to the query and returns the
// surviving documents. When a top-k cap is active the result is ordered by
// descending similarity; otherwise the documents keep their input order.
func (f EmbeddingsFilter) CompressDocuments(
	ctx context.Context,
	docs []documents.Document,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(docs) == 0 {
		return []documents.Document{}, nil
	}

	queryVector, err := f.embeddings.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	texts := make([]string, len(docs))
	for i, doc := range docs {
		texts[i] = doc.PageContent
	}
	docVectors, err := f.embeddings.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("embed documents: %w", err)
	}
	if len(docVectors) != len(docs) {
		return nil, fmt.Errorf("embedding count mismatch: got %d want %d", len(docVectors), len(docs))
	}
	similarities := make([]float64, len(docs))
	for i, vector := range docVectors {
		similarities[i] = embeddingsFilterCosine(queryVector, vector)
	}

	indices := make([]int, len(docs))
	for i := range indices {
		indices[i] = i
	}
	if f.k > 0 {
		// Sort by descending similarity; the original index breaks ties so the
		// outcome is deterministic.
		slices.SortStableFunc(indices, func(a, b int) int {
			if similarities[a] != similarities[b] {
				if similarities[a] > similarities[b] {
					return -1
				}
				return 1
			}
			return a - b
		})
		if len(indices) > f.k {
			indices = indices[:f.k]
		}
	}
	if f.threshold != nil {
		kept := make([]int, 0, len(indices))
		for _, idx := range indices {
			if similarities[idx] > *f.threshold {
				kept = append(kept, idx)
			}
		}
		indices = kept
	}

	out := make([]documents.Document, 0, len(indices))
	for _, idx := range indices {
		doc := docs[idx].Clone()
		if doc.Metadata == nil {
			doc.Metadata = map[string]any{}
		}
		doc.Metadata["query_similarity_score"] = similarities[idx]
		out = append(out, doc)
	}
	return out, nil
}

// embeddingsFilterCosine returns the cosine similarity of two vectors. Zero
// vectors and vectors of different length score 0.
func embeddingsFilterCosine(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
