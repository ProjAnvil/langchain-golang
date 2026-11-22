package retrievers

import (
	"context"
	"fmt"
	"slices"

	"github.com/projanvil/langchain-golang/core/documents"
)

// CrossEncoder scores query/document text pairs. It is the Go form of
// langchain_core.cross_encoders.BaseCrossEncoder (score(text_pairs)), with
// pairs represented as [2]string{query, documentContent}.
type CrossEncoder interface {
	Score(pairs [][2]string) ([]float64, error)
}

// CrossEncoderReranker is a DocumentCompressor that reranks documents with a
// CrossEncoder and keeps the top_n highest-scoring ones. It builds one
// (query, pageContent) pair per document, asks the model to score them, sorts
// by score descending, and returns the first top_n documents.
//
// Python parity: langchain_classic.retrievers.document_compressors.
// cross_encoder_rerank.CrossEncoderReranker (top_n defaults to 3).
type CrossEncoderReranker struct {
	model CrossEncoder
	topN  int
}

// CrossEncoderRerankerOption configures a CrossEncoderReranker.
type CrossEncoderRerankerOption func(*CrossEncoderReranker)

// WithCrossEncoderTopN sets the number of documents to return (default 3).
func WithCrossEncoderTopN(n int) CrossEncoderRerankerOption {
	return func(r *CrossEncoderReranker) {
		r.topN = n
	}
}

// NewCrossEncoderReranker creates a reranker using model. The model is
// required.
func NewCrossEncoderReranker(
	model CrossEncoder,
	opts ...CrossEncoderRerankerOption,
) (CrossEncoderReranker, error) {
	if model == nil {
		return CrossEncoderReranker{}, fmt.Errorf("cross encoder model is required")
	}
	reranker := CrossEncoderReranker{model: model, topN: 3}
	for _, opt := range opts {
		if opt != nil {
			opt(&reranker)
		}
	}
	return reranker, nil
}

var _ DocumentCompressor = CrossEncoderReranker{}

// CompressDocuments reranks documents by cross-encoder score and returns the
// top_n. Ties keep input order (stable sort).
func (r CrossEncoderReranker) CompressDocuments(
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

	pairs := make([][2]string, len(docs))
	for i, doc := range docs {
		pairs[i] = [2]string{query, doc.PageContent}
	}
	scores, err := r.model.Score(pairs)
	if err != nil {
		return nil, fmt.Errorf("score query/document pairs: %w", err)
	}
	if len(scores) != len(docs) {
		return nil, fmt.Errorf(
			"cross encoder returned %d scores for %d documents",
			len(scores), len(docs),
		)
	}

	type scored struct {
		idx   int
		score float64
	}
	ranked := make([]scored, len(docs))
	for i := range docs {
		ranked[i] = scored{idx: i, score: scores[i]}
	}
	slices.SortStableFunc(ranked, func(a, b scored) int {
		if a.score != b.score {
			if a.score > b.score {
				return -1
			}
			return 1
		}
		return a.idx - b.idx
	})

	limit := min(r.topN, len(ranked))
	if limit < 0 {
		limit = 0
	}
	out := make([]documents.Document, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, docs[ranked[i].idx].Clone())
	}
	return out, nil
}
