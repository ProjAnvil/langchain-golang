package retrievers

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/projanvil/langchain-golang/core/documents"
)

// ensembleRankConstant is the reciprocal-rank-fusion constant, mirroring
// Python's EnsembleRetriever rank_constant c = 60: each member's contribution
// to a document's fused score is weight / (rankConstant + rank) with a
// 1-based rank (Python: enumerate(doc_list, start=1)), so the top-ranked
// document contributes weight / 61.
const ensembleRankConstant = 60

// ensembleIDKey mirrors Python's EnsembleRetriever id_key ("id"): the
// metadata key identifying a document across member retrievers. Documents
// without it fall back to page content.
const ensembleIDKey = "id"

// EnsembleRetriever fuses the ranked results of several retrievers (e.g. a
// dense vector retriever and a sparse keyword retriever) with reciprocal rank
// fusion: score(doc) = sum over members of weight / (60 + rank), where rank
// is the document's 1-based position in that member's result list. Documents
// are deduplicated by their id metadata (or page content), the first-seen
// document wins, and the output is sorted by fused score descending with ties
// keeping first-seen order.
//
// Python parity: langchain_community.retrievers.EnsembleRetriever
// (rank_fusion; equal weights default; the fused score is recorded in the
// output metadata under "score").
type EnsembleRetriever struct {
	retrievers []Retriever
	weights    []float64
}

// EnsembleOption configures an EnsembleRetriever.
type EnsembleOption func(*EnsembleRetriever)

// WithWeights sets the per-retriever fusion weights (default: equal weights).
// The slice length must match the retriever count and weights must be
// non-negative.
func WithWeights(weights []float64) EnsembleOption {
	return func(r *EnsembleRetriever) {
		r.weights = weights
	}
}

// NewEnsembleRetriever creates a retriever fusing the ranked results of the
// given retrievers. At least one retriever is required; nil members are
// rejected.
func NewEnsembleRetriever(
	retrievers []Retriever,
	opts ...EnsembleOption,
) (EnsembleRetriever, error) {
	if len(retrievers) == 0 {
		return EnsembleRetriever{}, fmt.Errorf("at least one retriever is required")
	}
	for i, member := range retrievers {
		if member == nil {
			return EnsembleRetriever{}, fmt.Errorf("retriever at index %d is nil", i)
		}
	}
	retriever := EnsembleRetriever{
		retrievers: slices.Clone(retrievers),
		weights:    equalWeights(len(retrievers)),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&retriever)
		}
	}
	if len(retriever.weights) != len(retriever.retrievers) {
		return EnsembleRetriever{}, fmt.Errorf(
			"got %d weights for %d retrievers",
			len(retriever.weights),
			len(retriever.retrievers),
		)
	}
	for i, weight := range retriever.weights {
		if math.IsNaN(weight) || weight < 0 {
			return EnsembleRetriever{}, fmt.Errorf("weight at index %d must be non-negative, got %v", i, weight)
		}
	}
	return retriever, nil
}

var _ Retriever = EnsembleRetriever{}

// GetRelevantDocuments queries every member retriever for the query, fuses
// their ranked lists with reciprocal rank fusion, and returns the deduplicated
// documents ordered by fused score.
func (r EnsembleRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	type fused struct {
		doc   documents.Document
		score float64
	}
	byKey := make(map[string]int)
	fusedDocs := make([]fused, 0)

	for i, member := range r.retrievers {
		docs, err := member.GetRelevantDocuments(ctx, query)
		if err != nil {
			return nil, err
		}
		for rank, doc := range docs {
			key := ensembleDocKey(doc)
			index, seen := byKey[key]
			if !seen {
				index = len(fusedDocs)
				byKey[key] = index
				fusedDocs = append(fusedDocs, fused{doc: doc.Clone()})
			}
			fusedDocs[index].score += r.weights[i] / float64(ensembleRankConstant+rank+1)
		}
	}

	// Stable descending sort keeps first-seen order on score ties, matching
	// Python's stable sort over the fused score dict.
	slices.SortStableFunc(fusedDocs, func(a, b fused) int {
		return cmp.Compare(b.score, a.score)
	})

	out := make([]documents.Document, len(fusedDocs))
	for i, entry := range fusedDocs {
		if entry.doc.Metadata == nil {
			entry.doc.Metadata = map[string]any{}
		}
		entry.doc.Metadata["score"] = entry.score
		out[i] = entry.doc
	}
	return out, nil
}

// ensembleDocKey identifies a document across member retrievers: the "id"
// metadata when present (stringified for non-string values, mirroring
// Python's use of the raw metadata value), otherwise the page content.
func ensembleDocKey(doc documents.Document) string {
	if value, ok := doc.Metadata[ensembleIDKey]; ok && value != nil {
		if id, isString := value.(string); isString {
			return id
		}
		return fmt.Sprint(value)
	}
	return doc.PageContent
}

func equalWeights(count int) []float64 {
	weights := make([]float64, count)
	for i := range weights {
		weights[i] = 1.0 / float64(count)
	}
	return weights
}
