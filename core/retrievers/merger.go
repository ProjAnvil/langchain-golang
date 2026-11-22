package retrievers

import (
	"context"
	"fmt"
	"slices"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// MergerRetriever merges the results of several retrievers in round-robin
// (interleaved) order: it takes the first document from each retriever, then
// the second from each, and so on, until every result list is exhausted. It
// performs no deduplication, no scoring, and no reordering beyond the
// round-robin interleaving.
//
// Python parity: langchain_classic.retrievers.merger_retriever.MergerRetriever
// (merge_documents: `for i in range(max_docs): for doc in retriever_docs: ...`).
// The Go form retrieves from every member concurrently (via
// runnables.ParallelMap) while keeping the member result lists in input order,
// so the merged output is deterministic regardless of completion order.
type MergerRetriever struct {
	retrievers []Retriever
}

// NewMergerRetriever creates a retriever that round-robin merges the results
// of the given retrievers. At least one retriever is required and nil members
// are rejected.
func NewMergerRetriever(retrievers []Retriever) (MergerRetriever, error) {
	if len(retrievers) == 0 {
		return MergerRetriever{}, fmt.Errorf("at least one retriever is required")
	}
	for i, member := range retrievers {
		if member == nil {
			return MergerRetriever{}, fmt.Errorf("retriever at index %d is nil", i)
		}
	}
	return MergerRetriever{retrievers: slices.Clone(retrievers)}, nil
}

var _ Retriever = MergerRetriever{}

// GetRelevantDocuments retrieves from every member concurrently and returns
// their documents interleaved in round-robin order. An error from any member
// aborts the merge and is returned.
func (r MergerRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	results, err := runnables.ParallelMap(
		ctx,
		runnables.NewConfig(),
		r.retrievers,
		func(ctx context.Context, member Retriever) ([]documents.Document, error) {
			return member.GetRelevantDocuments(ctx, query)
		},
	)
	if err != nil {
		return nil, err
	}

	maxDocs := 0
	for _, docs := range results {
		if len(docs) > maxDocs {
			maxDocs = len(docs)
		}
	}
	merged := make([]documents.Document, 0)
	for i := 0; i < maxDocs; i++ {
		for _, docs := range results {
			if i < len(docs) {
				merged = append(merged, docs[i].Clone())
			}
		}
	}
	return merged, nil
}
