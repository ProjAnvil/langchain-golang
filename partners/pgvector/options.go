package pgvector

import (
	"context"
	"fmt"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// SimilaritySearchWithOptions implements vectorstores.OptionSearcher: the
// declarative filter is translated to parameterized jsonb SQL and applied
// server-side. ScoreThreshold is applied client-side after converting the
// pgvector distance to a [0, 1] relevance score with the helper selected for
// the configured distance metric — the same strategy langchain-postgres uses
// for its score_threshold filtering.
func (s *Store) SimilaritySearchWithOptions(
	ctx context.Context,
	query string,
	opts vectorstores.SearchOptions,
) ([]documents.Document, error) {
	if err := vectorstores.ValidateFilter(opts.Filter); err != nil {
		return nil, err
	}
	k := opts.K
	if k <= 0 {
		k = 4
	}
	sql, args, err := s.searchWithFilterSQL(ctx, query, opts.Filter, false)
	if err != nil {
		return nil, err
	}
	results, err := s.queryResults(ctx, sql, args, k)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, 0, len(results))
	for _, result := range results {
		if opts.ScoreThreshold > 0 && s.relevanceScore(result.Score) < opts.ScoreThreshold {
			continue
		}
		docs = append(docs, result.Document)
	}
	return docs, nil
}

// MMRSearchWithOptions implements vectorstores.OptionSearcher: FetchK
// candidates (with their stored embeddings) are fetched under the translated
// filter, then re-ranked client-side with maximal marginal relevance. lambda
// defaults to the Python SearchArgs value of 0.5.
func (s *Store) MMRSearchWithOptions(
	ctx context.Context,
	query string,
	opts vectorstores.SearchOptions,
) ([]documents.Document, error) {
	if err := vectorstores.ValidateFilter(opts.Filter); err != nil {
		return nil, err
	}
	k := opts.K
	if k <= 0 {
		k = 4
	}
	fetchK := opts.FetchK
	if fetchK <= 0 {
		fetchK = 20
	}

	queryVector, err := s.embedQuery(ctx, query)
	if err != nil {
		return nil, err
	}
	// Build the candidate query straight from queryVector: going through
	// searchWithFilterSQL would embed the query a second time.
	sql, args, err := s.searchWithFilterSQLByVector(queryVector, opts.Filter, true)
	if err != nil {
		return nil, err
	}
	args = append(args, fetchK)
	sql = fmt.Sprintf("%s LIMIT $%d", sql, len(args))

	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type candidate struct {
		doc       documents.Document
		embedding []float64
	}
	candidates := make([]candidate, 0, fetchK)
	for rows.Next() {
		var (
			id        string
			content   string
			rawMeta   []byte
			rawVector string
			distance  float64
		)
		if err := rows.Scan(&id, &content, &rawMeta, &rawVector, &distance); err != nil {
			return nil, err
		}
		embedding, err := parseVector(rawVector)
		if err != nil {
			return nil, err
		}
		candidates = append(candidates, candidate{
			doc: documents.Document{
				ID:          id,
				PageContent: content,
				Metadata:    decodeMetadata(rawMeta),
			},
			embedding: embedding,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	embeddings := make([][]float64, len(candidates))
	for i, item := range candidates {
		embeddings[i] = item.embedding
	}
	selected := vectorstores.MaximalMarginalRelevance(queryVector, embeddings, 0.5, k)
	docs := make([]documents.Document, 0, len(selected))
	for _, index := range selected {
		if index >= 0 && index < len(candidates) {
			docs = append(docs, candidates[index].doc)
		}
	}
	return docs, nil
}
