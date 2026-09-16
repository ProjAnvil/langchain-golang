package chroma

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"slices"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

var _ vectorstores.OptionSearcher = (*Store)(nil)

// Chroma where-clause support for the shared declarative filter DSL
// (core/vectorstores/filter.go, operator semantics per langchain-postgres
// SearchArgs.filter):
//
//   - $eq/$ne/$gt/$gte/$lt/$lte/$in/$nin map to the same-named chroma where
//     operators.
//   - $between [low, high] decomposes into "$gte low AND $lte high" (both
//     bounds inclusive, matching the langchain-postgres translation).
//   - $exists has NO chroma mapping: the chroma where grammar has no
//     key-existence operator. Chroma's $contains is a where_document operator
//     that matches document text, not metadata, so it cannot express $exists.
//   - $like has NO chroma mapping: chroma metadata filtering has no
//     LIKE/wildcard operator ($contains again applies to document text only).
//
// Unsupported operators fail loudly instead of being dropped silently.
const chromaUnsupportedFilterOperators = "$exists, $like"

// SimilaritySearchWithOptions implements vectorstores.OptionSearcher: the
// declarative filter is translated to a chroma where clause and executed
// server-side. ScoreThreshold is applied client-side to the returned chroma
// distances via the cosine relevance helper; this assumes a cosine-space
// collection (configure WithCollectionConfiguration with hnsw space "cosine"),
// mirroring how langchain-chroma selects relevance score functions.
func (s *Store) SimilaritySearchWithOptions(
	ctx context.Context,
	query string,
	opts vectorstores.SearchOptions,
) ([]documents.Document, error) {
	where, err := translateFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	results, err := s.SimilaritySearchWithScoreOptions(ctx, query, QueryOptions{
		K:     opts.K,
		Where: where,
	})
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, 0, len(results))
	for _, result := range results {
		if opts.ScoreThreshold > 0 &&
			vectorstores.CosineRelevanceScore(result.Score) < opts.ScoreThreshold {
			continue
		}
		docs = append(docs, result.Document)
	}
	return docs, nil
}

// MMRSearchWithOptions implements vectorstores.OptionSearcher: the translated
// filter restricts the server-side candidate prefetch. SearchOptions carries
// no lambda_mult, so the Python SearchArgs default of 0.5 is applied.
func (s *Store) MMRSearchWithOptions(
	ctx context.Context,
	query string,
	opts vectorstores.SearchOptions,
) ([]documents.Document, error) {
	where, err := translateFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	return s.MaxMarginalRelevanceSearch(ctx, query, MMROptions{
		K:          opts.K,
		FetchK:     opts.FetchK,
		LambdaMult: 0.5,
		Where:      where,
	})
}

// translateFilter converts the shared declarative filter DSL into a chroma
// where clause. Multiple conditions are joined with chroma's $and (the where
// grammar allows at most one key per dict unless it is $and/$or).
func translateFilter(filter map[string]any) (map[string]any, error) {
	if err := vectorstores.ValidateFilter(filter); err != nil {
		return nil, err
	}
	if len(filter) == 0 {
		return nil, nil
	}

	conditions := make([]map[string]any, 0, len(filter))
	for _, field := range slices.Sorted(maps.Keys(filter)) {
		operator, operand := splitCondition(filter[field])
		switch operator {
		case vectorstores.FilterBetween:
			bounds, ok := anySlice(operand)
			if !ok || len(bounds) != 2 {
				// Unreachable for filters that passed ValidateFilter.
				return nil, fmt.Errorf("invalid %s bounds for field %q", vectorstores.FilterBetween, field)
			}
			conditions = append(conditions,
				map[string]any{field: map[string]any{vectorstores.FilterGte: bounds[0]}},
				map[string]any{field: map[string]any{vectorstores.FilterLte: bounds[1]}},
			)
		case vectorstores.FilterExists, vectorstores.FilterLike:
			return nil, fmt.Errorf(
				"chroma where clauses do not support the %s operator (unsupported: %s)",
				operator,
				chromaUnsupportedFilterOperators,
			)
		default:
			conditions = append(conditions, map[string]any{field: map[string]any{operator: operand}})
		}
	}
	if len(conditions) == 1 {
		return conditions[0], nil
	}
	return map[string]any{"$and": conditions}, nil
}

// splitCondition normalizes an already-validated condition into its operator
// and operand; literal values are the equality shorthand.
func splitCondition(condition any) (string, any) {
	if operators, ok := condition.(map[string]any); ok {
		for operator, operand := range operators {
			return operator, operand
		}
	}
	return vectorstores.FilterEq, condition
}

// anySlice widens []any and typed scalar slices to []any so $between bounds
// written as natural Go literals survive translation.
func anySlice(value any) ([]any, bool) {
	if items, ok := value.([]any); ok {
		return items, true
	}
	reflected := reflect.ValueOf(value)
	if reflected.Kind() != reflect.Slice && reflected.Kind() != reflect.Array {
		return nil, false
	}
	out := make([]any, reflected.Len())
	for i := range out {
		out[i] = reflected.Index(i).Interface()
	}
	return out, true
}
