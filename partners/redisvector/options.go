package redisvector

import (
	"context"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Declarative filter DSL support (core/vectorstores/filter.go, operator
// semantics per langchain-postgres SearchArgs.filter) mapped onto RediSearch
// query clauses. RediSearch filters address schema attributes — FT.SEARCH
// does not fully support JSONPath expressions in queries — so a filter field
// must be declared with WithMetadataField:
//
//	numeric field (MetadataNumeric):
//	  $eq n            -> @f:[n n]
//	  $ne n            -> -@f:[n n] -ismissing(@f)
//	  $gt/$gte/$lt/$lte-> @f:[(n +inf] / @f:[n +inf] / @f:[-inf (n] / @f:[-inf n]
//	  $in [a b]        -> (@f:[a a]|@f:[b b])
//	  $nin [a b]       -> -(@f:[a a]|@f:[b b]) -ismissing(@f)
//	  $between [lo hi] -> @f:[lo hi]
//	tag field (MetadataTag; strings and booleans):
//	  $eq v            -> @f:{v}
//	  $ne v            -> -@f:{v} -ismissing(@f)
//	  $in [a b]        -> @f:{a|b}
//	  $nin [a b]       -> -@f:{a|b} -ismissing(@f)
//	both:
//	  $exists true     -> -ismissing(@f)
//	  $exists false    -> ismissing(@f)
//
// ismissing() requires DIALECT 2 plus the INDEXMISSING flag the index schema
// sets on every declared metadata field, and mirrors the shared DSL
// semantics: absent fields never match $ne/$nin.
//
// Explicitly unsupported (fail loudly rather than silently mis-filter):
//   - $like: RediSearch tag clauses only support trailing-prefix wildcards
//     (`@f:{pre*}`); SQL LIKE patterns (%anywhere%, _one char) cannot be
//     expressed, so $like errors.
//   - string range predicates ($gt/$gte/$lt/$lte/$between on tag fields):
//     TAG attributes have no lexicographic ranges.
//   - operands whose type does not match the declared field type (string on
//     numeric, number on tag).
//   - undeclared metadata fields: the value is not indexed, so any query
//     would silently match nothing or everything.
const redisvectorUnsupportedFilterOperators = "$like, string range predicates on tag fields"

// SimilaritySearchWithOptions implements vectorstores.OptionSearcher: the
// declarative filter becomes a hybrid KNN pre-filter
// `(<filter>)=>[KNN k @embedding $BLOB AS __embedding_score]`. ScoreThreshold
// is applied client-side after converting the RediSearch distance to a
// relevance score for the configured metric.
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
	hits, err := s.knnSearch(ctx, query, opts.Filter, k)
	if err != nil {
		return nil, err
	}
	docs := make([]documents.Document, 0, len(hits))
	for _, hit := range hits {
		if opts.ScoreThreshold > 0 && s.relevanceScore(hit.score) < opts.ScoreThreshold {
			continue
		}
		docs = append(docs, hit.doc)
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
	hits, err := s.knnSearch(ctx, query, opts.Filter, fetchK)
	if err != nil {
		return nil, err
	}

	embeddings := make([][]float64, len(hits))
	for i, hit := range hits {
		embeddings[i] = hit.embedding
	}
	selected := vectorstores.MaximalMarginalRelevance(queryVector, embeddings, 0.5, k)
	docs := make([]documents.Document, 0, len(selected))
	for _, index := range selected {
		if index >= 0 && index < len(hits) {
			docs = append(docs, hits[index].doc)
		}
	}
	return docs, nil
}

// DeleteWithFilter removes every document matching the declarative filter:
// FT.SEARCH with NOCONTENT resolves the matching ids, then DEL removes them.
func (s *Store) DeleteWithFilter(ctx context.Context, filter map[string]any) error {
	filterClause, err := s.buildFilterQuery(filter)
	if err != nil {
		return err
	}
	if filterClause == "" {
		return fmt.Errorf("refusing to delete with an empty filter")
	}

	totalReply := s.client.Do(ctx, "FT.SEARCH", s.indexName, filterClause,
		"NOCONTENT", "LIMIT", 0, 0, "DIALECT", 2)
	if err := totalReply.Err(); err != nil {
		return err
	}
	totalRaw, err := totalReply.Result()
	if err != nil {
		return err
	}
	total, ok := replyFirstInt(totalRaw)
	if !ok {
		return fmt.Errorf("unexpected FT.SEARCH count reply: %#v", totalRaw)
	}
	if total == 0 {
		return nil
	}

	idsReply := s.client.Do(ctx, "FT.SEARCH", s.indexName, filterClause,
		"NOCONTENT", "LIMIT", 0, total, "DIALECT", 2)
	if err := idsReply.Err(); err != nil {
		return err
	}
	raw, err := idsReply.Result()
	if err != nil {
		return err
	}
	hits, err := parseSearchReply(raw, s.keyPrefix)
	if err != nil {
		return err
	}
	if len(hits) == 0 {
		return nil
	}

	keys := make([]any, 0, len(hits)+1)
	keys = append(keys, "DEL")
	for _, hit := range hits {
		keys = append(keys, s.keyPrefix+hit.doc.ID)
	}
	return s.client.Do(ctx, keys...).Err()
}

// buildFilterQuery translates a validated declarative filter into a
// RediSearch query clause (top-level conditions ANDed). An empty filter
// yields an empty clause.
func (s *Store) buildFilterQuery(filter map[string]any) (string, error) {
	if err := vectorstores.ValidateFilter(filter); err != nil {
		return "", err
	}
	if len(filter) == 0 {
		return "", nil
	}

	clauses := make([]string, 0, len(filter))
	for _, field := range slices.Sorted(maps.Keys(filter)) {
		clause, err := s.buildConditionClause(field, filter[field])
		if err != nil {
			return "", err
		}
		clauses = append(clauses, clause)
	}
	if len(clauses) == 1 {
		// A lone clause needs no grouping, mirroring the chroma filter
		// translation convention.
		return clauses[0], nil
	}
	wrapped := make([]string, len(clauses))
	for i, clause := range clauses {
		wrapped[i] = "(" + clause + ")"
	}
	return strings.Join(wrapped, " "), nil
}

func (s *Store) buildConditionClause(field string, condition any) (string, error) {
	declared, ok := s.metadataFields[field]
	if !ok {
		return "", fmt.Errorf(
			"metadata field %q is not declared for filtering: pass WithMetadataField(%q, MetadataTag|MetadataNumeric) when creating the store",
			field, field,
		)
	}
	operator, operand := splitCondition(condition)

	switch operator {
	case vectorstores.FilterExists:
		existence := "ismissing(@" + field + ")"
		if operand == true {
			existence = "-" + existence
		}
		return existence, nil
	case vectorstores.FilterLike:
		return "", fmt.Errorf(
			"RediSearch does not support the %s operator (unsupported: %s)",
			vectorstores.FilterLike, redisvectorUnsupportedFilterOperators,
		)
	}

	if declared == MetadataNumeric {
		return numericCondition(field, operator, operand)
	}
	return tagCondition(field, operator, operand)
}

// numericCondition renders predicates for NUMERIC attributes as inclusive
// ranges with "(" marking exclusive bounds, per RediSearch range syntax.
func numericCondition(field, operator string, operand any) (string, error) {
	// List operators validate their items individually.
	switch operator {
	case vectorstores.FilterBetween:
		bounds, err := betweenBounds(operand)
		if err != nil {
			return "", fmt.Errorf("field %q: %w", field, err)
		}
		return "@" + field + ":[" + bounds[0] + " " + bounds[1] + "]", nil
	case vectorstores.FilterIn:
		return numericMembership(field, operand, false)
	case vectorstores.FilterNin:
		return numericMembership(field, operand, true)
	}

	value, ok := numericValue(operand)
	if !ok {
		return "", fmt.Errorf(
			"metadata field %q is declared numeric but the %s operand is %T; declare it WithMetadataField(%q, MetadataTag) for string values",
			field, operator, operand, field,
		)
	}
	number := strconv.FormatFloat(value, 'g', -1, 64)
	switch operator {
	case vectorstores.FilterEq:
		return "@" + field + ":[" + number + " " + number + "]", nil
	case vectorstores.FilterNe:
		return "-@" + field + ":[" + number + " " + number + "] -ismissing(@" + field + ")", nil
	case vectorstores.FilterGt:
		return "@" + field + ":[(" + number + " +inf]", nil
	case vectorstores.FilterGte:
		return "@" + field + ":[" + number + " +inf]", nil
	case vectorstores.FilterLt:
		return "@" + field + ":[-inf (" + number + "]", nil
	case vectorstores.FilterLte:
		return "@" + field + ":[-inf " + number + "]", nil
	default:
		return "", fmt.Errorf("unsupported filter operator: %s", operator)
	}
}

// numericMembership renders $in/$nin as unions of point ranges.
func numericMembership(field string, operand any, negate bool) (string, error) {
	items, ok := toAnySlice(operand)
	if !ok || len(items) == 0 {
		return "", fmt.Errorf(
			"invalid membership operand for field %q: expected a non-empty list", field,
		)
	}
	predicates := make([]string, 0, len(items))
	for _, item := range items {
		value, isNumeric := numericValue(item)
		if !isNumeric {
			return "", fmt.Errorf(
				"metadata field %q is declared numeric but the list contains a %T", field, item,
			)
		}
		number := strconv.FormatFloat(value, 'g', -1, 64)
		predicates = append(predicates, "@"+field+":["+number+" "+number+"]")
	}
	joined := strings.Join(predicates, "|")
	if negate {
		return "-(" + joined + ") -ismissing(@" + field + ")", nil
	}
	return "(" + joined + ")", nil
}

// tagCondition renders predicates for TAG attributes; tag values escape the
// RediSearch tag specials (space, |, {, }, \).
func tagCondition(field, operator string, operand any) (string, error) {
	switch operator {
	case vectorstores.FilterEq:
		value, err := tagValue(operand)
		if err != nil {
			return "", fmt.Errorf("field %q: %w", field, err)
		}
		return "@" + field + ":{" + value + "}", nil
	case vectorstores.FilterNe:
		value, err := tagValue(operand)
		if err != nil {
			return "", fmt.Errorf("field %q: %w", field, err)
		}
		return "-@" + field + ":{" + value + "} -ismissing(@" + field + ")", nil
	case vectorstores.FilterIn, vectorstores.FilterNin:
		items, ok := toAnySlice(operand)
		if !ok || len(items) == 0 {
			return "", fmt.Errorf(
				"invalid membership operand for field %q: expected a non-empty list", field,
			)
		}
		values := make([]string, 0, len(items))
		for _, item := range items {
			value, err := tagValue(item)
			if err != nil {
				return "", fmt.Errorf("field %q: %w", field, err)
			}
			values = append(values, value)
		}
		joined := "@" + field + ":{" + strings.Join(values, "|") + "}"
		if operator == vectorstores.FilterNin {
			return "-" + joined + " -ismissing(@" + field + ")", nil
		}
		return joined, nil
	case vectorstores.FilterGt, vectorstores.FilterGte,
		vectorstores.FilterLt, vectorstores.FilterLte, vectorstores.FilterBetween:
		return "", fmt.Errorf(
			"metadata field %q is declared tag but %s needs a range: RediSearch TAG attributes have no lexicographic ranges (unsupported: %s)",
			field, operator, redisvectorUnsupportedFilterOperators,
		)
	default:
		return "", fmt.Errorf("unsupported filter operator: %s", operator)
	}
}

// tagValue renders one tag operand: strings and booleans map to their tag
// text with specials escaped; numbers are rejected because the field was
// declared tag.
func tagValue(operand any) (string, error) {
	switch typed := operand.(type) {
	case string:
		return escapeTag(typed), nil
	case bool:
		return strconv.FormatBool(typed), nil
	default:
		if _, isNumeric := numericValue(operand); isNumeric {
			return "", fmt.Errorf(
				"operand is numeric but the field is declared tag; declare it WithMetadataField(..., MetadataNumeric) for numbers",
			)
		}
		return "", fmt.Errorf("expected a string or boolean operand, got %T", operand)
	}
}

// betweenBounds validates a [low, high] numeric list and renders both
// numbers.
func betweenBounds(operand any) ([]string, error) {
	items, ok := toAnySlice(operand)
	if !ok || len(items) != 2 {
		return nil, fmt.Errorf("invalid %s operand: expected [low, high]", vectorstores.FilterBetween)
	}
	out := make([]string, 2)
	for i, item := range items {
		value, isNumeric := numericValue(item)
		if !isNumeric {
			return nil, fmt.Errorf(
				"invalid %s operand: bounds must be numeric on numeric fields", vectorstores.FilterBetween,
			)
		}
		out[i] = strconv.FormatFloat(value, 'g', -1, 64)
	}
	return out, nil
}

// replyFirstInt extracts the total count from an FT.SEARCH reply: a bare
// integer, a RESP2 array head ([total, ...]), or a RESP3 envelope map's
// "total_results".
func replyFirstInt(raw any) (int, bool) {
	switch typed := raw.(type) {
	case int64:
		return int(typed), true
	case []any:
		if len(typed) > 0 {
			if value, ok := typed[0].(int64); ok {
				return int(value), true
			}
		}
	case map[any]any:
		if value, ok := typed["total_results"].(int64); ok {
			return int(value), true
		}
	}
	return 0, false
}

// escapeTag backslash-escapes the characters RediSearch treats as tag
// separators or clause syntax (space included for pre-2.4 safety).
func escapeTag(value string) string {
	replacer := strings.NewReplacer(
		`\`, `\\`,
		`|`, `\|`,
		`{`, `\{`,
		`}`, `\}`,
		` `, `\ `,
	)
	return replacer.Replace(value)
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

// numericValue reports whether value is one of Go's numeric types.
func numericValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int8:
		return float64(typed), true
	case int16:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint8:
		return float64(typed), true
	case uint16:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case float32:
		return float64(typed), true
	case float64:
		return typed, true
	default:
		return 0, false
	}
}

// toAnySlice widens []any and typed scalar slices to []any so filters written
// with natural Go literals (for example []string{"a", "b"}) translate too.
func toAnySlice(value any) ([]any, bool) {
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
