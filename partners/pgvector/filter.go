package pgvector

import (
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// buildFilterClause translates the shared declarative filter DSL
// (core/vectorstores/filter.go, langchain-postgres SearchArgs.filter baseline)
// into a parameterized jsonb SQL predicate joined by AND, appending every
// operand to args so callers can bind them as query parameters. It returns an
// empty clause for an empty filter.
//
// Every operand is bound as a parameter ($n placeholders); the only values
// interpolated into the SQL text are metadata field names, which are embedded
// as SQL string literals with single quotes doubled (see escapeSQLLiteral).
// Existence checks use the jsonb `?` operator, whose right-hand side is a
// bound parameter, so field names never need interpolation there.
func buildFilterClause(filter map[string]any, args *[]any) (string, error) {
	if err := vectorstores.ValidateFilter(filter); err != nil {
		return "", err
	}
	if len(filter) == 0 {
		return "", nil
	}

	clauses := make([]string, 0, len(filter))
	for _, field := range slices.Sorted(maps.Keys(filter)) {
		clause, err := buildConditionClause(field, filter[field], args)
		if err != nil {
			return "", err
		}
		clauses = append(clauses, "("+clause+")")
	}
	return strings.Join(clauses, " AND "), nil
}

func buildConditionClause(field string, condition any, args *[]any) (string, error) {
	operator, operand := splitCondition(condition)

	switch operator {
	case vectorstores.FilterEq:
		return equalityPredicate(field, operand, "=", args)
	case vectorstores.FilterNe:
		// A missing (or JSON-null) field yields SQL NULL for `<>`, which
		// never matches — the langchain-postgres $ne semantics.
		return equalityPredicate(field, operand, "<>", args)
	case vectorstores.FilterGt, vectorstores.FilterGte,
		vectorstores.FilterLt, vectorstores.FilterLte:
		return comparisonPredicate(field, operator, operand, args)
	case vectorstores.FilterIn:
		return inPredicate(field, operand, args, false)
	case vectorstores.FilterNin:
		// A missing (or JSON-null) field must not match $nin, so require
		// key existence alongside the negated membership tests. The field
		// placeholder is bound before the item operands to keep the clause
		// readable ($1 is always the field name).
		existence := "metadata ? " + appendPlaceholder(args, field)
		predicate, err := inPredicate(field, operand, args, true)
		if err != nil {
			return "", err
		}
		return existence + " AND " + predicate, nil
	case vectorstores.FilterBetween:
		return betweenPredicate(field, operand, args)
	case vectorstores.FilterExists:
		placeholder := appendPlaceholder(args, field)
		if operand == true {
			return "metadata ? " + placeholder, nil
		}
		return "NOT (metadata ? " + placeholder + ")", nil
	case vectorstores.FilterLike:
		pattern, ok := operand.(string)
		if !ok {
			return "", fmt.Errorf(
				"invalid %s operand for field %q: expected a string pattern, got %T",
				vectorstores.FilterLike, field, operand,
			)
		}
		return "metadata->>'" + escapeSQLLiteral(field) + "' LIKE " + appendPlaceholder(args, pattern), nil
	default:
		// Unreachable for filters that passed ValidateFilter.
		return "", fmt.Errorf("unsupported filter operator: %s", operator)
	}
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

// equalityPredicate renders $eq/$ne for one operand (symbol "=" or "<>").
// The binding shape follows the stored JSON type: text via ->>, numbers via
// ::numeric cast (so int 5 matches jsonb 5.0), booleans via their JSON text
// form, and nil via a jsonb null comparison.
func equalityPredicate(field string, operand any, symbol string, args *[]any) (string, error) {
	textPath := "metadata->>'" + escapeSQLLiteral(field) + "'"
	switch typed := operand.(type) {
	case nil:
		return "metadata->'" + escapeSQLLiteral(field) + "' " + symbol + " " +
			appendPlaceholder(args, "null") + "::jsonb", nil
	case string:
		return textPath + " " + symbol + " " + appendPlaceholder(args, typed), nil
	case bool:
		return textPath + " " + symbol + " " + appendPlaceholder(args, strconv.FormatBool(typed)), nil
	default:
		if _, isNumeric := numericOperand(operand); isNumeric {
			return "(" + textPath + ")::numeric " + symbol + " " + appendPlaceholder(args, operand), nil
		}
		return "", fmt.Errorf(
			"invalid equality operand for field %q: got %T", field, operand,
		)
	}
}

// comparisonPredicate renders $gt/$gte/$lt/$lte. Numeric operands compare via
// ::numeric; string operands compare lexicographically through ->>.
func comparisonPredicate(field, operator string, operand any, args *[]any) (string, error) {
	symbol, ok := map[string]string{
		vectorstores.FilterGt:  ">",
		vectorstores.FilterGte: ">=",
		vectorstores.FilterLt:  "<",
		vectorstores.FilterLte: "<=",
	}[operator]
	if !ok {
		return "", fmt.Errorf("unsupported comparison operator: %s", operator)
	}
	textPath := "metadata->>'" + escapeSQLLiteral(field) + "'"
	switch typed := operand.(type) {
	case string:
		return textPath + " " + symbol + " " + appendPlaceholder(args, typed), nil
	default:
		if _, isNumeric := numericOperand(operand); isNumeric {
			return "(" + textPath + ")::numeric " + symbol + " " + appendPlaceholder(args, operand), nil
		}
		return "", fmt.Errorf(
			"invalid %s operand for field %q: expected a string or number, got %T",
			operator, field, operand,
		)
	}
}

// inPredicate renders $in as a disjunction of equality predicates (and $nin
// as their negation). Per-item rendering keeps mixed string/numeric lists
// working without array-typed parameters.
func inPredicate(field string, operand any, args *[]any, negate bool) (string, error) {
	items, ok := toAnySlice(operand)
	if !ok || len(items) == 0 {
		return "", fmt.Errorf(
			"invalid membership operand for field %q: expected a non-empty list", field,
		)
	}
	predicates := make([]string, 0, len(items))
	for _, item := range items {
		predicate, err := equalityPredicate(field, item, "=", args)
		if err != nil {
			return "", err
		}
		predicates = append(predicates, predicate)
	}
	joined := "(" + strings.Join(predicates, " OR ") + ")"
	if negate {
		return "NOT " + joined, nil
	}
	return joined, nil
}

// betweenPredicate renders the inclusive [low, high] range as >= low AND <=
// high, matching the langchain-postgres translation.
func betweenPredicate(field string, operand any, args *[]any) (string, error) {
	bounds, ok := toAnySlice(operand)
	if !ok || len(bounds) != 2 {
		return "", fmt.Errorf(
			"invalid %s operand for field %q: expected [low, high]",
			vectorstores.FilterBetween, field,
		)
	}
	low, err := comparisonPredicate(field, vectorstores.FilterGte, bounds[0], args)
	if err != nil {
		return "", err
	}
	high, err := comparisonPredicate(field, vectorstores.FilterLte, bounds[1], args)
	if err != nil {
		return "", err
	}
	return low + " AND " + high, nil
}

// appendPlaceholder appends value to args and returns its $n placeholder.
func appendPlaceholder(args *[]any, value any) string {
	*args = append(*args, value)
	return "$" + strconv.Itoa(len(*args))
}

// escapeSQLLiteral doubles single quotes so an interpolated metadata field
// name cannot break out of its SQL string literal.
func escapeSQLLiteral(value string) string {
	return strings.ReplaceAll(value, "'", "''")
}

// numericOperand reports whether value is one of Go's numeric types.
func numericOperand(value any) (float64, bool) {
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
