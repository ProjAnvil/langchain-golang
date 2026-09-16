package vectorstores

import (
	"fmt"
	"maps"
	"reflect"
	"regexp"
	"slices"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// Filter operator names of the declarative metadata filter DSL. The operator
// set and semantics follow langchain-postgres SearchArgs.filter
// (langchain_postgres/vectorstores.py), which is the baseline for all vector
// store integrations in this repo.
const (
	// FilterEq matches documents whose field equals the operand.
	FilterEq = "$eq"
	// FilterNe matches documents whose field differs from the operand. A
	// missing field never matches (langchain-postgres parity: jsonb_path_match
	// yields NULL for absent keys).
	FilterNe = "$ne"
	// FilterGt matches documents whose field is greater than the operand.
	FilterGt = "$gt"
	// FilterGte matches documents whose field is greater than or equal to the
	// operand.
	FilterGte = "$gte"
	// FilterLt matches documents whose field is less than the operand.
	FilterLt = "$lt"
	// FilterLte matches documents whose field is less than or equal to the
	// operand.
	FilterLte = "$lte"
	// FilterIn matches documents whose field equals any list element.
	FilterIn = "$in"
	// FilterNin matches documents whose field equals no list element. A
	// missing field never matches (langchain-postgres parity).
	FilterNin = "$nin"
	// FilterBetween matches documents whose field lies within the inclusive
	// two-element range [low, high] (langchain-postgres: >= low AND <= high).
	FilterBetween = "$between"
	// FilterExists matches documents where the field is present (true) or
	// absent (false).
	FilterExists = "$exists"
	// FilterLike matches string fields against a case-sensitive SQL LIKE
	// pattern where "%" matches any sequence and "_" matches one character.
	FilterLike = "$like"
)

// supportedFilterOperators lists every operator accepted by ValidateFilter.
var supportedFilterOperators = []string{
	FilterEq, FilterNe, FilterGt, FilterGte, FilterLt, FilterLte,
	FilterIn, FilterNin, FilterBetween, FilterExists, FilterLike,
}

// ValidateFilter reports whether filter is a well-formed declarative filter.
//
// A filter maps field names to either a literal value (equality shorthand) or
// a single-entry map of operator to operand:
//
//	{"group": "a", "page": {"$gt": 2}, "name": {"$like": "%al%"}}
//
// Top-level fields are ANDed. One operator per condition (langchain-postgres
// rejects multi-operator conditions as well). Field names must be non-empty
// and must not start with "$" (that prefix is reserved for operators).
//
// ValidateFilter returns an error for structurally invalid filters so stores
// can fail loudly before executing a search; use MatchFilter for evaluation.
func ValidateFilter(filter map[string]any) error {
	for _, field := range slices.Sorted(maps.Keys(filter)) {
		if field == "" {
			return fmt.Errorf("invalid filter: field name must not be empty")
		}
		if strings.HasPrefix(field, "$") {
			return fmt.Errorf(
				"invalid filter condition. Expected a field but got an operator: %s",
				field,
			)
		}
		if err := validateCondition(field, filter[field]); err != nil {
			return err
		}
	}
	return nil
}

func validateCondition(field string, condition any) error {
	op, operand, err := splitCondition(condition)
	if err != nil {
		return fmt.Errorf("invalid filter condition for field %q: %w", field, err)
	}
	switch op {
	case FilterEq, FilterNe:
		if !isScalarValue(operand) {
			return fmt.Errorf(
				"invalid filter condition for field %q: %s requires a scalar operand, use %s for membership",
				field, op, FilterIn,
			)
		}
	case FilterGt, FilterGte, FilterLt, FilterLte:
		if _, ok := numericValue(operand); !ok {
			if _, isString := operand.(string); !isString {
				return fmt.Errorf(
					"invalid filter condition for field %q: %s requires a string or numeric operand, got %T",
					field, op, operand,
				)
			}
		}
	case FilterIn, FilterNin:
		items, err := operandList(operand)
		if err != nil {
			return fmt.Errorf("invalid filter condition for field %q: %s: %w", field, op, err)
		}
		for _, item := range items {
			// langchain-postgres only allows str/int/float in $in/$nin.
			if _, isNumeric := numericValue(item); !isNumeric {
				if _, isString := item.(string); !isString {
					return fmt.Errorf(
						"invalid filter condition for field %q: %s only supports string and numeric elements, got %T",
						field, op, item,
					)
				}
			}
		}
	case FilterBetween:
		items, err := operandList(operand)
		if err != nil {
			return fmt.Errorf("invalid filter condition for field %q: %s: %w", field, op, err)
		}
		if len(items) != 2 {
			return fmt.Errorf(
				"invalid filter condition for field %q: %s requires exactly two bounds [low, high], got %d",
				field, op, len(items),
			)
		}
		_, lowIsNumeric := numericValue(items[0])
		_, highIsNumeric := numericValue(items[1])
		if lowIsNumeric != highIsNumeric {
			return fmt.Errorf(
				"invalid filter condition for field %q: %s bounds must both be numeric or both be strings",
				field, op,
			)
		}
		if !lowIsNumeric &&
			(!isStringOrNumeric(items[0]) || !isStringOrNumeric(items[1])) {
			return fmt.Errorf(
				"invalid filter condition for field %q: %s bounds must both be numeric or both be strings",
				field, op,
			)
		}
	case FilterExists:
		if _, isBool := operand.(bool); !isBool {
			return fmt.Errorf(
				"invalid filter condition for field %q: %s requires a boolean operand, got %T",
				field, op, operand,
			)
		}
	case FilterLike:
		if _, isString := operand.(string); !isString {
			return fmt.Errorf(
				"invalid filter condition for field %q: %s requires a string pattern operand, got %T",
				field, op, operand,
			)
		}
	default:
		return fmt.Errorf(
			"invalid operator: %s. Expected one of %s",
			op, strings.Join(supportedFilterOperators, ", "),
		)
	}
	return nil
}

// splitCondition normalizes a condition into an operator and its operand. A
// non-map condition is the equality shorthand.
func splitCondition(condition any) (string, any, error) {
	switch typed := condition.(type) {
	case map[string]any:
		if len(typed) != 1 {
			return "", nil, fmt.Errorf(
				"expected a value which is a dictionary with a single key that corresponds to an operator but got %d keys",
				len(typed),
			)
		}
		op := slices.Sorted(maps.Keys(typed))[0]
		if !slices.Contains(supportedFilterOperators, op) {
			return op, typed[op], fmt.Errorf(
				"invalid operator: %s. Expected one of %s",
				op, strings.Join(supportedFilterOperators, ", "),
			)
		}
		return op, typed[op], nil
	default:
		if !isScalarValue(condition) {
			return "", nil, fmt.Errorf(
				"expected a scalar value or a single-operator dictionary, got %T",
				condition,
			)
		}
		return FilterEq, condition, nil
	}
}

// MatchFilter reports whether doc satisfies the declarative filter. All
// top-level field conditions must hold (logical AND). Structurally invalid
// filters (unknown operators, wrong operand shapes) never match; call
// ValidateFilter first to surface such mistakes with explicit errors.
//
// Semantics follow langchain-postgres SQL translation: every operator except
// $exists evaluates to false when the metadata field is absent (or nil),
// including $ne and $nin. Type mismatches between the stored value and the
// operand also evaluate to false rather than erroring.
func MatchFilter(doc documents.Document, filter map[string]any) bool {
	for field, condition := range filter {
		// Field names rejected by ValidateFilter never match here either.
		if field == "" || strings.HasPrefix(field, "$") {
			return false
		}
		value, present := doc.Metadata[field]
		op, operand, ok := matchCondition(condition)
		if !ok {
			return false
		}
		if !matchOperator(op, operand, value, present) {
			return false
		}
	}
	return true
}

// matchCondition mirrors splitCondition for evaluation: it returns ok=false
// for structurally invalid conditions.
func matchCondition(condition any) (op string, operand any, ok bool) {
	switch typed := condition.(type) {
	case map[string]any:
		if len(typed) != 1 {
			return "", nil, false
		}
		for key, value := range typed {
			if !slices.Contains(supportedFilterOperators, key) {
				return "", nil, false
			}
			return key, value, true
		}
		return "", nil, false
	default:
		if !isScalarValue(condition) {
			return "", nil, false
		}
		return FilterEq, condition, true
	}
}

func matchOperator(op string, operand, value any, present bool) bool {
	switch op {
	case FilterEq:
		return present && value != nil && valuesEqual(value, operand)
	case FilterNe:
		return present && value != nil && !valuesEqual(value, operand)
	case FilterGt, FilterGte, FilterLt, FilterLte:
		if !present || value == nil {
			return false
		}
		order, ok := compareValues(value, operand)
		if !ok {
			return false
		}
		return switchOrder(op, order)
	case FilterIn:
		if !present || value == nil {
			return false
		}
		items, ok := toAnySlice(operand)
		if !ok {
			return false
		}
		for _, item := range items {
			if !isStringOrNumeric(item) {
				continue
			}
			if valuesEqual(value, item) {
				return true
			}
		}
		return false
	case FilterNin:
		if !present || value == nil {
			return false
		}
		items, ok := toAnySlice(operand)
		if !ok {
			return false
		}
		for _, item := range items {
			if !isStringOrNumeric(item) {
				continue
			}
			if valuesEqual(value, item) {
				return false
			}
		}
		return true
	case FilterBetween:
		if !present || value == nil {
			return false
		}
		items, ok := toAnySlice(operand)
		if !ok || len(items) != 2 {
			return false
		}
		lowOrder, ok := compareValues(value, items[0])
		if !ok {
			return false
		}
		highOrder, ok := compareValues(value, items[1])
		if !ok {
			return false
		}
		return lowOrder >= 0 && highOrder <= 0
	case FilterExists:
		want, isBool := operand.(bool)
		if !isBool {
			return false
		}
		return want == present
	case FilterLike:
		pattern, isString := operand.(string)
		if !isString {
			return false
		}
		text, isString := value.(string)
		if !isString || !present {
			return false
		}
		return likeMatch(pattern, text)
	default:
		return false
	}
}

func switchOrder(op string, order int) bool {
	switch op {
	case FilterGt:
		return order > 0
	case FilterGte:
		return order >= 0
	case FilterLt:
		return order < 0
	case FilterLte:
		return order <= 0
	default:
		return false
	}
}

// valuesEqual compares scalar metadata values with numeric normalization so
// that int 5, int64 5, and float64 5 compare equal regardless of which side
// they appear on.
func valuesEqual(a, b any) bool {
	if aNumeric, ok := numericValue(a); ok {
		bNumeric, ok := numericValue(b)
		return ok && aNumeric == bNumeric
	}
	aString, aIsString := a.(string)
	bString, bIsString := b.(string)
	if aIsString && bIsString {
		return aString == bString
	}
	aBool, aIsBool := a.(bool)
	bBool, bIsBool := b.(bool)
	return aIsBool && bIsBool && aBool == bBool
}

// compareValues returns -1/0/+1 when both values are numbers or both are
// strings, and ok=false otherwise.
func compareValues(a, b any) (int, bool) {
	if aNumeric, ok := numericValue(a); ok {
		bNumeric, ok := numericValue(b)
		if !ok {
			return 0, false
		}
		switch {
		case aNumeric < bNumeric:
			return -1, true
		case aNumeric > bNumeric:
			return 1, true
		default:
			return 0, true
		}
	}
	aString, aIsString := a.(string)
	bString, bIsString := b.(string)
	if aIsString && bIsString {
		return strings.Compare(aString, bString), true
	}
	return 0, false
}

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

// isScalarValue reports whether value is a valid $eq operand: string, number,
// bool, or nil.
func isScalarValue(value any) bool {
	switch value.(type) {
	case string, bool, nil:
		return true
	default:
		_, isNumeric := numericValue(value)
		return isNumeric
	}
}

// operandList converts list operands ($in/$nin/$between) to []any. It accepts
// []any plus typed scalar slices so callers can write natural Go literals.
func operandList(operand any) ([]any, error) {
	items, ok := toAnySlice(operand)
	if !ok {
		return nil, fmt.Errorf("requires a list operand, got %T", operand)
	}
	return items, nil
}

func isStringOrNumeric(value any) bool {
	if _, isString := value.(string); isString {
		return true
	}
	_, isNumeric := numericValue(value)
	return isNumeric
}

func toAnySlice(value any) ([]any, bool) {
	switch typed := value.(type) {
	case []any:
		return typed, true
	case []string, []int, []int32, []int64, []float32, []float64, []uint:
		reflected := reflect.ValueOf(typed)
		out := make([]any, reflected.Len())
		for i := range out {
			out[i] = reflected.Index(i).Interface()
		}
		return out, true
	default:
		return nil, false
	}
}

// likeMatch implements case-sensitive SQL LIKE matching where "%" matches any
// sequence (including empty) and "_" matches exactly one character, mirroring
// the SQL LIKE translation used by langchain-postgres for $like.
func likeMatch(pattern, text string) bool {
	var builder strings.Builder
	builder.WriteString("^")
	for _, char := range pattern {
		switch char {
		case '%':
			builder.WriteString(".*")
		case '_':
			builder.WriteString(".")
		default:
			builder.WriteString(regexp.QuoteMeta(string(char)))
		}
	}
	builder.WriteString("$")
	matched, err := regexp.MatchString(builder.String(), text)
	return err == nil && matched
}
