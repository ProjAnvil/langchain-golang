package utils

import (
	"fmt"
	"reflect"
	"strings"
)

// MergeDicts merges maps left-to-right, mirroring Python's merge_dicts
// (utils/_merge.py:6). Rules: a key missing from the merged result, or
// holding a nil value while the right value is non-nil, adopts the right
// value; nil right values are skipped; mismatched concrete types are an
// error; strings concatenate (except lc_-prefixed "index" values and equal
// "id"/"output_version"/"model_provider" values); nested maps and slices
// recurse; equal values pass through; ints sum except the "index"/"created"/
// "timestamp" keys which are last-wins; anything else is an error. Inputs
// are never mutated.
//
// Port note: Python's json.loads yields int for integral numbers while Go's
// encoding/json yields float64, so numeric values are aligned to the existing
// value's type (losslessly) before the type check — see coerceNumeric.
func MergeDicts(left map[string]any, others ...map[string]any) (map[string]any, error) {
	merged := cloneMap(left)
	for _, right := range others {
		for key, rightValue := range right {
			existing, present := merged[key]
			if !present || (rightValue != nil && existing == nil) {
				merged[key] = rightValue
				continue
			}
			if rightValue == nil {
				continue
			}
			existing, rightValue = coerceNumeric(existing, rightValue)
			if reflect.TypeOf(existing) != reflect.TypeOf(rightValue) {
				return nil, fmt.Errorf(
					"additional_kwargs[%q] already exists in this message, but with a different type",
					key)
			}
			switch existingValue := existing.(type) {
			case string:
				rightString := rightValue.(string)
				if key == "index" && strings.HasPrefix(existingValue, "lc_") {
					continue
				}
				if (key == "id" || key == "output_version" || key == "model_provider") && existingValue == rightString {
					continue
				}
				merged[key] = existingValue + rightString
			case map[string]any:
				next, err := MergeDicts(existingValue, rightValue.(map[string]any))
				if err != nil {
					return nil, err
				}
				merged[key] = next
			case []any:
				next, err := MergeLists(existingValue, rightValue.([]any))
				if err != nil {
					return nil, err
				}
				merged[key] = next
			default:
				if reflect.DeepEqual(existing, rightValue) {
					continue
				}
				if existingInt, ok := existing.(int); ok {
					rightInt := rightValue.(int)
					if key == "index" || key == "created" || key == "timestamp" {
						// Identification and temporal fields are last-wins
						// instead of summed (_merge.py:71-79).
						merged[key] = rightInt
					} else {
						merged[key] = existingInt + rightInt
					}
					continue
				}
				return nil, fmt.Errorf(
					"additional kwargs key %s already exists in left dict and value has unsupported type %T",
					key, existing)
			}
		}
	}
	return merged, nil
}

// MergeLists merges lists left-to-right, mirroring Python's merge_lists
// (utils/_merge.py:89). nil stands in for Python's None: nil others are
// skipped, and an all-nil merge returns nil. Dict elements with an "index"
// key (int/float64-integral, or an "lc_"-prefixed string) merge into the
// first existing element with the same index when their IDs are not in
// conflict (either side missing/empty, or equal); all other elements append.
// Inputs are never mutated.
func MergeLists(left []any, others ...[]any) ([]any, error) {
	var merged []any
	if left != nil {
		merged = make([]any, len(left))
		copy(merged, left)
	}
	for _, other := range others {
		if other == nil {
			continue
		}
		if merged == nil {
			merged = make([]any, len(other))
			copy(merged, other)
			continue
		}
		for _, element := range other {
			elementMap, ok := element.(map[string]any)
			if !ok {
				merged = append(merged, element)
				continue
			}
			elementIndex, ok := mergeableIndex(elementMap)
			if !ok {
				merged = append(merged, element)
				continue
			}
			matchIndex := -1
			for i, existing := range merged {
				existingMap, ok := existing.(map[string]any)
				if !ok {
					continue
				}
				existingIndex, ok := existingMap["index"]
				if !ok || !indexValuesEqual(existingIndex, elementIndex) {
					continue
				}
				// IDs must not be inconsistent (_merge.py:123-128).
				existingID, existingHasID := existingMap["id"]
				elementID, elementHasID := elementMap["id"]
				if !existingHasID || !elementHasID || isEmptyID(existingID) || isEmptyID(elementID) ||
					reflect.DeepEqual(existingID, elementID) {
					matchIndex = i
					break
				}
			}
			if matchIndex < 0 {
				merged = append(merged, element)
				continue
			}
			existingMap := merged[matchIndex].(map[string]any)
			newElement := mergeIncomingChunk(existingMap, elementMap)
			mergedDict, err := MergeDicts(existingMap, newElement)
			if err != nil {
				return nil, err
			}
			merged[matchIndex] = mergedDict
		}
	}
	return merged, nil
}

// mergeableIndex reports whether e's "index" makes it mergeable
// (_merge.py:108-116): an int/float64-integral value, or an "lc_"-prefixed
// string.
func mergeableIndex(e map[string]any) (any, bool) {
	index, ok := e["index"]
	if !ok {
		return nil, false
	}
	if _, ok := indexAsFloat(index); ok {
		return index, true
	}
	if text, ok := index.(string); ok && strings.HasPrefix(text, "lc_") {
		return index, true
	}
	return nil, false
}

// coerceNumeric aligns the right value's numeric type with the existing
// value's type when both are numeric and the conversion is lossless. This
// keeps Python parity for JSON-decoded data: json.loads yields int for
// integral numbers, encoding/json yields float64.
func coerceNumeric(existing any, rightValue any) (any, any) {
	_, existingOK := numericAsFloat(existing)
	rightFloat, rightOK := numericAsFloat(rightValue)
	if !existingOK || !rightOK {
		return existing, rightValue
	}
	switch existing.(type) {
	case int:
		if rightFloat == float64(int64(rightFloat)) {
			return existing, int(rightFloat)
		}
	case int64:
		if rightFloat == float64(int64(rightFloat)) {
			return existing, int64(rightFloat)
		}
	case float64:
		return existing, rightFloat
	}
	return existing, rightValue
}

// numericAsFloat converts int/int64/float64 to float64.
func numericAsFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case float64:
		return typed, true
	default:
		return 0, false
	}
}

// indexValuesEqual compares index values across numeric Go types (int from
// Go-built chunks, float64 from JSON-decoded chunks), matching Python's
// e_left["index"] == e["index"].
func indexValuesEqual(a any, b any) bool {
	if af, ok := indexAsFloat(a); ok {
		bf, ok := indexAsFloat(b)
		return ok && af == bf
	}
	aText, aOK := a.(string)
	bText, bOK := b.(string)
	return aOK && bOK && aText == bText
}

func indexAsFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case float64:
		if typed == float64(int64(typed)) {
			return typed, true
		}
		return 0, false
	default:
		return 0, false
	}
}

func isEmptyID(value any) bool {
	if value == nil {
		return true
	}
	text, ok := value.(string)
	return ok && text == ""
}

// mergeIncomingChunk prepares an incoming indexed dict for merging into an
// existing entry (_merge.py:130-161): a "non_standard" incoming chunk folds
// its "value" into "extras" (standard existing entry) or into "value"
// (non_standard existing entry, keeping the index); otherwise the incoming
// "type" key is stripped so it does not overwrite the existing one.
func mergeIncomingChunk(existing map[string]any, incoming map[string]any) map[string]any {
	existingType, hasExistingType := existing["type"].(string)
	incomingType, _ := incoming["type"].(string)
	incomingValue, hasIncomingValue := incoming["value"]
	if hasExistingType && existingType != "" && incomingType == "non_standard" && hasIncomingValue {
		valueMap, _ := incomingValue.(map[string]any)
		if existingType != "non_standard" {
			// standard + non_standard
			extras := map[string]any{}
			for key, value := range valueMap {
				if key != "type" {
					extras[key] = value
				}
			}
			return map[string]any{"extras": extras}
		}
		// non_standard + non_standard
		values := map[string]any{}
		for key, value := range valueMap {
			if key != "type" {
				values[key] = value
			}
		}
		newElement := map[string]any{"value": values}
		if index, ok := incoming["index"]; ok {
			newElement["index"] = index
		}
		return newElement
	}
	if _, hasType := incoming["type"]; hasType {
		newElement := make(map[string]any, len(incoming))
		for key, value := range incoming {
			if key != "type" {
				newElement[key] = value
			}
		}
		return newElement
	}
	return incoming
}
