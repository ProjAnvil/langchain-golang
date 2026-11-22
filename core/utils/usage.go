package utils

import (
	"errors"
	"fmt"
	"reflect"
	"slices"
)

// DictIntOp recursively applies an integer binary operation to corresponding
// values of two nested maps and returns a fresh map. Keys present in only one
// input use the integer default 0 (or an empty nested map) on the other side,
// mirroring LangChain's _dict_int_op. Only integer values and nested
// map[string]any values are supported; any other type is reported as an error.
//
// maxDepth bounds recursion: the top-level maps are at depth 0, each nested
// map increments depth, and an error is returned once depth reaches maxDepth
// (or immediately if maxDepth is not positive). A nil op is also an error.
// The function returns errors instead of panicking, never mutates its inputs,
// and never calls op on mismatched types.
func DictIntOp(left, right map[string]any, op func(int, int) int, maxDepth int) (map[string]any, error) {
	if op == nil {
		return nil, errors.New("dict int op: nil operation")
	}
	if maxDepth < 0 {
		return nil, fmt.Errorf("dict int op: maxDepth must be >= 0, got %d", maxDepth)
	}
	return smallADictIntOp(left, right, op, 0, maxDepth)
}

// smallADictIntOp is the recursive worker behind DictIntOp.
func smallADictIntOp(left, right map[string]any, op func(int, int) int, depth, maxDepth int) (map[string]any, error) {
	if depth >= maxDepth {
		return nil, fmt.Errorf("dict int op: max depth %d exceeded, unable to combine dicts", maxDepth)
	}

	keys := make([]string, 0, len(left)+len(right))
	seen := make(map[string]bool, len(left)+len(right))
	for k := range left {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	for k := range right {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	slices.Sort(keys)

	combined := make(map[string]any, len(keys))
	for _, k := range keys {
		lv, lok := left[k]
		rv, rok := right[k]

		// Integer branch: a missing key defaults to the integer 0.
		lIntDefault := any(int(0))
		rIntDefault := any(int(0))
		if lok {
			lIntDefault = lv
		}
		if rok {
			rIntDefault = rv
		}
		if li, ok := smallAAsInt(lIntDefault); ok {
			if ri, ok := smallAAsInt(rIntDefault); ok {
				combined[k] = op(li, ri)
				continue
			}
		}

		// Nested-map branch: a missing key defaults to an empty map.
		lDictDefault := any(map[string]any{})
		rDictDefault := any(map[string]any{})
		if lok {
			lDictDefault = lv
		}
		if rok {
			rDictDefault = rv
		}
		if ld, ok := smallAAsMap(lDictDefault); ok {
			if rd, ok := smallAAsMap(rDictDefault); ok {
				sub, err := smallADictIntOp(ld, rd, op, depth+1, maxDepth)
				if err != nil {
					return nil, err
				}
				combined[k] = sub
				continue
			}
		}

		return nil, fmt.Errorf("dict int op: unsupported value types for key %q (left %T, right %T); only dict and int values are supported", k, lv, rv)
	}
	return combined, nil
}

// smallAAsInt reports whether value is any Go integer type and returns it as
// an int. Booleans, floats, and non-integer types are intentionally rejected.
func smallAAsInt(value any) (int, bool) {
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return int(rv.Uint()), true
	default:
		return 0, false
	}
}

// smallAAsMap reports whether value is a supported nested map type.
func smallAAsMap(value any) (map[string]any, bool) {
	m, ok := value.(map[string]any)
	return m, ok
}
