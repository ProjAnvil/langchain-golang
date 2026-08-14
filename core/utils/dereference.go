package utils

import (
	"fmt"
	"strconv"
	"strings"
)

// DereferenceRefs resolves JSON Schema `$ref` pointers in schemaObj by
// inlining the referenced definitions. It mirrors Python's
// `langchain_core.utils.json_schema.dereference_refs`.
//
// fullSchema is the schema against which `$ref` pointers are resolved; when
// nil it defaults to schemaObj (for self-contained schemas). skipKeys lists
// keys under which recursion is skipped (the subtree is copied as-is);
// a nil skipKeys means the default `["$defs"]`, matching Python's shallow
// mode. The input is never mutated; a new map is returned.
func DereferenceRefs(schemaObj map[string]any, fullSchema map[string]any, skipKeys []string) (map[string]any, error) {
	if fullSchema == nil {
		fullSchema = schemaObj
	}
	if skipKeys == nil {
		skipKeys = []string{"$defs"}
	}
	skip := make(map[string]bool, len(skipKeys))
	for _, k := range skipKeys {
		skip[k] = true
	}

	out, err := derefHelper(schemaObj, fullSchema, map[string]bool{}, skip)
	if err != nil {
		return nil, err
	}
	res, ok := out.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("utils: dereference_refs result is not an object: %T", out)
	}
	return res, nil
}

// derefHelper is the recursive core of DereferenceRefs. processing tracks the
// set of `$ref` paths currently being resolved for cycle breaking.
func derefHelper(obj any, fullSchema map[string]any, processing map[string]bool, skip map[string]bool) (any, error) {
	switch v := obj.(type) {
	case map[string]any:
		if ref, ok := v["$ref"]; ok {
			refStr, ok := ref.(string)
			if !ok {
				return nil, fmt.Errorf("utils: $ref must be a string, got %T", ref)
			}
			additional := make(map[string]any, len(v))
			for k, val := range v {
				if k != "$ref" {
					additional[k] = val
				}
			}

			// Cycle: if we are already resolving this ref, return only the
			// additional properties (which override nothing) to break the loop.
			if processing[refStr] {
				return derefDictProps(additional, fullSchema, processing, skip)
			}

			processing[refStr] = true
			refObj, err := retrieveRef(refStr, fullSchema)
			if err != nil {
				delete(processing, refStr)
				return nil, err
			}
			resolved, err := derefHelper(refObj, fullSchema, processing, skip)
			delete(processing, refStr)
			if err != nil {
				return nil, err
			}

			if len(additional) == 0 {
				return resolved, nil
			}

			// Mixed $ref: merge resolved with additional (additional wins).
			merged := map[string]any{}
			if rm, ok := resolved.(map[string]any); ok {
				for k, val := range rm {
					merged[k] = val
				}
			}
			procAdditional, err := derefDictProps(additional, fullSchema, processing, skip)
			if err != nil {
				return nil, err
			}
			for k, val := range procAdditional {
				merged[k] = val
			}
			return merged, nil
		}
		return derefDictProps(v, fullSchema, processing, skip)
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			res, err := derefHelper(item, fullSchema, processing, skip)
			if err != nil {
				return nil, err
			}
			out[i] = res
		}
		return out, nil
	default:
		return obj, nil
	}
}

func derefDictProps(props map[string]any, fullSchema map[string]any, processing map[string]bool, skip map[string]bool) (map[string]any, error) {
	result := make(map[string]any, len(props))
	for k, val := range props {
		if skip[k] {
			result[k] = val
			continue
		}
		res, err := derefHelper(val, fullSchema, processing, skip)
		if err != nil {
			return nil, err
		}
		result[k] = res
	}
	return result, nil
}

// retrieveRef resolves a JSON pointer (e.g. "#/$defs/MyType") against schema.
func retrieveRef(path string, schema map[string]any) (any, error) {
	components := strings.Split(path, "/")
	if components[0] != "#" {
		return nil, fmt.Errorf("utils: ref path must be a URI fragment starting with '#': %q", path)
	}
	var out any = schema
	for _, component := range components[1:] {
		switch cur := out.(type) {
		case map[string]any:
			val, ok := cur[component]
			if !ok {
				return nil, fmt.Errorf("utils: reference %q not found (missing %q)", path, component)
			}
			out = val
		case []any:
			idx, err := strconv.Atoi(component)
			if err != nil || idx < 0 || idx >= len(cur) {
				return nil, fmt.Errorf("utils: reference %q not found (bad index %q)", path, component)
			}
			out = cur[idx]
		default:
			return nil, fmt.Errorf("utils: reference %q not found (cannot traverse %q through %T)", path, component, out)
		}
	}
	return out, nil
}
