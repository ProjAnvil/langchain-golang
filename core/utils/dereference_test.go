package utils

import (
	"reflect"
	"testing"
)

func TestDereferenceRefsBasic(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"$ref": "#/$defs/string_type"},
		},
		"$defs": map[string]any{
			"string_type": map[string]any{"type": "string"},
		},
	}

	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}

	props := got["properties"].(map[string]any)
	name := props["name"].(map[string]any)
	if name["type"] != "string" {
		t.Fatalf("name.type = %v, want string", name["type"])
	}
	// $defs is preserved (skip_keys default) without inlining.
	if _, ok := got["$defs"]; !ok {
		t.Fatalf("$defs missing from result")
	}
}

func TestDereferenceRefsMixedRef(t *testing.T) {
	schema := map[string]any{
		"properties": map[string]any{
			"name": map[string]any{
				"$ref":        "#/$defs/base",
				"description": "User name",
			},
		},
		"$defs": map[string]any{
			"base": map[string]any{"type": "string", "minLength": 1},
		},
	}

	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}

	name := got["properties"].(map[string]any)["name"].(map[string]any)
	if name["type"] != "string" || name["minLength"] != 1 {
		t.Fatalf("resolved base fields wrong: %#v", name)
	}
	if name["description"] != "User name" {
		t.Fatalf("additional description = %v, want 'User name'", name["description"])
	}
}

func TestDereferenceRefsAdditionalOverrides(t *testing.T) {
	schema := map[string]any{
		"properties": map[string]any{
			"name": map[string]any{
				"$ref":        "#/$defs/base",
				"description": "Override",
			},
		},
		"$defs": map[string]any{
			"base": map[string]any{"type": "string", "description": "Base desc"},
		},
	}

	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}

	name := got["properties"].(map[string]any)["name"].(map[string]any)
	if name["description"] != "Override" {
		t.Fatalf("description = %v, want 'Override' (additional wins)", name["description"])
	}
}

func TestDereferenceRefsCircular(t *testing.T) {
	schema := map[string]any{
		"properties": map[string]any{
			"user": map[string]any{"$ref": "#/$defs/User"},
		},
		"$defs": map[string]any{
			"User": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"friend": map[string]any{"$ref": "#/$defs/User"},
				},
			},
		},
	}

	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	// Must terminate without infinite recursion.
	_ = got
}

func TestDereferenceRefsArrayIndex(t *testing.T) {
	schema := map[string]any{
		"type":  "array",
		"items": map[string]any{"$ref": "#/$defs/tuple/0"},
		"$defs": map[string]any{
			"tuple": []any{
				map[string]any{"type": "string"},
			},
		},
	}

	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	items := got["items"].(map[string]any)
	if items["type"] != "string" {
		t.Fatalf("items.type = %v, want string", items["type"])
	}
}

func TestDereferenceRefsBadPath(t *testing.T) {
	schema := map[string]any{"x": map[string]any{"$ref": "not-a-fragment"}}
	if _, err := DereferenceRefs(schema, nil, nil); err == nil {
		t.Fatalf("expected error for non-# ref, got nil")
	}
}

func TestDereferenceRefsMissingRef(t *testing.T) {
	schema := map[string]any{"x": map[string]any{"$ref": "#/$defs/nope"}}
	if _, err := DereferenceRefs(schema, nil, nil); err == nil {
		t.Fatalf("expected error for missing ref, got nil")
	}
}

func TestDereferenceRefsDeepInlinesWhenSkipKeysProvided(t *testing.T) {
	// With an explicit (even empty) skipKeys, Python switches to deep mode:
	// no default "$defs" skip, so nested refs are fully inlined.
	schema := map[string]any{
		"$defs": map[string]any{
			"a": map[string]any{"type": "string"},
		},
		"x": map[string]any{"$ref": "#/$defs/a"},
	}

	got, err := DereferenceRefs(schema, nil, []string{})
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	x := got["x"].(map[string]any)
	if x["type"] != "string" {
		t.Fatalf("x.type = %v, want string", x["type"])
	}
	if _, ok := got["$defs"]; ok {
		// With empty skipKeys, "$defs" is recursed into (no skip), so its
		// content is preserved but processed; the key should still exist.
		if _, ok2 := got["$defs"]; !ok2 {
			t.Fatalf("$defs disappeared")
		}
	}
}

func TestDereferenceRefsDoesNotMutateInput(t *testing.T) {
	schema := map[string]any{
		"properties": map[string]any{
			"name": map[string]any{"$ref": "#/$defs/string_type"},
		},
		"$defs": map[string]any{"string_type": map[string]any{"type": "string"}},
	}
	before := map[string]any{
		"properties": map[string]any{
			"name": map[string]any{"$ref": "#/$defs/string_type"},
		},
		"$defs": map[string]any{"string_type": map[string]any{"type": "string"}},
	}
	if _, err := DereferenceRefs(schema, nil, nil); err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	if !reflect.DeepEqual(schema, before) {
		t.Fatalf("input was mutated:\n got %#v\nwant %#v", schema, before)
	}
}

func TestDereferenceRefsNonStringRef(t *testing.T) {
	schema := map[string]any{"x": map[string]any{"$ref": 42}}
	if _, err := DereferenceRefs(schema, nil, nil); err == nil {
		t.Fatal("expected error for non-string $ref")
	}
}

func TestDereferenceRefsTopLevelNonObjectResult(t *testing.T) {
	// A top-level $ref with no sibling keys resolving to a scalar leaves a
	// non-object result. fullSchema is explicit so the ref target exists even
	// though the processed schema carries no $defs of its own.
	schema := map[string]any{"$ref": "#/$defs/scalar"}
	full := map[string]any{
		"$defs": map[string]any{
			"scalar": "just a string",
		},
	}
	if _, err := DereferenceRefs(schema, full, nil); err == nil {
		t.Fatal("expected error for non-object dereference result")
	}
}

func TestDereferenceRefsBadArrayIndex(t *testing.T) {
	for _, ref := range []string{"#/$defs/tuple/5", "#/$defs/tuple/abc", "#/$defs/tuple/-1"} {
		schema := map[string]any{
			"items": map[string]any{"$ref": ref},
			"$defs": map[string]any{
				"tuple": []any{map[string]any{"type": "string"}},
			},
		}
		if _, err := DereferenceRefs(schema, nil, nil); err == nil {
			t.Fatalf("expected error for ref %q", ref)
		}
	}
}

func TestDereferenceRefsCannotTraverseScalar(t *testing.T) {
	schema := map[string]any{
		"x": map[string]any{"$ref": "#/a/b"},
		"a": "scalar",
	}
	if _, err := DereferenceRefs(schema, nil, nil); err == nil {
		t.Fatal("expected error when traversing through a scalar")
	}
}

func TestDereferenceRefsCircularWithAdditionalProps(t *testing.T) {
	// A circular $ref carrying additional properties breaks the cycle by
	// returning only the additional properties.
	schema := map[string]any{
		"properties": map[string]any{
			"user": map[string]any{"$ref": "#/$defs/User"},
		},
		"$defs": map[string]any{
			"User": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"friend": map[string]any{
						"$ref":        "#/$defs/User",
						"description": "a friend",
					},
				},
			},
		},
	}
	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	user := got["properties"].(map[string]any)["user"].(map[string]any)
	friend := user["properties"].(map[string]any)["friend"].(map[string]any)
	if friend["description"] != "a friend" {
		t.Fatalf("cycle-breaking additional props wrong: %#v", friend)
	}
	if _, ok := friend["type"]; ok {
		t.Fatalf("circular ref should not inline resolved fields: %#v", friend)
	}
}

func TestDereferenceRefsMixedRefWithNonMapResolved(t *testing.T) {
	// When the resolved $ref target is not an object, only the additional
	// properties survive the merge.
	schema := map[string]any{
		"x": map[string]any{
			"$ref":        "#/$defs/scalar",
			"description": "extra",
		},
		"$defs": map[string]any{
			"scalar": "not an object",
		},
	}
	got, err := DereferenceRefs(schema, nil, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	x := got["x"].(map[string]any)
	if !reflect.DeepEqual(x, map[string]any{"description": "extra"}) {
		t.Fatalf("mixed ref with scalar resolved: %#v", x)
	}
}

func TestDereferenceRefsErrorInsideArray(t *testing.T) {
	schema := map[string]any{
		"anyOf": []any{
			map[string]any{"type": "string"},
			map[string]any{"$ref": "#/$defs/missing"},
		},
	}
	if _, err := DereferenceRefs(schema, nil, nil); err == nil {
		t.Fatal("expected error for bad ref inside array")
	}
}

func TestDereferenceRefsErrorInsideMixedRefAdditional(t *testing.T) {
	schema := map[string]any{
		"x": map[string]any{
			"$ref":   "#/$defs/base",
			"nested": map[string]any{"$ref": "#/$defs/missing"},
		},
		"$defs": map[string]any{
			"base": map[string]any{"type": "string"},
		},
	}
	if _, err := DereferenceRefs(schema, nil, nil); err == nil {
		t.Fatal("expected error for bad ref inside additional properties")
	}
}

func TestDereferenceRefsExplicitFullSchema(t *testing.T) {
	// Refs resolve against fullSchema, not the subschema being processed.
	sub := map[string]any{"x": map[string]any{"$ref": "#/$defs/a"}}
	full := map[string]any{
		"$defs": map[string]any{
			"a": map[string]any{"type": "integer"},
		},
	}
	got, err := DereferenceRefs(sub, full, nil)
	if err != nil {
		t.Fatalf("DereferenceRefs() error = %v", err)
	}
	if got["x"].(map[string]any)["type"] != "integer" {
		t.Fatalf("explicit fullSchema not used: %#v", got)
	}
}
