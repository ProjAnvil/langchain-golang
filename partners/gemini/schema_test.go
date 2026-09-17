package gemini

// Schema-conversion tests: the recursive walk from core JSON-schema maps to
// the genai SDK's typed Schema — type spellings (string, list, missing),
// constraint copying, and the malformed shapes that must fail binding
// instead of silently dropping constraints.

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
	"google.golang.org/genai"
)

func TestMapToGenAISchemaFullWalk(t *testing.T) {
	got, err := mapToGenAISchema(map[string]any{
		"type":        "object",
		"description": "a record",
		"format":      "date-time",
		"nullable":    true,
		"default":     "none",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"tags": map[string]any{
				"type":     "array",
				"minItems": 1,
				"maxItems": 9,
				"items":    map[string]any{"type": "string"},
			},
			"score": map[string]any{
				"type":    "number",
				"minimum": 0.5,
				"maximum": 10,
			},
		},
		"required":  []any{"name"},
		"enum":      []any{"a", "b"},
		"anyOf":     []any{map[string]any{"type": "string"}},
		"minLength": 1,
		"maxLength": 64,
	})
	if err != nil {
		t.Fatalf("mapToGenAISchema() error = %v", err)
	}
	if got.Type != genai.TypeObject || got.Description != "a record" || got.Format != "date-time" {
		t.Fatalf("scalar fields = %+v", got)
	}
	if got.Nullable == nil || !*got.Nullable {
		t.Fatalf("Nullable = %v, want true", got.Nullable)
	}
	if got.Default != "none" {
		t.Fatalf("Default = %v", got.Default)
	}
	if len(got.Enum) != 2 || got.Enum[0] != "a" {
		t.Fatalf("Enum = %v", got.Enum)
	}
	if len(got.Properties) != 3 {
		t.Fatalf("Properties = %+v", got.Properties)
	}
	if tags := got.Properties["tags"]; tags.Type != genai.TypeArray ||
		tags.Items == nil || tags.Items.Type != genai.TypeString ||
		*tags.MinItems != 1 || *tags.MaxItems != 9 {
		t.Fatalf("tags property = %+v", tags)
	}
	if score := got.Properties["score"]; *score.Minimum != 0.5 || *score.Maximum != 10 {
		t.Fatalf("score property = %+v", score)
	}
	if len(got.Required) != 1 || got.Required[0] != "name" {
		t.Fatalf("Required = %v", got.Required)
	}
	if len(got.AnyOf) != 1 || got.AnyOf[0].Type != genai.TypeString {
		t.Fatalf("AnyOf = %+v", got.AnyOf)
	}
	if *got.MinLength != 1 || *got.MaxLength != 64 {
		t.Fatalf("length bounds = %v/%v", got.MinLength, got.MaxLength)
	}
}

func TestMapToGenAISchemaConstraintSpellings(t *testing.T) {
	// Numeric bounds accept the int spellings too; string-list constraints
	// accept []string (the schema.Object convention); non-string enum /
	// required entries and non-numeric bounds are skipped, not fatal.
	got, err := mapToGenAISchema(map[string]any{
		"minimum":    int(2),
		"maximum":    int64(8),
		"minItems":   float64(1),
		"maxItems":   int(4),
		"minLength":  int64(2),
		"maxLength":  float64(32),
		"required":   []string{"a", "b"},
		"enum":       []string{"x"},
		"items":      "not-a-map",
		"properties": map[string]any{},
		"anyOf":      []any{},
		// Malformed values for optional fields are tolerated (leniency):
		"nullable": "yes",
	})
	if err != nil {
		t.Fatalf("mapToGenAISchema() error = %v", err)
	}
	if *got.Minimum != 2 || *got.Maximum != 8 {
		t.Fatalf("numeric bounds = %v/%v", got.Minimum, got.Maximum)
	}
	if *got.MinItems != 1 || *got.MaxItems != 4 || *got.MinLength != 2 || *got.MaxLength != 32 {
		t.Fatalf("item/length bounds = %v %v %v %v", got.MinItems, got.MaxItems, got.MinLength, got.MaxLength)
	}
	if len(got.Required) != 2 || got.Required[1] != "b" {
		t.Fatalf("Required = %v", got.Required)
	}
	if len(got.Enum) != 1 || got.Enum[0] != "x" {
		t.Fatalf("Enum = %v", got.Enum)
	}
	if got.Items != nil || got.Properties != nil || got.AnyOf != nil || got.Nullable != nil {
		t.Fatalf("malformed optional fields must be dropped: %+v", got)
	}
}

func TestMapToGenAISchemaErrors(t *testing.T) {
	cases := []struct {
		name    string
		m       map[string]any
		wantErr string
	}{
		{
			name:    "bad scalar type",
			m:       map[string]any{"type": "widget"},
			wantErr: `unsupported schema type "widget"`,
		},
		{
			name:    "type not a string or list",
			m:       map[string]any{"type": 7},
			wantErr: "unsupported schema type value",
		},
		{
			name:    "property not an object",
			m:       map[string]any{"properties": map[string]any{"p": "plain"}},
			wantErr: `schema property "p" is not an object`,
		},
		{
			name:    "property with bad nested type",
			m:       map[string]any{"properties": map[string]any{"p": map[string]any{"type": "widget"}}},
			wantErr: `schema property "p"`,
		},
		{
			name:    "anyOf entry not an object",
			m:       map[string]any{"anyOf": []any{"plain"}},
			wantErr: "anyOf[0] is not an object",
		},
		{
			name:    "anyOf entry with bad nested type",
			m:       map[string]any{"anyOf": []any{map[string]any{"type": []any{"widget"}}}},
			wantErr: "anyOf[0]",
		},
		{
			name:    "items with bad nested type",
			m:       map[string]any{"items": map[string]any{"type": "widget"}},
			wantErr: `unsupported schema type "widget"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := mapToGenAISchema(tc.m)
			if err == nil {
				t.Fatalf("mapToGenAISchema(%v) succeeded, want error", tc.m)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestApplySchemaType(t *testing.T) {
	// A type list reduces to its first non-"null" entry.
	out := &genai.Schema{}
	if err := applySchemaType(out, []any{"null", "string"}); err != nil {
		t.Fatalf("applySchemaType(list) error = %v", err)
	}
	if out.Type != genai.TypeString {
		t.Fatalf("Type = %v, want string", out.Type)
	}

	// Non-string entries are skipped; a null-only list leaves the type unset.
	out = &genai.Schema{}
	if err := applySchemaType(out, []any{42, "null", "integer"}); err != nil {
		t.Fatalf("applySchemaType(mixed list) error = %v", err)
	}
	if out.Type != genai.TypeInteger {
		t.Fatalf("Type = %v, want integer", out.Type)
	}
	out = &genai.Schema{}
	if err := applySchemaType(out, []any{"null"}); err != nil {
		t.Fatalf("applySchemaType(null list) error = %v", err)
	}
	if out.Type != "" {
		t.Fatalf("Type = %v, want unset (zero value)", out.Type)
	}

	// A nil type value infers from shape; a bad list member fails.
	out = &genai.Schema{}
	if err := applySchemaType(out, nil); err != nil {
		t.Fatalf("applySchemaType(nil) error = %v", err)
	}
	if err := applySchemaType(out, []any{"widget"}); err == nil {
		t.Fatal("applySchemaType(bad list member) must fail")
	}
}

func TestGenaiTypeMapping(t *testing.T) {
	for name, want := range map[string]genai.Type{
		"string":  genai.TypeString,
		"number":  genai.TypeNumber,
		"integer": genai.TypeInteger,
		"boolean": genai.TypeBoolean,
		"array":   genai.TypeArray,
		"object":  genai.TypeObject,
		"null":    genai.TypeNULL,
	} {
		got, err := genaiType(name)
		if err != nil || got != want {
			t.Fatalf("genaiType(%q) = %v, %v; want %v", name, got, err, want)
		}
	}
	if _, err := genaiType("tuple"); err == nil {
		t.Fatal("genaiType(unknown) must fail")
	}
}

func TestSchemaStringSliceNormalization(t *testing.T) {
	if got, err := schemaStringSlice([]string{"a"}); err != nil || got[0] != "a" {
		t.Fatalf("schemaStringSlice([]string) = %v, %v", got, err)
	}
	if got, err := schemaStringSlice([]any{"a", "b"}); err != nil || len(got) != 2 {
		t.Fatalf("schemaStringSlice([]any) = %v, %v", got, err)
	}
	if _, err := schemaStringSlice([]any{"a", 2}); err == nil {
		t.Fatal("schemaStringSlice(non-string entry) must fail")
	}
	if _, err := schemaStringSlice("a"); err == nil {
		t.Fatal("schemaStringSlice(non-list) must fail")
	}
}

func TestNumericAndIntValueSpellings(t *testing.T) {
	if v, ok := numericValue(float64(1.5)); !ok || v != 1.5 {
		t.Fatalf("numericValue(float64) = %v, %v", v, ok)
	}
	if v, ok := numericValue(int(2)); !ok || v != 2 {
		t.Fatalf("numericValue(int) = %v, %v", v, ok)
	}
	if v, ok := numericValue(int64(3)); !ok || v != 3 {
		t.Fatalf("numericValue(int64) = %v, %v", v, ok)
	}
	if _, ok := numericValue("4"); ok {
		t.Fatal("numericValue(string) must not match")
	}

	if v, ok := intValue(float64(5)); !ok || v != 5 {
		t.Fatalf("intValue(float64) = %v, %v", v, ok)
	}
	if v, ok := intValue(int(6)); !ok || v != 6 {
		t.Fatalf("intValue(int) = %v, %v", v, ok)
	}
	if v, ok := intValue(int64(7)); !ok || v != 7 {
		t.Fatalf("intValue(int64) = %v, %v", v, ok)
	}
	if _, ok := intValue(true); ok {
		t.Fatal("intValue(bool) must not match")
	}
}

func TestAsSchemaMap(t *testing.T) {
	plain := map[string]any{"type": "string"}
	if m, ok := asSchemaMap(plain); !ok || m["type"] != "string" {
		t.Fatalf("asSchemaMap(map) = %v, %v", m, ok)
	}
	named := schema.Schema{"type": "string"}
	if m, ok := asSchemaMap(named); !ok || m["type"] != "string" {
		t.Fatalf("asSchemaMap(schema.Schema) = %v, %v", m, ok)
	}
	if _, ok := asSchemaMap("string"); ok {
		t.Fatal("asSchemaMap(string) must not match")
	}
}
