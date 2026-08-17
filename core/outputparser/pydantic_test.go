package outputparser

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/lcerrors"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestPydanticParserParsesTypedStruct(t *testing.T) {
	parser := NewPydanticParser[parsedPerson](schema.Object(map[string]schema.Schema{
		"name": schema.String("person name"),
		"age":  schema.Integer("person age"),
		"tags": {
			"type":  "array",
			"items": schema.String("tag"),
		},
	}, "name", "age"))

	got, err := parser.Parse(context.Background(), `{"name":"Ada","age":37,"tags":["math","code"]}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Name != "Ada" || got.Age != 37 || len(got.Tags) != 2 || got.Tags[1] != "code" {
		t.Fatalf("got %+v", got)
	}
}

func TestPydanticParserRejectsMissingRequiredField(t *testing.T) {
	parser := NewPydanticParser[parsedPerson](schema.Object(map[string]schema.Schema{
		"name": schema.String("person name"),
		"age":  schema.Integer("person age"),
	}, "name", "age"))

	_, err := parser.Parse(context.Background(), `{"name":"Ada"}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) {
		t.Fatalf("err: got %v", err)
	}
	if !strings.Contains(err.Error(), "$.age") {
		t.Fatalf("err should include field path: %v", err)
	}
}

func TestPydanticParserRejectsWrongType(t *testing.T) {
	parser := NewPydanticParser[parsedPerson](schema.Object(map[string]schema.Schema{
		"name": schema.String("person name"),
		"age":  schema.Integer("person age"),
	}, "name", "age"))

	_, err := parser.Parse(context.Background(), `{"name":"Ada","age":"old"}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) {
		t.Fatalf("err: got %v", err)
	}
	if !strings.Contains(err.Error(), "expected integer") {
		t.Fatalf("err should include type mismatch: %v", err)
	}
}

func TestPydanticParserEnumConstAndNullable(t *testing.T) {
	parser := NewPydanticParser[map[string]any](schema.Object(map[string]schema.Schema{
		"status": {
			"type": "string",
			"enum": []any{"ok", "pending"},
		},
		"kind": {
			"const": "event",
		},
		"note": {
			"type": []any{"string", "null"},
		},
	}, "status", "kind", "note"))

	got, err := parser.Parse(context.Background(), `{"status":"ok","kind":"event","note":null}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got["note"] != nil {
		t.Fatalf("note: %#v", got)
	}
	_, err = parser.Parse(context.Background(), `{"status":"bad","kind":"event","note":null}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "enum") {
		t.Fatalf("enum err: %v", err)
	}
	_, err = parser.Parse(context.Background(), `{"status":"ok","kind":"other","note":null}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "const") {
		t.Fatalf("const err: %v", err)
	}
}

func TestPydanticParserStringNumberArrayConstraints(t *testing.T) {
	parser := NewPydanticParser[map[string]any](schema.Object(map[string]schema.Schema{
		"code": {
			"type":      "string",
			"minLength": 2,
			"maxLength": 4,
			"pattern":   `^[A-Z]+$`,
		},
		"score": {
			"type":    "number",
			"minimum": 0,
			"maximum": 10,
		},
		"items": {
			"type":     "array",
			"minItems": 1,
			"maxItems": 2,
			"items":    schema.String("item"),
		},
	}, "code", "score", "items"))

	if _, err := parser.Parse(context.Background(), `{"code":"AB","score":7.5,"items":["x"]}`); err != nil {
		t.Fatalf("parse valid: %v", err)
	}
	_, err := parser.Parse(context.Background(), `{"code":"a","score":7.5,"items":["x"]}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "at least") {
		t.Fatalf("string err: %v", err)
	}
	_, err = parser.Parse(context.Background(), `{"code":"AB","score":11,"items":["x"]}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("number err: %v", err)
	}
	_, err = parser.Parse(context.Background(), `{"code":"AB","score":7.5,"items":["x","y","z"]}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "items") {
		t.Fatalf("array err: %v", err)
	}
}

func TestPydanticParserCombinatorsAndAdditionalProperties(t *testing.T) {
	parser := NewPydanticParser[map[string]any](schema.Schema{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"allOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"minLength": 3},
				},
			},
			"id": map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
			"flag": map[string]any{
				"oneOf": []any{
					map[string]any{"const": true},
					map[string]any{"const": false},
				},
			},
		},
		"required":             []string{"name", "id", "flag"},
		"additionalProperties": false,
	})

	if _, err := parser.Parse(context.Background(), `{"name":"Ada","id":1,"flag":true}`); err != nil {
		t.Fatalf("parse valid: %v", err)
	}
	_, err := parser.Parse(context.Background(), `{"name":"Ada","id":{},"flag":true}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "anyOf") {
		t.Fatalf("anyOf err: %v", err)
	}
	_, err = parser.Parse(context.Background(), `{"name":"Ada","id":1,"flag":true,"extra":"no"}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) || !strings.Contains(err.Error(), "additional") {
		t.Fatalf("additional err: %v", err)
	}
}

func TestPydanticParserFormatInstructions(t *testing.T) {
	parser := NewPydanticParser[parsedPerson](schema.Schema{
		"title": "Person",
		"type":  "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"required": []string{"name"},
	})

	instructions := parser.FormatInstructions()
	if !strings.Contains(instructions, "JSON schema") {
		t.Fatalf("instructions: %s", instructions)
	}
	if strings.Contains(instructions, `"title":"Person"`) || strings.Contains(instructions, `"type":"object"`) {
		t.Fatalf("instructions should remove top-level title/type: %s", instructions)
	}
	if !strings.Contains(instructions, `"required":["name"]`) {
		t.Fatalf("instructions should include required fields: %s", instructions)
	}
}

func TestPydanticParserJSONModeFormatInstructions(t *testing.T) {
	parser := NewPydanticParserWithOptions[parsedPerson](
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
		}, "name"),
		WithPydanticInstructionStyle(PydanticInstructionJSONMode),
		WithPydanticInstructionIndentedSchema(true),
	)

	instructions := parser.FormatInstructions()
	if !strings.Contains(instructions, "Return only a valid JSON object") {
		t.Fatalf("instructions: %s", instructions)
	}
	if strings.Contains(instructions, "```") || strings.Contains(instructions, "well-formatted instance") {
		t.Fatalf("json mode instructions should be concise: %s", instructions)
	}
	if !strings.Contains(instructions, "\n  \"properties\"") {
		t.Fatalf("schema should be indented: %s", instructions)
	}
}

func TestPydanticParserProviderNativeFormatInstructions(t *testing.T) {
	parser := NewPydanticParserWithOptions[parsedPerson](
		schema.Object(map[string]schema.Schema{
			"name": schema.String("person name"),
		}, "name"),
		WithPydanticInstructionStyle(PydanticInstructionProviderNative),
		WithPydanticInstructionName("person_output"),
		WithPydanticInstructionStrict(true),
		WithPydanticInstructionSchema(false),
	)

	instructions := parser.FormatInstructions()
	if !strings.Contains(instructions, `provider-native structured output in strict mode named "person_output"`) {
		t.Fatalf("instructions: %s", instructions)
	}
	if strings.Contains(instructions, `"properties"`) || strings.Contains(instructions, "```") {
		t.Fatalf("provider native instructions should not inline schema: %s", instructions)
	}
}

type parsedPerson struct {
	Name string   `json:"name"`
	Age  int      `json:"age"`
	Tags []string `json:"tags"`
}

func TestPydanticParserInvalidJSON(t *testing.T) {
	parser := NewPydanticParser[map[string]any](schema.Schema{})
	_, err := parser.Parse(context.Background(), `not json at all`)
	if err == nil {
		t.Fatal("expected parse error")
	}
	if !strings.Contains(err.Error(), "parse pydantic json output") {
		t.Fatalf("err: %v", err)
	}
}

func TestPydanticParserDecodeTypedError(t *testing.T) {
	// Empty schema passes validation, but decoding into the typed struct fails.
	parser := NewPydanticParser[parsedPerson](schema.Schema{})
	_, err := parser.Parse(context.Background(), `{"name":"Ada","age":"old"}`)
	if !errors.Is(err, lcerrors.ErrSchemaValidation) {
		t.Fatalf("err: got %v", err)
	}
	if !strings.Contains(err.Error(), "decode typed pydantic output") {
		t.Fatalf("err should mention typed decode: %v", err)
	}
}

func TestPydanticParserJSONModeWithoutSchema(t *testing.T) {
	parser := NewPydanticParserWithOptions[parsedPerson](
		schema.Object(map[string]schema.Schema{"name": schema.String("name")}, "name"),
		WithPydanticInstructionStyle(PydanticInstructionJSONMode),
		WithPydanticInstructionSchema(false),
	)
	instructions := parser.FormatInstructions()
	if !strings.Contains(instructions, "Return only a valid JSON object.") {
		t.Fatalf("instructions: %s", instructions)
	}
	if strings.Contains(instructions, `"properties"`) {
		t.Fatalf("schema should be omitted: %s", instructions)
	}
}

func TestPydanticParserProviderNativeDefaults(t *testing.T) {
	parser := NewPydanticParserWithOptions[parsedPerson](
		schema.Object(map[string]schema.Schema{"name": schema.String("name")}, "name"),
		WithPydanticInstructionStyle(PydanticInstructionProviderNative),
	)
	instructions := parser.FormatInstructions()
	if !strings.Contains(instructions, `provider-native structured output named "structured_output"`) {
		t.Fatalf("instructions: %s", instructions)
	}
	if strings.Contains(instructions, "strict mode") {
		t.Fatalf("non-strict instructions should not mention strict mode: %s", instructions)
	}
}

func TestPydanticParserProviderNativeWithSchema(t *testing.T) {
	parser := NewPydanticParserWithOptions[parsedPerson](
		schema.Object(map[string]schema.Schema{"name": schema.String("name")}, "name"),
		WithPydanticInstructionStyle(PydanticInstructionProviderNative),
		WithPydanticInstructionSchema(true),
	)
	instructions := parser.FormatInstructions()
	if !strings.Contains(instructions, "conforms to this JSON schema") {
		t.Fatalf("instructions: %s", instructions)
	}
	if !strings.Contains(instructions, `"properties"`) {
		t.Fatalf("schema should be inlined: %s", instructions)
	}
}

func TestPydanticParserFormatInstructionsUnmarshalableSchema(t *testing.T) {
	parser := NewPydanticParser[map[string]any](schema.Schema{
		"properties": map[string]any{"bad": func() {}},
	})
	if got := parser.FormatInstructions(); got != "Return valid JSON matching the provided schema." {
		t.Fatalf("got %q", got)
	}
}

func TestValidateSchemaNullAndAny(t *testing.T) {
	if err := validateSchema(nil, schema.Schema{"type": "null"}, "$"); err != nil {
		t.Fatalf("null value: %v", err)
	}
	if err := validateSchema("x", schema.Schema{"type": "null"}, "$"); err == nil ||
		!strings.Contains(err.Error(), "expected null") {
		t.Fatalf("expected null mismatch error: %v", err)
	}
	if err := validateSchema(nil, schema.Schema{"type": []any{"string", "null"}}, "$"); err != nil {
		t.Fatalf("nullable null value: %v", err)
	}
	if err := validateSchema("x", schema.Schema{}, "$"); err != nil {
		t.Fatalf("empty spec: %v", err)
	}
	if err := validateSchema(42, schema.Schema{"type": "any"}, "$"); err != nil {
		t.Fatalf("any type: %v", err)
	}
}

func TestValidateSchemaBooleanAndUnsupported(t *testing.T) {
	if err := validateSchema(true, schema.Schema{"type": "boolean"}, "$"); err != nil {
		t.Fatalf("boolean: %v", err)
	}
	if err := validateSchema("x", schema.Schema{"type": "boolean"}, "$"); err == nil ||
		!strings.Contains(err.Error(), "expected boolean") {
		t.Fatalf("expected boolean mismatch: %v", err)
	}
	if err := validateSchema("x", schema.Schema{"type": "date"}, "$"); err == nil ||
		!strings.Contains(err.Error(), "unsupported schema type") {
		t.Fatalf("expected unsupported type error: %v", err)
	}
}

func TestValidateNumberExclusiveBounds(t *testing.T) {
	spec := schema.Schema{"type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": 5}
	if err := validateSchema(json.Number("3"), spec, "$"); err != nil {
		t.Fatalf("in range: %v", err)
	}
	if err := validateSchema(json.Number("1"), spec, "$"); err == nil ||
		!strings.Contains(err.Error(), "greater than") {
		t.Fatalf("exclusiveMinimum: %v", err)
	}
	if err := validateSchema(json.Number("5"), spec, "$"); err == nil ||
		!strings.Contains(err.Error(), "less than") {
		t.Fatalf("exclusiveMaximum: %v", err)
	}
}

func TestValidateStringPatternAndLength(t *testing.T) {
	if err := validateSchema("AB", schema.Schema{"type": "string", "pattern": "["}, "$"); err == nil ||
		!strings.Contains(err.Error(), "invalid pattern") {
		t.Fatalf("invalid pattern: %v", err)
	}
	if err := validateSchema("ab", schema.Schema{"type": "string", "pattern": `^[A-Z]+$`}, "$"); err == nil ||
		!strings.Contains(err.Error(), "does not match pattern") {
		t.Fatalf("pattern mismatch: %v", err)
	}
	if err := validateSchema("abcde", schema.Schema{"type": "string", "maxLength": 3}, "$"); err == nil ||
		!strings.Contains(err.Error(), "at most") {
		t.Fatalf("maxLength: %v", err)
	}
}

func TestValidateObjectAdditionalPropertiesSchema(t *testing.T) {
	spec := schema.Schema{
		"type":                 "object",
		"properties":           map[string]any{"a": map[string]any{"type": "string"}},
		"additionalProperties": map[string]any{"type": "integer"},
	}
	if err := validateSchema(map[string]any{"a": "x", "b": json.Number("2")}, spec, "$"); err != nil {
		t.Fatalf("valid additional: %v", err)
	}
	if err := validateSchema(map[string]any{"a": "x", "b": "nope"}, spec, "$"); err == nil {
		t.Fatal("expected additional property schema error")
	}

	nonSchemaProps := schema.Schema{
		"type":       "object",
		"properties": map[string]any{"a": "not a schema"},
	}
	if err := validateSchema(map[string]any{"a": 1}, nonSchemaProps, "$"); err != nil {
		t.Fatalf("non-schema property should be skipped: %v", err)
	}

	nonMapProps := schema.Schema{
		"type":                 "object",
		"properties":           "not a map",
		"additionalProperties": false,
	}
	if err := validateSchema(map[string]any{"a": 1}, nonMapProps, "$"); err == nil ||
		!strings.Contains(err.Error(), "additional property not allowed") {
		t.Fatalf("additionalProperties=false: %v", err)
	}
}

func TestValidateArrayEdgeCases(t *testing.T) {
	if err := validateSchema("x", schema.Schema{"type": "array"}, "$"); err == nil ||
		!strings.Contains(err.Error(), "expected array") {
		t.Fatalf("non-array: %v", err)
	}
	spec := schema.Schema{"type": "array", "items": map[string]any{"type": "string"}}
	if err := validateSchema([]any{"ok", json.Number("1")}, spec, "$"); err == nil ||
		!strings.Contains(err.Error(), "$[1]") {
		t.Fatalf("item error should include index: %v", err)
	}
	if err := validateSchema([]any{1}, schema.Schema{"type": "array", "items": "not a schema"}, "$"); err != nil {
		t.Fatalf("non-schema items should be skipped: %v", err)
	}
	if err := validateSchema([]any{}, schema.Schema{"type": "array", "minItems": 1}, "$"); err == nil ||
		!strings.Contains(err.Error(), "at least") {
		t.Fatalf("minItems: %v", err)
	}
}

func TestValidateCombinatorsEdgeCases(t *testing.T) {
	allOf := schema.Schema{"allOf": []any{map[string]any{"type": "string"}}}
	if err := validateSchema(json.Number("1"), allOf, "$"); err == nil ||
		!strings.Contains(err.Error(), "allOf failed") {
		t.Fatalf("allOf: %v", err)
	}
	oneOfNone := schema.Schema{"oneOf": []any{
		map[string]any{"type": "string"},
		map[string]any{"type": "boolean"},
	}}
	if err := validateSchema(json.Number("1"), oneOfNone, "$"); err == nil ||
		!strings.Contains(err.Error(), "exactly one match, got 0") {
		t.Fatalf("oneOf zero matches: %v", err)
	}
	oneOfTwo := schema.Schema{"oneOf": []any{
		map[string]any{"type": "string"},
		map[string]any{"minLength": 1},
	}}
	if err := validateSchema("x", oneOfTwo, "$"); err == nil ||
		!strings.Contains(err.Error(), "exactly one match, got 2") {
		t.Fatalf("oneOf two matches: %v", err)
	}
}

func TestNumericConstraintTypes(t *testing.T) {
	if v, ok := numericConstraint(json.Number("1.5")); !ok || v != 1.5 {
		t.Fatalf("json.Number: %v %v", v, ok)
	}
	if _, ok := numericConstraint(json.Number("bad")); ok {
		t.Fatal("invalid json.Number should fail")
	}
	if v, ok := numericConstraint(float32(2.5)); !ok || v != 2.5 {
		t.Fatalf("float32: %v %v", v, ok)
	}
	if v, ok := numericConstraint(int8(3)); !ok || v != 3 {
		t.Fatalf("int8: %v %v", v, ok)
	}
	if v, ok := numericConstraint(uint64(4)); !ok || v != 4 {
		t.Fatalf("uint64: %v %v", v, ok)
	}
	if _, ok := numericConstraint("x"); ok {
		t.Fatal("string should fail")
	}
}

func TestNumberFloatTypes(t *testing.T) {
	if v, ok := numberFloat(json.Number("2.5")); !ok || v != 2.5 {
		t.Fatalf("json.Number: %v %v", v, ok)
	}
	if _, ok := numberFloat(json.Number("bad")); ok {
		t.Fatal("invalid json.Number should fail")
	}
	if v, ok := numberFloat(int64(7)); !ok || v != 7 {
		t.Fatalf("int64: %v %v", v, ok)
	}
	if _, ok := numberFloat(true); ok {
		t.Fatal("bool should fail")
	}
}

func TestRequiredKeysAndAnySlice(t *testing.T) {
	if got := requiredKeys([]string{"a", "b"}); len(got) != 2 || got[0] != "a" {
		t.Fatalf("[]string: %#v", got)
	}
	if got := requiredKeys([]any{"a", 1, "b"}); len(got) != 2 || got[1] != "b" {
		t.Fatalf("[]any filters non-strings: %#v", got)
	}
	if got := requiredKeys("x"); got != nil {
		t.Fatalf("default: %#v", got)
	}
	if got := anySlice([]string{"a"}); len(got) != 1 || got[0] != "a" {
		t.Fatalf("[]string: %#v", got)
	}
	if got := anySlice([]schema.Schema{{"type": "string"}}); len(got) != 1 {
		t.Fatalf("[]schema.Schema: %#v", got)
	}
	if got := anySlice(42); got != nil {
		t.Fatalf("default: %#v", got)
	}
}

func TestIsIntegerAndIsNumber(t *testing.T) {
	if !isInteger(json.Number("3.0")) {
		t.Fatal("3.0 should be an integer")
	}
	if isInteger(json.Number("3.5")) {
		t.Fatal("3.5 should not be an integer")
	}
	if isInteger(json.Number("bad")) {
		t.Fatal("invalid json.Number should not be an integer")
	}
	if !isInteger(float64(2)) || isInteger(2.5) {
		t.Fatal("float64 integer check failed")
	}
	if !isInteger(5) || isInteger("x") {
		t.Fatal("int/string integer check failed")
	}
	if isNumber(json.Number("bad")) {
		t.Fatal("invalid json.Number should not be a number")
	}
	if !isNumber(uint(3)) || isNumber(true) {
		t.Fatal("number check failed")
	}
}

func TestJSONEqual(t *testing.T) {
	if !jsonEqual(map[string]any{"a": 1}, map[string]any{"a": 1}) {
		t.Fatal("equal maps should match")
	}
	if jsonEqual(map[string]any{"a": 1}, map[string]any{"a": 2}) {
		t.Fatal("different maps should not match")
	}
	// json.Marshal fails on funcs, exercising the fmt.Sprint fallback.
	if jsonEqual(func() {}, "x") {
		t.Fatal("unmarshalable values should fall back to fmt.Sprint comparison")
	}
}

func TestAsSchemaAndSchemaTypes(t *testing.T) {
	if _, ok := asSchema(schema.Schema{}); !ok {
		t.Fatal("schema.Schema should convert")
	}
	if _, ok := asSchema(map[string]any{}); !ok {
		t.Fatal("map[string]any should convert")
	}
	if _, ok := asSchema("x"); ok {
		t.Fatal("string should not convert")
	}
	if got := schemaTypes("string"); len(got) != 1 || got[0] != "string" {
		t.Fatalf("string: %#v", got)
	}
	if got := schemaTypes([]string{"a", "b"}); len(got) != 2 {
		t.Fatalf("[]string: %#v", got)
	}
	if got := schemaTypes([]any{"a", 1}); len(got) != 1 || got[0] != "a" {
		t.Fatalf("[]any filters non-strings: %#v", got)
	}
	if got := schemaTypes(1); got != nil {
		t.Fatalf("default: %#v", got)
	}
}

func TestNumericConstraintAllNumericTypes(t *testing.T) {
	values := []struct {
		name  string
		value any
		want  float64
	}{
		{"float64", float64(1), 1},
		{"float32", float32(2), 2},
		{"int", int(3), 3},
		{"int8", int8(4), 4},
		{"int16", int16(5), 5},
		{"int32", int32(6), 6},
		{"int64", int64(7), 7},
		{"uint", uint(8), 8},
		{"uint8", uint8(9), 9},
		{"uint16", uint16(10), 10},
		{"uint32", uint32(11), 11},
		{"uint64", uint64(12), 12},
	}
	for _, tc := range values {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := numericConstraint(tc.value)
			if !ok || got != tc.want {
				t.Fatalf("numericConstraint(%T): got %v ok=%v", tc.value, got, ok)
			}
			fgot, ok := numberFloat(tc.value)
			if !ok || fgot != tc.want {
				t.Fatalf("numberFloat(%T): got %v ok=%v", tc.value, fgot, ok)
			}
		})
	}
}

func TestPydanticParserDefaultStyleKeepsExplicitSchemaOption(t *testing.T) {
	// A non-zero options struct with the default style still strips the
	// top-level title/type from the emitted schema.
	parser := NewPydanticParserWithOptions[parsedPerson](
		schema.Schema{
			"title": "Person",
			"type":  "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
			},
		},
		WithPydanticInstructionSchema(true),
	)
	instructions := parser.FormatInstructions()
	if strings.Contains(instructions, `"title":"Person"`) || strings.Contains(instructions, `"type":"object"`) {
		t.Fatalf("instructions should remove top-level title/type: %s", instructions)
	}
	if !strings.Contains(instructions, `"properties"`) {
		t.Fatalf("instructions should include schema: %s", instructions)
	}
}
