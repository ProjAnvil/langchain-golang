package outputparser

import (
	"context"
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
