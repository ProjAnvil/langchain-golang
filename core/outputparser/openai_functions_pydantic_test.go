package outputparser

import (
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func pydanticFunctionMessage(name string, arguments string) messages.Message {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
	return msg
}

func cookieSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"name": schema.String("cookie name"),
		"age":  schema.Integer("cookie age"),
	}, "name", "age")
}

func dogSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"species": schema.String("dog species"),
	}, "species")
}

func TestPydanticOutputFunctionsParserSingleSchema(t *testing.T) {
	parser := PydanticOutputFunctionsParser{Schema: cookieSchema(), ArgsOnly: true}
	got, err := parser.ParseMessage(pydanticFunctionMessage("cookie", `{"name":"chip","age":3}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	object, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("result is not an object: %#v", got)
	}
	if object["name"] != "chip" || object["age"].(float64) != 3 {
		t.Fatalf("object: %#v", object)
	}
}

func TestPydanticOutputFunctionsParserMultipleSchemas(t *testing.T) {
	parser := PydanticOutputFunctionsParser{
		Schemas: map[string]schema.Schema{
			"cookie": cookieSchema(),
			"dog":    dogSchema(),
		},
	}
	got, err := parser.ParseMessage(pydanticFunctionMessage("dog", `{"species":"lab"}`))
	if err != nil {
		t.Fatalf("parse dog: %v", err)
	}
	if got.(map[string]any)["species"] != "lab" {
		t.Fatalf("dog object: %#v", got)
	}

	if _, err := parser.ParseMessage(pydanticFunctionMessage("unknown", `{}`)); err == nil ||
		!strings.Contains(err.Error(), "no pydantic schema") {
		t.Fatalf("unknown name: %v", err)
	}
}

func TestPydanticOutputFunctionsParserValidationErrors(t *testing.T) {
	parser := PydanticOutputFunctionsParser{Schema: cookieSchema(), ArgsOnly: true}

	if _, err := parser.ParseMessage(pydanticFunctionMessage("cookie", `{"name":"chip"}`)); err == nil ||
		!strings.Contains(err.Error(), "$.age") {
		t.Fatalf("missing required field: %v", err)
	}
	if _, err := parser.ParseMessage(pydanticFunctionMessage("cookie", `{"name":"chip","age":"old"}`)); err == nil ||
		!strings.Contains(err.Error(), "expected integer") {
		t.Fatalf("wrong type: %v", err)
	}
	if _, err := parser.ParseMessage(pydanticFunctionMessage("cookie", `{bad`)); err == nil ||
		!strings.Contains(err.Error(), "parse function call arguments") {
		t.Fatalf("invalid JSON: %v", err)
	}

	// A schema that accepts any value exposes the non-object guard.
	lenient := PydanticOutputFunctionsParser{Schema: schema.Schema{"type": "any"}, ArgsOnly: true}
	if _, err := lenient.ParseMessage(pydanticFunctionMessage("x", `[1,2]`)); err == nil ||
		!strings.Contains(err.Error(), "not a JSON object") {
		t.Fatalf("non-object arguments: %v", err)
	}
}

func TestPydanticOutputFunctionsParserConfigErrors(t *testing.T) {
	parser := PydanticOutputFunctionsParser{
		Schema:   cookieSchema(),
		Schemas:  map[string]schema.Schema{"cookie": cookieSchema()},
		ArgsOnly: true,
	}
	if _, err := parser.ParseMessage(pydanticFunctionMessage("cookie", `{"name":"chip","age":3}`)); err == nil ||
		!strings.Contains(err.Error(), "ArgsOnly should be False") {
		t.Fatalf("ArgsOnly with multiple schemas: %v", err)
	}

	empty := PydanticOutputFunctionsParser{}
	if _, err := empty.ParseMessage(pydanticFunctionMessage("cookie", `{}`)); err == nil ||
		!strings.Contains(err.Error(), "pydantic schema is required") {
		t.Fatalf("missing schema: %v", err)
	}

	if _, err := parser.ParseMessage(messages.AI("")); err == nil ||
		!strings.Contains(err.Error(), "missing function_call") {
		t.Fatalf("missing function_call: %v", err)
	}
}

func TestPydanticOutputFunctionsParserFunctionCallVariant(t *testing.T) {
	msg := messages.AI("")
	msg.AdditionalKwargs = map[string]any{
		"function_call": FunctionCall{Name: "cookie", Arguments: `{"name":"chip","age":3}`},
	}
	parser := PydanticOutputFunctionsParser{Schema: cookieSchema(), ArgsOnly: true}
	got, err := parser.ParseMessage(msg)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.(map[string]any)["name"] != "chip" {
		t.Fatalf("object: %#v", got)
	}
}

func TestPydanticAttrOutputFunctionsParser(t *testing.T) {
	parser := PydanticAttrOutputFunctionsParser{
		PydanticOutputFunctionsParser: PydanticOutputFunctionsParser{
			Schema:   cookieSchema(),
			ArgsOnly: true,
		},
		AttrName: "name",
	}
	got, err := parser.ParseMessage(pydanticFunctionMessage("cookie", `{"name":"chip","age":3}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != "chip" {
		t.Fatalf("attribute: %#v", got)
	}

	missing := PydanticAttrOutputFunctionsParser{
		PydanticOutputFunctionsParser: PydanticOutputFunctionsParser{
			Schema:   cookieSchema(),
			ArgsOnly: true,
		},
		AttrName: "absent",
	}
	if _, err := missing.ParseMessage(pydanticFunctionMessage("cookie", `{"name":"chip","age":3}`)); err == nil ||
		!strings.Contains(err.Error(), "not found") {
		t.Fatalf("missing attribute: %v", err)
	}
}

func TestPydanticAttrOutputFunctionsParserWithMultipleSchemas(t *testing.T) {
	parser := PydanticAttrOutputFunctionsParser{
		PydanticOutputFunctionsParser: PydanticOutputFunctionsParser{
			Schemas: map[string]schema.Schema{"dog": dogSchema()},
		},
		AttrName: "species",
	}
	got, err := parser.ParseMessage(pydanticFunctionMessage("dog", `{"species":"poodle"}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != "poodle" {
		t.Fatalf("attribute: %#v", got)
	}
}
