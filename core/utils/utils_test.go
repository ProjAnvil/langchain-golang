package utils

import (
	"context"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestMergeMapsRecursive(t *testing.T) {
	got := MergeMaps(
		map[string]any{"a": map[string]any{"x": 1}, "b": 1},
		map[string]any{"a": map[string]any{"y": 2}},
	)
	nested := got["a"].(map[string]any)
	if nested["x"] != 1 || nested["y"] != 2 || got["b"] != 1 {
		t.Fatalf("unexpected merge: %#v", got)
	}
}

func TestIteratorHelpers(t *testing.T) {
	iter := NewSliceIterator([]string{"a", "b"})
	values, err := CollectIterator(context.Background(), iter)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(values, []string{"a", "b"}) {
		t.Fatalf("values: %#v", values)
	}

	values[0] = "changed"
	again, err := CollectIterator(context.Background(), NewSliceIterator([]string{"a"}))
	if err != nil {
		t.Fatal(err)
	}
	if again[0] != "a" {
		t.Fatalf("slice iterator did not copy values: %#v", again)
	}
}

func TestIteratorHelpersRespectContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := CollectIterator(ctx, NewSliceIterator([]int{1}))
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}

func TestIteratorToChannel(t *testing.T) {
	values, errs := IteratorToChannel(context.Background(), NewSliceIterator([]int{1, 2}), 1)
	got := []int{}
	for value := range values {
		got = append(got, value)
	}
	if !reflect.DeepEqual(got, []int{1, 2}) {
		t.Fatalf("values: %#v", got)
	}
	if err, ok := <-errs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	_, nilErrs := IteratorToChannel[int](context.Background(), nil, 0)
	if err := <-nilErrs; err == nil {
		t.Fatal("expected nil iterator error")
	}
}

func TestEnvHelpers(t *testing.T) {
	t.Setenv("LC_TEST_KEY", "value")
	if got := GetFromEnv([]string{"MISSING", "LC_TEST_KEY"}, "default"); got != "value" {
		t.Fatalf("GetFromEnv = %q", got)
	}
	if _, err := MustGetFromEnv("MISSING"); err == nil {
		t.Fatal("expected missing env error")
	}
}

func TestUUIDAndEnsureID(t *testing.T) {
	if got, err := EnsureID("existing"); err != nil || got != "existing" {
		t.Fatalf("EnsureID existing = %q, %v", got, err)
	}
	got, err := EnsureID("")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len("lc_00000000-0000-0000-0000-000000000000") || got[:3] != "lc_" {
		t.Fatalf("unexpected generated id: %q", got)
	}
}

func TestFunctionCallingHelpers(t *testing.T) {
	params := schema.Object(map[string]schema.Schema{"query": schema.String("query")}, "query")
	fn, err := NewFunctionSpec("search_tool", "searches", params)
	if err != nil {
		t.Fatal(err)
	}
	tool := ConvertToOpenAITool(fn)
	if tool.Type != "function" || tool.Function.Name != "search_tool" {
		t.Fatalf("unexpected tool: %#v", tool)
	}
	fn.Parameters["type"] = "changed"
	if params["type"] != "object" {
		t.Fatal("parameters were not copied")
	}
	if _, err := NewFunctionSpec("bad name", "", params); err == nil {
		t.Fatal("expected invalid name error")
	}
	if _, err := NewFunctionSpec("bad", "", schema.String("not object")); err == nil {
		t.Fatal("expected invalid schema error")
	}
}

func TestRemoveJSONSchemaDefinitions(t *testing.T) {
	got := RemoveJSONSchemaDefinitions(schema.Schema{
		"type":        "object",
		"$defs":       map[string]any{"X": "Y"},
		"definitions": map[string]any{"A": "B"},
	})
	if _, ok := got["$defs"]; ok {
		t.Fatal("$defs was not removed")
	}
	if _, ok := got["definitions"]; ok {
		t.Fatal("definitions was not removed")
	}
	if got["type"] != "object" {
		t.Fatalf("unexpected schema: %#v", got)
	}
}

func TestSchemaOrientedPydanticHelpers(t *testing.T) {
	input := schema.Object(map[string]schema.Schema{
		"name": schema.String("old name"),
		"age":  schema.Integer("age"),
		"city": schema.String("city"),
	}, "name", "age")
	input["description"] = "person schema"

	if !IsObjectSchema(input) {
		t.Fatal("object schema not detected")
	}
	props := SchemaProperties(input)
	props["name"]["description"] = "changed"
	originalProps := input["properties"].(map[string]any)
	if originalProps["name"].(schema.Schema)["description"] != "old name" {
		t.Fatal("properties were not defensively copied")
	}
	if got := SchemaRequired(input); !reflect.DeepEqual(got, []string{"name", "age"}) {
		t.Fatalf("required: %#v", got)
	}

	subset, err := CreateSubsetSchema(
		"PersonSubset",
		input,
		[]string{"name", "city"},
		map[string]string{"name": "new name"},
		"subset description",
	)
	if err != nil {
		t.Fatalf("subset schema: %v", err)
	}
	if subset["title"] != "PersonSubset" || subset["description"] != "subset description" {
		t.Fatalf("metadata: %#v", subset)
	}
	subsetProps := SchemaProperties(subset)
	if len(subsetProps) != 2 || subsetProps["name"]["description"] != "new name" {
		t.Fatalf("subset properties: %#v", subsetProps)
	}
	if got := SchemaRequired(subset); !reflect.DeepEqual(got, []string{"name"}) {
		t.Fatalf("subset required: %#v", got)
	}
	if _, err := CreateSubsetSchema("", input, []string{"missing"}, nil, ""); err == nil {
		t.Fatal("expected missing field error")
	}
}

func TestMustacheTemplateVariables(t *testing.T) {
	got := MustacheTemplateVariables("Hello {{user.name}}, role {{user.role}}, raw {{{html}}}, skip {{#items}}{{name}}{{/items}}")
	want := []string{"html", "name", "user"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("variables: got %#v want %#v", got, want)
	}
}

func TestRenderSimpleMustache(t *testing.T) {
	got := RenderSimpleMustache(
		"Hello {{user.name}} {{missing}} {{html}} {{{html}}} {{& html}} {{! comment}} {{#items}}x{{/items}}",
		map[string]any{
			"user": map[string]any{"name": "Ada"},
			"html": "<b>ok</b>",
		},
	)
	want := "Hello Ada  &lt;b&gt;ok&lt;/b&gt; <b>ok</b> <b>ok</b>  {{#items}}x{{/items}}"
	if got != want {
		t.Fatalf("render: got %q want %q", got, want)
	}
}
