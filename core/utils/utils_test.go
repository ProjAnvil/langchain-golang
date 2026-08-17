package utils

import (
	"context"
	"reflect"
	"strings"
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

	// A negative buffer is normalized to unbuffered and still delivers values.
	negValues, negErrs := IteratorToChannel(context.Background(), NewSliceIterator([]int{7}), -1)
	if got := <-negValues; got != 7 {
		t.Fatalf("negative buffer values: %#v", got)
	}
	if _, ok := <-negValues; ok {
		t.Fatal("values channel should be closed")
	}
	if err, ok := <-negErrs; ok && err != nil {
		t.Fatalf("unexpected error: %v", err)
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

// errIterator fails on the first Next call.
type errIterator struct{ err error }

func (e *errIterator) Next(context.Context) (int, bool, error) { return 0, false, e.err }
func (e *errIterator) Close() error                            { return nil }

type stringerValue struct{ text string }

func (s stringerValue) String() string { return s.text }

func TestCollectIteratorNilAndError(t *testing.T) {
	if _, err := CollectIterator[int](context.Background(), nil); err == nil {
		t.Fatal("expected nil iterator error")
	}
	wantErr := errIterator{err: context.DeadlineExceeded}
	if _, err := CollectIterator(context.Background(), &wantErr); err != context.DeadlineExceeded {
		t.Fatalf("expected Next error to propagate, got %v", err)
	}
}

func TestIteratorToChannelContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	// Unbuffered channel with no reader: the goroutine blocks on send until
	// the context is canceled.
	values, errs := IteratorToChannel(ctx, NewSliceIterator([]int{1, 2, 3}), 0)
	cancel()
	if err := <-errs; err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	// Drain values in case a send raced the cancellation.
	for range values {
	}
}

func TestIteratorToChannelNextError(t *testing.T) {
	iter := &errIterator{err: context.DeadlineExceeded}
	values, errs := IteratorToChannel(context.Background(), iter, 1)
	if err := <-errs; err != context.DeadlineExceeded {
		t.Fatalf("expected Next error, got %v", err)
	}
	if _, ok := <-values; ok {
		t.Fatal("values channel should be closed without values")
	}
}

func TestMustGetFromEnvSuccess(t *testing.T) {
	t.Setenv("LC_TEST_REQUIRED", "present")
	got, err := MustGetFromEnv("MISSING", "LC_TEST_REQUIRED")
	if err != nil || got != "present" {
		t.Fatalf("MustGetFromEnv = %q, %v", got, err)
	}
}

func TestToJSONString(t *testing.T) {
	got, err := ToJSONString(map[string]any{"a": 1})
	if err != nil {
		t.Fatal(err)
	}
	if got != `{"a":1}` {
		t.Fatalf("ToJSONString = %q", got)
	}
	if _, err := ToJSONString(map[string]any{"f": func() {}}); err == nil {
		t.Fatal("expected marshal error for func value")
	}
}

func TestStringifyValue(t *testing.T) {
	if got := StringifyValue("plain"); got != "plain" {
		t.Fatalf("string: %q", got)
	}
	if got := StringifyValue(stringerValue{text: "stringer"}); got != "stringer" {
		t.Fatalf("Stringer: %q", got)
	}
	if got := StringifyValue(42); got != "42" {
		t.Fatalf("default: %q", got)
	}
}

func TestMergeMapsNilBase(t *testing.T) {
	got := MergeMaps(nil, map[string]any{"a": 1})
	if got["a"] != 1 {
		t.Fatalf("unexpected merge: %#v", got)
	}
}

func TestNewFunctionSpecValidationAndDefaults(t *testing.T) {
	if _, err := NewFunctionSpec("", "", nil); err == nil {
		t.Fatal("expected error for empty name")
	}
	if _, err := NewFunctionSpec(strings.Repeat("a", 65), "", nil); err == nil {
		t.Fatal("expected error for name longer than 64 characters")
	}
	fn, err := NewFunctionSpec("ok_name", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if fn.Parameters["type"] != "object" {
		t.Fatalf("nil parameters should default to object schema: %#v", fn.Parameters)
	}
}

func TestCloneSchemaNil(t *testing.T) {
	if got := CloneSchema(nil); got != nil {
		t.Fatalf("CloneSchema(nil) = %#v, want nil", got)
	}
}

func TestSchemaPropertiesVariants(t *testing.T) {
	if got := SchemaProperties(schema.Schema{"type": "object"}); len(got) != 0 {
		t.Fatalf("missing properties: %#v", got)
	}
	if got := SchemaProperties(schema.Schema{"properties": nil}); len(got) != 0 {
		t.Fatalf("nil properties: %#v", got)
	}
	typed := schema.Schema{
		"properties": map[string]schema.Schema{
			"name": schema.String("name"),
		},
	}
	got := SchemaProperties(typed)
	if got["name"]["type"] != "string" {
		t.Fatalf("typed properties: %#v", got)
	}
	plain := schema.Schema{
		"properties": map[string]any{
			"age": map[string]any{"type": "integer"},
			"raw": "not a schema",
		},
	}
	got = SchemaProperties(plain)
	if got["age"]["type"] != "integer" {
		t.Fatalf("plain map properties: %#v", got)
	}
	if _, ok := got["raw"]; ok {
		t.Fatalf("non-schema values should be skipped: %#v", got)
	}
}

func TestSchemaRequiredVariants(t *testing.T) {
	if got := SchemaRequired(schema.Schema{"type": "object"}); got != nil {
		t.Fatalf("missing required: %#v", got)
	}
	if got := SchemaRequired(schema.Schema{"required": nil}); got != nil {
		t.Fatalf("nil required: %#v", got)
	}
	got := SchemaRequired(schema.Schema{"required": []any{"a", 1, "b"}})
	if !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("[]any required: %#v", got)
	}
	if got := SchemaRequired(schema.Schema{"required": "name"}); got != nil {
		t.Fatalf("non-list required: %#v", got)
	}
}

func TestCreateSubsetSchemaErrorsAndDescriptionFallback(t *testing.T) {
	if _, err := CreateSubsetSchema("", schema.String("not an object"), nil, nil, ""); err == nil {
		t.Fatal("expected error for non-object schema")
	}
	input := schema.Object(map[string]schema.Schema{"name": schema.String("name")})
	input["description"] = "original description"
	subset, err := CreateSubsetSchema("", input, []string{"name"}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if subset["description"] != "original description" {
		t.Fatalf("description fallback: %#v", subset)
	}
	if _, ok := subset["title"]; ok {
		t.Fatalf("empty name should not set title: %#v", subset)
	}
}

func TestMustacheEdgeCases(t *testing.T) {
	// {{& name}} contributes the variable in template-variable extraction.
	if got := MustacheTemplateVariables("{{& html}}"); !reflect.DeepEqual(got, []string{"html"}) {
		t.Fatalf("variables: %#v", got)
	}
	// Empty tags are not variables and are left unchanged when rendering.
	if got := MustacheTemplateVariables("{{ }} {{name}}"); !reflect.DeepEqual(got, []string{"name"}) {
		t.Fatalf("variables with empty tag: %#v", got)
	}
	got := RenderSimpleMustache("a {{ }} b", map[string]any{})
	if got != "a {{ }} b" {
		t.Fatalf("empty tag render: %q", got)
	}
	// Dotted lookup through a non-map value resolves to empty.
	got = RenderSimpleMustache("{{user.name.first}}", map[string]any{"user": map[string]any{"name": "Ada"}})
	if got != "" {
		t.Fatalf("non-map dotted lookup: %q", got)
	}
	// Explicit nil values render as empty.
	got = RenderSimpleMustache("{{value}}", map[string]any{"value": nil})
	if got != "" {
		t.Fatalf("nil value render: %q", got)
	}
}
