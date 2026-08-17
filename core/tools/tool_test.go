package tools

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestFuncToolInvoke(t *testing.T) {
	tool, err := NewFunc(
		"adder",
		"adds two integers",
		schema.Object(
			map[string]schema.Schema{
				"a": schema.Integer("left side"),
				"b": schema.Integer("right side"),
			},
			"a",
			"b",
		),
		func(_ context.Context, input map[string]any) (Result, error) {
			a, ok := input["a"].(int)
			if !ok {
				return Result{}, fmt.Errorf("a must be int")
			}
			b, ok := input["b"].(int)
			if !ok {
				return Result{}, fmt.Errorf("b must be int")
			}
			return Result{Content: fmt.Sprintf("%d", a+b)}, nil
		},
	)
	if err != nil {
		t.Fatalf("new tool: %v", err)
	}

	got, err := tool.Invoke(context.Background(), map[string]any{"a": 2, "b": 3})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.Content != "5" {
		t.Fatalf("content: got %q want %q", got.Content, "5")
	}
}

func TestSimpleToolParity(t *testing.T) {
	tool, err := NewSimple("echo", "echoes input", func(_ context.Context, input string) (Result, error) {
		return Result{Content: "echo: " + input}, nil
	})
	if err != nil {
		t.Fatalf("new simple tool: %v", err)
	}
	if tool.Name() != "echo" || tool.Description() != "echoes input" {
		t.Fatalf("metadata: %s %s", tool.Name(), tool.Description())
	}
	args := tool.ArgsSchema()
	properties := args["properties"].(map[string]any)
	if _, ok := properties["tool_input"]; !ok {
		t.Fatalf("args schema: %#v", args)
	}

	got, err := tool.InvokeString(context.Background(), "hello")
	if err != nil {
		t.Fatalf("invoke string: %v", err)
	}
	if got.Content != "echo: hello" {
		t.Fatalf("content: %q", got.Content)
	}
	got, err = tool.Invoke(context.Background(), map[string]any{"tool_input": "world"})
	if err != nil {
		t.Fatalf("invoke map: %v", err)
	}
	if got.Content != "echo: world" {
		t.Fatalf("content: %q", got.Content)
	}
	got, err = tool.Invoke(context.Background(), map[string]any{"query": "compat"})
	if err != nil {
		t.Fatalf("invoke compat map: %v", err)
	}
	if got.Content != "echo: compat" {
		t.Fatalf("content: %q", got.Content)
	}
	if _, err := tool.Invoke(context.Background(), map[string]any{"a": "1", "b": "2"}); err == nil {
		t.Fatal("expected too many arguments error")
	}
	if _, err := tool.Invoke(context.Background(), map[string]any{"tool_input": 3}); err == nil {
		t.Fatal("expected non-string input error")
	}
}

func TestStructuredFuncConstructorParity(t *testing.T) {
	tool, err := NewStructuredFunc(
		"adder",
		"adds",
		schema.Object(map[string]schema.Schema{
			"a": schema.Integer("left"),
			"b": schema.Integer("right"),
		}, "a", "b"),
		func(_ context.Context, input map[string]any) (Result, error) {
			return Result{Content: fmt.Sprintf("%v%v", input["a"], input["b"])}, nil
		},
	)
	if err != nil {
		t.Fatalf("new structured tool: %v", err)
	}
	if tool.Name() != "adder" || tool.ArgsSchema()["type"] != "object" {
		t.Fatalf("tool: %#v", tool)
	}
}

func TestFuncInvokeRequiresFunction(t *testing.T) {
	var tool Func
	if _, err := tool.Invoke(context.Background(), map[string]any{}); err == nil {
		t.Fatal("expected missing function error")
	}
}

func TestRenderTools(t *testing.T) {
	tool, err := NewFunc(
		"search",
		"searches docs",
		schema.Object(map[string]schema.Schema{"query": schema.String("query")}, "query"),
		func(context.Context, map[string]any) (Result, error) { return Result{}, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := RenderTextDescription([]Tool{tool}); got != "search - searches docs" {
		t.Fatalf("RenderTextDescription = %q", got)
	}
	got := RenderTextDescriptionAndArgs([]Tool{tool})
	if !strings.Contains(got, "search - searches docs, args:") || !strings.Contains(got, `"query"`) {
		t.Fatalf("unexpected args rendering: %q", got)
	}
}

func TestToolConversionHelpers(t *testing.T) {
	tool, err := NewFunc(
		"search",
		"searches docs",
		schema.Object(map[string]schema.Schema{"query": schema.String("query")}, "query"),
		func(context.Context, map[string]any) (Result, error) { return Result{}, nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := ToFunctionSpec(tool)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Name != "search" || spec.Parameters["type"] != "object" {
		t.Fatalf("unexpected spec: %#v", spec)
	}
	openAITool, err := ToOpenAIToolSpec(tool)
	if err != nil {
		t.Fatal(err)
	}
	if openAITool.Type != "function" || openAITool.Function.Name != "search" {
		t.Fatalf("unexpected OpenAI tool: %#v", openAITool)
	}
}

func TestCreateRetrieverTool(t *testing.T) {
	retriever := retrievers.Static{Documents: []documents.Document{
		documents.New("alpha", map[string]any{"source": "a"}),
		documents.New("beta", map[string]any{"source": "b"}),
	}}
	tool, err := CreateRetrieverTool(retriever, "lookup", "looks up docs", RetrieverToolOptions{
		DocumentSeparator: " | ",
		IncludeArtifact:   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := tool.Invoke(context.Background(), map[string]any{"query": "anything"})
	if err != nil {
		t.Fatal(err)
	}
	if got.Content != "alpha | beta" {
		t.Fatalf("Content = %q", got.Content)
	}
	if _, ok := got.Artifact.([]documents.Document); !ok {
		t.Fatalf("Artifact type = %T, want []documents.Document", got.Artifact)
	}
}

func TestValidateArgsSchemaRejectsNonObject(t *testing.T) {
	_, err := NewFunc("bad", "", schema.String("nope"), func(context.Context, map[string]any) (Result, error) {
		return Result{}, nil
	})
	if err == nil {
		t.Fatal("expected schema validation error")
	}
}

func TestNewSimpleErrors(t *testing.T) {
	fn := func(context.Context, string) (Result, error) { return Result{}, nil }
	if _, err := NewSimple("", "desc", fn); err == nil {
		t.Fatal("expected error for empty name")
	}
	if _, err := NewSimple("x", "desc", nil); err == nil {
		t.Fatal("expected error for nil function")
	}
}

func TestNewFuncErrors(t *testing.T) {
	fn := func(context.Context, map[string]any) (Result, error) { return Result{}, nil }
	valid := schema.Object(map[string]schema.Schema{"a": schema.String("a")})
	if _, err := NewFunc("", "desc", valid, fn); err == nil {
		t.Fatal("expected error for empty name")
	}
	if _, err := NewFunc("x", "desc", valid, nil); err == nil {
		t.Fatal("expected error for nil function")
	}
	// properties must be an object when present.
	bad := schema.Schema{"type": "object", "properties": "not-a-map"}
	if _, err := NewFunc("x", "desc", bad, fn); err == nil {
		t.Fatal("expected error for non-object properties")
	}
}

func TestSimpleInvokeStringRequiresFunction(t *testing.T) {
	var tool Simple
	if _, err := tool.InvokeString(context.Background(), "hi"); err == nil {
		t.Fatal("expected missing function error")
	}
}

func TestValidateArgsSchema(t *testing.T) {
	if err := ValidateArgsSchema(nil); err != nil {
		t.Fatalf("nil schema should be valid, got %v", err)
	}
	if err := ValidateArgsSchema(schema.Object(nil)); err != nil {
		t.Fatalf("object schema should be valid, got %v", err)
	}
	ok := schema.Schema{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}}
	if err := ValidateArgsSchema(ok); err != nil {
		t.Fatalf("object properties should be valid, got %v", err)
	}
	if err := ValidateArgsSchema(schema.String("nope")); err == nil {
		t.Fatal("expected error for non-object type")
	}
	badProps := schema.Schema{"type": "object", "properties": 42}
	if err := ValidateArgsSchema(badProps); err == nil {
		t.Fatal("expected error for non-object properties")
	}
}

// unmarshalableSchemaTool returns a schema that encoding/json cannot marshal,
// exercising the "{}" fallback in RenderTextDescriptionAndArgs.
type unmarshalableSchemaTool struct{}

func (unmarshalableSchemaTool) Name() string        { return "bad" }
func (unmarshalableSchemaTool) Description() string { return "unmarshalable schema" }
func (unmarshalableSchemaTool) ArgsSchema() schema.Schema {
	return schema.Schema{"type": "object", "bad": make(chan int)}
}
func (unmarshalableSchemaTool) Invoke(context.Context, map[string]any) (Result, error) {
	return Result{}, nil
}

func TestRenderTextDescriptionAndArgsFallback(t *testing.T) {
	got := RenderTextDescriptionAndArgs([]Tool{unmarshalableSchemaTool{}})
	if !strings.Contains(got, "bad - unmarshalable schema, args: {}") {
		t.Fatalf("expected {} fallback, got %q", got)
	}
}

func TestToFunctionSpecErrors(t *testing.T) {
	if _, err := ToFunctionSpec(nil); err == nil {
		t.Fatal("expected error for nil tool")
	}
	// A tool whose name fails function-name validation propagates the error.
	tool, err := NewFunc("bad name!", "desc", nil,
		func(context.Context, map[string]any) (Result, error) { return Result{}, nil })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ToFunctionSpec(tool); err == nil {
		t.Fatal("expected error for invalid function name")
	}
	if _, err := ToOpenAIToolSpec(tool); err == nil {
		t.Fatal("expected ToOpenAIToolSpec to propagate the spec error")
	}
	if _, err := ToOpenAIToolSpec(nil); err == nil {
		t.Fatal("expected error for nil tool")
	}
}

// errRetriever always fails, exercising the retriever-error path.
type errRetriever struct{ err error }

func (r errRetriever) GetRelevantDocuments(context.Context, string) ([]documents.Document, error) {
	return nil, r.err
}

func TestCreateRetrieverToolValidation(t *testing.T) {
	if _, err := CreateRetrieverTool(nil, "x", "x", RetrieverToolOptions{}); err == nil {
		t.Fatal("expected error for nil retriever")
	}
	tool, err := CreateRetrieverTool(retrievers.Static{}, "lookup", "desc", RetrieverToolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for name, input := range map[string]map[string]any{
		"missing query": {},
		"empty query":   {"query": ""},
		"non-string":    {"query": 3},
	} {
		if _, err := tool.Invoke(context.Background(), input); err == nil {
			t.Fatalf("expected query validation error for %s", name)
		}
	}
}

func TestCreateRetrieverToolErrorPropagation(t *testing.T) {
	want := fmt.Errorf("backend down")
	tool, err := CreateRetrieverTool(errRetriever{err: want}, "lookup", "desc", RetrieverToolOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tool.Invoke(context.Background(), map[string]any{"query": "q"}); err == nil {
		t.Fatal("expected retriever error to propagate")
	}
}

func TestCreateRetrieverToolDefaults(t *testing.T) {
	retriever := retrievers.Static{Documents: []documents.Document{
		documents.New("alpha", nil),
		documents.New("beta", nil),
	}}
	tool, err := CreateRetrieverTool(retriever, "lookup", "desc", RetrieverToolOptions{
		FormatDocument: func(doc documents.Document) string { return "<" + doc.PageContent + ">" },
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := tool.Invoke(context.Background(), map[string]any{"query": "q"})
	if err != nil {
		t.Fatal(err)
	}
	// Default separator is "\n\n" and the custom formatter is applied.
	if got.Content != "<alpha>\n\n<beta>" {
		t.Fatalf("Content = %q", got.Content)
	}
	if got.Artifact != nil {
		t.Fatalf("Artifact should be nil without IncludeArtifact, got %v", got.Artifact)
	}
}
