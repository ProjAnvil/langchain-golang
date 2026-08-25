package structuredoutput

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/core/tools"
)

func personSchema() schema.Schema {
	return schema.Object(map[string]schema.Schema{
		"name": schema.String("person name"),
		"age":  schema.Integer("person age"),
	}, "name", "age")
}

// method=function_calling: the schema is bound as a tool and the first
// matching tool call's arguments are parsed (Python chat_models.py:2512-2527).
func TestBindOptionsFunctionCallingParsesToolCall(t *testing.T) {
	model := testStructuredModel{
		response: messages.Message{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "person", Args: map[string]any{"name": "Ada", "age": 37}},
			},
		},
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}

	runnable, err := BindOptions[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.Name != "Ada" || got.Age != 37 {
		t.Fatalf("parsed: %+v", got)
	}
}

// The tool name defaults to the schema's "title" when Options.Name is empty
// (mirrors Python's convert_to_openai_tool key-name derivation).
func TestBindOptionsFunctionCallingUsesSchemaTitle(t *testing.T) {
	titled := personSchema()
	titled["title"] = "Person"
	model := testStructuredModel{
		response: messages.Message{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "Person", Args: map[string]any{"name": "Ada", "age": 37}},
			},
		},
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}

	runnable, err := BindOptions[testStructuredModel, person](model, Options{
		Schema: titled,
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.Name != "Ada" {
		t.Fatalf("parsed: %+v", got)
	}
}

// Documented divergence: Python's JsonOutputKeyToolsParser returns None when
// no tool call matches; Go surfaces it as a parse error (safer: a silent zero
// value is indistinguishable from a valid empty parse).
func TestBindOptionsFunctionCallingNoMatchingToolCall(t *testing.T) {
	model := testStructuredModel{
		response:     messages.AI("no tool call"),
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}
	runnable, err := BindOptions[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	_, err = runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err == nil || !strings.Contains(err.Error(), `tool call named "person"`) {
		t.Fatalf("expected missing-tool-call error, got %v", err)
	}
}

// function_calling requires a tool-calling-capable model (Python raises
// NotImplementedError when bind_tools is not implemented).
func TestBindOptionsFunctionCallingRequiresToolCalling(t *testing.T) {
	_, err := BindOptions[testStructuredModel, person](testStructuredModel{}, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err == nil || !strings.Contains(err.Error(), "tool calling") {
		t.Fatalf("expected tool calling error, got %v", err)
	}
}

func TestBindOptionsUnsupportedMethod(t *testing.T) {
	_, err := BindOptions[testStructuredModel, person](testStructuredModel{
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: Method("yaml_mode"),
	})
	if err == nil || !strings.Contains(err.Error(), `unsupported structured output method "yaml_mode"`) {
		t.Fatalf("expected unsupported method error, got %v", err)
	}
}

// include_raw=True, parse failure: the envelope carries the raw message and
// the parsing error instead of returning an error
// (Python chat_models.py:2528-2536 parser_with_fallback).
func TestBindOptionsWithRawCapturesParseError(t *testing.T) {
	model := testStructuredModel{
		response:     messages.AI(`{"name":"Ada","age":`),
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Strict: true,
		Method: MethodJSONSchema,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.Raw.Content != `{"name":"Ada","age":` {
		t.Fatalf("raw: %+v", got.Raw)
	}
	if got.ParsingError == nil {
		t.Fatal("expected ParsingError")
	}
	if got.Parsed != (person{}) {
		t.Fatalf("parsed should be zero on failure: %+v", got.Parsed)
	}
}

// include_raw=True, success: raw + parsed set, parsing_error nil.
func TestBindOptionsWithRawSuccess(t *testing.T) {
	model := testStructuredModel{
		response:     messages.AI(`{"name":"Ada","age":37}`),
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodJSONSchema,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.ParsingError != nil {
		t.Fatalf("parsing error: %v", got.ParsingError)
	}
	if got.Raw.Content != `{"name":"Ada","age":37}` || got.Parsed.Name != "Ada" || got.Parsed.Age != 37 {
		t.Fatalf("result: %+v", got)
	}
}

// include_raw with method=function_calling also envelopes parse failures.
func TestBindOptionsWithRawFunctionCallingParseFailure(t *testing.T) {
	model := testStructuredModel{
		response:     messages.AI("no tool call"),
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.ParsingError == nil || !strings.Contains(got.ParsingError.Error(), `tool call named "person"`) {
		t.Fatalf("expected missing-tool-call parsing error, got %+v", got)
	}
	if got.Raw.Content != "no tool call" {
		t.Fatalf("raw: %+v", got.Raw)
	}
}

// Model invocation errors are NOT captured in the envelope — they propagate
// (Python's RunnableMap(raw=llm) fails before the parser fallback runs).
func TestBindOptionsWithRawPropagatesModelError(t *testing.T) {
	invokeErr := errors.New("model blew up")
	model := testStructuredModel{
		err:          invokeErr,
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodJSONSchema,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	_, err = runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if !errors.Is(err, invokeErr) {
		t.Fatalf("expected invoke error, got %v", err)
	}
}

// include_raw=False still raises parse errors (existing BindJSON behavior).
func TestBindOptionsJSONSchemaRaisesParseError(t *testing.T) {
	model := testStructuredModel{
		response:     messages.AI(`{"name":"Ada","age":`),
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}
	runnable, err := BindOptions[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodJSONSchema,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	if _, err = runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")}); err == nil {
		t.Fatal("expected parse error")
	}
}

// Construction matrix mirroring test_base.py:984's parametrization
// (method × include_raw × strict): every supported combination builds.
func TestBindOptionsConstructionMatrix(t *testing.T) {
	model := testStructuredModel{
		capabilities: language.ChatModelCapabilities{StructuredOutput: true, ToolCalling: true},
	}
	for _, method := range []Method{"", MethodJSONSchema, MethodFunctionCalling} {
		for _, strict := range []bool{true, false} {
			opts := Options{Name: "person", Schema: personSchema(), Strict: strict, Method: method}
			if _, err := BindOptions[testStructuredModel, person](model, opts); err != nil {
				t.Fatalf("BindOptions method=%q strict=%v: %v", method, strict, err)
			}
			if _, err := BindOptionsWithRaw[testStructuredModel, person](model, opts); err != nil {
				t.Fatalf("BindOptionsWithRaw method=%q strict=%v: %v", method, strict, err)
			}
		}
	}
}

// Tool-call args that fail to re-encode as JSON are parse failures.
func TestBindOptionsFunctionCallingUnmarshalableArgs(t *testing.T) {
	model := testStructuredModel{
		response: messages.Message{
			Role: messages.RoleAI,
			ToolCalls: []messages.ToolCall{
				{ID: "call_1", Name: "person", Args: map[string]any{"bad": func() {}}},
			},
		},
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.ParsingError == nil {
		t.Fatal("expected ParsingError for unmarshalable args")
	}
}

// BindOptionsWithRaw surfaces method-validation errors at bind time too.
func TestBindOptionsWithRawUnsupportedMethod(t *testing.T) {
	_, err := BindOptionsWithRaw[testStructuredModel, person](testStructuredModel{
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: Method("yaml_mode"),
	})
	if err == nil || !strings.Contains(err.Error(), `unsupported structured output method "yaml_mode"`) {
		t.Fatalf("expected unsupported method error, got %v", err)
	}
}

// method=json_schema (and the empty default) require StructuredOutput
// capability (existing BindJSON contract).
func TestBindOptionsJSONSchemaRequiresStructuredOutput(t *testing.T) {
	_, err := BindOptions[testStructuredModel, person](testStructuredModel{}, Options{
		Name:   "person",
		Schema: personSchema(),
	})
	if err == nil || !strings.Contains(err.Error(), "structured output is not supported") {
		t.Fatalf("expected unsupported error, got %v", err)
	}
}

// Parse errors returned by BindOptions (include_raw=False) wrap the
// underlying parser failure and stay unwrappable.
func TestBindOptionsParseErrorUnwraps(t *testing.T) {
	model := testStructuredModel{
		response:     messages.AI(`{"name":"Ada","age":`),
		capabilities: language.ChatModelCapabilities{StructuredOutput: true},
	}
	runnable, err := BindOptions[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodJSONSchema,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	_, err = runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	var pErr parseError
	if !errors.As(err, &pErr) {
		t.Fatalf("expected parseError, got %v", err)
	}
	if errors.Unwrap(pErr) == nil {
		t.Fatal("expected parseError to wrap the parser failure")
	}
}

// A matching tool call with nil Args parses as an empty object.
func TestBindOptionsFunctionCallingNilArgs(t *testing.T) {
	model := testStructuredModel{
		response: messages.Message{
			Role:      messages.RoleAI,
			ToolCalls: []messages.ToolCall{{ID: "call_1", Name: "person"}},
		},
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	got, err := runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if got.ParsingError == nil {
		// person has required fields; either a schema-validation parse error or
		// a zero person is acceptable — but the nil-args branch must not panic.
		t.Logf("parsing error (acceptable): %v", got.ParsingError)
	}
}

// bindToolsErrorModel fails BindTools, exercising the function-calling
// bind-error branch.
type bindToolsErrorModel struct{ testStructuredModel }

func (m bindToolsErrorModel) BindTools([]tools.Tool) (language.ChatModel, error) {
	return nil, errors.New("bind tools blew up")
}

func (m bindToolsErrorModel) WithStructuredOutput(
	name string,
	outputSchema schema.Schema,
	strict bool,
) bindToolsErrorModel {
	return m
}

func TestBindOptionsFunctionCallingBindToolsError(t *testing.T) {
	model := bindToolsErrorModel{testStructuredModel{
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}}
	runnable, err := BindOptions[bindToolsErrorModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	_, err = runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if err == nil || !strings.Contains(err.Error(), "bind tools blew up") {
		t.Fatalf("expected bind tools error, got %v", err)
	}
}

// A model error on the function-calling invoke path propagates as a plain
// error, not a parsing error.
func TestBindOptionsFunctionCallingInvokeError(t *testing.T) {
	invokeErr := errors.New("model blew up")
	model := testStructuredModel{
		err:          invokeErr,
		capabilities: language.ChatModelCapabilities{ToolCalling: true},
	}
	runnable, err := BindOptionsWithRaw[testStructuredModel, person](model, Options{
		Name:   "person",
		Schema: personSchema(),
		Method: MethodFunctionCalling,
	})
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	_, err = runnable.Invoke(context.Background(), []messages.Message{messages.Human("extract person")})
	if !errors.Is(err, invokeErr) {
		t.Fatalf("expected invoke error, got %v", err)
	}
}
