package prebuilt

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
	"github.com/projanvil/langchain-golang/langchain/tools"
	"github.com/projanvil/langchain-golang/langgraph/channels"
	"github.com/projanvil/langchain-golang/langgraph/graph"
	"github.com/projanvil/langchain-golang/langgraph/runtime"
	"github.com/projanvil/langchain-golang/langgraph/types"
)

// selectNumber mirrors the SelectNumber/my_tool schema of
// langgraph/libs/prebuilt/tests/test_validation_node.py: fields some_val
// (integer) and some_other_val (string), both required.
func selectNumberTool(t *testing.T) tools.Tool {
	t.Helper()
	tool, err := tools.NewFunc("SelectNumber", "select a number",
		schema.Object(map[string]schema.Schema{
			"some_val":       schema.Integer("the number"),
			"some_other_val": schema.String("the label"),
		}, "some_val", "some_other_val"),
		func(context.Context, map[string]any) (tools.Result, error) {
			// ValidationNode never runs tools (tool_validator.py:54-57).
			return tools.Result{}, nil
		})
	if err != nil {
		t.Fatalf("NewFunc() error = %v", err)
	}
	return tool
}

// invokeValidation runs the node directly on a {"messages": ...} state.
func invokeValidation(t *testing.T, node graph.NodeFunc, msgs []messages.Message) []messages.Message {
	t.Helper()
	out, err := node(runtime.NewRuntime(context.Background()), map[string]any{"messages": msgs})
	if err != nil {
		t.Fatalf("ValidationNode error = %v", err)
	}
	update, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("output = %T, want map[string]any", out)
	}
	result, ok := update["messages"].([]messages.Message)
	if !ok {
		t.Fatalf("update[messages] = %T, want []messages.Message", update["messages"])
	}
	return result
}

// Mirrors test_validation_node (test_validation_node.py:51-88): one valid and
// one invalid (wrong type for some_val) tool call; the first ToolMessage
// carries the validated args as JSON, the second is flagged is_error.
func TestValidationNode(t *testing.T) {
	node, err := NewValidationNode([]tools.Tool{selectNumberTool(t)})
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	result := invokeValidation(t, node, []messages.Message{
		aiWithCalls(
			messages.ToolCall{ID: "some 0", Name: "SelectNumber", Args: map[string]any{"some_val": 1, "some_other_val": "foo"}},
			messages.ToolCall{ID: "some 1", Name: "SelectNumber", Args: map[string]any{"some_val": "bar", "some_other_val": "foo"}},
		),
	})
	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}
	for i, msg := range result {
		if msg.Role != messages.RoleTool {
			t.Errorf("result[%d].Role = %q, want tool", i, msg.Role)
		}
	}
	if result[0].AdditionalKwargs["is_error"] == true {
		t.Errorf("result[0] unexpectedly flagged is_error: %+v", result[0])
	}
	// json.Marshal orders map keys alphabetically.
	if want := `{"some_other_val":"foo","some_val":1}`; result[0].Content != want {
		t.Errorf("result[0].Content = %q, want %q", result[0].Content, want)
	}
	if result[0].Name != "SelectNumber" || result[0].ToolCallID != "some 0" {
		t.Errorf("result[0] = %+v, want name=SelectNumber tool_call_id=\"some 0\"", result[0])
	}
	if result[1].AdditionalKwargs["is_error"] != true {
		t.Errorf("result[1].AdditionalKwargs = %v, want is_error=true", result[1].AdditionalKwargs)
	}
	// Default format mirrors _default_format_error (tool_validator.py:34-40).
	if !strings.Contains(result[1].Content, "some_val") ||
		!strings.HasSuffix(result[1].Content, "Respond after fixing all validation errors.") {
		t.Errorf("result[1].Content = %q, want a validation error ending in the re-prompt instruction", result[1].Content)
	}
	if result[1].ToolCallID != "some 1" {
		t.Errorf("result[1].ToolCallID = %q, want \"some 1\"", result[1].ToolCallID)
	}
}

// Mirrors the use_message_key=False parametrization only in spirit: Go nodes
// always receive dict-shaped state, so the custom-key variant is exercised
// instead (WithValidationMessagesKey).
func TestValidationNodeCustomMessagesKey(t *testing.T) {
	node, err := NewValidationNode([]tools.Tool{selectNumberTool(t)}, WithValidationMessagesKey("chat_history"))
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	out, err := node(runtime.NewRuntime(context.Background()), map[string]any{"chat_history": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "c1", Name: "SelectNumber", Args: map[string]any{"some_val": 1, "some_other_val": "x"}}),
	}})
	if err != nil {
		t.Fatalf("ValidationNode error = %v", err)
	}
	result := out.(map[string]any)["chat_history"].([]messages.Message)
	if len(result) != 1 || result[0].AdditionalKwargs["is_error"] == true {
		t.Fatalf("result = %+v, want one non-error tool message", result)
	}
}

// Mirrors Python's "No message found in input" (tool_validator.py:178) and
// "Last message is not an AIMessage" (:181).
func TestValidationNodeInputErrors(t *testing.T) {
	node, err := NewValidationNode([]tools.Tool{selectNumberTool(t)})
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	if _, err := node(runtime.NewRuntime(context.Background()), map[string]any{}); err == nil ||
		!strings.Contains(err.Error(), "no message found in input") {
		t.Errorf("missing key: error = %v, want a 'no message found in input' error", err)
	}
	if _, err := node(runtime.NewRuntime(context.Background()), map[string]any{"messages": []messages.Message{messages.Human("hi")}}); err == nil ||
		!strings.Contains(err.Error(), "last message to be an AI message") {
		t.Errorf("last not AI: error = %v, want a 'last message to be an AI message' error", err)
	}
}

// Mirrors Python's KeyError on schemas_by_name[call["name"]]
// (tool_validator.py:191) as a descriptive Go error.
func TestValidationNodeUnknownTool(t *testing.T) {
	node, err := NewValidationNode([]tools.Tool{selectNumberTool(t)})
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	_, err = node(runtime.NewRuntime(context.Background()), map[string]any{"messages": []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "c1", Name: "Nope", Args: map[string]any{}}),
	}})
	if err == nil || !strings.Contains(err.Error(), `no schema for tool "Nope"`) {
		t.Fatalf("error = %v, want a 'no schema for tool' error", err)
	}
}

// A missing required field is a validation error, mirroring pydantic's
// required-field enforcement in test_validation_node's schema.
func TestValidationNodeMissingRequiredField(t *testing.T) {
	node, err := NewValidationNode([]tools.Tool{selectNumberTool(t)})
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	result := invokeValidation(t, node, []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "c1", Name: "SelectNumber", Args: map[string]any{"some_val": 1}}),
	})
	if result[0].AdditionalKwargs["is_error"] != true ||
		!strings.Contains(result[0].Content, "some_other_val") {
		t.Fatalf("result[0] = %+v, want an is_error message naming some_other_val", result[0])
	}
}

// Custom format_error, mirroring ValidationNode(format_error=...)
// (tool_validator.py:120-121).
func TestValidationNodeCustomFormatError(t *testing.T) {
	node, err := NewValidationNode(
		[]tools.Tool{selectNumberTool(t)},
		WithFormatError(func(err error, call messages.ToolCall, tool tools.Tool) string {
			return "custom: " + tool.Name() + "/" + call.ID + ": " + err.Error()
		}),
	)
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	result := invokeValidation(t, node, []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "c1", Name: "SelectNumber", Args: map[string]any{"some_val": "bar", "some_other_val": "x"}}),
	})
	if !strings.HasPrefix(result[0].Content, "custom: SelectNumber/c1: ") {
		t.Fatalf("result[0].Content = %q, want the custom-formatted error", result[0].Content)
	}
}

// A tool without an args schema validates any args (Python's BaseTool branch
// requires args_schema; Go's schema-less tools accept everything instead —
// documented divergence since core/tools.NewFunc allows a nil schema).
func TestValidationNodeNilSchemaAccepts(t *testing.T) {
	free, err := tools.NewFunc("Free", "no schema", nil,
		func(context.Context, map[string]any) (tools.Result, error) { return tools.Result{}, nil })
	if err != nil {
		t.Fatalf("NewFunc() error = %v", err)
	}
	node, err := NewValidationNode([]tools.Tool{free})
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	result := invokeValidation(t, node, []messages.Message{
		aiWithCalls(messages.ToolCall{ID: "c1", Name: "Free", Args: map[string]any{"anything": true}}),
	})
	if result[0].AdditionalKwargs["is_error"] == true {
		t.Fatalf("result[0] = %+v, want success", result[0])
	}
}

func TestValidationNodeConstructionErrors(t *testing.T) {
	if _, err := NewValidationNode(nil); err == nil {
		t.Error("NewValidationNode(nil) error = nil, want an error")
	}
	if _, err := NewValidationNode([]tools.Tool{nil}); err == nil {
		t.Error("NewValidationNode([nil]) error = nil, want an error")
	}
	tool := selectNumberTool(t)
	if _, err := NewValidationNode([]tools.Tool{tool, tool}); err == nil {
		t.Error("duplicate tool names: error = nil, want an error")
	}
	if _, err := NewValidationNode([]tools.Tool{tool}, WithValidationMessagesKey("")); err == nil {
		t.Error("empty messages key: error = nil, want an error")
	}
	if _, err := NewValidationNode([]tools.Tool{tool}, WithFormatError(nil)); err != nil {
		t.Errorf("WithFormatError(nil) should fall back to the default, got error %v", err)
	}
}

// End-to-end re-prompt loop, mirroring the docstring example in
// tool_validator.py:64-113: model -> validation -> (is_error ? model : END).
func TestValidationNodeInGraph(t *testing.T) {
	node, err := NewValidationNode([]tools.Tool{selectNumberTool(t)})
	if err != nil {
		t.Fatalf("NewValidationNode() error = %v", err)
	}
	g := graph.NewStateGraph()
	g.AddReducer("messages", channels.MessagesReducer)
	g.AddNode("model", modelNodeStub(aiWithCalls(
		messages.ToolCall{ID: "some 0", Name: "SelectNumber", Args: map[string]any{"some_val": 1, "some_other_val": "foo"}},
	)))
	g.AddNode("validation", node)
	g.AddEdge(types.START, "model")
	g.AddEdge("model", "validation")
	g.AddEdge("validation", types.END)
	compiled, err := g.Compile()
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	res, err := compiled.Invoke(context.Background(), nil)
	if err != nil {
		t.Fatalf("Invoke() error = %v", err)
	}
	msgs := stateMessages(t, res.Values, "messages")
	if len(msgs) != 2 || msgs[1].Role != messages.RoleTool || msgs[1].ToolCallID != "some 0" {
		t.Fatalf("messages = %+v, want ai + validated tool message", msgs)
	}
}

// TestValidateArgsAgainstSchema covers the schema-subset arms the mirrored
// Python tests never reach, keeping the package at the ≥95% coverage gate:
// every remaining jsonTypeMatches type name (boolean, number, object, array,
// integer-valued float32/float64, and the unknown-type default), the []any
// (JSON-decoded) and non-slice default arms of requiredKeys, and the
// "!present || typ == \"\"" continue that skips absent and untyped properties.
// Properties are written as literal map[string]any values, the JSON-decoded
// shape validateArgsAgainstSchema asserts on. (Named TestValidate... so the
// `go test -run TestValidationNode` steps below still count 9 tests.)
func TestValidateArgsAgainstSchema(t *testing.T) {
	prop := func(typ string) map[string]any { return map[string]any{"type": typ} }
	cases := []struct {
		name    string
		schema  schema.Schema
		args    map[string]any
		wantErr string // empty means the args validate
	}{
		{"boolean accepts a bool", schema.Schema{"properties": map[string]any{"flag": prop("boolean")}}, map[string]any{"flag": true}, ""},
		{"boolean rejects a string", schema.Schema{"properties": map[string]any{"flag": prop("boolean")}}, map[string]any{"flag": "yes"}, `field "flag": expected boolean`},
		{"number accepts a float64", schema.Schema{"properties": map[string]any{"n": prop("number")}}, map[string]any{"n": 1.5}, ""},
		{"number rejects a bool", schema.Schema{"properties": map[string]any{"n": prop("number")}}, map[string]any{"n": true}, `field "n": expected number`},
		{"object accepts a map", schema.Schema{"properties": map[string]any{"o": prop("object")}}, map[string]any{"o": map[string]any{"k": "v"}}, ""},
		{"object rejects a slice", schema.Schema{"properties": map[string]any{"o": prop("object")}}, map[string]any{"o": []any{1}}, `field "o": expected object`},
		{"array accepts a slice", schema.Schema{"properties": map[string]any{"a": prop("array")}}, map[string]any{"a": []any{1}}, ""},
		{"array rejects a map", schema.Schema{"properties": map[string]any{"a": prop("array")}}, map[string]any{"a": map[string]any{}}, `field "a": expected array`},
		{"integer accepts an integral float32", schema.Schema{"properties": map[string]any{"i": prop("integer")}}, map[string]any{"i": float32(3)}, ""},
		{"integer accepts an integral float64", schema.Schema{"properties": map[string]any{"i": prop("integer")}}, map[string]any{"i": float64(3)}, ""},
		{"integer rejects a non-integral float64", schema.Schema{"properties": map[string]any{"i": prop("integer")}}, map[string]any{"i": 2.5}, `field "i": expected integer`},
		{"unknown type name is not checked", schema.Schema{"properties": map[string]any{"u": prop("duration")}}, map[string]any{"u": 42}, ""},
		{"required as []any (JSON-decoded shape)", schema.Schema{"required": []any{"x"}, "properties": map[string]any{"x": prop("string")}}, map[string]any{}, `field "x" is required but missing`},
		{"non-slice required is ignored", schema.Schema{"required": "x", "properties": map[string]any{"x": prop("string")}}, map[string]any{}, ""},
		{"declared-but-absent property is skipped", schema.Schema{"properties": map[string]any{"y": prop("string")}}, map[string]any{}, ""},
		{"property without a type is skipped", schema.Schema{"properties": map[string]any{"z": map[string]any{"description": "untyped"}}}, map[string]any{"z": 42}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateArgsAgainstSchema(tc.schema, tc.args)
			if tc.wantErr == "" && err != nil {
				t.Fatalf("validateArgsAgainstSchema() error = %v, want nil", err)
			}
			if tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
				t.Fatalf("validateArgsAgainstSchema() error = %v, want one containing %q", err, tc.wantErr)
			}
		})
	}
}
