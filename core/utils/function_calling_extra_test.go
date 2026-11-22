package utils

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestLargeUtilParseGoogleDocstringWithArgs(t *testing.T) {
	docstring := "Search the web for a query.\n\n" +
		"Args:\n" +
		"    query (str): The search query.\n" +
		"    limit: Maximum number of results\n" +
		"        to return.\n" +
		"    verbose: Whether to print debug info.\n\n" +
		"Returns:\n" +
		"    A list of results."

	description, args, err := ParseGoogleDocstring(docstring)
	if err != nil {
		t.Fatal(err)
	}
	if description != "Search the web for a query." {
		t.Fatalf("description = %q", description)
	}
	want := map[string]string{
		"query":   "The search query.",
		"limit":   "Maximum number of results to return.",
		"verbose": "Whether to print debug info.",
	}
	if !reflect.DeepEqual(args, want) {
		t.Fatalf("args = %#v, want %#v", args, want)
	}
}

func TestLargeUtilParseGoogleDocstringWithoutArgs(t *testing.T) {
	description, args, err := ParseGoogleDocstring("Just a plain description.\n\nThis is a second paragraph.")
	if err != nil {
		t.Fatal(err)
	}
	if description != "Just a plain description. This is a second paragraph." {
		t.Fatalf("description = %q", description)
	}
	if len(args) != 0 {
		t.Fatalf("expected no args, got %#v", args)
	}

	// Empty docstring yields an empty description and an empty (non-nil) map.
	description, args, err = ParseGoogleDocstring("")
	if err != nil {
		t.Fatal(err)
	}
	if description != "" || args == nil || len(args) != 0 {
		t.Fatalf("empty docstring = (%q, %#v)", description, args)
	}
}

func TestLargeUtilParseGoogleDocstringArgsAfterReturns(t *testing.T) {
	docstring := "Does a thing.\n\n" +
		"Returns:\n" +
		"    Nothing.\n\n" +
		"Args:\n" +
		"    x: The x value."

	description, args, err := ParseGoogleDocstring(docstring)
	if err != nil {
		t.Fatal(err)
	}
	if description != "Does a thing." {
		t.Fatalf("description = %q", description)
	}
	if !reflect.DeepEqual(args, map[string]string{"x": "The x value."}) {
		t.Fatalf("args = %#v", args)
	}
}

func TestLargeUtilRemoveTitles(t *testing.T) {
	input := map[string]any{
		"title": "TopLevel",
		"type":  "object",
		"properties": map[string]any{
			"name": map[string]any{
				"title": "Name",
				"type":  "string",
			},
			"nested": map[string]any{
				"title": "Nested",
				"properties": map[string]any{
					"leaf": map[string]any{"title": "Leaf", "type": "integer"},
				},
			},
		},
	}
	got := RemoveTitles(input)
	if _, ok := got["title"]; ok {
		t.Fatal("top-level title was not removed")
	}
	props := got["properties"].(map[string]any)
	name := props["name"].(map[string]any)
	if _, ok := name["title"]; ok || name["type"] != "string" {
		t.Fatalf("name property = %#v", name)
	}
	nested := props["nested"].(map[string]any)
	if _, ok := nested["title"]; ok {
		t.Fatal("nested title was not removed")
	}
	leaf := nested["properties"].(map[string]any)["leaf"].(map[string]any)
	if _, ok := leaf["title"]; ok || leaf["type"] != "integer" {
		t.Fatalf("leaf property = %#v", leaf)
	}

	// A "title" key that sits directly under "properties" with an object value
	// is preserved (mirrors _rm_titles).
	preserved := RemoveTitles(map[string]any{
		"properties": map[string]any{
			"title": map[string]any{"type": "string"},
		},
	})
	if _, ok := preserved["properties"].(map[string]any)["title"]; !ok {
		t.Fatal("title under properties should be preserved")
	}
}

func TestLargeUtilRemoveTitlesSchemaValues(t *testing.T) {
	// Values built with schema.Schema must be handled by the recursion too.
	input := map[string]any{
		"title": "X",
		"properties": map[string]any{
			"a": schema.Schema{"title": "A", "type": "string"},
		},
	}
	got := RemoveTitles(input)
	prop := got["properties"].(map[string]any)["a"].(map[string]any)
	if _, ok := prop["title"]; ok || prop["type"] != "string" {
		t.Fatalf("schema-valued property = %#v", prop)
	}
}

func TestLargeUtilToolExampleToMessages(t *testing.T) {
	msgs, err := ToolExampleToMessages(
		"Who is Ada?",
		[]ToolCallExample{
			{Name: "Person", Arguments: map[string]any{"name": "Ada"}},
		},
		ToolExampleOptions{AIResponse: "Found Ada."},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(msgs))
	}
	if msgs[0].Role != messages.RoleHuman || msgs[0].Content != "Who is Ada?" {
		t.Fatalf("first message = %#v", msgs[0])
	}
	if msgs[1].Role != messages.RoleAI || msgs[1].Content != "" {
		t.Fatalf("second message = %#v", msgs[1])
	}
	calls, ok := msgs[1].AdditionalKwargs["tool_calls"].([]map[string]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %#v", msgs[1].AdditionalKwargs)
	}
	if calls[0]["type"] != "function" {
		t.Fatalf("tool call type = %#v", calls[0])
	}
	id, _ := calls[0]["id"].(string)
	if id == "" {
		t.Fatal("tool call id is empty")
	}
	fn := calls[0]["function"].(map[string]any)
	if fn["name"] != "Person" {
		t.Fatalf("function name = %#v", fn)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(fn["arguments"].(string)), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded["name"] != "Ada" {
		t.Fatalf("arguments = %#v", decoded)
	}
	if msgs[2].Role != messages.RoleTool || msgs[2].ToolCallID != id || msgs[2].Content != "You have correctly called this tool." {
		t.Fatalf("third message = %#v", msgs[2])
	}
	if msgs[3].Role != messages.RoleAI || msgs[3].Content != "Found Ada." {
		t.Fatalf("fourth message = %#v", msgs[3])
	}
}

func TestLargeUtilToolExampleToMessagesOutputsAndErrors(t *testing.T) {
	msgs, err := ToolExampleToMessages(
		"input",
		[]ToolCallExample{{Name: "T", Arguments: "{}"}},
		ToolExampleOptions{ToolOutputs: []string{"custom output"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages (no final AI), got %d", len(msgs))
	}
	if msgs[2].Content != "custom output" {
		t.Fatalf("tool output = %q", msgs[2].Content)
	}

	if _, err := ToolExampleToMessages(
		"input",
		[]ToolCallExample{{Name: "T", Arguments: nil}},
		ToolExampleOptions{ToolOutputs: []string{"a", "b"}},
	); err == nil {
		t.Fatal("expected tool_outputs length mismatch error")
	}
}
