package outputparser

import (
	"context"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/outputs"
	"github.com/projanvil/langchain-golang/core/schema"
)

func TestStringParser(t *testing.T) {
	parser := StringParser{}
	got, err := parser.Parse(context.Background(), "hello")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got != "hello" {
		t.Fatalf("got %q", got)
	}
}

func TestJSONParser(t *testing.T) {
	parser := NewJSONParser[map[string]any]("")
	got, err := parser.Parse(context.Background(), `{"name":"Ada","age":37}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got["name"] != "Ada" {
		t.Fatalf("name: got %v", got["name"])
	}
	if got["age"].(jsonNumber).String() != "37" {
		t.Fatalf("age: got %v", got["age"])
	}
}

type jsonNumber interface {
	String() string
}

func TestJSONParserInvalid(t *testing.T) {
	parser := NewJSONParser[map[string]any]("")
	_, err := parser.Parse(context.Background(), `{bad json}`)
	if err == nil {
		t.Fatal("expected parse error")
	}
}

func TestJSONParserMarkdownAndResult(t *testing.T) {
	parser := NewJSONParser[map[string]any]("")
	got, err := parser.Parse(context.Background(), "Here:\n```json\n{\"name\":\"Ada\"}\n```")
	if err != nil {
		t.Fatal(err)
	}
	if got["name"] != "Ada" {
		t.Fatalf("got %#v", got)
	}
	result, ok, err := parser.ParseResult(context.Background(), []outputs.Generation{
		outputs.NewGeneration(`{"answer": 42}`, nil),
	}, false)
	if err != nil || !ok {
		t.Fatalf("ParseResult ok=%v err=%v", ok, err)
	}
	if result["answer"].(jsonNumber).String() != "42" {
		t.Fatalf("result %#v", result)
	}
	partial, ok, err := parser.ParseResult(context.Background(), []outputs.Generation{
		outputs.NewGeneration(`{"answer": 42`, nil),
	}, true)
	if err != nil || !ok {
		t.Fatalf("partial ok=%v err=%v", ok, err)
	}
	if partial["answer"].(jsonNumber).String() != "42" {
		t.Fatalf("partial %#v", partial)
	}
}

func TestJSONParserFormatInstructionsWithSchema(t *testing.T) {
	parser := NewJSONParserWithSchema[map[string]any](schema.Schema{
		"type":  "object",
		"title": "Ignored",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
		},
		"$defs": map[string]any{"Unused": "x"},
	})
	instructions := parser.FormatInstructions()
	for _, want := range []string{"STRICT OUTPUT FORMAT", "properties", "name"} {
		if !strings.Contains(instructions, want) {
			t.Fatalf("instructions missing %q: %s", want, instructions)
		}
	}
	if strings.Contains(instructions, "$defs") || strings.Contains(instructions, "Ignored") {
		t.Fatalf("instructions did not reduce schema: %s", instructions)
	}
}

func TestCommaSeparatedListParser(t *testing.T) {
	parser := CommaSeparatedListParser{}
	got, err := parser.Parse(context.Background(), `foo, "bar, baz", qux`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := []string{"foo", "bar, baz", "qux"}
	if len(got) != len(want) {
		t.Fatalf("items: got %#v want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("item %d: got %q want %q", i, got[i], want[i])
		}
	}
}

func TestNumberedListParser(t *testing.T) {
	parser := NumberedListParser{}
	got, err := parser.Parse(context.Background(), "1. alpha\n2. beta\nnot an item")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
		t.Fatalf("items: %#v", got)
	}
}

func TestMarkdownListParser(t *testing.T) {
	parser := MarkdownListParser{}
	got, err := parser.Parse(context.Background(), "- alpha\n* beta\nplain")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
		t.Fatalf("items: %#v", got)
	}
}

func TestXMLParser(t *testing.T) {
	parser := XMLParser{Tags: []string{"foo", "bar", "baz"}}
	got, err := parser.Parse(context.Background(), "```xml\n<foo><bar><baz>ok</baz></bar></foo>\n```")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	foo := got["foo"].([]any)
	bar := foo[0].(map[string]any)["bar"].([]any)
	baz := bar[0].(map[string]any)["baz"]
	if baz != "ok" {
		t.Fatalf("baz: got %#v", baz)
	}
}

func TestXMLParserInvalid(t *testing.T) {
	parser := XMLParser{}
	_, err := parser.Parse(context.Background(), "<foo><bar></foo>")
	if err == nil {
		t.Fatal("expected parse error")
	}
}
