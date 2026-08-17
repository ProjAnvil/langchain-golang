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

func TestParserFormatInstructions(t *testing.T) {
	cases := []struct {
		name   string
		parser interface{ FormatInstructions() string }
		want   string
	}{
		{"string", StringParser{}, "plain text"},
		{"comma separated", CommaSeparatedListParser{}, "comma separated values"},
		{"numbered list", NumberedListParser{}, "numbered list"},
		{"markdown list", MarkdownListParser{}, "markdown list"},
		{"xml with tags", XMLParser{Tags: []string{"foo", "bar"}}, "Expected tags: foo, bar."},
		{"xml without tags", XMLParser{}, "make them on your own"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.parser.FormatInstructions(); !strings.Contains(got, tc.want) {
				t.Fatalf("instructions %q missing %q", got, tc.want)
			}
		})
	}
}

func TestCommaSeparatedListParserFallback(t *testing.T) {
	parser := CommaSeparatedListParser{}
	got, err := parser.Parse(context.Background(), `a, "unclosed, c`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	want := []string{"a", `"unclosed`, "c"}
	if len(got) != len(want) {
		t.Fatalf("items: got %#v want %#v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("item %d: got %q want %q", i, got[i], want[i])
		}
	}
}

func TestJSONParserParseResultEmpty(t *testing.T) {
	parser := NewJSONParser[map[string]any]("")
	_, ok, err := parser.ParseResult(context.Background(), nil, false)
	if err == nil || ok {
		t.Fatalf("expected error for empty result, ok=%v err=%v", ok, err)
	}
}

func TestJSONParserParseResultPartialIncomplete(t *testing.T) {
	parser := NewJSONParser[map[string]any]("")
	_, ok, err := parser.ParseResult(context.Background(), []outputs.Generation{
		outputs.NewGeneration(`tru`, nil),
	}, true)
	if err != nil {
		t.Fatalf("partial parse should not error: %v", err)
	}
	if ok {
		t.Fatal("expected incomplete partial JSON to report ok=false")
	}
}

func TestJSONFormatInstructionsEmptySchema(t *testing.T) {
	if got := JSONFormatInstructions(nil); got != "Return a JSON object." {
		t.Fatalf("got %q", got)
	}
}

func TestExtractJSONMarkdown(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"array in prose", "result: [1, 2] done", "[1, 2]"},
		{"no brackets", "plain text", "plain text"},
		{"closing before opening", "}{", "}{"},
		{"object only", `{"a":1}`, `{"a":1}`},
		{"array before object", `[{"a":1}]`, `[{"a":1}]`},
		{"object before array", `{"a":[1]}`, `{"a":[1]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := extractJSONMarkdown(tc.input); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestXMLParserWithEncodingDeclaration(t *testing.T) {
	parser := XMLParser{}
	got, err := parser.Parse(context.Background(), "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<foo>bar</foo>")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got["foo"] != "bar" {
		t.Fatalf("foo: got %#v", got["foo"])
	}
}

func TestParseXMLNodeErrors(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		// The encoding/xml decoder rejects unbalanced documents before
		// parseXMLNode's own tag-matching branches run, so these inputs only
		// assert that an error surfaces.
		{"unexpected closing tag", "</foo>", ""},
		{"mismatched closing tag", "<a></b>", ""},
		{"missing root", "plain text", "missing XML root"},
		{"truncated document", "<a", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseXMLNode(tc.input)
			if err == nil {
				t.Fatal("expected error")
			}
			if tc.want != "" && !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err %q missing %q", err, tc.want)
			}
		})
	}
}

func TestXMLNodeValueWithChildrenAndText(t *testing.T) {
	parser := XMLParser{}
	got, err := parser.Parse(context.Background(), "<foo><bar>1</bar>tail</foo>")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	children, ok := got["foo"].([]any)
	if !ok || len(children) != 1 {
		t.Fatalf("foo children: %#v", got["foo"])
	}
}
