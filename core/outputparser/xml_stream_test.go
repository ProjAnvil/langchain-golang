package outputparser

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func xmlJSON(t *testing.T, value any) string {
	t.Helper()
	var buf bytes.Buffer
	encoder := json.NewEncoder(&buf)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(value); err != nil {
		t.Fatalf("marshal xml result: %v", err)
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

func TestStreamingXMLParserTopLevelLeaf(t *testing.T) {
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), []string{"<answer>42</answer>"})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if want := `[{"answer":"42"}]`; xmlJSON(t, got) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, got), want)
	}
}

func TestStreamingXMLParserChunkBoundariesAcrossTags(t *testing.T) {
	chunks := []string{"<fo", "o><ba", "r>o", "k</ba", "r></fo", "o>"}
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), chunks)
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	want := `[{"foo":[{"bar":"ok"}]}]`
	if xmlJSON(t, got) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, got), want)
	}
}

func TestStreamingXMLParserIncrementalParseEmitsOnce(t *testing.T) {
	chunks := []string{"<foo><bar>ok</bar></foo>"}
	parser := NewStreamingXMLParser()
	var collected []map[string]any
	emitCount := 0
	for _, chunk := range chunks {
		emitted, err := parser.Parse(t.Context(), chunk)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if len(emitted) > 0 {
			emitCount++
		}
		collected = append(collected, emitted...)
	}
	if err := parser.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if emitCount != 1 {
		t.Fatalf("expected a single emission chunk, got %d", emitCount)
	}
	if want := `[{"foo":[{"bar":"ok"}]}]`; xmlJSON(t, collected) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, collected), want)
	}
}

func TestStreamingXMLParserNestedElements(t *testing.T) {
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), []string{"<a><b><c>deep</c></b></a>"})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	want := `[{"a":[{"b":[{"c":"deep"}]}]}]`
	if xmlJSON(t, got) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, got), want)
	}
}

func TestStreamingXMLParserMultipleLeafSiblings(t *testing.T) {
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), []string{"<r><x>1</x><y>2</y></r>"})
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	want := `[{"r":[{"x":"1"}]},{"r":[{"y":"2"}]}]`
	if xmlJSON(t, got) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, got), want)
	}
}

func TestStreamingXMLParserLeadingProseAndDeclaration(t *testing.T) {
	chunks := []string{
		"Here is the result: ",
		"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n",
		"<!-- an ignored comment -->",
		"<foo>bar</foo>",
	}
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), chunks)
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if want := `[{"foo":"bar"}]`; xmlJSON(t, got) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, got), want)
	}
}

func TestStreamingXMLParserIncompleteTrailingChunks(t *testing.T) {
	// A complete root followed by a truncated second root: only the first
	// leaf is emitted and Close tolerates the incomplete tail.
	chunks := []string{"<a>1</a><b>2</b", " still incomplete"}
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), chunks)
	if err != nil {
		t.Fatalf("transform: %v", err)
	}
	if want := `[{"a":"1"}]`; xmlJSON(t, got) != want {
		t.Fatalf("got %s want %s", xmlJSON(t, got), want)
	}

	// An entirely truncated document yields nothing but must not fail.
	parser = NewStreamingXMLParser()
	got, err = parser.Transform(t.Context(), []string{"<foo><bar>value"})
	if err != nil {
		t.Fatalf("transform truncated: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected no emissions, got %s", xmlJSON(t, got))
	}
}

func TestStreamingXMLParserSelfClosingAndEmpty(t *testing.T) {
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), []string{"<foo/>"})
	if err != nil {
		t.Fatalf("transform self-closing: %v", err)
	}
	if want := `[{"foo":null}]`; xmlJSON(t, got) != want {
		t.Fatalf("self-closing got %s want %s", xmlJSON(t, got), want)
	}

	parser = NewStreamingXMLParser()
	got, err = parser.Transform(t.Context(), []string{"<foo></foo>"})
	if err != nil {
		t.Fatalf("transform empty: %v", err)
	}
	if want := `[{"foo":null}]`; xmlJSON(t, got) != want {
		t.Fatalf("empty got %s want %s", xmlJSON(t, got), want)
	}
}

func TestStreamingXMLParserAttributesAndCDATA(t *testing.T) {
	parser := NewStreamingXMLParser()
	got, err := parser.Transform(t.Context(), []string{`<foo a="x>y">ok</foo>`})
	if err != nil {
		t.Fatalf("transform attribute: %v", err)
	}
	if want := `[{"foo":"ok"}]`; xmlJSON(t, got) != want {
		t.Fatalf("attribute got %s want %s", xmlJSON(t, got), want)
	}

	parser = NewStreamingXMLParser()
	got, err = parser.Transform(t.Context(), []string{"<foo><![CDATA[a<b]]></foo>"})
	if err != nil {
		t.Fatalf("transform cdata: %v", err)
	}
	if want := `[{"foo":"a<b"}]`; xmlJSON(t, got) != want {
		t.Fatalf("cdata got %s want %s", xmlJSON(t, got), want)
	}
}

func TestNestedXMLElementHelper(t *testing.T) {
	got := NestedXMLElement([]string{}, "leaf", "v")
	if !reflect.DeepEqual(got, map[string]any{"leaf": "v"}) {
		t.Fatalf("root helper: %#v", got)
	}
	got = NestedXMLElement([]string{"a", "b"}, "leaf", nil)
	want := map[string]any{
		"a": []any{
			map[string]any{"b": []any{map[string]any{"leaf": nil}}},
		},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("nested helper: %#v", got)
	}
}

func TestStreamingXMLParserContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	parser := NewStreamingXMLParser()
	if _, err := parser.Parse(ctx, "<foo>bar</foo>"); err == nil {
		t.Fatal("expected context error from Parse")
	}
	if _, err := parser.Transform(ctx, []string{"<foo>bar</foo>"}); err == nil {
		t.Fatal("expected context error from Transform")
	}
}

func TestStreamingXMLParserFormatInstructions(t *testing.T) {
	parser := NewStreamingXMLParser()
	if !strings.Contains(parser.FormatInstructions(), "XML") {
		t.Fatalf("instructions missing XML: %q", parser.FormatInstructions())
	}
}
