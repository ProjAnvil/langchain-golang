package textsplitters

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

const markdownSyntaxFixture = "# My Header 1\n" +
	"Content for header 1\n" +
	"## Header 2\n" +
	"Content for header 2\n" +
	"## Header 2 Again\n" +
	"This should be tagged with Header 1 and Header 2 Again\n" +
	"```python\n" +
	"def func():\n" +
	"   print(\"hi\")\n" +
	"```\n" +
	"# Header 1 again\n" +
	"We should split on the horizontal line\n" +
	"----\n" +
	"This is a new doc with the same header metadata"

func TestMarkdownSyntaxSplitsByHeadersAndCodeFences(t *testing.T) {
	splitter := NewMarkdownSyntax(nil, false, true)
	docs := splitter.SplitText(markdownSyntaxFixture)

	want := []documents.Document{
		documents.New("Content for header 1\n", map[string]any{"Header 1": "My Header 1"}),
		documents.New("Content for header 2\n", map[string]any{
			"Header 1": "My Header 1",
			"Header 2": "Header 2",
		}),
		documents.New("This should be tagged with Header 1 and Header 2 Again\n", map[string]any{
			"Header 1": "My Header 1",
			"Header 2": "Header 2 Again",
		}),
		documents.New("```python\ndef func():\n   print(\"hi\")\n```\n", map[string]any{
			"Code":    "python",
			"Header 1": "My Header 1",
			"Header 2": "Header 2 Again",
		}),
		documents.New("We should split on the horizontal line\n", map[string]any{"Header 1": "Header 1 again"}),
		documents.New("This is a new doc with the same header metadata", map[string]any{"Header 1": "Header 1 again"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestMarkdownSyntaxStripHeadersFalse(t *testing.T) {
	splitter := NewMarkdownSyntax(nil, false, false)
	docs := splitter.SplitText(markdownSyntaxFixture)

	want := []documents.Document{
		documents.New("# My Header 1\nContent for header 1\n", map[string]any{"Header 1": "My Header 1"}),
		documents.New("## Header 2\nContent for header 2\n", map[string]any{
			"Header 1": "My Header 1",
			"Header 2": "Header 2",
		}),
		documents.New("## Header 2 Again\nThis should be tagged with Header 1 and Header 2 Again\n", map[string]any{
			"Header 1": "My Header 1",
			"Header 2": "Header 2 Again",
		}),
		documents.New("```python\ndef func():\n   print(\"hi\")\n```\n", map[string]any{
			"Code":    "python",
			"Header 1": "My Header 1",
			"Header 2": "Header 2 Again",
		}),
		documents.New("# Header 1 again\nWe should split on the horizontal line\n", map[string]any{"Header 1": "Header 1 again"}),
		documents.New("This is a new doc with the same header metadata", map[string]any{"Header 1": "Header 1 again"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestMarkdownSyntaxReturnEachLine(t *testing.T) {
	splitter := NewMarkdownSyntax(nil, true, true)
	docs := splitter.SplitText(markdownSyntaxFixture)

	codeMeta := map[string]any{
		"Code":    "python",
		"Header 1": "My Header 1",
		"Header 2": "Header 2 Again",
	}
	want := []documents.Document{
		documents.New("Content for header 1", map[string]any{"Header 1": "My Header 1"}),
		documents.New("Content for header 2", map[string]any{
			"Header 1": "My Header 1",
			"Header 2": "Header 2",
		}),
		documents.New("This should be tagged with Header 1 and Header 2 Again", map[string]any{
			"Header 1": "My Header 1",
			"Header 2": "Header 2 Again",
		}),
		documents.New("```python", codeMeta),
		documents.New("def func():", codeMeta),
		documents.New("   print(\"hi\")", codeMeta),
		documents.New("```", codeMeta),
		documents.New("We should split on the horizontal line", map[string]any{"Header 1": "Header 1 again"}),
		documents.New("This is a new doc with the same header metadata", map[string]any{"Header 1": "Header 1 again"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestMarkdownSyntaxHeaderConfiguration(t *testing.T) {
	splitter := NewMarkdownSyntax([]Header{{Marker: "#", Name: "Encabezamiento 1"}}, false, true)
	docs := splitter.SplitText(markdownSyntaxFixture)

	want := []documents.Document{
		documents.New(
			"Content for header 1\n## Header 2\nContent for header 2\n## Header 2 Again\nThis should be tagged with Header 1 and Header 2 Again\n",
			map[string]any{"Encabezamiento 1": "My Header 1"},
		),
		documents.New("```python\ndef func():\n   print(\"hi\")\n```\n", map[string]any{
			"Code":             "python",
			"Encabezamiento 1": "My Header 1",
		}),
		documents.New("We should split on the horizontal line\n", map[string]any{"Encabezamiento 1": "Header 1 again"}),
		documents.New("This is a new doc with the same header metadata", map[string]any{"Encabezamiento 1": "Header 1 again"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}
