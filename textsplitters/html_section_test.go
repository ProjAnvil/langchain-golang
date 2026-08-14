package textsplitters

import (
	"reflect"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestHTMLSectionSplitsByH1H2(t *testing.T) {
	splitter := NewHTMLSection([]Header{
		{Marker: "h1", Name: "Header 1"},
		{Marker: "h2", Name: "Header 2"},
	})
	docs := splitter.SplitText(`<body>
<h1>Introduction</h1>
<p>Welcome to the guide.</p>
<h2>Background</h2>
<p>Some background material.</p>
</body>`)

	want := []documents.Document{
		documents.New("Introduction \n Welcome to the guide.", map[string]any{"Header 1": "Introduction"}),
		documents.New("Background \n Some background material.", map[string]any{"Header 2": "Background"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestHTMLSectionNestedHeaders(t *testing.T) {
	splitter := NewHTMLSection([]Header{
		{Marker: "h1", Name: "Header 1"},
		{Marker: "h2", Name: "Header 2"},
	})
	docs := splitter.SplitText(`<html><body>
<div>
<h1>Foo</h1>
<p>Some intro text about Foo.</p>
<div>
<h2>Bar main section</h2>
<p>Some intro text about Bar.</p>
</div>
<div>
<h2>Baz</h2>
<p>Some text about Baz</p>
</div>
<p>Some concluding text about Foo</p>
</div>
</body></html>`)

	want := []documents.Document{
		documents.New("Foo \n Some intro text about Foo.", map[string]any{"Header 1": "Foo"}),
		documents.New("Bar main section \n Some intro text about Bar.", map[string]any{"Header 2": "Bar main section"}),
		documents.New("Baz \n Some text about Baz \n \n Some concluding text about Foo", map[string]any{"Header 2": "Baz"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestHTMLSectionFontSizeToHeader(t *testing.T) {
	splitter := NewHTMLSection([]Header{
		{Marker: "h1", Name: "Header 1"},
		{Marker: "h2", Name: "Header 2"},
	})
	docs := splitter.SplitText(`<body>
<span style="font-size: 22px">Big Title</span>
<p>Content here.</p>
<span style="font-size: 12px">Small text</span>
</body>`)

	want := []documents.Document{
		documents.New("Big Title \n Content here. \n Small text", map[string]any{"Header 1": "Big Title"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestHTMLSectionTitlePlaceholderForPreamble(t *testing.T) {
	splitter := NewHTMLSection([]Header{
		{Marker: "h1", Name: "Header 1"},
	})
	docs := splitter.SplitText(`<body><p>Preamble text.</p><h1>Intro</h1><p>Welcome.</p></body>`)

	want := []documents.Document{
		documents.New("Preamble text.", map[string]any{"Header 1": "#TITLE#"}),
		documents.New("Intro Welcome.", map[string]any{"Header 1": "Intro"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}

func TestHTMLSectionCreateDocumentsReplacesTitle(t *testing.T) {
	splitter := NewHTMLSection([]Header{
		{Marker: "h1", Name: "Header 1"},
	})
	docs := splitter.CreateDocuments(
		[]string{`<body><p>Preamble text.</p><h1>Intro</h1><p>Welcome.</p></body>`},
		[]map[string]any{{"Title": "My Document"}},
	)

	want := []documents.Document{
		documents.New("Preamble text.", map[string]any{"Header 1": "My Document", "Title": "My Document"}),
		documents.New("Intro Welcome.", map[string]any{"Header 1": "Intro", "Title": "My Document"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs:\n got %#v\nwant %#v", docs, want)
	}
}
