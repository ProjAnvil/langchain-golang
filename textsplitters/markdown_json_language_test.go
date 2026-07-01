package textsplitters

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestRecursiveCharacterFromLanguage(t *testing.T) {
	separators, err := SeparatorsForLanguage(LanguageGo)
	if err != nil {
		t.Fatalf("separators: %v", err)
	}
	if len(separators) == 0 || separators[0] != "\nfunc " {
		t.Fatalf("go separators: %#v", separators)
	}

	splitter, err := NewRecursiveCharacterFromLanguage(LanguagePython, Config{
		ChunkSize:       24,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new language splitter: %v", err)
	}
	chunks := splitter.SplitText("class A:\n    pass\n\ndef b():\n    pass")
	if len(chunks) < 2 {
		t.Fatalf("expected language-aware chunks, got %#v", chunks)
	}
}

func TestAdditionalPythonLanguageSeparators(t *testing.T) {
	tests := []struct {
		language Language
		first    string
		contains string
	}{
		{language: LanguageR, first: "\nfunction ", contains: "\nsetClass\\("},
		{language: LanguageElixir, first: "\ndef ", contains: "\ndefmodule "},
		{language: LanguageCOBOL, first: "\nIDENTIFICATION DIVISION.", contains: "\nSTOP RUN."},
	}
	for _, tt := range tests {
		t.Run(string(tt.language), func(t *testing.T) {
			separators, err := SeparatorsForLanguage(tt.language)
			if err != nil {
				t.Fatalf("separators: %v", err)
			}
			if len(separators) == 0 || separators[0] != tt.first || separators[len(separators)-1] != "" {
				t.Fatalf("separators: %#v", separators)
			}
			found := false
			for _, separator := range separators {
				if separator == tt.contains {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("missing separator %q in %#v", tt.contains, separators)
			}
		})
	}
}

func TestMarkdownHeaderTextSplitter(t *testing.T) {
	splitter := NewMarkdownHeader([]Header{
		{Marker: "#", Name: "Header 1"},
		{Marker: "##", Name: "Header 2"},
	}, false, true)

	docs := splitter.SplitText("# Title\nintro\n\n## Section\nbody\n```go\n# not a header\n```\nmore")
	if len(docs) != 2 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[0].PageContent != "intro" || docs[0].Metadata["Header 1"] != "Title" {
		t.Fatalf("first doc: %#v", docs[0])
	}
	if !strings.Contains(docs[1].PageContent, "# not a header") {
		t.Fatalf("code block was not preserved: %#v", docs[1].PageContent)
	}
	wantMetadata := map[string]any{"Header 1": "Title", "Header 2": "Section"}
	if !reflect.DeepEqual(docs[1].Metadata, wantMetadata) {
		t.Fatalf("metadata: got %#v want %#v", docs[1].Metadata, wantMetadata)
	}
}

func TestHTMLHeaderTextSplitter(t *testing.T) {
	splitter := NewHTMLHeader([]Header{
		{Marker: "h1", Name: "Header 1"},
		{Marker: "h2", Name: "Header 2"},
	}, false)

	docs := splitter.SplitText(`
<html><body>
<h1>Introduction</h1>
<p>Welcome to the introduction section.</p>
<h2>Background</h2>
<p>Some background details here.</p>
<h1>Conclusion</h1>
<p>Final thoughts.</p>
</body></html>`)
	if len(docs) != 6 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[0].PageContent != "Introduction" || docs[0].Metadata["Header 1"] != "Introduction" {
		t.Fatalf("first header: %#v", docs[0])
	}
	if docs[1].PageContent != "Welcome to the introduction section." || docs[1].Metadata["Header 1"] != "Introduction" {
		t.Fatalf("intro body: %#v", docs[1])
	}
	wantMetadata := map[string]any{"Header 1": "Introduction", "Header 2": "Background"}
	if docs[3].PageContent != "Some background details here." || !reflect.DeepEqual(docs[3].Metadata, wantMetadata) {
		t.Fatalf("nested body: %#v", docs[3])
	}
	if _, ok := docs[4].Metadata["Header 2"]; ok {
		t.Fatalf("h2 metadata should be cleared after new h1: %#v", docs[4])
	}
}

func TestHTMLHeaderTextSplitterReturnEachElement(t *testing.T) {
	splitter := NewHTMLHeader([]Header{{Marker: "h1", Name: "Header 1"}}, true)
	docs := splitter.SplitText(`<h1>Title</h1><p>First</p><p>Second</p>`)
	if len(docs) != 3 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[1].PageContent != "First" || docs[2].PageContent != "Second" {
		t.Fatalf("element docs: %#v", docs)
	}
}

func TestHTMLHeaderTextSplitterFallbackNoHeaders(t *testing.T) {
	splitter := NewHTMLHeader([]Header{{Marker: "h1", Name: "Header 1"}}, false)
	docs := splitter.SplitText(`<div>Hello <strong>world</strong></div>`)
	if len(docs) != 1 || docs[0].PageContent != "Hello world" || len(docs[0].Metadata) != 0 {
		t.Fatalf("fallback docs: %#v", docs)
	}
}

func TestHTMLSemanticPreservingSplitter(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving([]Header{
		{Marker: "h1", Name: "Header 1"},
		{Marker: "h2", Name: "Header 2"},
	}, []string{"section", "article"}, Config{ChunkSize: 120})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Guide</h1>
<section><p>Intro text.</p><p>Still the same section.</p></section>
<h2>Details</h2>
<article><p>Deep detail.</p></article>`)
	if len(docs) != 4 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[1].PageContent != "Intro text. Still the same section." || docs[1].Metadata["tag"] != "section" {
		t.Fatalf("section doc: %#v", docs[1])
	}
	wantMetadata := map[string]any{"Header 1": "Guide", "Header 2": "Details", "tag": "article"}
	if !reflect.DeepEqual(docs[3].Metadata, wantMetadata) {
		t.Fatalf("article metadata: got %#v want %#v", docs[3].Metadata, wantMetadata)
	}
}

func TestHTMLSemanticPreservingSplitterSplitsLargeBlockAndMergesMetadata(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving([]Header{{Marker: "h1", Name: "Header 1"}}, []string{"section"}, Config{
		ChunkSize:       30,
		ChunkOverlap:    0,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new splitter: %v", err)
	}

	docs := splitter.CreateDocuments([]string{
		`<h1>Title</h1><section>alpha beta gamma delta epsilon zeta eta theta</section>`,
	}, []map[string]any{{"source": "unit"}})
	if len(docs) < 3 {
		t.Fatalf("expected header plus split section chunks: %#v", docs)
	}
	if docs[1].Metadata["source"] != "unit" || docs[1].Metadata["Header 1"] != "Title" || docs[1].Metadata["tag"] != "section" {
		t.Fatalf("metadata not merged: %#v", docs[1])
	}
	for _, doc := range docs[1:] {
		if len([]rune(doc.PageContent)) > 30 {
			t.Fatalf("chunk too large: %#v", doc.PageContent)
		}
	}
}

func TestRecursiveJSONSplitter(t *testing.T) {
	splitter := NewRecursiveJSON(45, 20)
	chunks := splitter.SplitJSON(map[string]any{
		"a": strings.Repeat("x", 30),
		"b": map[string]any{
			"c": strings.Repeat("y", 30),
		},
	}, false)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks: %#v", chunks)
	}

	texts, err := splitter.SplitText(map[string]any{
		"items": []any{"a", "b"},
	}, true)
	if err != nil {
		t.Fatalf("split text: %v", err)
	}
	if len(texts) != 1 {
		t.Fatalf("texts: %#v", texts)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(texts[0]), &parsed); err != nil {
		t.Fatalf("json output: %v", err)
	}
	items, ok := parsed["items"].(map[string]any)
	if !ok || items["0"] != "a" || items["1"] != "b" {
		t.Fatalf("converted list: %#v", parsed)
	}

	docs, err := splitter.CreateDocuments([]map[string]any{{"name": "langchain"}}, false, []map[string]any{{"source": "unit"}})
	if err != nil {
		t.Fatalf("create docs: %v", err)
	}
	if len(docs) != 1 || docs[0].Metadata["source"] != "unit" {
		t.Fatalf("docs: %#v", docs)
	}
}
