package textsplitters

import (
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestSentenceSplitterPythonFixtureSeparator(t *testing.T) {
	text := "This is sentence one. And this is sentence two."
	splitter, err := NewSentence(func(text string, language string) ([]string, error) {
		if language != "english" {
			t.Fatalf("language: got %q want english", language)
		}
		return []string{"This is sentence one.", "And this is sentence two."}, nil
	}, "|||", "english", Config{
		ChunkSize:       100,
		ChunkOverlap:    0,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new sentence splitter: %v", err)
	}

	got, err := splitter.SplitText(text)
	if err != nil {
		t.Fatalf("split text: %v", err)
	}
	want := []string{"This is sentence one.|||And this is sentence two."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestSentenceSpanSplitterPythonFixturePreservesBoundaryWhitespace(t *testing.T) {
	text := "This is sentence one. And this is sentence two."
	splitter, err := NewSentenceSpans(func(text string, language string) ([]SentenceSpan, error) {
		return []SentenceSpan{
			{Start: 0, End: len("This is sentence one.")},
			{Start: len("This is sentence one. "), End: len(text)},
		}, nil
	}, "english", Config{
		ChunkSize:       100,
		ChunkOverlap:    0,
		StripWhitespace: false,
	})
	if err != nil {
		t.Fatalf("new sentence span splitter: %v", err)
	}

	got, err := splitter.SplitText(text)
	if err != nil {
		t.Fatalf("split text: %v", err)
	}
	want := []string{text}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("chunks: got %#v want %#v", got, want)
	}
}

func TestHTMLSemanticPythonFixtureNestedElements(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(
		[]Header{{Marker: "h1", Name: "Header 1"}},
		[]string{"div"},
		Config{ChunkSize: 1000, StripWhitespace: true},
	)
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Main Section</h1>
<div>
    <p>Some text here.</p>
    <div>
        <p>Nested content.</p>
    </div>
</div>`)
	if len(docs) != 2 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[1].PageContent != "Some text here. Nested content." {
		t.Fatalf("content: %#v", docs[1])
	}
	if docs[1].Metadata["Header 1"] != "Main Section" || docs[1].Metadata["tag"] != "div" {
		t.Fatalf("metadata: %#v", docs[1].Metadata)
	}
}

func TestHTMLSemanticPythonFixtureNoFurtherSplits(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(
		[]Header{{Marker: "h1", Name: "Header 1"}},
		[]string{"p"},
		Config{ChunkSize: 1000, StripWhitespace: true},
	)
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Section 1</h1>
<p>Some content here.</p>
<h1>Section 2</h1>
<p>More content here.</p>`)
	if len(docs) != 4 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[1].PageContent != "Some content here." || docs[1].Metadata["Header 1"] != "Section 1" {
		t.Fatalf("first section body: %#v", docs[1])
	}
	if docs[3].PageContent != "More content here." || docs[3].Metadata["Header 1"] != "Section 2" {
		t.Fatalf("second section body: %#v", docs[3])
	}
}

func TestHTMLSemanticPythonFixtureSmallChunkSize(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(
		[]Header{{Marker: "h1", Name: "Header 1"}},
		[]string{"p"},
		Config{
			ChunkSize:       20,
			ChunkOverlap:    5,
			StripWhitespace: true,
		},
	)
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Section 1</h1>
<p>This is some long text that should be split into multiple chunks due to the small chunk size.</p>`)
	if len(docs) < 3 {
		t.Fatalf("expected split body chunks: %#v", docs)
	}
	for _, doc := range docs[1:] {
		if len([]rune(doc.PageContent)) > 20 {
			t.Fatalf("chunk too large: %#v", doc.PageContent)
		}
		if doc.Metadata["Header 1"] != "Section 1" || doc.Metadata["tag"] != "p" {
			t.Fatalf("metadata: %#v", doc.Metadata)
		}
	}
	joined := strings.Join(documentContents(docs[1:]), " ")
	for _, phrase := range []string{"This is some", "small chunk", "size."} {
		if !strings.Contains(joined, phrase) {
			t.Fatalf("missing phrase %q in chunks %#v", phrase, docs)
		}
	}
}

func TestHTMLSemanticPythonFixtureNoHeaders(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(nil, []string{"p"}, Config{
		ChunkSize:       1000,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<p>This is content without any headers.</p>
<p>It should still produce a valid document.</p>`)
	if len(docs) != 2 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[0].PageContent != "This is content without any headers." || len(docs[0].Metadata) != 1 || docs[0].Metadata["tag"] != "p" {
		t.Fatalf("first doc: %#v", docs[0])
	}
	if docs[1].PageContent != "It should still produce a valid document." || docs[1].Metadata["tag"] != "p" {
		t.Fatalf("second doc: %#v", docs[1])
	}
}

func TestHTMLSemanticPythonFixtureScriptStyleIgnored(t *testing.T) {
	splitter, err := NewHTMLSemanticPreserving(nil, []string{"p"}, Config{
		ChunkSize:       1000,
		StripWhitespace: true,
	})
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<script><p>hidden script</p></script>
<style><p>hidden style</p></style>
<p>Visible &amp; normalized text.</p>`)
	if len(docs) != 1 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	if docs[0].PageContent != "Visible & normalized text." {
		t.Fatalf("content: %#v", docs[0].PageContent)
	}
}

func TestHTMLSemanticOptionsPythonFixtureCustomExtractor(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(
		[]Header{{Marker: "h1", Name: "Header 1"}},
		Config{ChunkSize: 1000, StripWhitespace: true},
		HTMLSemanticOptions{
			CustomHandlers: map[string]HTMLCustomHandler{
				"iframe": func(attrs map[string]string, _ string) string {
					src := attrs["src"]
					return "[iframe:" + src + "](" + src + ")"
				},
			},
		},
	)
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Section 1</h1>
<p>This is an iframe:</p>
<iframe src="http://example.com"></iframe>`)
	want := []documents.Document{
		documents.New("This is an iframe: [iframe:http://example.com](http://example.com)", map[string]any{"Header 1": "Section 1"}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs: got %#v want %#v", docs, want)
	}
}

func TestHTMLSemanticOptionsPythonFixtureLinksAndMedia(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(
		[]Header{{Marker: "h1", Name: "Header 1"}},
		Config{ChunkSize: 1000, StripWhitespace: true},
		HTMLSemanticOptions{
			PreserveLinks:  true,
			PreserveImages: true,
			PreserveVideos: true,
			PreserveAudio:  true,
		},
	)
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Section 1</h1>
<p>This is a link to <a href="http://example.com">example.com</a></p>
<p>This is an image:</p>
<img src="http://example.com/image.png" />
<p>This is a video:</p>
<video src="http://example.com/video.mp4"></video>
<p>This is audio:</p>
<audio src="http://example.com/audio.mp3"></audio>`)
	if len(docs) != 1 {
		t.Fatalf("docs: got %d %#v", len(docs), docs)
	}
	for _, fragment := range []string{
		"This is a link to [example.com](http://example.com)",
		"![image:http://example.com/image.png](http://example.com/image.png)",
		"![video:http://example.com/video.mp4](http://example.com/video.mp4)",
		"![audio:http://example.com/audio.mp3](http://example.com/audio.mp3)",
	} {
		if !strings.Contains(docs[0].PageContent, fragment) {
			t.Fatalf("missing %q in %q", fragment, docs[0].PageContent)
		}
	}
	if !reflect.DeepEqual(docs[0].Metadata, map[string]any{"Header 1": "Section 1"}) {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestHTMLSemanticOptionsPythonFixtureFiltersMetadataAndNormalization(t *testing.T) {
	splitter, err := NewHTMLSemanticPreservingWithOptions(
		[]Header{{Marker: "h1", Name: "Header 1"}},
		Config{ChunkSize: 1000, StripWhitespace: true},
		HTMLSemanticOptions{
			AllowlistTags:    []string{"p", "span"},
			DenylistTags:     []string{"em"},
			ExternalMetadata: map[string]any{"source": "example.com"},
			NormalizeText:    true,
		},
	)
	if err != nil {
		t.Fatalf("new html semantic splitter: %v", err)
	}

	docs := splitter.SplitText(`
<h1>Section 1</h1>
<p>This paragraph should be kept!</p>
<span>This SPAN should be kept.</span>
<div>This div should be removed.</div>
<p>Drop <em>THIS</em> word.</p>`)
	want := []documents.Document{
		documents.New("this paragraph should be kept this span should be kept drop word", map[string]any{
			"Header 1": "Section 1",
			"source":   "example.com",
		}),
	}
	if !reflect.DeepEqual(docs, want) {
		t.Fatalf("docs: got %#v want %#v", docs, want)
	}
}

func documentContents(docs []documents.Document) []string {
	out := make([]string, len(docs))
	for i, doc := range docs {
		out[i] = doc.PageContent
	}
	return out
}
