package documentloaders

import (
	"os"
	"strings"
	"testing"
)

// goldenArticleText is the extraction contract for testdata/article.html:
// non-content tags (script/style/noscript/nav/aside/footer/header/template)
// removed, block-level boundaries rendered as newlines, inline whitespace
// collapsed to single spaces.
const goldenArticleText = `Sample Article
First paragraph with bold and italic text.
Second paragraph mentions
an explicit line break.
Alpha
Beta`

func readTestFixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return data
}

func TestHTMLLoaderExtractsBodyText(t *testing.T) {
	loader := NewHTMLLoader(strings.NewReader(string(readTestFixture(t, "article.html"))))

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].PageContent != goldenArticleText {
		t.Fatalf("content mismatch:\nwant: %q\ngot:  %q", goldenArticleText, docs[0].PageContent)
	}
	if docs[0].Metadata["title"] != "Sample Article" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestHTMLLoaderMetadata(t *testing.T) {
	html := `<html><head><title>Doc Title</title></head><body><p>Hello</p></body></html>`
	loader := NewHTMLLoader(strings.NewReader(html), WithMetadata(map[string]any{
		"source": "input.html",
		"tier":   "gold",
	}))

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	meta := docs[0].Metadata
	if meta["source"] != "input.html" || meta["tier"] != "gold" || meta["title"] != "Doc Title" {
		t.Fatalf("metadata: %#v", meta)
	}
}

func TestHTMLLoaderUserTitleWinsOverDocumentTitle(t *testing.T) {
	html := `<html><head><title>Doc Title</title></head><body><p>Hello</p></body></html>`
	loader := NewHTMLLoader(strings.NewReader(html), WithMetadata(map[string]any{
		"title": "User Title",
	}))

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if docs[0].Metadata["title"] != "User Title" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestHTMLLoaderWithoutTitle(t *testing.T) {
	loader := NewHTMLLoader(strings.NewReader("<p>No title here</p>"))

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if docs[0].PageContent != "No title here" {
		t.Fatalf("content: %q", docs[0].PageContent)
	}
	if _, ok := docs[0].Metadata["title"]; ok {
		t.Fatalf("unexpected title metadata: %#v", docs[0].Metadata)
	}
}

func TestHTMLLoaderMultilineTitleNormalized(t *testing.T) {
	html := `<html><head><title>
  Multi
  line
  title
</title></head><body><p>Body.</p></body></html>`
	loader := NewHTMLLoader(strings.NewReader(html))

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if docs[0].Metadata["title"] != "Multi line title" {
		t.Fatalf("title: %#v", docs[0].Metadata)
	}
}

func TestHTMLLoaderMetadataNotShared(t *testing.T) {
	metadata := map[string]any{"tier": "gold"}
	loader := NewHTMLLoader(strings.NewReader("<p>One</p>"), WithMetadata(metadata))
	metadata["tier"] = "changed"

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if docs[0].Metadata["tier"] != "gold" {
		t.Fatalf("metadata: %#v", docs[0].Metadata)
	}
}

func TestHTMLLoaderLoadAndSplit(t *testing.T) {
	loader := NewHTMLLoader(strings.NewReader("<p>alpha beta</p>"))

	docs, err := LoadAndSplit(t.Context(), loader, fakeSplitter{})
	if err != nil {
		t.Fatalf("load and split: %v", err)
	}
	if len(docs) != 2 || docs[0].PageContent != "a" || docs[1].PageContent != "b" {
		t.Fatalf("docs: %#v", docs)
	}
}

func TestHTMLLoaderReaderError(t *testing.T) {
	loader := NewHTMLLoader(errReader{err: errTest})
	if _, err := loader.LazyLoad(t.Context()); err == nil {
		t.Fatal("expected read error")
	}
}

type errReader struct {
	err error
}

func (r errReader) Read([]byte) (int, error) { return 0, r.err }
