package documentloaders

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestLoad(t *testing.T) {
	loader := fakeLoader{docs: []documents.Document{
		documents.New("a", nil),
		documents.New("b", nil),
	}}
	got, err := Load(context.Background(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(got) != 2 || got[0].PageContent != "a" || got[1].PageContent != "b" {
		t.Fatalf("docs: %#v", got)
	}
}

func TestLoadAndSplit(t *testing.T) {
	loader := fakeLoader{docs: []documents.Document{
		documents.New("a b", map[string]any{"source": "unit"}),
	}}
	got, err := LoadAndSplit(context.Background(), loader, fakeSplitter{})
	if err != nil {
		t.Fatalf("load and split: %v", err)
	}
	if len(got) != 2 || got[1].PageContent != "b" || got[1].Metadata["source"] != "unit" {
		t.Fatalf("docs: %#v", got)
	}
}

func TestLoadAndSplitDefaultSplitterFactory(t *testing.T) {
	RegisterDefaultTextSplitterFactory(nil)
	t.Cleanup(func() { RegisterDefaultTextSplitterFactory(nil) })

	loader := fakeLoader{docs: []documents.Document{documents.New("a b", map[string]any{"source": "unit"})}}
	if _, err := LoadAndSplit(context.Background(), loader, nil); err == nil {
		t.Fatal("expected missing splitter error")
	}

	RegisterDefaultTextSplitterFactory(func() (TextSplitter, error) {
		return fakeSplitter{}, nil
	})
	got, err := LoadAndSplit(context.Background(), loader, nil)
	if err != nil {
		t.Fatalf("load and split with default: %v", err)
	}
	if len(got) != 2 || got[0].Metadata["source"] != "unit" {
		t.Fatalf("docs: %#v", got)
	}
}

func TestParseBlob(t *testing.T) {
	blob := Blob{Data: []byte("hello"), Path: "memory.txt", Metadata: map[string]any{"source": "blob"}}
	data, err := io.ReadAll(blob.Reader())
	if err != nil {
		t.Fatalf("read blob: %v", err)
	}
	if string(data) != "hello" {
		t.Fatalf("blob data: %q", data)
	}

	got, err := Parse(context.Background(), fakeParser{}, blob)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got) != 1 || got[0].PageContent != "hello" || got[0].Metadata["source"] != "blob" {
		t.Fatalf("docs: %#v", got)
	}
}

func TestBlobConstructorsAndSource(t *testing.T) {
	metadata := map[string]any{"source": "memory"}
	blob := NewBlobFromData([]byte("hello"), "text/plain", metadata)
	metadata["source"] = "changed"
	if blob.Source() != "memory" || blob.AsString() != "hello" {
		t.Fatalf("unexpected data blob: %#v", blob)
	}
	data := blob.AsBytes()
	data[0] = 'j'
	if blob.AsString() != "hello" {
		t.Fatal("AsBytes returned internal slice")
	}

	path := filepath.Join(t.TempDir(), "note.txt")
	if err := os.WriteFile(path, []byte("file"), 0o600); err != nil {
		t.Fatal(err)
	}
	fileBlob, err := NewBlobFromPath(path, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if fileBlob.Source() != path || fileBlob.AsString() != "file" {
		t.Fatalf("unexpected file blob: %#v", fileBlob)
	}
	if fileBlob.Mimetype == "" {
		t.Fatal("expected guessed mimetype")
	}
}

type fakeLoader struct {
	docs []documents.Document
}

func (l fakeLoader) LazyLoad(context.Context) (DocumentIterator, error) {
	return NewSliceIterator(l.docs), nil
}

type fakeSplitter struct{}

func (fakeSplitter) SplitDocuments(docs []documents.Document) []documents.Document {
	out := []documents.Document{}
	for _, doc := range docs {
		out = append(out, documents.New("a", doc.Metadata), documents.New("b", doc.Metadata))
	}
	return out
}

type fakeParser struct{}

func (fakeParser) LazyParse(_ context.Context, blob Blob) (DocumentIterator, error) {
	return NewSliceIterator([]documents.Document{
		documents.New(string(blob.Data), blob.Metadata),
	}), nil
}
