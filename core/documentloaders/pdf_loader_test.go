package documentloaders

import (
	"strings"
	"testing"
)

func TestPDFLoaderExtractsPages(t *testing.T) {
	loader := NewPDFLoader(strings.NewReader(string(readTestFixture(t, "article.pdf"))))

	docs, err := Load(t.Context(), loader)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].PageContent != "PDF article page one.\nIt has two lines." {
		t.Fatalf("page 0 content: %q", docs[0].PageContent)
	}
	if docs[1].PageContent != "PDF article page two." {
		t.Fatalf("page 1 content: %q", docs[1].PageContent)
	}
	// Page numbers are 0-based, matching Python's PyPDFLoader which
	// enumerates pdf.pages from 0.
	if docs[0].Metadata["page"] != 0 || docs[1].Metadata["page"] != 1 {
		t.Fatalf("page metadata: %#v / %#v", docs[0].Metadata, docs[1].Metadata)
	}
	if docs[0].Metadata["total_pages"] != 2 || docs[1].Metadata["total_pages"] != 2 {
		t.Fatalf("total_pages metadata: %#v / %#v", docs[0].Metadata, docs[1].Metadata)
	}
}

func TestPDFLoaderMetadataNotShared(t *testing.T) {
	data := string(readTestFixture(t, "article.pdf"))

	first, err := Load(t.Context(), NewPDFLoader(strings.NewReader(data)))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	first[0].Metadata["mutated"] = true

	second, err := Load(t.Context(), NewPDFLoader(strings.NewReader(data)))
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if _, ok := second[0].Metadata["mutated"]; ok {
		t.Fatal("loader returned shared metadata")
	}
}

func TestPDFLoaderCorruptInput(t *testing.T) {
	loader := NewPDFLoader(strings.NewReader("this is not a pdf"))
	if _, err := loader.LazyLoad(t.Context()); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestPDFLoaderReaderError(t *testing.T) {
	loader := NewPDFLoader(errReader{err: errTest})
	if _, err := loader.LazyLoad(t.Context()); err == nil {
		t.Fatal("expected read error")
	}
}

func TestPDFLoaderLoadAndSplit(t *testing.T) {
	loader := NewPDFLoader(strings.NewReader(string(readTestFixture(t, "article.pdf"))))

	docs, err := LoadAndSplit(t.Context(), loader, fakeSplitter{})
	if err != nil {
		t.Fatalf("load and split: %v", err)
	}
	// fakeSplitter doubles every document: 2 pages -> 4 chunks.
	if len(docs) != 4 {
		t.Fatalf("docs: %#v", docs)
	}
	if docs[0].Metadata["page"] != 0 || docs[2].Metadata["page"] != 1 {
		t.Fatalf("metadata: %#v / %#v", docs[0].Metadata, docs[2].Metadata)
	}
}
