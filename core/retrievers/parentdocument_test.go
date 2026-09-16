package retrievers

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/stores"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// paragraphSplitter is a child splitter that splits page content on blank
// lines, mirroring what a real text splitter produces for these fixtures.
func paragraphSplitter(
	ctx context.Context,
	docs []documents.Document,
) ([]documents.Document, error) {
	out := make([]documents.Document, 0, len(docs))
	for _, doc := range docs {
		for _, part := range strings.Split(doc.PageContent, "\n\n") {
			if part == "" {
				continue
			}
			// Children deliberately carry no ID: like Python's splitter
			// output they are fresh documents identified by their parent-id
			// metadata, and sharing the parent's ID would collapse them in
			// ID-keyed stores.
			out = append(out, documents.New(part, doc.Metadata))
		}
	}
	return out, nil
}

func newParentDocumentStack(t *testing.T, opts ...ParentDocumentOption) (
	ParentDocumentRetriever,
	*vectorstores.InMemory,
	*stores.InMemoryStore[documents.Document],
) {
	t.Helper()
	store := vectorstores.NewInMemory(embeddings.NewFake(8))
	docstore := stores.NewInMemoryStore[documents.Document]()
	retriever, err := NewParentDocumentRetriever(store, docstore, opts...)
	if err != nil {
		t.Fatalf("new parent-document retriever: %v", err)
	}
	return retriever, store, docstore
}

// Python parity: langchain_community ParentDocumentRetriever.add_documents
// splits parents into children tagged with the parent doc id, stores parents
// whole in the docstore, and retrieval returns full parent documents for
// child hits (deduplicated by parent id).
func TestParentDocumentRetrieverEndToEnd(t *testing.T) {
	retriever, store, docstore := newParentDocumentStack(
		t,
		WithChildSplitter(documents.TransformerFunc(paragraphSplitter)),
	)

	parents := []documents.Document{
		documents.New("parent one first\n\nparent one second", map[string]any{"source": "p1"}).
			WithID("parent-1"),
		documents.New("parent two first\n\nparent two second", map[string]any{"source": "p2"}).
			WithID("parent-2"),
	}
	if err := retriever.AddDocuments(t.Context(), parents); err != nil {
		t.Fatalf("add documents: %v", err)
	}

	keys, err := docstore.YieldKeys(t.Context(), "")
	if err != nil {
		t.Fatalf("yield docstore keys: %v", err)
	}
	slices.Sort(keys)
	if !slices.Equal(keys, []string{"parent-1", "parent-2"}) {
		t.Fatalf("docstore keys: got %v", keys)
	}
	values, err := docstore.MGet(t.Context(), keys)
	if err != nil {
		t.Fatalf("mget parents: %v", err)
	}
	if !values[0].Found || values[0].Value.PageContent != "parent one first\n\nparent one second" {
		t.Fatalf("stored parent: %+v", values[0])
	}

	// All four children must be indexed, each tagged with its parent id and
	// holding only the child chunk text.
	children, err := store.SimilaritySearch(t.Context(), "parent", 10)
	if err != nil {
		t.Fatalf("list children: %v", err)
	}
	if len(children) != 4 {
		t.Fatalf("indexed children: got %d want 4", len(children))
	}
	for _, child := range children {
		if strings.Contains(child.PageContent, "\n\n") {
			t.Fatalf("child must hold a chunk, got full text %q", child.PageContent)
		}
		if got := child.Metadata["doc_id"]; got != "parent-1" && got != "parent-2" {
			t.Fatalf("child doc_id metadata: got %v", got)
		}
	}

	// Retrieval (k=4 hits every child) must return the two full parents,
	// deduplicated even though multiple children of each parent hit.
	docs, err := retriever.GetRelevantDocuments(t.Context(), "parent one")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	var contents []string
	for _, doc := range docs {
		contents = append(contents, doc.PageContent)
	}
	slices.Sort(contents)
	want := []string{
		"parent one first\n\nparent one second",
		"parent two first\n\nparent two second",
	}
	if !slices.Equal(contents, want) {
		t.Fatalf("retrieved parents: got %q want %q", contents, want)
	}
	if docs[0].Metadata["source"] == "" {
		t.Fatalf("parent metadata must survive: %v", docs[0].Metadata)
	}
}

func TestParentDocumentRetrieverCustomIDKey(t *testing.T) {
	retriever, store, _ := newParentDocumentStack(
		t,
		WithChildSplitter(documents.TransformerFunc(paragraphSplitter)),
		WithIDKey("parent_id"),
	)

	err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("only parent body", nil).WithID("p-1"),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}
	children, err := store.SimilaritySearch(t.Context(), "only", 10)
	if err != nil {
		t.Fatalf("list children: %v", err)
	}
	if len(children) != 1 || children[0].Metadata["parent_id"] != "p-1" {
		t.Fatalf("child metadata: %+v", children)
	}
}

// Deviation from Python (which requires child_splitter and raises ValueError
// in add_documents): a nil splitter indexes each parent as a single child.
func TestParentDocumentRetrieverNilSplitterIndexesWholeParents(t *testing.T) {
	retriever, store, docstore := newParentDocumentStack(t)

	err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("whole document body", nil).WithID("whole-1"),
	})
	if err != nil {
		t.Fatalf("add documents: %v", err)
	}
	children, err := store.SimilaritySearch(t.Context(), "whole", 10)
	if err != nil {
		t.Fatalf("list children: %v", err)
	}
	if len(children) != 1 || children[0].PageContent != "whole document body" {
		t.Fatalf("indexed child: %+v", children)
	}
	keys, err := docstore.YieldKeys(t.Context(), "")
	if err != nil || len(keys) != 1 || keys[0] != "whole-1" {
		t.Fatalf("docstore keys: %v err %v", keys, err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "whole document")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "whole document body" {
		t.Fatalf("docs: %+v", docs)
	}
}

// Python parity: parents missing from the docstore are skipped (`[d for d in
// docs if d is not None]`).
func TestParentDocumentRetrieverMissingParentSkipped(t *testing.T) {
	retriever, _, docstore := newParentDocumentStack(
		t,
		WithChildSplitter(documents.TransformerFunc(paragraphSplitter)),
	)

	if err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("kept parent first\n\nkept parent second", nil).WithID("kept"),
		documents.New("dropped parent first\n\ndropped parent second", nil).WithID("dropped"),
	}); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	if err := docstore.MDelete(t.Context(), []string{"dropped"}); err != nil {
		t.Fatalf("delete parent: %v", err)
	}

	docs, err := retriever.GetRelevantDocuments(t.Context(), "parent")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "kept parent first\n\nkept parent second" {
		t.Fatalf("docs: %+v", docs)
	}
}

func TestParentDocumentRetrieverGeneratesParentIDs(t *testing.T) {
	retriever, _, docstore := newParentDocumentStack(t)

	if err := retriever.AddDocuments(t.Context(), []documents.Document{
		documents.New("body one", nil),
		documents.New("body two", nil),
	}); err != nil {
		t.Fatalf("add documents: %v", err)
	}
	keys, err := docstore.YieldKeys(t.Context(), "")
	if err != nil {
		t.Fatalf("yield keys: %v", err)
	}
	if len(keys) != 2 {
		t.Fatalf("generated parent ids: got %v", keys)
	}
	for _, key := range keys {
		if key == "" {
			t.Fatalf("parent ids must be non-empty, got %v", keys)
		}
	}
}

func TestParentDocumentRetrieverErrors(t *testing.T) {
	t.Run("constructor requires store and docstore", func(t *testing.T) {
		if _, err := NewParentDocumentRetriever(nil, stores.NewInMemoryStore[documents.Document]()); err == nil {
			t.Fatal("nil vector store must error")
		}
		if _, err := NewParentDocumentRetriever(
			vectorstores.NewInMemory(embeddings.NewFake(4)),
			nil,
		); err == nil {
			t.Fatal("nil docstore must error")
		}
	})

	t.Run("splitter error propagates from add", func(t *testing.T) {
		splitErr := errors.New("split boom")
		retriever, _, _ := newParentDocumentStack(
			t,
			WithChildSplitter(documents.TransformerFunc(
				func(context.Context, []documents.Document) ([]documents.Document, error) {
					return nil, splitErr
				},
			)),
		)
		err := retriever.AddDocuments(t.Context(), []documents.Document{documents.New("x", nil)})
		if !errors.Is(err, splitErr) {
			t.Fatalf("want splitter error, got %v", err)
		}
	})

	t.Run("empty add is a no-op", func(t *testing.T) {
		retriever, store, docstore := newParentDocumentStack(t)
		if err := retriever.AddDocuments(t.Context(), nil); err != nil {
			t.Fatalf("add nil: %v", err)
		}
		keys, err := docstore.YieldKeys(t.Context(), "")
		if err != nil || len(keys) != 0 {
			t.Fatalf("docstore must stay empty: %v %v", keys, err)
		}
		children, err := store.SimilaritySearch(t.Context(), "x", 10)
		if err != nil || len(children) != 0 {
			t.Fatalf("vector store must stay empty: %v %v", children, err)
		}
	})

	t.Run("canceled context", func(t *testing.T) {
		retriever, _, _ := newParentDocumentStack(t)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := retriever.GetRelevantDocuments(ctx, "q"); !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	})
}
