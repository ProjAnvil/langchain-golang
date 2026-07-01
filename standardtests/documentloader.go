package standardtests

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/documentloaders"
	"github.com/projanvil/langchain-golang/core/documents"
)

// DocumentLoaderFactory creates a fresh lazy loader for standard tests.
type DocumentLoaderFactory func(t testing.TB) documentloaders.LazyLoader

// RunDocumentLoaderBasics verifies lazy/eager equivalence and metadata copy
// behavior expected from document loaders.
func RunDocumentLoaderBasics(t *testing.T, factory DocumentLoaderFactory) {
	t.Helper()

	t.Run("lazy eager equivalence", func(t *testing.T) {
		loader := factory(t)
		eager, err := documentloaders.Load(context.Background(), loader)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		iter, err := factory(t).LazyLoad(context.Background())
		if err != nil {
			t.Fatalf("lazy load: %v", err)
		}
		defer iter.Close()
		var lazy []documents.Document
		for {
			doc, ok, err := iter.Next(context.Background())
			if err != nil {
				t.Fatalf("next: %v", err)
			}
			if !ok {
				break
			}
			lazy = append(lazy, doc)
		}
		if len(eager) != len(lazy) {
			t.Fatalf("doc count eager=%d lazy=%d", len(eager), len(lazy))
		}
		for i := range eager {
			if eager[i].PageContent != lazy[i].PageContent || eager[i].Source() != lazy[i].Source() {
				t.Fatalf("doc %d eager=%#v lazy=%#v", i, eager[i], lazy[i])
			}
		}
	})

	t.Run("metadata copy", func(t *testing.T) {
		docs, err := documentloaders.Load(context.Background(), factory(t))
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if len(docs) == 0 {
			t.Fatal("expected at least one document")
		}
		docs[0].Metadata["mutated"] = true
		again, err := documentloaders.Load(context.Background(), factory(t))
		if err != nil {
			t.Fatalf("reload: %v", err)
		}
		if _, ok := again[0].Metadata["mutated"]; ok {
			t.Fatal("loader returned shared metadata")
		}
	})
}
