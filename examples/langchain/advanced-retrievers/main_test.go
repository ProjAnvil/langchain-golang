package main

import (
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func TestKeywordRetrieverPreservesTieOrderAndMetadata(t *testing.T) {
	retriever := keywordRetriever{docs: []documents.Document{
		{ID: "lower", PageContent: "alpha"},
		{ID: "first", PageContent: "alpha beta"},
		{ID: "second", PageContent: "beta alpha"},
		{ID: "third", PageContent: "alpha beta"},
	}}
	got, err := retriever.GetRelevantDocuments(t.Context(), "alpha beta")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"first", "second", "third"}
	if len(got) != len(want) {
		t.Fatalf("got %d documents, want %d", len(got), len(want))
	}
	for i, id := range want {
		if got[i].ID != id || got[i].Metadata["rank"] != i+1 || got[i].Metadata["member"] != "keyword" {
			t.Fatalf("result[%d] = %#v, want %q with rank %d", i, got[i], id, i+1)
		}
	}
	for _, doc := range retriever.docs {
		if doc.Metadata != nil {
			t.Fatalf("source metadata was mutated: %#v", doc)
		}
	}
}
