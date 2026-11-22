package retrievers_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

func staticWith(contents ...string) retrievers.Static {
	docs := make([]documents.Document, len(contents))
	for i, content := range contents {
		docs[i] = documents.New(content, nil)
	}
	return retrievers.Static{Documents: docs}
}

func TestMergerRetrieverRoundRobinInterleaving(t *testing.T) {
	first := staticWith("a1", "a2", "a3")
	second := staticWith("b1", "b2")
	third := staticWith("c1")
	merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{first, second, third})
	if err != nil {
		t.Fatalf("new merger retriever: %v", err)
	}
	docs, err := merger.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	want := []string{"a1", "b1", "c1", "a2", "b2", "a3"}
	got := docContents(docs)
	if len(got) != len(want) {
		t.Fatalf("merged = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("merged = %v, want %v", got, want)
		}
	}
}

func TestMergerRetrieverNoDeduplication(t *testing.T) {
	first := staticWith("dup", "x")
	second := staticWith("dup")
	merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{first, second})
	if err != nil {
		t.Fatalf("new merger retriever: %v", err)
	}
	docs, err := merger.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	want := []string{"dup", "dup", "x"}
	got := docContents(docs)
	if len(got) != len(want) || got[0] != "dup" || got[1] != "dup" || got[2] != "x" {
		t.Fatalf("merged = %v, want %v (duplicates preserved)", got, want)
	}
}

func TestMergerRetrieverEmptyAndMixedResults(t *testing.T) {
	t.Run("no retrievers is an error", func(t *testing.T) {
		if _, err := retrievers.NewMergerRetriever(nil); err == nil {
			t.Fatal("empty retriever list must error")
		}
	})

	t.Run("all empty results", func(t *testing.T) {
		merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{
			staticWith(), staticWith(),
		})
		if err != nil {
			t.Fatalf("new merger retriever: %v", err)
		}
		docs, err := merger.GetRelevantDocuments(t.Context(), "query")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		if len(docs) != 0 {
			t.Fatalf("docs = %v, want empty", docContents(docs))
		}
	})

	t.Run("empty member does not disturb interleaving", func(t *testing.T) {
		merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{
			staticWith(), staticWith("b1", "b2"), staticWith(),
		})
		if err != nil {
			t.Fatalf("new merger retriever: %v", err)
		}
		docs, err := merger.GetRelevantDocuments(t.Context(), "query")
		if err != nil {
			t.Fatalf("retrieve: %v", err)
		}
		got := docContents(docs)
		if len(got) != 2 || got[0] != "b1" || got[1] != "b2" {
			t.Fatalf("merged = %v, want [b1 b2]", got)
		}
	})
}

func TestMergerRetrieverConstructorRejectsNilMember(t *testing.T) {
	if _, err := retrievers.NewMergerRetriever([]retrievers.Retriever{staticWith("a"), nil}); err == nil {
		t.Fatal("nil retriever member must error")
	}
}

func TestMergerRetrieverForwardsQueryToAllMembersConcurrently(t *testing.T) {
	var mu sync.Mutex
	seen := make([]string, 0, 3)
	makeRecorder := func(tag string) stubRetriever {
		return func(_ context.Context, query string) ([]documents.Document, error) {
			mu.Lock()
			seen = append(seen, tag+":"+query)
			mu.Unlock()
			return []documents.Document{documents.New(tag, nil)}, nil
		}
	}
	merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{
		makeRecorder("r1"), makeRecorder("r2"), makeRecorder("r3"),
	})
	if err != nil {
		t.Fatalf("new merger retriever: %v", err)
	}
	if _, err := merger.GetRelevantDocuments(t.Context(), "the query"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(seen) != 3 {
		t.Fatalf("members called %d times, want 3: %v", len(seen), seen)
	}
	for _, entry := range seen {
		if entry != "r1:the query" && entry != "r2:the query" && entry != "r3:the query" {
			t.Fatalf("unexpected recorded call %q", entry)
		}
	}
}

func TestMergerRetrieverErrorPropagation(t *testing.T) {
	wantErr := errors.New("retriever failed")
	merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{
		staticWith("a"),
		stubRetriever(func(context.Context, string) ([]documents.Document, error) {
			return nil, wantErr
		}),
		staticWith("c"),
	})
	if err != nil {
		t.Fatalf("new merger retriever: %v", err)
	}
	if _, err := merger.GetRelevantDocuments(t.Context(), "query"); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

func TestMergerRetrieverCanceledContext(t *testing.T) {
	merger, err := retrievers.NewMergerRetriever([]retrievers.Retriever{staticWith("a")})
	if err != nil {
		t.Fatalf("new merger retriever: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := merger.GetRelevantDocuments(ctx, "query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled retrieval error = %v, want context.Canceled", err)
	}
}
