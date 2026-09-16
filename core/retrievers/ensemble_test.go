package retrievers

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
)

func docsOf(contents ...string) []documents.Document {
	docs := make([]documents.Document, len(contents))
	for i, content := range contents {
		docs[i] = documents.New(content, nil)
	}
	return docs
}

// Python parity: langchain_community EnsembleRetriever fuses ranked lists
// with reciprocal rank fusion, score(doc) = sum(weight / (rank + 60)) over
// 1-based ranks (enumerate(doc_list, start=1)), sorted descending with the
// first-seen document winning ties and duplicates.
func TestEnsembleRetrieverRRFOrdering(t *testing.T) {
	first := Static{Documents: docsOf("doc-a", "doc-b", "doc-c")}
	second := Static{Documents: docsOf("doc-c", "doc-d")}

	retriever, err := NewEnsembleRetriever([]Retriever{first, second})
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}

	var contents []string
	for _, doc := range docs {
		contents = append(contents, doc.PageContent)
	}
	// R1: [a, b, c], R2: [c, d], equal weights 0.5, 1-based ranks:
	//   c: 0.5/63 + 0.5/61 (rank 3 in R1, rank 1 in R2) -> highest
	//   a: 0.5/61
	//   b: 0.5/62, d: 0.5/62 (tie; b seen first via R1)
	want := []string{"doc-c", "doc-a", "doc-b", "doc-d"}
	if !slices.Equal(contents, want) {
		t.Fatalf("fused docs: got %v want %v", contents, want)
	}

	wantScoreC := 0.5/(60+3) + 0.5/(60+1)
	gotScore, ok := docs[0].Metadata["score"].(float64)
	if !ok || math.Abs(gotScore-wantScoreC) > 1e-12 {
		t.Fatalf("fused score metadata: got %v want %v", docs[0].Metadata["score"], wantScoreC)
	}
}

func TestEnsembleRetrieverWeightsBiasFusion(t *testing.T) {
	first := Static{Documents: docsOf("doc-a")}
	second := Static{Documents: docsOf("doc-b")}

	heavyFirst, err := NewEnsembleRetriever(
		[]Retriever{first, second},
		WithWeights([]float64{0.9, 0.1}),
	)
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err := heavyFirst.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 2 || docs[0].PageContent != "doc-a" {
		t.Fatalf("docs: got %v", docs)
	}

	heavySecond, err := NewEnsembleRetriever(
		[]Retriever{first, second},
		WithWeights([]float64{0.2, 0.8}),
	)
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err = heavySecond.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 2 || docs[0].PageContent != "doc-b" {
		t.Fatalf("docs: got %v", docs)
	}
}

// Python parity: documents sharing the id_key metadata are deduplicated to
// the first retriever's instance but accumulate every retriever's rank
// contribution.
func TestEnsembleRetrieverDeduplicatesByIDKey(t *testing.T) {
	first := Static{Documents: []documents.Document{
		documents.New("content-from-first", map[string]any{"id": "shared"}),
	}}
	second := Static{Documents: []documents.Document{
		documents.New("content-from-second", map[string]any{"id": "shared"}),
	}}

	retriever, err := NewEnsembleRetriever([]Retriever{first, second})
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "content-from-first" {
		t.Fatalf("docs: got %v", docs)
	}
	wantScore := 0.5/61 + 0.5/61
	gotScore, ok := docs[0].Metadata["score"].(float64)
	if !ok || math.Abs(gotScore-wantScore) > 1e-12 {
		t.Fatalf("fused score: got %v want %v", docs[0].Metadata["score"], wantScore)
	}
}

func TestEnsembleRetrieverEmptyMemberResults(t *testing.T) {
	empty := Static{}
	nonEmpty := Static{Documents: docsOf("doc-a")}

	retriever, err := NewEnsembleRetriever([]Retriever{empty, nonEmpty})
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "doc-a" {
		t.Fatalf("docs: got %v", docs)
	}

	allEmpty, err := NewEnsembleRetriever([]Retriever{empty})
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err = allEmpty.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs: got %d want 0", len(docs))
	}
}

// Ensemble output must not mutate member retriever documents when annotating
// fused scores.
func TestEnsembleRetrieverDoesNotMutateMemberDocs(t *testing.T) {
	shared := Static{Documents: docsOf("doc-a")}

	retriever, err := NewEnsembleRetriever([]Retriever{shared, shared})
	if err != nil {
		t.Fatalf("new ensemble retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if _, polluted := shared.Documents[0].Metadata["score"]; polluted {
		t.Fatalf("member retriever document was mutated: %v", shared.Documents[0].Metadata)
	}
	if _, ok := docs[0].Metadata["score"]; !ok {
		t.Fatalf("fused score metadata missing: %v", docs[0].Metadata)
	}
}

func TestEnsembleRetrieverErrors(t *testing.T) {
	t.Run("constructor requires retrievers", func(t *testing.T) {
		if _, err := NewEnsembleRetriever(nil); err == nil {
			t.Fatal("empty retriever list must error")
		}
		if _, err := NewEnsembleRetriever([]Retriever{Static{}, nil}); err == nil {
			t.Fatal("nil retriever must error")
		}
	})

	t.Run("weights must match retriever count", func(t *testing.T) {
		_, err := NewEnsembleRetriever(
			[]Retriever{Static{}, Static{}},
			WithWeights([]float64{0.5}),
		)
		if err == nil {
			t.Fatal("weight count mismatch must error")
		}
	})

	t.Run("weights must be non-negative", func(t *testing.T) {
		_, err := NewEnsembleRetriever(
			[]Retriever{Static{}, Static{}},
			WithWeights([]float64{0.5, -0.5}),
		)
		if err == nil {
			t.Fatal("negative weight must error")
		}
	})

	t.Run("member retriever error propagates", func(t *testing.T) {
		memberErr := errors.New("member boom")
		retriever, err := NewEnsembleRetriever([]Retriever{
			Static{},
			fnRetriever(func(context.Context, string) ([]documents.Document, error) {
				return nil, memberErr
			}),
		})
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, memberErr) {
			t.Fatalf("want member error, got %v", err)
		}
	})

	t.Run("canceled context", func(t *testing.T) {
		retriever, err := NewEnsembleRetriever([]Retriever{Static{}})
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := retriever.GetRelevantDocuments(ctx, "q"); !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	})
}
