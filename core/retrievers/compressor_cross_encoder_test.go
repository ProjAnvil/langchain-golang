package retrievers_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

// stubCrossEncoder records the pairs it was asked to score and returns
// configured scores or an error.
type stubCrossEncoder struct {
	scores []float64
	err    error
	pairs  [][2]string
	calls  int
}

func (e *stubCrossEncoder) Score(pairs [][2]string) ([]float64, error) {
	e.calls++
	e.pairs = pairs
	if e.err != nil {
		return nil, e.err
	}
	return append([]float64(nil), e.scores...), nil
}

func crossEncoderDocs() []documents.Document {
	return []documents.Document{
		documents.New("doc zero", nil),
		documents.New("doc one", nil),
		documents.New("doc two", nil),
		documents.New("doc three", nil),
	}
}

func TestCrossEncoderRerankerTopNOrdering(t *testing.T) {
	model := &stubCrossEncoder{scores: []float64{0.1, 0.9, 0.5, 0.7}}
	reranker, err := retrievers.NewCrossEncoderReranker(
		model,
		retrievers.WithCrossEncoderTopN(3),
	)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	docs := crossEncoderDocs()
	out, err := reranker.CompressDocuments(t.Context(), docs, "the query")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	// Scores descending: 1 (0.9), 3 (0.7), 2 (0.5).
	got := docContents(out)
	want := []string{"doc one", "doc three", "doc two"}
	if len(got) != len(want) {
		t.Fatalf("compressed = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("compressed = %v, want %v", got, want)
		}
	}
	// One (query, content) pair per document, in input order.
	if model.calls != 1 || len(model.pairs) != 4 {
		t.Fatalf("Score called %d times with %d pairs, want 1 call and 4 pairs", model.calls, len(model.pairs))
	}
	for i, pair := range model.pairs {
		if pair[0] != "the query" || pair[1] != docs[i].PageContent {
			t.Fatalf("pair %d = %v, want [the query %s]", i, pair, docs[i].PageContent)
		}
	}
}

func TestCrossEncoderRerankerDefaultTopN(t *testing.T) {
	model := &stubCrossEncoder{scores: []float64{0.1, 0.9, 0.5, 0.7}}
	reranker, err := retrievers.NewCrossEncoderReranker(model)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), crossEncoderDocs(), "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("default top_n returned %d docs, want 3", len(out))
	}
}

func TestCrossEncoderRerankerTopNLargerThanInput(t *testing.T) {
	model := &stubCrossEncoder{scores: []float64{0.2, 0.8}}
	reranker, err := retrievers.NewCrossEncoderReranker(
		model,
		retrievers.WithCrossEncoderTopN(10),
	)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), []documents.Document{
		documents.New("low", nil),
		documents.New("high", nil),
	}, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 2 || got[0] != "high" || got[1] != "low" {
		t.Fatalf("compressed = %v, want [high low]", got)
	}
}

func TestCrossEncoderRerankerTiesKeepInputOrder(t *testing.T) {
	model := &stubCrossEncoder{scores: []float64{0.5, 0.5, 0.5}}
	reranker, err := retrievers.NewCrossEncoderReranker(
		model,
		retrievers.WithCrossEncoderTopN(2),
	)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), crossEncoderDocs()[:3], "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 2 || got[0] != "doc zero" || got[1] != "doc one" {
		t.Fatalf("compressed = %v, want first two in input order", got)
	}
}

func TestCrossEncoderRerankerEmptyInputSkipsModel(t *testing.T) {
	model := &stubCrossEncoder{}
	reranker, err := retrievers.NewCrossEncoderReranker(model)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), nil, "q")
	if err != nil {
		t.Fatalf("compress empty: %v", err)
	}
	if len(out) != 0 || model.calls != 0 {
		t.Fatalf("out=%v calls=%d, want empty result and no Score call", docContents(out), model.calls)
	}
}

func TestCrossEncoderRerankerErrors(t *testing.T) {
	wantErr := errors.New("score failed")
	scoring := &stubCrossEncoder{err: wantErr}
	reranker, err := retrievers.NewCrossEncoderReranker(scoring)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	if _, err := reranker.CompressDocuments(t.Context(), crossEncoderDocs(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}

	mismatch := &stubCrossEncoder{scores: []float64{0.5}} // one score for four docs
	reranker, err = retrievers.NewCrossEncoderReranker(mismatch)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	if _, err := reranker.CompressDocuments(t.Context(), crossEncoderDocs(), "q"); err == nil {
		t.Fatal("score count mismatch must error")
	}

	if _, err := retrievers.NewCrossEncoderReranker(nil); err == nil {
		t.Fatal("nil model must error")
	}
}

func TestCrossEncoderRerankerCanceledContext(t *testing.T) {
	model := &stubCrossEncoder{scores: []float64{0.5, 0.6, 0.7, 0.8}}
	reranker, err := retrievers.NewCrossEncoderReranker(model)
	if err != nil {
		t.Fatalf("new cross encoder reranker: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := reranker.CompressDocuments(ctx, crossEncoderDocs(), "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled compress error = %v, want context.Canceled", err)
	}
}
