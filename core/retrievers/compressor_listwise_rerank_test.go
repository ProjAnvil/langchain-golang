package retrievers_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

func listwiseDocs() []documents.Document {
	return []documents.Document{
		documents.New("doc zero", nil),
		documents.New("doc one", nil),
		documents.New("doc two", nil),
		documents.New("doc three", nil),
		documents.New("doc four", nil),
	}
}

func TestLLMListwiseRerankRanksByIds(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("2, 0, 4"))
	reranker, err := retrievers.NewLLMListwiseRerank(
		model,
		retrievers.WithListwiseRerankTopN(3),
	)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), listwiseDocs(), "who is steve")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	got := docContents(out)
	want := []string{"doc two", "doc zero", "doc four"}
	if len(got) != len(want) {
		t.Fatalf("reranked = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("reranked = %v, want %v", got, want)
		}
	}
}

func TestLLMListwiseRerankUsesOneLLMCall(t *testing.T) {
	calls := 0
	var captured string
	model := textModelFunc(func(_ context.Context, prompt string) (string, error) {
		calls++
		captured = prompt
		return "4, 3, 2, 1, 0", nil
	})
	reranker, err := retrievers.NewLLMListwiseRerank(model)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	if _, err := reranker.CompressDocuments(t.Context(), listwiseDocs(), "the query"); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if calls != 1 {
		t.Fatalf("model invoked %d times, want exactly one listwise call", calls)
	}
	if !strings.Contains(captured, "the query") {
		t.Fatalf("prompt missing query: %q", captured)
	}
	for _, content := range []string{"doc zero", "doc one", "doc two", "doc three", "doc four"} {
		if !strings.Contains(captured, content) {
			t.Fatalf("prompt missing document %q: %q", content, captured)
		}
	}
	if !strings.Contains(captured, "Document ID: 0") {
		t.Fatalf("prompt missing document ids: %q", captured)
	}
}

func TestLLMListwiseRerankDropsInvalidAndDuplicateIds(t *testing.T) {
	// 2 kept, duplicate 2 skipped, 99 out of range skipped, 0 kept, 1 kept.
	model := language.NewFakeLLM(language.WithLLMResponses("2, 2, 99, 0, 1"))
	reranker, err := retrievers.NewLLMListwiseRerank(
		model,
		retrievers.WithListwiseRerankTopN(5),
	)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), listwiseDocs(), "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	got := docContents(out)
	want := []string{"doc two", "doc zero", "doc one"}
	if len(got) != len(want) {
		t.Fatalf("reranked = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("reranked = %v, want %v", got, want)
		}
	}
}

func TestLLMListwiseRerankTopNTruncates(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("0, 1, 2, 3, 4"))
	reranker, err := retrievers.NewLLMListwiseRerank(
		model,
		retrievers.WithListwiseRerankTopN(2),
	)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), listwiseDocs(), "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 2 || got[0] != "doc zero" || got[1] != "doc one" {
		t.Fatalf("reranked = %v, want [doc zero doc one]", got)
	}
}

func TestLLMListwiseRerankCustomPrompt(t *testing.T) {
	var captured string
	model := textModelFunc(func(_ context.Context, prompt string) (string, error) {
		captured = prompt
		return "0", nil
	})
	reranker, err := retrievers.NewLLMListwiseRerank(
		model,
		retrievers.WithListwiseRerankPrompt("Q={query} D={documents}"),
	)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	if _, err := reranker.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("only doc", nil)},
		"the query",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if !strings.HasPrefix(captured, "Q=the query D=Document ID: 0") {
		t.Fatalf("custom prompt = %q, want substituted placeholders", captured)
	}
}

func TestLLMListwiseRerankEmptyInputSkipsModel(t *testing.T) {
	calls := 0
	model := textModelFunc(func(context.Context, string) (string, error) {
		calls++
		return "0", nil
	})
	reranker, err := retrievers.NewLLMListwiseRerank(model)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	out, err := reranker.CompressDocuments(t.Context(), nil, "q")
	if err != nil {
		t.Fatalf("compress empty: %v", err)
	}
	if len(out) != 0 || calls != 0 {
		t.Fatalf("out=%v calls=%d, want empty result and no model call", docContents(out), calls)
	}
}

func TestLLMListwiseRerankErrors(t *testing.T) {
	wantErr := errors.New("model failed")
	model := textModelFunc(func(context.Context, string) (string, error) {
		return "", wantErr
	})
	reranker, err := retrievers.NewLLMListwiseRerank(model)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	if _, err := reranker.CompressDocuments(t.Context(), listwiseDocs(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}

	if _, err := retrievers.NewLLMListwiseRerank(nil); err == nil {
		t.Fatal("nil model must error")
	}
}

func TestLLMListwiseRerankCanceledContext(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("0"))
	reranker, err := retrievers.NewLLMListwiseRerank(model)
	if err != nil {
		t.Fatalf("new listwise reranker: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := reranker.CompressDocuments(ctx, listwiseDocs(), "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled compress error = %v, want context.Canceled", err)
	}
}
