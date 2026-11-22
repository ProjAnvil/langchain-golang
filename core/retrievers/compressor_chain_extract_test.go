package retrievers_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

func TestLLMChainExtractorExtractsAndDropsNoOutput(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses(
		"part one",
		"NO_OUTPUT",
		"  part three  ",
	))
	extractor, err := retrievers.NewLLMChainExtractor(model)
	if err != nil {
		t.Fatalf("new chain extractor: %v", err)
	}
	docs := []documents.Document{
		documents.New("doc zero", map[string]any{"src": "a"}).WithID("id0"),
		documents.New("doc one", nil).WithID("id1"),
		documents.New("doc two", nil).WithID("id2"),
	}
	out, err := extractor.CompressDocuments(t.Context(), docs, "the question")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 2 || got[0] != "part one" || got[1] != "part three" {
		t.Fatalf("compressed = %v, want [part one part three]", got)
	}
	// Metadata and id of kept documents are preserved.
	if out[0].ID != "id0" || out[0].Metadata["src"] != "a" {
		t.Fatalf("first doc metadata/id not preserved: %+v", out[0])
	}
	if out[1].ID != "id2" {
		t.Fatalf("second doc id not preserved: %+v", out[1])
	}
	// Inputs are not mutated.
	if docs[0].PageContent != "doc zero" {
		t.Fatalf("input document was mutated: %q", docs[0].PageContent)
	}
}

func TestLLMChainExtractorAllNoOutputDropsEverything(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("NO_OUTPUT"))
	extractor, err := retrievers.NewLLMChainExtractor(model)
	if err != nil {
		t.Fatalf("new chain extractor: %v", err)
	}
	out, err := extractor.CompressDocuments(t.Context(), []documents.Document{
		documents.New("a", nil),
		documents.New("b", nil),
	}, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("docs = %v, want empty", docContents(out))
	}
}

func TestLLMChainExtractorCustomPrompt(t *testing.T) {
	var captured string
	model := textModelFunc(func(_ context.Context, prompt string) (string, error) {
		captured = prompt
		return "kept", nil
	})
	extractor, err := retrievers.NewLLMChainExtractor(
		model,
		retrievers.WithChainExtractorPrompt("Q={question} C={context} NIL={no_output_str}"),
	)
	if err != nil {
		t.Fatalf("new chain extractor: %v", err)
	}
	if _, err := extractor.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("the context", nil)},
		"the question",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if captured != "Q=the question C=the context NIL=NO_OUTPUT" {
		t.Fatalf("prompt = %q, want all placeholders substituted", captured)
	}
}

func TestLLMChainExtractorEmptyInputSkipsModel(t *testing.T) {
	calls := 0
	model := textModelFunc(func(context.Context, string) (string, error) {
		calls++
		return "x", nil
	})
	extractor, err := retrievers.NewLLMChainExtractor(model)
	if err != nil {
		t.Fatalf("new chain extractor: %v", err)
	}
	out, err := extractor.CompressDocuments(t.Context(), nil, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 0 || calls != 0 {
		t.Fatalf("out=%v calls=%d, want empty result and no model call", docContents(out), calls)
	}
}

func TestLLMChainExtractorErrors(t *testing.T) {
	wantErr := errors.New("model failed")
	model := textModelFunc(func(context.Context, string) (string, error) {
		return "", wantErr
	})
	extractor, err := retrievers.NewLLMChainExtractor(model)
	if err != nil {
		t.Fatalf("new chain extractor: %v", err)
	}
	if _, err := extractor.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("a", nil)},
		"q",
	); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}

	if _, err := retrievers.NewLLMChainExtractor(nil); err == nil {
		t.Fatal("nil model must error")
	}
}

func TestLLMChainExtractorCanceledContext(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("x"))
	extractor, err := retrievers.NewLLMChainExtractor(model)
	if err != nil {
		t.Fatalf("new chain extractor: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := extractor.CompressDocuments(
		ctx,
		[]documents.Document{documents.New("a", nil)},
		"q",
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled compress error = %v, want context.Canceled", err)
	}
}
