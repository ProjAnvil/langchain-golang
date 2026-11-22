package retrievers_test

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/retrievers"
)

func TestLLMChainFilterKeepsYesDropsNo(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("YES", "NO", "yes, keep"))
	filter, err := retrievers.NewLLMChainFilter(model)
	if err != nil {
		t.Fatalf("new chain filter: %v", err)
	}
	docs := []documents.Document{
		documents.New("keep zero", map[string]any{"src": "a"}).WithID("id0"),
		documents.New("drop one", nil).WithID("id1"),
		documents.New("keep two", nil).WithID("id2"),
	}
	out, err := filter.CompressDocuments(t.Context(), docs, "the question")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if got := docContents(out); len(got) != 2 || got[0] != "keep zero" || got[1] != "keep two" {
		t.Fatalf("filtered = %v, want [keep zero keep two]", got)
	}
	if out[0].ID != "id0" || out[0].Metadata["src"] != "a" {
		t.Fatalf("kept doc metadata/id not preserved: %+v", out[0])
	}
}

func TestLLMChainFilterAllNoDropsEverything(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("NO"))
	filter, err := retrievers.NewLLMChainFilter(model)
	if err != nil {
		t.Fatalf("new chain filter: %v", err)
	}
	out, err := filter.CompressDocuments(t.Context(), []documents.Document{
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

func TestLLMChainFilterInvalidAndAmbiguousOutputErrors(t *testing.T) {
	cases := []struct {
		name     string
		response string
	}{
		{name: "neither token", response: "maybe"},
		{name: "ambiguous both tokens", response: "YES and NO"},
		{name: "empty output", response: ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := language.NewFakeLLM(language.WithLLMResponses(tc.response))
			filter, err := retrievers.NewLLMChainFilter(model)
			if err != nil {
				t.Fatalf("new chain filter: %v", err)
			}
			if _, err := filter.CompressDocuments(
				t.Context(),
				[]documents.Document{documents.New("a", nil)},
				"q",
			); err == nil {
				t.Fatalf("response %q should produce a parse error", tc.response)
			}
		})
	}
}

func TestLLMChainFilterCustomPrompt(t *testing.T) {
	var captured string
	model := textModelFunc(func(_ context.Context, prompt string) (string, error) {
		captured = prompt
		return "YES", nil
	})
	filter, err := retrievers.NewLLMChainFilter(
		model,
		retrievers.WithChainFilterPrompt("Q={question} C={context}"),
	)
	if err != nil {
		t.Fatalf("new chain filter: %v", err)
	}
	if _, err := filter.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("the context", nil)},
		"the question",
	); err != nil {
		t.Fatalf("compress: %v", err)
	}
	if captured != "Q=the question C=the context" {
		t.Fatalf("prompt = %q, want placeholders substituted", captured)
	}
}

func TestLLMChainFilterEmptyInputSkipsModel(t *testing.T) {
	calls := 0
	model := textModelFunc(func(context.Context, string) (string, error) {
		calls++
		return "YES", nil
	})
	filter, err := retrievers.NewLLMChainFilter(model)
	if err != nil {
		t.Fatalf("new chain filter: %v", err)
	}
	out, err := filter.CompressDocuments(t.Context(), nil, "q")
	if err != nil {
		t.Fatalf("compress: %v", err)
	}
	if len(out) != 0 || calls != 0 {
		t.Fatalf("out=%v calls=%d, want empty result and no model call", docContents(out), calls)
	}
}

func TestLLMChainFilterErrors(t *testing.T) {
	wantErr := errors.New("model failed")
	model := textModelFunc(func(context.Context, string) (string, error) {
		return "", wantErr
	})
	filter, err := retrievers.NewLLMChainFilter(model)
	if err != nil {
		t.Fatalf("new chain filter: %v", err)
	}
	if _, err := filter.CompressDocuments(
		t.Context(),
		[]documents.Document{documents.New("a", nil)},
		"q",
	); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}

	if _, err := retrievers.NewLLMChainFilter(nil); err == nil {
		t.Fatal("nil model must error")
	}
}

func TestLLMChainFilterCanceledContext(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("YES"))
	filter, err := retrievers.NewLLMChainFilter(model)
	if err != nil {
		t.Fatalf("new chain filter: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := filter.CompressDocuments(
		ctx,
		[]documents.Document{documents.New("a", nil)},
		"q",
	); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled compress error = %v, want context.Canceled", err)
	}
}
