package retrievers_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// chatModelTextLLM adapts a chat model to retrievers.TextLLM, showing the
// consumer-side text interface is not tied to a single model kind.
type chatModelTextLLM struct {
	model language.ChatModel
}

func (c chatModelTextLLM) Invoke(
	ctx context.Context,
	prompt string,
	_ ...runnables.Option,
) (string, error) {
	msg, err := c.model.Invoke(ctx, []messages.Message{messages.Human(prompt)})
	if err != nil {
		return "", err
	}
	return msg.Content, nil
}

func TestRePhraseQueryRetrieverAcceptsAdaptedChatModel(t *testing.T) {
	chat := language.NewFakeChatModel(language.WithResponses(messages.AI("chat rewritten")))
	var received string
	base := stubRetriever(func(_ context.Context, query string) ([]documents.Document, error) {
		received = query
		return nil, nil
	})
	retriever, err := retrievers.NewRePhraseQueryRetriever(base, chatModelTextLLM{model: chat})
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if received != "chat rewritten" {
		t.Fatalf("base retriever received %q, want %q", received, "chat rewritten")
	}
}

func TestRePhraseQueryRetrieverRewritesOnceAndRetrieves(t *testing.T) {
	modelCalls := 0
	model := textModelFunc(func(_ context.Context, _ string) (string, error) {
		modelCalls++
		return "rewritten query", nil
	})
	var received string
	base := stubRetriever(func(_ context.Context, query string) ([]documents.Document, error) {
		received = query
		return []documents.Document{documents.New("result", nil)}, nil
	})

	retriever, err := retrievers.NewRePhraseQueryRetriever(base, model)
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "original question")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if modelCalls != 1 {
		t.Fatalf("model invoked %d times, want exactly 1", modelCalls)
	}
	if received != "rewritten query" {
		t.Fatalf("base retriever received %q, want the rephrased query", received)
	}
	if got := docContents(docs); len(got) != 1 || got[0] != "result" {
		t.Fatalf("docs = %v, want [result]", got)
	}
}

func TestRePhraseQueryRetrieverUsesFakeLLM(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("vector query"))
	var received string
	base := stubRetriever(func(_ context.Context, query string) ([]documents.Document, error) {
		received = query
		return nil, nil
	})
	retriever, err := retrievers.NewRePhraseQueryRetriever(base, model)
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	if _, err := retriever.GetRelevantDocuments(t.Context(), "who is steve"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if received != "vector query" {
		t.Fatalf("base retriever received %q, want %q", received, "vector query")
	}
}

func TestRePhraseQueryRetrieverCustomPrompt(t *testing.T) {
	var captured string
	model := textModelFunc(func(_ context.Context, prompt string) (string, error) {
		captured = prompt
		return "q", nil
	})
	base := stubRetriever(func(context.Context, string) ([]documents.Document, error) {
		return nil, nil
	})
	retriever, err := retrievers.NewRePhraseQueryRetriever(
		base,
		model,
		retrievers.WithRephrasePrompt("Rewrite this: {question}"),
	)
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	if _, err := retriever.GetRelevantDocuments(t.Context(), "the question"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if captured != "Rewrite this: the question" {
		t.Fatalf("prompt = %q, want the {question} placeholder substituted", captured)
	}
	if !strings.Contains(retrievers.DefaultRephraseTemplate, "{question}") {
		t.Fatal("default template must contain a {question} placeholder")
	}
}

func TestRePhraseQueryRetrieverConstructorErrors(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("q"))
	if _, err := retrievers.NewRePhraseQueryRetriever(nil, model); err == nil {
		t.Fatal("nil base retriever must error")
	}
	if _, err := retrievers.NewRePhraseQueryRetriever(staticWith("a"), nil); err == nil {
		t.Fatal("nil model must error")
	}
}

func TestRePhraseQueryRetrieverModelError(t *testing.T) {
	wantErr := errors.New("model failed")
	model := textModelFunc(func(context.Context, string) (string, error) {
		return "", wantErr
	})
	retriever, err := retrievers.NewRePhraseQueryRetriever(staticWith("a"), model)
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

func TestRePhraseQueryRetrieverBaseError(t *testing.T) {
	wantErr := errors.New("base failed")
	model := language.NewFakeLLM(language.WithLLMResponses("q"))
	base := stubRetriever(func(context.Context, string) ([]documents.Document, error) {
		return nil, wantErr
	})
	retriever, err := retrievers.NewRePhraseQueryRetriever(base, model)
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

func TestRePhraseQueryRetrieverCanceledContext(t *testing.T) {
	model := language.NewFakeLLM(language.WithLLMResponses("q"))
	retriever, err := retrievers.NewRePhraseQueryRetriever(staticWith("a"), model)
	if err != nil {
		t.Fatalf("new re-phrase retriever: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := retriever.GetRelevantDocuments(ctx, "q"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled retrieval error = %v, want context.Canceled", err)
	}
}
