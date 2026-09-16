package retrievers_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/language"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// funcRetriever adapts a function to the Retriever interface for tests.
type funcRetriever func(context.Context, string) ([]documents.Document, error)

func (f funcRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	return f(ctx, query)
}

// recordingModel captures the prompts a MultiQueryRetriever sends and replies
// through a delegate chat model.
type recordingModel struct {
	language.ChatModel
	mu     sync.Mutex
	inputs []string
}

func (m *recordingModel) Invoke(
	ctx context.Context,
	input []messages.Message,
	opts ...runnables.Option,
) (messages.Message, error) {
	m.mu.Lock()
	for _, msg := range input {
		m.inputs = append(m.inputs, msg.Content)
	}
	m.mu.Unlock()
	return m.ChatModel.Invoke(ctx, input, opts...)
}

// queryRecorder collects the queries a base retriever saw, safely across
// concurrent retrieval goroutines.
type queryRecorder struct {
	mu      sync.Mutex
	queries []string
}

func (q *queryRecorder) record(query string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.queries = append(q.queries, query)
}

func (q *queryRecorder) sorted() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := append([]string(nil), q.queries...)
	slices.Sort(out)
	return out
}

// Python parity: langchain_community.retrievers.MultiQueryRetriever generates
// query variants with a chat model (one per line, blank lines dropped),
// retrieves for each variant concurrently, and merges the hits deduplicated
// by page content.
func TestMultiQueryRetrieverModelVariantsMergeAndDedup(t *testing.T) {
	model := &recordingModel{
		ChatModel: language.NewFakeChatModel(
			language.WithResponses(
				messages.AI("q-variant-1\nq-variant-2\n\n   q-variant-3   \n"),
			),
		),
	}
	recorder := &queryRecorder{}
	base := funcRetriever(func(ctx context.Context, query string) ([]documents.Document, error) {
		recorder.record(query)
		switch query {
		case "q-variant-1":
			return []documents.Document{documents.New("doc-a", nil), documents.New("doc-b", nil)}, nil
		case "q-variant-2":
			return []documents.Document{documents.New("doc-b", nil), documents.New("doc-c", nil)}, nil
		case "q-variant-3":
			return []documents.Document{documents.New("doc-d", nil)}, nil
		}
		return nil, nil
	})

	retriever, err := retrievers.NewMultiQueryRetriever(base, retrievers.WithQueryModel(model))
	if err != nil {
		t.Fatalf("new multi-query retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "what is LangChain")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}

	var contents []string
	for _, doc := range docs {
		contents = append(contents, doc.PageContent)
	}
	want := []string{"doc-a", "doc-b", "doc-c", "doc-d"}
	if !slices.Equal(contents, want) {
		t.Fatalf("merged docs: got %v want %v", contents, want)
	}

	gotQueries := recorder.sorted()
	if !slices.Equal(gotQueries, []string{"q-variant-1", "q-variant-2", "q-variant-3"}) {
		t.Fatalf("base retriever queries: got %v", gotQueries)
	}

	model.mu.Lock()
	prompts := append([]string(nil), model.inputs...)
	model.mu.Unlock()
	if len(prompts) != 1 {
		t.Fatalf("model invocations: got %d want 1", len(prompts))
	}
	for _, want := range []string{"what is LangChain", "different versions", "separated by newlines"} {
		if !strings.Contains(prompts[0], want) {
			t.Fatalf("prompt %q must mention %q", prompts[0], want)
		}
	}
}

func TestMultiQueryRetrieverCustomPrompt(t *testing.T) {
	model := &recordingModel{
		ChatModel: language.NewFakeChatModel(
			language.WithResponses(messages.AI("only-variant")),
		),
	}
	base := funcRetriever(func(context.Context, string) ([]documents.Document, error) {
		return nil, nil
	})

	retriever, err := retrievers.NewMultiQueryRetriever(
		base,
		retrievers.WithQueryModel(model),
		retrievers.WithPrompt("Rewrite in French: {question}"),
	)
	if err != nil {
		t.Fatalf("new multi-query retriever: %v", err)
	}
	if _, err := retriever.GetRelevantDocuments(t.Context(), "what color is the sky"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}

	model.mu.Lock()
	defer model.mu.Unlock()
	if len(model.inputs) != 1 || model.inputs[0] != "Rewrite in French: what color is the sky" {
		t.Fatalf("model input: got %v", model.inputs)
	}
}

// Python parity: MultiQueryRetriever(include_original=True) appends the
// original query after the generated variants.
func TestMultiQueryRetrieverIncludeOriginalQuery(t *testing.T) {
	recorder := &queryRecorder{}
	base := funcRetriever(func(ctx context.Context, query string) ([]documents.Document, error) {
		recorder.record(query)
		return []documents.Document{documents.New("hit-"+query, nil)}, nil
	})

	retriever, err := retrievers.NewMultiQueryRetriever(
		base,
		retrievers.WithQueryVariants(func(context.Context, string) ([]string, error) {
			return []string{"paraphrase-a", "paraphrase-b"}, nil
		}),
		retrievers.WithIncludeOriginalQuery(true),
	)
	if err != nil {
		t.Fatalf("new multi-query retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "original question")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}

	gotQueries := recorder.sorted()
	wantQueries := []string{"original question", "paraphrase-a", "paraphrase-b"}
	if !slices.Equal(gotQueries, wantQueries) {
		t.Fatalf("queries: got %v want %v", gotQueries, wantQueries)
	}
	if len(docs) != 3 {
		t.Fatalf("docs: got %d want 3", len(docs))
	}
}

// Concurrent retrieval must not reorder the merge: results are merged in
// variant order even when later variants finish first.
func TestMultiQueryRetrieverMergedOrderIsStableUnderSkewedLatency(t *testing.T) {
	base := funcRetriever(func(ctx context.Context, query string) ([]documents.Document, error) {
		switch query {
		case "v1":
			time.Sleep(60 * time.Millisecond)
			return []documents.Document{documents.New("doc-1", nil)}, nil
		case "v2":
			time.Sleep(30 * time.Millisecond)
			return []documents.Document{documents.New("doc-2", nil)}, nil
		default:
			return []documents.Document{documents.New("doc-3", nil)}, nil
		}
	})

	retriever, err := retrievers.NewMultiQueryRetriever(
		base,
		retrievers.WithQueryVariants(func(context.Context, string) ([]string, error) {
			return []string{"v1", "v2", "v3"}, nil
		}),
	)
	if err != nil {
		t.Fatalf("new multi-query retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	var contents []string
	for _, doc := range docs {
		contents = append(contents, doc.PageContent)
	}
	if !slices.Equal(contents, []string{"doc-1", "doc-2", "doc-3"}) {
		t.Fatalf("merged docs must follow variant order, got %v", contents)
	}
}

func TestMultiQueryRetrieverWhitespaceOnlyVariantsRetrieveNothing(t *testing.T) {
	model := &recordingModel{
		ChatModel: language.NewFakeChatModel(
			language.WithResponses(messages.AI("\n   \n")),
		),
	}
	recorder := &queryRecorder{}
	base := funcRetriever(func(ctx context.Context, query string) ([]documents.Document, error) {
		recorder.record(query)
		return []documents.Document{documents.New("unexpected", nil)}, nil
	})

	retriever, err := retrievers.NewMultiQueryRetriever(base, retrievers.WithQueryModel(model))
	if err != nil {
		t.Fatalf("new multi-query retriever: %v", err)
	}
	docs, err := retriever.GetRelevantDocuments(t.Context(), "q")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("docs: got %d want 0", len(docs))
	}
	if len(recorder.sorted()) != 0 {
		t.Fatalf("base retriever must not be called, saw %v", recorder.sorted())
	}
}

func TestMultiQueryRetrieverErrors(t *testing.T) {
	t.Run("constructor requires base retriever", func(t *testing.T) {
		if _, err := retrievers.NewMultiQueryRetriever(nil); err == nil {
			t.Fatal("nil base retriever must error")
		}
	})

	t.Run("constructor requires model or variants", func(t *testing.T) {
		if _, err := retrievers.NewMultiQueryRetriever(funcRetriever(nil)); err == nil {
			t.Fatal("missing model and variants must error")
		}
	})

	t.Run("base retriever error propagates", func(t *testing.T) {
		baseErr := errors.New("retrieval boom")
		retriever, err := retrievers.NewMultiQueryRetriever(
			funcRetriever(func(context.Context, string) ([]documents.Document, error) {
				return nil, baseErr
			}),
			retrievers.WithQueryVariants(func(context.Context, string) ([]string, error) {
				return []string{"v1"}, nil
			}),
		)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, baseErr) {
			t.Fatalf("want base error, got %v", err)
		}
	})

	t.Run("model error propagates", func(t *testing.T) {
		modelErr := errors.New("model boom")
		retriever, err := retrievers.NewMultiQueryRetriever(
			funcRetriever(nil),
			retrievers.WithQueryModel(errModel{err: modelErr}),
		)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, modelErr) {
			t.Fatalf("want model error, got %v", err)
		}
	})

	t.Run("variants error propagates", func(t *testing.T) {
		variantsErr := errors.New("variants boom")
		retriever, err := retrievers.NewMultiQueryRetriever(
			funcRetriever(nil),
			retrievers.WithQueryVariants(func(context.Context, string) ([]string, error) {
				return nil, variantsErr
			}),
		)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		if _, err := retriever.GetRelevantDocuments(t.Context(), "q"); !errors.Is(err, variantsErr) {
			t.Fatalf("want variants error, got %v", err)
		}
	})

	t.Run("canceled context", func(t *testing.T) {
		retriever, err := retrievers.NewMultiQueryRetriever(
			funcRetriever(nil),
			retrievers.WithQueryVariants(func(context.Context, string) ([]string, error) {
				return []string{"v1"}, nil
			}),
		)
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

// errModel is a chat model that always fails.
type errModel struct {
	err error
}

func (m errModel) Invoke(
	context.Context,
	[]messages.Message,
	...runnables.Option,
) (messages.Message, error) {
	return messages.Message{}, m.err
}
