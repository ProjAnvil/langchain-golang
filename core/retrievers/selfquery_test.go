package retrievers_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/retrievers"
	"github.com/projanvil/langchain-golang/core/structuredquery"
)

// Compile-time checks for the consumer-side functional adapters.
var (
	_ retrievers.QueryConstructor          = retrievers.QueryConstructorFunc(nil)
	_ retrievers.StructuredQueryTranslator = retrievers.StructuredQueryTranslatorFunc(nil)
	_ retrievers.Retriever                 = (*sqUtilRetriever)(nil)
)

// sqUtilRetriever is a self-contained retriever stub that records the query it
// receives so tests can assert the rewritten search string without depending
// on helpers owned by other test files in this package.
type sqUtilRetriever struct {
	docs      []documents.Document
	err       error
	gotQuery  string
	callCount int
}

func (s *sqUtilRetriever) GetRelevantDocuments(_ context.Context, query string) ([]documents.Document, error) {
	s.gotQuery = query
	s.callCount++
	if s.err != nil {
		return nil, s.err
	}
	return s.docs, nil
}

// sqUtilConstructor returns a QueryConstructor that always yields the supplied
// structured query and counts its invocations.
func sqUtilConstructor(sq *structuredquery.StructuredQuery, calls *int) retrievers.QueryConstructor {
	return retrievers.QueryConstructorFunc(
		func(_ context.Context, _ string) (*structuredquery.StructuredQuery, error) {
			*calls++
			return sq, nil
		},
	)
}

// sqUtilDefaultTranslator builds the generic translator backed by
// structuredquery.DefaultVisitor.
func sqUtilDefaultTranslator() retrievers.StructuredQueryTranslator {
	return retrievers.NewStructuredQueryTranslator(nil)
}

// sqUtilContents extracts page contents for concise assertions.
func sqUtilContents(docs []documents.Document) []string {
	out := make([]string, len(docs))
	for i, doc := range docs {
		out[i] = doc.PageContent
	}
	return out
}

func sqUtilNewRetriever(
	t *testing.T,
	base *sqUtilRetriever,
	constructor retrievers.QueryConstructor,
	translator retrievers.StructuredQueryTranslator,
	opts ...retrievers.SelfQueryOption,
) retrievers.SelfQueryRetriever {
	t.Helper()
	r, err := retrievers.NewSelfQueryRetriever(base, constructor, translator, opts...)
	if err != nil {
		t.Fatalf("new self-query retriever: %v", err)
	}
	return r
}

func TestSelfQueryRetrieverValidStructuredQueryWithFilter(t *testing.T) {
	sq := structuredquery.NewStructuredQuery(
		"rewritten query",
		structuredquery.NewComparison(structuredquery.ComparatorEq, "group", "a"),
		0,
	)
	constructorCalls := 0
	constructor := sqUtilConstructor(sq, &constructorCalls)
	base := &sqUtilRetriever{docs: []documents.Document{
		documents.New("a1", map[string]any{"group": "a"}),
		documents.New("b1", map[string]any{"group": "b"}),
		documents.New("a2", map[string]any{"group": "a"}),
	}}
	r := sqUtilNewRetriever(t, base, constructor, sqUtilDefaultTranslator())

	docs, err := r.GetRelevantDocuments(t.Context(), "find group a docs")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if constructorCalls != 1 {
		t.Fatalf("constructor invoked %d times, want 1", constructorCalls)
	}
	if base.gotQuery != "rewritten query" {
		t.Fatalf("base retriever received %q, want the rewritten query", base.gotQuery)
	}
	got := sqUtilContents(docs)
	if len(got) != 2 || got[0] != "a1" || got[1] != "a2" {
		t.Fatalf("filtered docs = %v, want [a1 a2]", got)
	}
}

func TestSelfQueryRetrieverRejectsInvalidFilter(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	constructorCalls := 0
	constructor := sqUtilConstructor(sq, &constructorCalls)

	// A valid structured query but a translator that emits a filter with an
	// unknown operator must be rejected by vectorstores.ValidateFilter.
	translator := retrievers.StructuredQueryTranslatorFunc(
		func(_ context.Context, _ *structuredquery.StructuredQuery) (string, map[string]any, error) {
			return "q", map[string]any{"group": map[string]any{"$nope": 1}}, nil
		},
	)
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, constructor, translator)

	_, err := r.GetRelevantDocuments(t.Context(), "some query")
	if err == nil {
		t.Fatal("invalid filter must be rejected")
	}
	if !strings.Contains(err.Error(), "invalid filter") {
		t.Fatalf("error = %q, want an invalid filter error", err)
	}
	if base.callCount != 0 {
		t.Fatalf("base retriever was called %d times with an invalid filter", base.callCount)
	}
}

func TestSelfQueryRetrieverConstructorErrorPropagation(t *testing.T) {
	wantErr := errors.New("constructor failed")
	constructor := retrievers.QueryConstructorFunc(
		func(_ context.Context, _ string) (*structuredquery.StructuredQuery, error) {
			return nil, wantErr
		},
	)
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, constructor, sqUtilDefaultTranslator())

	_, err := r.GetRelevantDocuments(t.Context(), "some query")
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
	if base.callCount != 0 {
		t.Fatalf("base retriever was called %d times after constructor error", base.callCount)
	}
}

func TestSelfQueryRetrieverTranslatorErrorPropagation(t *testing.T) {
	wantErr := errors.New("translator failed")
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	constructorCalls := 0
	translator := retrievers.StructuredQueryTranslatorFunc(
		func(_ context.Context, _ *structuredquery.StructuredQuery) (string, map[string]any, error) {
			return "", nil, wantErr
		},
	)
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(sq, &constructorCalls), translator)

	_, err := r.GetRelevantDocuments(t.Context(), "some query")
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
	if base.callCount != 0 {
		t.Fatalf("base retriever was called %d times after translator error", base.callCount)
	}
}

func TestSelfQueryRetrieverBaseErrorPropagation(t *testing.T) {
	wantErr := errors.New("base failed")
	sq := structuredquery.NewStructuredQuery("rewritten", nil, 0)
	constructorCalls := 0
	base := &sqUtilRetriever{err: wantErr}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(sq, &constructorCalls), sqUtilDefaultTranslator())

	_, err := r.GetRelevantDocuments(t.Context(), "some query")
	if !errors.Is(err, wantErr) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

func TestSelfQueryRetrieverInvalidStructuredQuery(t *testing.T) {
	// A negative limit is rejected by structuredquery.ValidateStructuredQuery
	// before the translator is ever invoked.
	sq := structuredquery.NewStructuredQuery("q", nil, -1)
	constructorCalls := 0
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(sq, &constructorCalls), sqUtilDefaultTranslator())

	_, err := r.GetRelevantDocuments(t.Context(), "some query")
	if err == nil {
		t.Fatal("invalid structured query must be rejected")
	}
	if !strings.Contains(err.Error(), "invalid structured query") {
		t.Fatalf("error = %q, want an invalid structured query error", err)
	}
	if base.callCount != 0 {
		t.Fatalf("base retriever was called %d times for an invalid structured query", base.callCount)
	}
}

func TestSelfQueryRetrieverNilStructuredQuery(t *testing.T) {
	constructorCalls := 0
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(nil, &constructorCalls), sqUtilDefaultTranslator())

	if _, err := r.GetRelevantDocuments(t.Context(), "some query"); err == nil {
		t.Fatal("nil structured query must be rejected")
	}
}

func TestSelfQueryRetrieverEmptyQuery(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	constructorCalls := 0
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(sq, &constructorCalls), sqUtilDefaultTranslator())

	for _, query := range []string{"", "   "} {
		if _, err := r.GetRelevantDocuments(t.Context(), query); err == nil {
			t.Fatalf("empty query %q must be rejected", query)
		}
	}
	if constructorCalls != 0 || base.callCount != 0 {
		t.Fatalf("empty query must fail before constructor/base (%d, %d)", constructorCalls, base.callCount)
	}
}

func TestSelfQueryRetrieverCanceledContext(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	constructorCalls := 0
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(sq, &constructorCalls), sqUtilDefaultTranslator())

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := r.GetRelevantDocuments(ctx, "some query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled retrieval error = %v, want context.Canceled", err)
	}
	if constructorCalls != 0 || base.callCount != 0 {
		t.Fatalf("canceled retrieval must not invoke constructor/base (%d, %d)", constructorCalls, base.callCount)
	}
}

func TestSelfQueryRetrieverCancellationBetweenStages(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("rewritten", nil, 0)
	ctx, cancel := context.WithCancel(t.Context())
	constructorCalls := 0
	constructor := retrievers.QueryConstructorFunc(
		func(_ context.Context, _ string) (*structuredquery.StructuredQuery, error) {
			constructorCalls++
			cancel()
			return sq, nil
		},
	)
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(t, base, constructor, sqUtilDefaultTranslator())

	if _, err := r.GetRelevantDocuments(ctx, "some query"); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled after constructor", err)
	}
	if base.callCount != 0 {
		t.Fatalf("base retriever was called %d times after cancellation", base.callCount)
	}
}

func TestSelfQueryRetrieverUseOriginalQuery(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("rewritten query", nil, 0)
	constructorCalls := 0
	base := &sqUtilRetriever{}
	r := sqUtilNewRetriever(
		t, base, sqUtilConstructor(sq, &constructorCalls), sqUtilDefaultTranslator(),
		retrievers.WithSelfQueryUseOriginalQuery(true),
	)

	if _, err := r.GetRelevantDocuments(t.Context(), "original question"); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if base.gotQuery != "original question" {
		t.Fatalf("base retriever received %q, want the original query", base.gotQuery)
	}
}

func TestSelfQueryRetrieverEmptyFilterReturnsAllDocuments(t *testing.T) {
	sq := structuredquery.NewStructuredQuery("rewritten", nil, 0)
	constructorCalls := 0
	translator := retrievers.StructuredQueryTranslatorFunc(
		func(_ context.Context, _ *structuredquery.StructuredQuery) (string, map[string]any, error) {
			return "rewritten", map[string]any{}, nil
		},
	)
	base := &sqUtilRetriever{docs: []documents.Document{
		documents.New("one", nil),
		documents.New("two", nil),
	}}
	r := sqUtilNewRetriever(t, base, sqUtilConstructor(sq, &constructorCalls), translator)

	docs, err := r.GetRelevantDocuments(t.Context(), "some query")
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if got := sqUtilContents(docs); len(got) != 2 || got[0] != "one" || got[1] != "two" {
		t.Fatalf("docs = %v, want [one two]", got)
	}
}

func TestNewSelfQueryRetrieverRequiresDependencies(t *testing.T) {
	validConstructor := sqUtilConstructor(structuredquery.NewStructuredQuery("q", nil, 0), new(int))
	base := &sqUtilRetriever{}

	if _, err := retrievers.NewSelfQueryRetriever(nil, validConstructor, sqUtilDefaultTranslator()); err == nil {
		t.Fatal("nil base retriever must error")
	}
	if _, err := retrievers.NewSelfQueryRetriever(base, nil, sqUtilDefaultTranslator()); err == nil {
		t.Fatal("nil query constructor must error")
	}
	if _, err := retrievers.NewSelfQueryRetriever(base, validConstructor, nil); err == nil {
		t.Fatal("nil translator must error")
	}
}

func TestNewStructuredQueryTranslatorRejectsNonMapVisitorOutput(t *testing.T) {
	// A visitor whose structured-query result is not a map[string]any must be
	// reported as an error rather than silently producing a bad filter.
	sq := structuredquery.NewStructuredQuery("q", nil, 0)
	translator := retrievers.NewStructuredQueryTranslator(sqUtilNonMapVisitor{})
	if _, _, err := translator.Translate(t.Context(), sq); err == nil {
		t.Fatal("non-map visitor output must error")
	}
}

// sqUtilNonMapVisitor is a structuredquery.Visitor whose VisitStructuredQuery
// returns a plain string instead of a filter map.
type sqUtilNonMapVisitor struct{}

func (sqUtilNonMapVisitor) VisitOperation(*structuredquery.Operation) (any, error) {
	return "op", nil
}

func (sqUtilNonMapVisitor) VisitComparison(*structuredquery.Comparison) (any, error) {
	return "cmp", nil
}

func (sqUtilNonMapVisitor) VisitStructuredQuery(*structuredquery.StructuredQuery) (any, error) {
	return "not a map", nil
}
