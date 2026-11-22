package retrievers

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/structuredquery"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// QueryConstructor parses a natural-language query into a structured query.
// It is the consumer-side counterpart of Python's query_constructor
// Runnable[dict, StructuredQuery] used by SelfQueryRetriever.from_llm.
//
// It is declared here, rather than importing core/language, because
// core/language -> core/tools -> core/retrievers would form an import cycle.
// Like TextLLM in rephrase.go it is a small structural interface that any
// injected LLM chain or a QueryConstructorFunc can satisfy.
type QueryConstructor interface {
	Construct(ctx context.Context, query string) (*structuredquery.StructuredQuery, error)
}

// QueryConstructorFunc adapts a plain function to QueryConstructor.
type QueryConstructorFunc func(ctx context.Context, query string) (*structuredquery.StructuredQuery, error)

// Construct implements QueryConstructor.
func (f QueryConstructorFunc) Construct(ctx context.Context, query string) (*structuredquery.StructuredQuery, error) {
	return f(ctx, query)
}

// StructuredQueryTranslator turns a parsed structured query into the
// declarative metadata filter consumed by core/vectorstores.ValidateFilter
// and the rewritten free-text search string forwarded to the base retriever.
// It is the Go counterpart of Python's structured_query_translator Visitor
// (visit_structured_query -> (new_query, kwargs)).
//
// LangChain's vendor-specific builtin translators (ChromaTranslator,
// PineconeTranslator, PGVectorTranslator, and the dozen others wired up by
// _get_builtin_translator) are intentionally NOT ported: they depend on
// third-party vector store packages that do not exist in this module. Callers
// inject a translator instead; NewStructuredQueryTranslator provides a generic
// one built on structuredquery.DefaultVisitor / structuredquery.ToFilter.
type StructuredQueryTranslator interface {
	Translate(ctx context.Context, query *structuredquery.StructuredQuery) (searchQuery string, filter map[string]any, err error)
}

// StructuredQueryTranslatorFunc adapts a plain function to
// StructuredQueryTranslator.
type StructuredQueryTranslatorFunc func(ctx context.Context, query *structuredquery.StructuredQuery) (string, map[string]any, error)

// Translate implements StructuredQueryTranslator.
func (f StructuredQueryTranslatorFunc) Translate(
	ctx context.Context,
	query *structuredquery.StructuredQuery,
) (string, map[string]any, error) {
	return f(ctx, query)
}

// NewStructuredQueryTranslator returns a translator that builds the filter
// with structuredquery.ToFilter (using structuredquery.DefaultVisitor when
// visitor is nil) and uses the StructuredQuery.Query field as the rewritten
// search string. Because ToFilter returns any, a non-map visitor result is
// reported as an error. This is the generic, store-agnostic replacement for
// Python's vendor-specific builtin translators.
func NewStructuredQueryTranslator(visitor structuredquery.Visitor) StructuredQueryTranslator {
	return StructuredQueryTranslatorFunc(
		func(ctx context.Context, query *structuredquery.StructuredQuery) (string, map[string]any, error) {
			if err := ctx.Err(); err != nil {
				return "", nil, err
			}
			if query == nil {
				return "", nil, fmt.Errorf("self-query: structured query is nil")
			}
			v := visitor
			if v == nil {
				v = &structuredquery.DefaultVisitor{}
			}
			raw, err := structuredquery.ToFilter(query, v)
			if err != nil {
				return "", nil, err
			}
			filter, ok := raw.(map[string]any)
			if !ok {
				return "", nil, fmt.Errorf(
					"self-query: translator produced %T, want map[string]any", raw,
				)
			}
			return query.Query, filter, nil
		},
	)
}

// SelfQueryOption configures a SelfQueryRetriever.
type SelfQueryOption func(*SelfQueryRetriever)

// WithSelfQueryUseOriginalQuery controls whether the original user query is
// used for retrieval instead of the rewritten search string produced by the
// translator (Python parity: SelfQueryRetriever.use_original_query). The
// default is false, i.e. use the rewritten query.
func WithSelfQueryUseOriginalQuery(use bool) SelfQueryOption {
	return func(r *SelfQueryRetriever) {
		r.useOriginalQuery = use
	}
}

// SelfQueryRetriever parses a natural-language query into a structured query,
// translates that structured query into a declarative metadata filter and a
// rewritten search string, then retrieves documents from a base retriever.
//
// Python parity: langchain_classic.retrievers.self_query.base.SelfQueryRetriever
// (_get_relevant_documents invokes query_constructor, calls
// structured_query_translator.visit_structured_query to obtain new_query and
// search_kwargs, and searches the vector store). Python's vendor-specific
// builtin translators selected by _get_builtin_translator are intentionally
// not ported; the translator is injected (see NewStructuredQueryTranslator).
//
// The base Retriever interface only accepts a query string, so the validated
// filter is applied store-agnostically with vectorstores.MatchFilter to the
// documents returned by the base retriever. This keeps the retriever usable
// with any Retriever (including VectorStoreRetriever) without depending on a
// concrete vector store implementation.
type SelfQueryRetriever struct {
	base             Retriever
	constructor      QueryConstructor
	translator       StructuredQueryTranslator
	useOriginalQuery bool
}

// NewSelfQueryRetriever creates a SelfQueryRetriever. base, constructor and
// translator are all required; a nil argument is reported as an error.
func NewSelfQueryRetriever(
	base Retriever,
	constructor QueryConstructor,
	translator StructuredQueryTranslator,
	opts ...SelfQueryOption,
) (SelfQueryRetriever, error) {
	if base == nil {
		return SelfQueryRetriever{}, fmt.Errorf("self-query: base retriever is required")
	}
	if constructor == nil {
		return SelfQueryRetriever{}, fmt.Errorf("self-query: query constructor is required")
	}
	if translator == nil {
		return SelfQueryRetriever{}, fmt.Errorf("self-query: structured query translator is required")
	}
	r := SelfQueryRetriever{
		base:        base,
		constructor: constructor,
		translator:  translator,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&r)
		}
	}
	return r, nil
}

var _ Retriever = SelfQueryRetriever{}

// GetRelevantDocuments parses query into a structured query, translates it
// into a filter and a rewritten search string, validates both the structured
// query and the filter, and returns the base retriever's documents filtered
// by the declarative filter. Errors from the constructor, translator, filter
// validation and base retriever are propagated, and ctx.Err() is checked
// between each stage.
func (r SelfQueryRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("self-query: query must not be empty")
	}

	structured, err := r.constructor.Construct(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("self-query: construct structured query: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if structured == nil {
		return nil, fmt.Errorf("self-query: query constructor returned a nil structured query")
	}
	if err := structuredquery.ValidateStructuredQuery(structured); err != nil {
		return nil, fmt.Errorf("self-query: invalid structured query: %w", err)
	}

	rewritten, filter, err := r.translator.Translate(ctx, structured)
	if err != nil {
		return nil, fmt.Errorf("self-query: translate structured query: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := vectorstores.ValidateFilter(filter); err != nil {
		return nil, fmt.Errorf("self-query: invalid filter: %w", err)
	}

	searchQuery := strings.TrimSpace(rewritten)
	if r.useOriginalQuery || searchQuery == "" {
		searchQuery = query
	}

	docs, err := r.base.GetRelevantDocuments(ctx, searchQuery)
	if err != nil {
		return nil, fmt.Errorf("self-query: base retriever: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(filter) == 0 {
		return docs, nil
	}
	out := make([]documents.Document, 0, len(docs))
	for _, doc := range docs {
		if vectorstores.MatchFilter(doc, filter) {
			out = append(out, doc.Clone())
		}
	}
	return out, nil
}
