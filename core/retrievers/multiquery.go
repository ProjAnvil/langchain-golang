package retrievers

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/messages"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// DefaultMultiQueryPrompt is the prompt used to generate query variants. It is
// semantically aligned with Python's DEFAULT_QUERY_PROMPT
// (langchain_community.retrievers.multi_query): generate three differently
// phrased versions of the user question, one per line. The {question}
// placeholder is replaced with the incoming query.
const DefaultMultiQueryPrompt = `You are an AI language model assistant. Your task is to generate 3
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}`

// QueryModel is the subset of language.ChatModel that MultiQueryRetriever
// needs to generate query variants (a single-shot invoke). It is declared
// consumer-side because core/language already depends on this package through
// core/tools; any language.ChatModel — including language.NewFakeChatModel and
// every partner chat model — satisfies it structurally.
type QueryModel interface {
	Invoke(
		ctx context.Context,
		input []messages.Message,
		opts ...runnables.Option,
	) (messages.Message, error)
}

// QueryVariantsFunc generates alternative phrasings of a query without a chat
// model. Blank variants are ignored.
type QueryVariantsFunc func(ctx context.Context, query string) ([]string, error)

// MultiQueryRetriever expands a query into several variants, retrieves for
// each variant concurrently, and merges the hits: an LLM paraphrases the
// question from multiple perspectives, which recovers documents that
// distance-based similarity on the original phrasing alone would miss.
//
// Python parity: langchain_community.retrievers.MultiQueryRetriever
// (generate queries from the model's response one per line, retrieve for
// every variant, merge deduplicated by page content with the first
// occurrence winning; include_original appends the raw query to the
// variants). Retrieval of the variants is bounded-concurrent via
// runnables.ParallelMap, and results are merged in variant order so the
// outcome stays deterministic regardless of completion order.
type MultiQueryRetriever struct {
	base            Retriever
	model           QueryModel
	prompt          string
	variantsFn      QueryVariantsFunc
	includeOriginal bool
}

// MultiQueryOption configures a MultiQueryRetriever.
type MultiQueryOption func(*MultiQueryRetriever)

// WithQueryModel sets the chat model that generates query variants. The model
// receives the prompt (DefaultMultiQueryPrompt unless overridden) formatted
// with the user question as a single human message, and its response is parsed
// one variant per non-blank line.
func WithQueryModel(model QueryModel) MultiQueryOption {
	return func(r *MultiQueryRetriever) {
		r.model = model
	}
}

// WithQueryVariants sets an injected query-variant function used instead of a
// chat model.
func WithQueryVariants(fn QueryVariantsFunc) MultiQueryOption {
	return func(r *MultiQueryRetriever) {
		r.variantsFn = fn
	}
}

// WithPrompt overrides the prompt sent to the model. The template must
// contain a {question} placeholder, which is replaced with the incoming
// query.
func WithPrompt(template string) MultiQueryOption {
	return func(r *MultiQueryRetriever) {
		r.prompt = template
	}
}

// WithIncludeOriginalQuery appends the original query after the generated
// variants, mirroring MultiQueryRetriever(include_original=True).
func WithIncludeOriginalQuery(include bool) MultiQueryOption {
	return func(r *MultiQueryRetriever) {
		r.includeOriginal = include
	}
}

// NewMultiQueryRetriever creates a multi-query retriever over base. Either a
// query model (WithQueryModel) or an injected variants function
// (WithQueryVariants) must be provided; the model takes precedence when both
// are set.
func NewMultiQueryRetriever(base Retriever, opts ...MultiQueryOption) (MultiQueryRetriever, error) {
	if base == nil {
		return MultiQueryRetriever{}, fmt.Errorf("base retriever is required")
	}
	retriever := MultiQueryRetriever{
		base:   base,
		prompt: DefaultMultiQueryPrompt,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&retriever)
		}
	}
	if retriever.model == nil && retriever.variantsFn == nil {
		return MultiQueryRetriever{}, fmt.Errorf(
			"a query model (WithQueryModel) or variants function (WithQueryVariants) is required",
		)
	}
	return retriever, nil
}

var _ Retriever = MultiQueryRetriever{}

// GetRelevantDocuments generates query variants, retrieves for each variant
// concurrently, and returns the merged, deduplicated hits ordered by first
// occurrence across the variants.
func (r MultiQueryRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	variants, err := r.generateQueries(ctx, query)
	if err != nil {
		return nil, err
	}
	// Python parity: include_original appends the raw query after the
	// generated variants (multi_query.py `_get_relevant_documents`).
	if r.includeOriginal {
		variants = append(variants, query)
	}
	if len(variants) == 0 {
		return []documents.Document{}, nil
	}

	results, err := runnables.ParallelMap(
		ctx,
		runnables.NewConfig(),
		variants,
		func(ctx context.Context, variant string) ([]documents.Document, error) {
			return r.base.GetRelevantDocuments(ctx, variant)
		},
	)
	if err != nil {
		return nil, err
	}

	// Merge in variant order deduplicating on page content: ParallelMap
	// preserves input order, so the first occurrence wins deterministically,
	// matching Python's unique_docs dict keyed by page content.
	seen := make(map[string]struct{})
	merged := make([]documents.Document, 0)
	for _, docs := range results {
		for _, doc := range docs {
			if _, duplicate := seen[doc.PageContent]; duplicate {
				continue
			}
			seen[doc.PageContent] = struct{}{}
			merged = append(merged, doc.Clone())
		}
	}
	return merged, nil
}

// generateQueries produces the query variants: from the chat model when one
// is configured, otherwise from the injected variants function.
func (r MultiQueryRetriever) generateQueries(
	ctx context.Context,
	query string,
) ([]string, error) {
	if r.model == nil {
		return r.variantsFn(ctx, query)
	}
	prompt := strings.ReplaceAll(r.prompt, "{question}", query)
	response, err := r.model.Invoke(ctx, []messages.Message{messages.Human(prompt)})
	if err != nil {
		return nil, fmt.Errorf("generate query variants: %w", err)
	}
	return parseQueryVariants(response.Content), nil
}

// parseQueryVariants extracts one variant per non-blank line, trimmed of
// surrounding whitespace (multi_query.py `generate_queries` line parsing).
func parseQueryVariants(text string) []string {
	var variants []string
	for line := range strings.SplitSeq(text, "\n") {
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			variants = append(variants, trimmed)
		}
	}
	return variants
}
