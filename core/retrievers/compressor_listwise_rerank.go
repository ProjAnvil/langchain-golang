package retrievers

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
)

// defaultListwiseRerankTemplate mirrors Python's listwise_rerank default
// system prompt: the documents are listed with integer ids, the model is asked
// to order them by relevance to the query, and it responds with the ordered
// list of integer ids. The {query} and {documents} placeholders are
// substituted before the single model call.
const defaultListwiseRerankTemplate = `Sort the Documents by their relevance to the Query. Return the document IDs ordered from most to least relevant as a comma-separated list of integers.

Query: {query}

{documents}`

// listwiseIDPattern extracts integer document ids from the model response.
var listwiseIDPattern = regexp.MustCompile(`\d+`)

// LLMListwiseRerank is a DocumentCompressor that uses a single LLM call to
// rerank a list of documents ("Zero-Shot Listwise Document Reranking"). It
// builds one prompt containing every document labeled by an integer id, asks
// the model to return the ids ordered from most to least relevant, and returns
// the corresponding documents (deduplicated, invalid ids skipped) truncated to
// top_n.
//
// Python parity: langchain_classic.retrievers.document_compressors.
// listwise_rerank.LLMListwiseRerank (one structured-output call producing
// ranked_document_ids; compress_documents returns results[:top_n]). The Go
// form parses the ordered ids from plain text so it works with any TextLLM.
type LLMListwiseRerank struct {
	model  TextLLM
	topN   int
	prompt string
}

// LLMListwiseRerankOption configures an LLMListwiseRerank.
type LLMListwiseRerankOption func(*LLMListwiseRerank)

// WithListwiseRerankTopN sets the number of documents to return (default 3).
func WithListwiseRerankTopN(n int) LLMListwiseRerankOption {
	return func(r *LLMListwiseRerank) {
		r.topN = n
	}
}

// WithListwiseRerankPrompt overrides the reranking prompt. The template may
// contain {query} and {documents} placeholders.
func WithListwiseRerankPrompt(template string) LLMListwiseRerankOption {
	return func(r *LLMListwiseRerank) {
		if template != "" {
			r.prompt = template
		}
	}
}

// NewLLMListwiseRerank creates a listwise reranker using model. The model is
// required.
func NewLLMListwiseRerank(
	model TextLLM,
	opts ...LLMListwiseRerankOption,
) (LLMListwiseRerank, error) {
	if model == nil {
		return LLMListwiseRerank{}, fmt.Errorf("reranking model is required")
	}
	reranker := LLMListwiseRerank{
		model:  model,
		topN:   3,
		prompt: defaultListwiseRerankTemplate,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&reranker)
		}
	}
	return reranker, nil
}

var _ DocumentCompressor = LLMListwiseRerank{}

// CompressDocuments reranks documents with one model call and returns the
// top_n. The model response is parsed for integer document ids in order.
func (r LLMListwiseRerank) CompressDocuments(
	ctx context.Context,
	docs []documents.Document,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(docs) == 0 {
		return []documents.Document{}, nil
	}

	prompt := strings.NewReplacer(
		"{query}", query,
		"{documents}", formatListwiseDocuments(docs),
	).Replace(r.prompt)
	raw, err := r.model.Invoke(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("listwise rerank: %w", err)
	}
	ids := parseListwiseRanking(raw, len(docs))

	limit := r.topN
	if limit < 0 {
		limit = 0
	}
	out := make([]documents.Document, 0, limit)
	for _, id := range ids {
		if id < 0 || id >= len(docs) {
			continue
		}
		if len(out) >= limit {
			break
		}
		out = append(out, docs[id].Clone())
	}
	return out, nil
}

// formatListwiseDocuments renders each document with its integer id, mirroring
// Python's _get_prompt_input context block.
func formatListwiseDocuments(docs []documents.Document) string {
	var sb strings.Builder
	for i, doc := range docs {
		fmt.Fprintf(&sb, "Document ID: %d\n```%s```\n\n", i, doc.PageContent)
	}
	fmt.Fprintf(&sb, "Documents = [Document ID: 0, ..., Document ID: %d]", len(docs)-1)
	return sb.String()
}

// parseListwiseRanking extracts integer ids from the model response in order,
// dropping out-of-range and duplicate ids so each document appears at most
// once.
func parseListwiseRanking(text string, docCount int) []int {
	seen := make(map[int]struct{})
	ids := make([]int, 0)
	for _, match := range listwiseIDPattern.FindAllString(text, -1) {
		id, err := strconv.Atoi(match)
		if err != nil || id < 0 || id >= docCount {
			continue
		}
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	return ids
}
