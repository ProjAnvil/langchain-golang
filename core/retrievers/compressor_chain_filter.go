package retrievers

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/outputparser"
)

// defaultChainFilterPrompt mirrors Python's chain_filter_prompt: the model is
// asked to answer YES or NO to whether the context is relevant to the
// question. The {question} and {context} placeholders are substituted before
// the prompt is sent to the model.
const defaultChainFilterPrompt = `Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):`

// LLMChainFilter is a DocumentCompressor that uses an LLM chain to drop
// documents that are not relevant to the query. For every input document it
// builds a prompt from the query and the document content, invokes the model
// once, and parses the response with outputparser.BooleanOutputParser: YES
// keeps the document, NO drops it. An ambiguous or invalid response (neither
// YES nor NO) is an error.
//
// Python parity: langchain_classic.retrievers.document_compressors.chain_filter.
// LLMChainFilter (compress_documents uses a BooleanOutputParser and keeps
// documents whose parsed output is truthy).
type LLMChainFilter struct {
	model  TextLLM
	prompt string
	parser outputparser.BooleanOutputParser
}

// LLMChainFilterOption configures an LLMChainFilter.
type LLMChainFilterOption func(*LLMChainFilter)

// WithChainFilterPrompt overrides the filtering prompt. The template may
// contain {question} and {context} placeholders.
func WithChainFilterPrompt(template string) LLMChainFilterOption {
	return func(c *LLMChainFilter) {
		if template != "" {
			c.prompt = template
		}
	}
}

// NewLLMChainFilter creates a filter that keeps/drops documents with model.
// The model is required.
func NewLLMChainFilter(
	model TextLLM,
	opts ...LLMChainFilterOption,
) (LLMChainFilter, error) {
	if model == nil {
		return LLMChainFilter{}, fmt.Errorf("filtering model is required")
	}
	filter := LLMChainFilter{
		model:  model,
		prompt: defaultChainFilterPrompt,
		parser: outputparser.NewBooleanOutputParser(),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&filter)
		}
	}
	return filter, nil
}

var _ DocumentCompressor = LLMChainFilter{}

// CompressDocuments keeps only the documents the model judges relevant. The
// kept documents are cloned and otherwise unmodified; the relative order of
// retained documents is preserved.
func (c LLMChainFilter) CompressDocuments(
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
	out := make([]documents.Document, 0, len(docs))
	for _, doc := range docs {
		prompt := strings.NewReplacer(
			"{question}", query,
			"{context}", doc.PageContent,
		).Replace(c.prompt)
		raw, err := c.model.Invoke(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("filter document %q: %w", doc.ID, err)
		}
		keep, err := c.parser.Parse(ctx, raw)
		if err != nil {
			return nil, fmt.Errorf("parse filter output for document %q: %w", doc.ID, err)
		}
		if keep {
			out = append(out, doc.Clone())
		}
	}
	return out, nil
}
