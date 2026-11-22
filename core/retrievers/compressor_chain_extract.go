package retrievers

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/outputparser"
)

// chainExtractNoOutput mirrors the sentinel formatted into Python's
// chain_extract_prompt.prompt_template and parsed by
// langchain_classic.retrievers.document_compressors.chain_extract.NoOutputParser.
const chainExtractNoOutput = "NO_OUTPUT"

// defaultChainExtractPrompt mirrors Python's chain_extract_prompt. The
// {no_output_str}, {question}, and {context} placeholders are substituted
// before the prompt is sent to the model.
const defaultChainExtractPrompt = `Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return {no_output_str}.

Remember, *DO NOT* edit the extracted parts of the context.

> Question: {question}
> Context:
>>>
{context}
>>>
Extracted relevant parts:`

// LLMChainExtractor is a DocumentCompressor that uses an LLM chain to extract
// the relevant parts of each document. For every input document it builds a
// prompt from the query and the document content, invokes the model once, and
// replaces the document's page content with the extracted text. Documents for
// which the model returns the NO_OUTPUT sentinel (or an empty/blank response)
// are dropped.
//
// Python parity: langchain_classic.retrievers.document_compressors.chain_extract.
// LLMChainExtractor (compress_documents iterates documents; empty output is
// skipped). The Go form uses the shared outputparser.NoOutputParser and invokes
// the model sequentially so responses map deterministically to documents.
type LLMChainExtractor struct {
	model  TextLLM
	prompt string
	parser outputparser.NoOutputParser
}

// LLMChainExtractorOption configures an LLMChainExtractor.
type LLMChainExtractorOption func(*LLMChainExtractor)

// WithChainExtractorPrompt overrides the extraction prompt. The template may
// contain {no_output_str}, {question}, and {context} placeholders.
func WithChainExtractorPrompt(template string) LLMChainExtractorOption {
	return func(c *LLMChainExtractor) {
		if template != "" {
			c.prompt = template
		}
	}
}

// NewLLMChainExtractor creates an extractor that compresses documents with
// model. The model is required.
func NewLLMChainExtractor(
	model TextLLM,
	opts ...LLMChainExtractorOption,
) (LLMChainExtractor, error) {
	if model == nil {
		return LLMChainExtractor{}, fmt.Errorf("extraction model is required")
	}
	extractor := LLMChainExtractor{
		model:  model,
		prompt: defaultChainExtractPrompt,
		parser: outputparser.NoOutputParser{NoOutputStr: chainExtractNoOutput},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&extractor)
		}
	}
	return extractor, nil
}

var _ DocumentCompressor = LLMChainExtractor{}

// CompressDocuments extracts the relevant part of each document's content.
// The original metadata and id are preserved; only the page content is
// replaced (defensively cloning the document before mutation).
func (c LLMChainExtractor) CompressDocuments(
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
			"{no_output_str}", chainExtractNoOutput,
			"{question}", query,
			"{context}", doc.PageContent,
		).Replace(c.prompt)
		raw, err := c.model.Invoke(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("extract from document %q: %w", doc.ID, err)
		}
		extracted, err := c.parser.Parse(ctx, raw)
		if err != nil {
			return nil, fmt.Errorf("parse extraction for document %q: %w", doc.ID, err)
		}
		if extracted == "" {
			continue
		}
		compressed := doc.Clone()
		compressed.PageContent = extracted
		out = append(out, compressed)
	}
	return out, nil
}
