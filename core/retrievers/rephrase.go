package retrievers

import (
	"context"
	"fmt"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/runnables"
)

// DefaultRephraseTemplate mirrors Python's DEFAULT_TEMPLATE
// (langchain_classic.retrievers.re_phraser): it asks the model to strip out
// information irrelevant to the retrieval task and contains a single
// {question} placeholder that is replaced with the incoming query.
const DefaultRephraseTemplate = `You are an assistant tasked with taking a natural language query from a user and converting it into a query for a vectorstore. In this process, you strip out information that is not relevant for the retrieval task. Here is the user query: {question}`

// TextLLM is the subset of language.LLM shared by the LLM-backed retrievers
// and document compressors in this package: a single text-in/text-out
// completion. It is declared consumer-side because core/language depends on
// core/tools, which depends on core/retrievers, so this package cannot import
// core/language without an import cycle. *language.FakeLLM (and every
// language.LLM) satisfies TextLLM structurally, which is what the tests use.
type TextLLM interface {
	Invoke(ctx context.Context, prompt string, opts ...runnables.Option) (string, error)
}

// RePhraseQueryRetriever uses an LLM to rewrite the user's query once into a
// retrieval-friendly query, then retrieves documents for the rewritten query.
//
// Python parity: langchain_classic.retrievers.re_phraser.RePhraseQueryRetriever
// (from_llm builds prompt | llm | StrOutputParser; _get_relevant_documents
// invokes the chain once and forwards the result to the base retriever).
type RePhraseQueryRetriever struct {
	retriever Retriever
	model     TextLLM
	prompt    string
}

// RePhraseOption configures a RePhraseQueryRetriever.
type RePhraseOption func(*RePhraseQueryRetriever)

// WithRephrasePrompt overrides the prompt sent to the model. The template must
// contain a {question} placeholder, which is replaced with the incoming query
// (Python: RePhraseQueryRetriever.from_llm(..., prompt=...)).
func WithRephrasePrompt(template string) RePhraseOption {
	return func(r *RePhraseQueryRetriever) {
		if template != "" {
			r.prompt = template
		}
	}
}

// NewRePhraseQueryRetriever creates a retriever that rewrites the query with
// model before delegating to base. Both base and model are required.
func NewRePhraseQueryRetriever(
	base Retriever,
	model TextLLM,
	opts ...RePhraseOption,
) (RePhraseQueryRetriever, error) {
	if base == nil {
		return RePhraseQueryRetriever{}, fmt.Errorf("base retriever is required")
	}
	if model == nil {
		return RePhraseQueryRetriever{}, fmt.Errorf("query rewriting model is required")
	}
	retriever := RePhraseQueryRetriever{
		retriever: base,
		model:     model,
		prompt:    DefaultRephraseTemplate,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&retriever)
		}
	}
	return retriever, nil
}

var _ Retriever = RePhraseQueryRetriever{}

// GetRelevantDocuments rewrites the query once via the configured model and
// retrieves documents for the rewritten query. The model response is trimmed
// of surrounding whitespace before it is used as the retrieval query.
func (r RePhraseQueryRetriever) GetRelevantDocuments(
	ctx context.Context,
	query string,
) ([]documents.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prompt := strings.ReplaceAll(r.prompt, "{question}", query)
	rephrased, err := r.model.Invoke(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("re-phrase query: %w", err)
	}
	rephrased = strings.TrimSpace(rephrased)
	return r.retriever.GetRelevantDocuments(ctx, rephrased)
}
