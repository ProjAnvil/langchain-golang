package exampleselectors

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// SortedValues returns the values of m ordered by their keys, mirroring
// langchain_core.example_selectors.semantic_similarity.sorted_values. Values
// are rendered with fmt.Sprint, so string, numeric, or any-valued maps all
// produce a stable list of strings.
func SortedValues[V any](values map[string]V) []string {
	keys := slices.Sorted(maps.Keys(values))
	out := make([]string, 0, len(keys))
	for _, key := range keys {
		out = append(out, fmt.Sprint(values[key]))
	}
	return out
}

// ExampleToText renders an example as a single space-joined string of its
// values sorted by key. When inputKeys is non-empty, only those keys are
// included (still ordered by key, matching sorted_values). It mirrors
// langchain_core's _example_to_text.
func ExampleToText(example Example, inputKeys []string) string {
	if len(inputKeys) > 0 {
		subset := make(Example, len(inputKeys))
		for _, key := range inputKeys {
			if value, ok := example[key]; ok {
				subset[key] = value
			}
		}
		return strings.Join(SortedValues(subset), " ")
	}
	return strings.Join(SortedValues(example), " ")
}

// mmrSelectorConfig holds the configuration for a
// MaxMarginalRelevanceExampleSelector.
type mmrSelectorConfig struct {
	k           int
	fetchK      int
	lambdaMult  float64
	inputKeys   []string
	exampleKeys []string
}

func defaultMMRConfig() mmrSelectorConfig {
	return mmrSelectorConfig{k: 4, fetchK: 20, lambdaMult: 0.5}
}

// MMROption configures a MaxMarginalRelevanceExampleSelector.
type MMROption func(*mmrSelectorConfig)

// WithK sets the number of examples to select (default 4).
func WithK(k int) MMROption {
	return func(c *mmrSelectorConfig) { c.k = k }
}

// WithFetchK sets the number of candidates fetched before MMR re-ranking
// (default 20).
func WithFetchK(fetchK int) MMROption {
	return func(c *mmrSelectorConfig) { c.fetchK = fetchK }
}

// WithLambdaMult sets the MMR trade-off: 1 favors query similarity, 0 favors
// diversity (default 0.5).
func WithLambdaMult(lambdaMult float64) MMROption {
	return func(c *mmrSelectorConfig) { c.lambdaMult = lambdaMult }
}

// WithInputKeys restricts the fields used to build the search text.
func WithInputKeys(keys ...string) MMROption {
	return func(c *mmrSelectorConfig) { c.inputKeys = append([]string(nil), keys...) }
}

// WithExampleKeys restricts the fields returned for each selected example.
func WithExampleKeys(keys ...string) MMROption {
	return func(c *mmrSelectorConfig) { c.exampleKeys = append([]string(nil), keys...) }
}

// MaxMarginalRelevanceExampleSelector selects examples using Max Marginal
// Relevance over an embedding-backed vector store, mirroring
// langchain_core.example_selectors.MaxMarginalRelevanceExampleSelector. It
// implements Selector. Each example is stored with its fields as document
// metadata, so selected documents can be turned back into Examples.
type MaxMarginalRelevanceExampleSelector struct {
	store       *vectorstores.InMemory
	k           int
	fetchK      int
	lambdaMult  float64
	inputKeys   []string
	exampleKeys []string
}

var _ Selector = (*MaxMarginalRelevanceExampleSelector)(nil)

// NewMaxMarginalRelevanceExampleSelector builds an MMR example selector over
// an existing in-memory vector store. Use FromExamples to build the store from
// a list of examples and an embedding model in one step.
func NewMaxMarginalRelevanceExampleSelector(
	store *vectorstores.InMemory,
	opts ...MMROption,
) (*MaxMarginalRelevanceExampleSelector, error) {
	if store == nil {
		return nil, fmt.Errorf("exampleselectors: vector store is required")
	}
	cfg := defaultMMRConfig()
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.k <= 0 {
		cfg.k = 4
	}
	if cfg.fetchK <= 0 {
		cfg.fetchK = 20
	}
	return &MaxMarginalRelevanceExampleSelector{
		store:       store,
		k:           cfg.k,
		fetchK:      cfg.fetchK,
		lambdaMult:  cfg.lambdaMult,
		inputKeys:   append([]string(nil), cfg.inputKeys...),
		exampleKeys: append([]string(nil), cfg.exampleKeys...),
	}, nil
}

// FromExamples builds a MaxMarginalRelevanceExampleSelector from a list of
// examples and an embedding model, mirroring the Python from_examples
// classmethod. Each example is embedded (as rendered by ExampleToText) and
// stored with its fields as document metadata.
func FromExamples(
	examples []Example,
	embedder embeddings.Embeddings,
	opts ...MMROption,
) (*MaxMarginalRelevanceExampleSelector, error) {
	if embedder == nil {
		return nil, fmt.Errorf("exampleselectors: embeddings are required")
	}
	cfg := defaultMMRConfig()
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	store := vectorstores.NewInMemory(embedder)
	if len(examples) > 0 {
		texts := make([]string, len(examples))
		metadatas := make([]map[string]any, len(examples))
		for i, example := range examples {
			texts[i] = ExampleToText(example, cfg.inputKeys)
			metadatas[i] = map[string]any(cloneExample(example))
		}
		if _, err := store.AddTexts(context.Background(), texts, metadatas, nil); err != nil {
			return nil, err
		}
	}
	return NewMaxMarginalRelevanceExampleSelector(store, opts...)
}

// AddExample embeds and stores a new example.
func (s *MaxMarginalRelevanceExampleSelector) AddExample(ctx context.Context, example Example) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.store == nil {
		return fmt.Errorf("exampleselectors: vector store is required")
	}
	_, err := s.store.AddTexts(ctx,
		[]string{ExampleToText(example, s.inputKeys)},
		[]map[string]any{map[string]any(cloneExample(example))},
		nil,
	)
	return err
}

// SelectExamples returns up to K examples selected for query similarity and
// diversity by Max Marginal Relevance.
func (s *MaxMarginalRelevanceExampleSelector) SelectExamples(
	ctx context.Context,
	inputVariables map[string]string,
) ([]Example, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.store == nil {
		return nil, fmt.Errorf("exampleselectors: vector store is required")
	}
	input := make(Example, len(inputVariables))
	for key, value := range inputVariables {
		input[key] = value
	}
	query := ExampleToText(input, s.inputKeys)
	docs, err := s.store.MaxMarginalRelevanceSearch(ctx, query, s.k, s.fetchK, s.lambdaMult, nil)
	if err != nil {
		return nil, err
	}
	return s.documentsToExamples(docs), nil
}

// documentsToExamples rebuilds examples from document metadata, optionally
// filtering to exampleKeys. It mirrors _documents_to_examples.
func (s *MaxMarginalRelevanceExampleSelector) documentsToExamples(docs []documents.Document) []Example {
	out := make([]Example, 0, len(docs))
	for _, doc := range docs {
		example := Example(doc.Metadata)
		if len(s.exampleKeys) > 0 {
			filtered := make(Example, len(s.exampleKeys))
			for _, key := range s.exampleKeys {
				if value, ok := example[key]; ok {
					filtered[key] = value
				}
			}
			example = filtered
		}
		out = append(out, cloneExample(example))
	}
	return out
}
