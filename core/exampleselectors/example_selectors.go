package exampleselectors

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

// Example is a prompt example keyed by input variable name.
type Example map[string]any

// Selector chooses examples for a prompt input.
type Selector interface {
	AddExample(ctx context.Context, example Example) error
	SelectExamples(ctx context.Context, inputVariables map[string]string) ([]Example, error)
}

// FormatFunc formats an example before length or semantic indexing.
type FormatFunc func(Example) (string, error)

// LengthFunc measures formatted prompt length.
type LengthFunc func(string) int

var defaultLengthRE = regexp.MustCompile(`\n| `)

// DefaultLength matches Python's length selector default: split on newline or
// space and count the resulting parts.
func DefaultLength(text string) int {
	return len(defaultLengthRE.Split(text, -1))
}

// KeyValueFormatter formats examples deterministically as "key: value" lines.
func KeyValueFormatter(example Example) (string, error) {
	keys := make([]string, 0, len(example))
	for key := range example {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	lines := make([]string, 0, len(keys))
	for _, key := range keys {
		lines = append(lines, fmt.Sprintf("%s: %v", key, example[key]))
	}
	return strings.Join(lines, "\n"), nil
}

// LengthBasedSelector selects examples in insertion order until max length.
type LengthBasedSelector struct {
	Examples    []Example
	Formatter   FormatFunc
	Length      LengthFunc
	MaxLength   int
	textLengths []int
}

// NewLengthBased creates a length-based selector.
func NewLengthBased(examples []Example, formatter FormatFunc, maxLength int) (*LengthBasedSelector, error) {
	if formatter == nil {
		formatter = KeyValueFormatter
	}
	if maxLength <= 0 {
		maxLength = 2048
	}
	selector := &LengthBasedSelector{
		Formatter: formatter,
		Length:    DefaultLength,
		MaxLength: maxLength,
	}
	for _, example := range examples {
		if err := selector.AddExample(context.Background(), example); err != nil {
			return nil, err
		}
	}
	return selector, nil
}

// AddExample appends an example and records its formatted length.
func (s *LengthBasedSelector) AddExample(ctx context.Context, example Example) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	text, err := s.Formatter(example)
	if err != nil {
		return err
	}
	length := s.Length
	if length == nil {
		length = DefaultLength
	}
	s.Examples = append(s.Examples, cloneExample(example))
	s.textLengths = append(s.textLengths, length(text))
	return nil
}

// SelectExamples returns examples that fit the remaining length budget.
func (s *LengthBasedSelector) SelectExamples(ctx context.Context, inputVariables map[string]string) ([]Example, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	values := make([]string, 0, len(inputVariables))
	for _, value := range inputVariables {
		values = append(values, value)
	}
	length := s.Length
	if length == nil {
		length = DefaultLength
	}
	remaining := s.MaxLength - length(strings.Join(values, " "))
	selected := []Example{}
	for i, example := range s.Examples {
		if remaining <= 0 {
			break
		}
		next := remaining - s.textLengths[i]
		if next < 0 {
			break
		}
		selected = append(selected, cloneExample(example))
		remaining = next
	}
	return selected, nil
}

// SemanticSimilaritySelector selects examples via an existing vector store.
type SemanticSimilaritySelector struct {
	Store     vectorstores.VectorStore
	K         int
	Formatter FormatFunc
}

func (s SemanticSimilaritySelector) AddExample(ctx context.Context, example Example) error {
	if s.Store == nil {
		return fmt.Errorf("vector store is required")
	}
	formatter := s.Formatter
	if formatter == nil {
		formatter = KeyValueFormatter
	}
	text, err := formatter(example)
	if err != nil {
		return err
	}
	_, err = s.Store.AddDocuments(ctx, []documents.Document{{
		PageContent: text,
		Metadata:    map[string]any{"example": cloneExample(example)},
	}})
	return err
}

func (s SemanticSimilaritySelector) SelectExamples(ctx context.Context, inputVariables map[string]string) ([]Example, error) {
	if s.Store == nil {
		return nil, fmt.Errorf("vector store is required")
	}
	k := s.K
	if k <= 0 {
		k = 4
	}
	values := make([]string, 0, len(inputVariables))
	for _, value := range inputVariables {
		values = append(values, value)
	}
	docs, err := s.Store.SimilaritySearch(ctx, strings.Join(values, " "), k)
	if err != nil {
		return nil, err
	}
	examples := make([]Example, 0, len(docs))
	for _, doc := range docs {
		if example, ok := doc.Metadata["example"].(Example); ok {
			examples = append(examples, cloneExample(example))
			continue
		}
		if example, ok := doc.Metadata["example"].(map[string]any); ok {
			examples = append(examples, cloneExample(example))
		}
	}
	return examples, nil
}

func cloneExample(example Example) Example {
	if example == nil {
		return nil
	}
	out := make(Example, len(example))
	for key, value := range example {
		out[key] = value
	}
	return out
}
