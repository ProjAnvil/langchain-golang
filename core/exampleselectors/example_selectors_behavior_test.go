package exampleselectors

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/projanvil/langchain-golang/core/documents"
	"github.com/projanvil/langchain-golang/core/vectorstores"
)

func TestDefaultLength(t *testing.T) {
	tests := []struct {
		name string
		text string
		want int
	}{
		{"empty", "", 1},
		{"single word", "hello", 1},
		{"words separated by spaces", "happy sad", 2},
		{"lines separated by newline", "a\nb", 2},
		{"mixed separators", "a b\nc", 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DefaultLength(tt.text); got != tt.want {
				t.Fatalf("DefaultLength(%q) = %d, want %d", tt.text, got, tt.want)
			}
		})
	}
}

func TestKeyValueFormatter(t *testing.T) {
	tests := []struct {
		name    string
		example Example
		want    string
	}{
		{"empty example", Example{}, ""},
		{"single key", Example{"input": "happy"}, "input: happy"},
		{
			"keys sorted alphabetically",
			Example{"output": "sad", "input": "happy"},
			"input: happy\noutput: sad",
		},
		{"non-string values", Example{"count": 3}, "count: 3"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := KeyValueFormatter(tt.example)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("KeyValueFormatter = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNewLengthBasedDefaults(t *testing.T) {
	selector, err := NewLengthBased(nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if selector.Formatter == nil {
		t.Fatal("expected default formatter to be set")
	}
	if selector.MaxLength != 2048 {
		t.Fatalf("MaxLength = %d, want default 2048", selector.MaxLength)
	}
	if len(selector.Examples) != 0 {
		t.Fatalf("expected no examples, got %d", len(selector.Examples))
	}

	// The default formatter must be usable through the selector.
	if err := selector.AddExample(context.Background(), Example{"b": 2, "a": 1}); err != nil {
		t.Fatal(err)
	}
	got, err := selector.SelectExamples(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("selected %d examples, want 1", len(got))
	}
}

func TestNewLengthBasedFormatterError(t *testing.T) {
	wantErr := errors.New("format failed")
	_, err := NewLengthBased([]Example{{"input": "x"}}, func(Example) (string, error) {
		return "", wantErr
	}, 10)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
}

func TestLengthBasedSelectorAddExample(t *testing.T) {
	selector, err := NewLengthBased(nil, KeyValueFormatter, 100)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if err := selector.AddExample(ctx, Example{"a": 1}); err == nil {
			t.Fatal("expected error for cancelled context")
		}
	})

	t.Run("formatter error propagates", func(t *testing.T) {
		wantErr := errors.New("boom")
		selector.Formatter = func(Example) (string, error) { return "", wantErr }
		if err := selector.AddExample(context.Background(), Example{"a": 1}); !errors.Is(err, wantErr) {
			t.Fatalf("err = %v, want %v", err, wantErr)
		}
		selector.Formatter = KeyValueFormatter
	})

	t.Run("stored example is cloned", func(t *testing.T) {
		example := Example{"input": "happy"}
		if err := selector.AddExample(context.Background(), example); err != nil {
			t.Fatal(err)
		}
		example["input"] = "mutated"
		if selector.Examples[len(selector.Examples)-1]["input"] != "happy" {
			t.Fatal("selector kept reference to caller's example map")
		}
	})

	t.Run("nil example is stored as nil", func(t *testing.T) {
		if err := selector.AddExample(context.Background(), nil); err != nil {
			t.Fatal(err)
		}
		if selector.Examples[len(selector.Examples)-1] != nil {
			t.Fatal("expected nil example to be stored as nil")
		}
	})

	t.Run("nil length falls back to default", func(t *testing.T) {
		selector.Length = nil
		if err := selector.AddExample(context.Background(), Example{"input": "x y"}); err != nil {
			t.Fatal(err)
		}
	})
}

func TestLengthBasedSelectorSelectExamples(t *testing.T) {
	formatter := func(e Example) (string, error) {
		return fmt.Sprintf("%v %v", e["input"], e["output"]), nil
	}
	newSelector := func(t *testing.T, maxLength int) *LengthBasedSelector {
		t.Helper()
		selector, err := NewLengthBased([]Example{
			{"input": "happy", "output": "sad"},  // length 2
			{"input": "tall", "output": "short"},  // length 2
			{"input": "fast", "output": "slow"},   // length 2
		}, formatter, maxLength)
		if err != nil {
			t.Fatal(err)
		}
		return selector
	}

	t.Run("cancelled context", func(t *testing.T) {
		selector := newSelector(t, 10)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := selector.SelectExamples(ctx, nil); err == nil {
			t.Fatal("expected error for cancelled context")
		}
	})

	t.Run("input consumes entire budget", func(t *testing.T) {
		selector := newSelector(t, 3)
		got, err := selector.SelectExamples(context.Background(), map[string]string{"input": "a b c"})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("selected %d examples, want 0", len(got))
		}
	})

	t.Run("stops at first example that does not fit", func(t *testing.T) {
		selector := newSelector(t, 3) // budget 3 fits one example of length 2, not two
		got, err := selector.SelectExamples(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 {
			t.Fatalf("selected %d examples, want 1", len(got))
		}
		if got[0]["input"] != "happy" {
			t.Fatalf("selected example %v, want first example", got[0])
		}
	})

	t.Run("all examples fit", func(t *testing.T) {
		selector := newSelector(t, 100)
		got, err := selector.SelectExamples(context.Background(), map[string]string{"input": "x"})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 3 {
			t.Fatalf("selected %d examples, want 3", len(got))
		}
	})

	t.Run("first example longer than remaining budget", func(t *testing.T) {
		// maxLength 2 minus length 1 for the (empty) joined input leaves
		// budget 1, so no example of length 2 fits.
		selector := newSelector(t, 2)
		got, err := selector.SelectExamples(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("selected %d examples, want 0", len(got))
		}
	})

	t.Run("nil length falls back to default", func(t *testing.T) {
		selector := newSelector(t, 100)
		selector.Length = nil
		got, err := selector.SelectExamples(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 3 {
			t.Fatalf("selected %d examples, want 3", len(got))
		}
	})
}

// fakeVectorStore is an in-memory VectorStore fake recording calls.
type fakeVectorStore struct {
	added      []documents.Document
	addErr     error
	searchDocs []documents.Document
	searchErr  error
	lastQuery  string
	lastK      int
}

func (f *fakeVectorStore) AddDocuments(_ context.Context, docs []documents.Document) ([]string, error) {
	if f.addErr != nil {
		return nil, f.addErr
	}
	ids := make([]string, len(docs))
	for i := range docs {
		ids[i] = fmt.Sprintf("id-%d", len(f.added)+i)
	}
	f.added = append(f.added, docs...)
	return ids, nil
}

func (f *fakeVectorStore) Delete(_ context.Context, _ []string) error { return nil }

func (f *fakeVectorStore) GetByIDs(_ context.Context, _ []string) ([]documents.Document, error) {
	return nil, nil
}

func (f *fakeVectorStore) SimilaritySearch(_ context.Context, query string, k int) ([]documents.Document, error) {
	f.lastQuery = query
	f.lastK = k
	if f.searchErr != nil {
		return nil, f.searchErr
	}
	return f.searchDocs, nil
}

func (f *fakeVectorStore) SimilaritySearchWithScore(_ context.Context, _ string, _ int) ([]vectorstores.SearchResult, error) {
	return nil, nil
}

func TestSemanticSimilaritySelectorAddExample(t *testing.T) {
	t.Run("nil store", func(t *testing.T) {
		selector := SemanticSimilaritySelector{}
		if err := selector.AddExample(context.Background(), Example{"a": 1}); err == nil {
			t.Fatal("expected error for nil store")
		}
	})

	t.Run("formatter error propagates", func(t *testing.T) {
		wantErr := errors.New("format failed")
		selector := SemanticSimilaritySelector{
			Store:     &fakeVectorStore{},
			Formatter: func(Example) (string, error) { return "", wantErr },
		}
		if err := selector.AddExample(context.Background(), Example{"a": 1}); !errors.Is(err, wantErr) {
			t.Fatalf("err = %v, want %v", err, wantErr)
		}
	})

	t.Run("store error propagates", func(t *testing.T) {
		wantErr := errors.New("store failed")
		selector := SemanticSimilaritySelector{Store: &fakeVectorStore{addErr: wantErr}}
		if err := selector.AddExample(context.Background(), Example{"a": 1}); !errors.Is(err, wantErr) {
			t.Fatalf("err = %v, want %v", err, wantErr)
		}
	})

	t.Run("adds formatted document with example metadata", func(t *testing.T) {
		store := &fakeVectorStore{}
		selector := SemanticSimilaritySelector{Store: store}
		example := Example{"input": "happy", "output": "sad"}
		if err := selector.AddExample(context.Background(), example); err != nil {
			t.Fatal(err)
		}
		if len(store.added) != 1 {
			t.Fatalf("store has %d documents, want 1", len(store.added))
		}
		doc := store.added[0]
		if doc.PageContent != "input: happy\noutput: sad" {
			t.Fatalf("PageContent = %q", doc.PageContent)
		}
		stored, ok := doc.Metadata["example"].(Example)
		if !ok {
			t.Fatalf("metadata example has type %T, want Example", doc.Metadata["example"])
		}
		example["input"] = "mutated"
		if stored["input"] != "happy" {
			t.Fatal("store kept reference to caller's example map")
		}
	})
}

func TestSemanticSimilaritySelectorSelectExamples(t *testing.T) {
	t.Run("nil store", func(t *testing.T) {
		selector := SemanticSimilaritySelector{}
		if _, err := selector.SelectExamples(context.Background(), nil); err == nil {
			t.Fatal("expected error for nil store")
		}
	})

	t.Run("store error propagates", func(t *testing.T) {
		wantErr := errors.New("search failed")
		selector := SemanticSimilaritySelector{Store: &fakeVectorStore{searchErr: wantErr}}
		if _, err := selector.SelectExamples(context.Background(), nil); !errors.Is(err, wantErr) {
			t.Fatalf("err = %v, want %v", err, wantErr)
		}
	})

	t.Run("defaults k to 4 and joins input values", func(t *testing.T) {
		store := &fakeVectorStore{}
		selector := SemanticSimilaritySelector{Store: store}
		if _, err := selector.SelectExamples(context.Background(), map[string]string{"input": "happy"}); err != nil {
			t.Fatal(err)
		}
		if store.lastK != 4 {
			t.Fatalf("k = %d, want default 4", store.lastK)
		}
		if store.lastQuery != "happy" {
			t.Fatalf("query = %q, want %q", store.lastQuery, "happy")
		}
	})

	t.Run("explicit k is used", func(t *testing.T) {
		store := &fakeVectorStore{}
		selector := SemanticSimilaritySelector{Store: store, K: 2}
		if _, err := selector.SelectExamples(context.Background(), nil); err != nil {
			t.Fatal(err)
		}
		if store.lastK != 2 {
			t.Fatalf("k = %d, want 2", store.lastK)
		}
	})

	t.Run("extracts examples from metadata", func(t *testing.T) {
		store := &fakeVectorStore{searchDocs: []documents.Document{
			{Metadata: map[string]any{"example": Example{"input": "happy"}}},
			{Metadata: map[string]any{"example": map[string]any{"input": "tall"}}},
			{Metadata: map[string]any{"example": "not an example"}},
			{Metadata: map[string]any{}},
		}}
		selector := SemanticSimilaritySelector{Store: store}
		got, err := selector.SelectExamples(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		want := []Example{{"input": "happy"}, {"input": "tall"}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("examples = %v, want %v", got, want)
		}
		got[0]["input"] = "mutated"
		again, err := selector.SelectExamples(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if again[0]["input"] != "happy" {
			t.Fatal("selector returned non-cloned example map")
		}
	})

	t.Run("empty result set", func(t *testing.T) {
		store := &fakeVectorStore{}
		selector := SemanticSimilaritySelector{Store: store}
		got, err := selector.SelectExamples(context.Background(), map[string]string{"input": "x"})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("selected %d examples, want 0", len(got))
		}
		if strings.Contains(store.lastQuery, "\x00") {
			t.Fatal("unexpected query content")
		}
	})
}
