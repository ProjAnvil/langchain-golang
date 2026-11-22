package exampleselectors_test

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/projanvil/langchain-golang/core/embeddings"
	"github.com/projanvil/langchain-golang/core/exampleselectors"
)

func TestSortedValuesByKey(t *testing.T) {
	got := exampleselectors.SortedValues(map[string]string{"c": "3", "a": "1", "b": "2"})
	want := []string{"1", "2", "3"}
	if len(got) != len(want) {
		t.Fatalf("SortedValues = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SortedValues = %v, want %v", got, want)
		}
	}

	// Works with any-valued maps too; values are rendered with fmt.Sprint.
	gotAny := exampleselectors.SortedValues(map[string]any{"z": 9, "y": true})
	wantAny := []string{"true", "9"}
	for i := range wantAny {
		if gotAny[i] != wantAny[i] {
			t.Fatalf("SortedValues(any) = %v, want %v", gotAny, wantAny)
		}
	}
}

func TestExampleToText(t *testing.T) {
	example := exampleselectors.Example{"input": "hello", "output": "world"}
	if got, want := exampleselectors.ExampleToText(example, nil), "hello world"; got != want {
		t.Fatalf("ExampleToText = %q, want %q", got, want)
	}
	// inputKeys limit the fields used, still ordered by key.
	if got, want := exampleselectors.ExampleToText(example, []string{"output"}), "world"; got != want {
		t.Fatalf("ExampleToText(inputKeys) = %q, want %q", got, want)
	}
}

// controlledMMRSetup returns a Static embedder and three examples where the
// second is a near-duplicate of the first (both highly similar to the query)
// while the third is diverse. DocumentVectors are unit-length directions at
// roughly 10, 12, and 40 degrees from the query [1, 0].
func controlledMMRSetup() (embeddings.Static, []exampleselectors.Example) {
	embedder := embeddings.Static{
		QueryVector: []float64{1, 0},
		DocumentVectors: [][]float64{
			{0.9848, 0.1736}, // A: closest to the query
			{0.9781, 0.2079}, // A2: near-duplicate of A
			{0.7660, 0.6428}, // C: diverse
		},
	}
	examples := []exampleselectors.Example{
		{"id": "A", "input": "alpha"},
		{"id": "A2", "input": "alpha-again"},
		{"id": "C", "input": "gamma"},
	}
	return embedder, examples
}

func selectedIDs(t *testing.T, results []exampleselectors.Example) []string {
	t.Helper()
	ids := make([]string, len(results))
	for i, result := range results {
		ids[i] = fmt.Sprint(result["id"])
	}
	return ids
}

func TestFromExamplesMMRSelectsDiverseExamples(t *testing.T) {
	embedder, examples := controlledMMRSetup()
	selector, err := exampleselectors.FromExamples(examples, embedder,
		exampleselectors.WithK(2), exampleselectors.WithLambdaMult(0.1))
	if err != nil {
		t.Fatalf("FromExamples: %v", err)
	}
	results, err := selector.SelectExamples(context.Background(), map[string]string{"input": "query"})
	if err != nil {
		t.Fatalf("SelectExamples: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d examples, want 2", len(results))
	}
	// MMR picks the best query match first, then prefers the diverse example
	// over the redundant near-duplicate. Pure similarity would pick A, A2.
	got := selectedIDs(t, results)
	if got[0] != "A" || got[1] != "C" {
		t.Fatalf("MMR order = %v, want [A C] (diversity beats the duplicate)", got)
	}
}

func TestFromExamplesMMRHighLambdaKeepsSimilarityOrder(t *testing.T) {
	embedder, examples := controlledMMRSetup()
	// lambdaMult=1 ignores diversity, so the near-duplicate is kept.
	selector, err := exampleselectors.FromExamples(examples, embedder,
		exampleselectors.WithK(2), exampleselectors.WithLambdaMult(1.0))
	if err != nil {
		t.Fatalf("FromExamples: %v", err)
	}
	results, err := selector.SelectExamples(context.Background(), map[string]string{"input": "query"})
	if err != nil {
		t.Fatalf("SelectExamples: %v", err)
	}
	got := selectedIDs(t, results)
	if got[0] != "A" || got[1] != "A2" {
		t.Fatalf("similarity order = %v, want [A A2]", got)
	}
}

func TestFromExamplesRespectsExampleKeys(t *testing.T) {
	embedder, examples := controlledMMRSetup()
	selector, err := exampleselectors.FromExamples(examples, embedder,
		exampleselectors.WithK(1), exampleselectors.WithExampleKeys("input"))
	if err != nil {
		t.Fatalf("FromExamples: %v", err)
	}
	results, err := selector.SelectExamples(context.Background(), map[string]string{"input": "query"})
	if err != nil {
		t.Fatalf("SelectExamples: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d examples, want 1", len(results))
	}
	if _, ok := results[0]["id"]; ok {
		t.Fatalf("exampleKeys filtered result still contains id: %#v", results[0])
	}
	if len(results[0]) != 1 || results[0]["input"] == nil {
		t.Fatalf("result = %#v, want only the input key", results[0])
	}
}

func TestMMRAddExampleThenSelect(t *testing.T) {
	embedder := embeddings.NewDeterministicFake(8)
	selector, err := exampleselectors.FromExamples(nil, embedder)
	if err != nil {
		t.Fatalf("FromExamples: %v", err)
	}
	example := exampleselectors.Example{"input": "hello", "output": "world"}
	if err := selector.AddExample(context.Background(), example); err != nil {
		t.Fatalf("AddExample: %v", err)
	}
	results, err := selector.SelectExamples(context.Background(), map[string]string{"input": "hello"})
	if err != nil {
		t.Fatalf("SelectExamples: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d examples, want 1", len(results))
	}
	if results[0]["input"] != "hello" || results[0]["output"] != "world" {
		t.Fatalf("result = %#v, want the added example", results[0])
	}
}

func TestMMRConstructorValidation(t *testing.T) {
	if _, err := exampleselectors.NewMaxMarginalRelevanceExampleSelector(nil); err == nil {
		t.Fatalf("expected an error for a nil store")
	}
	if _, err := exampleselectors.FromExamples(nil, nil); err == nil {
		t.Fatalf("expected an error for a nil embedder")
	}
}

func TestMMRRespectsContextCancellation(t *testing.T) {
	embedder := embeddings.NewDeterministicFake(8)
	selector, err := exampleselectors.FromExamples(nil, embedder)
	if err != nil {
		t.Fatalf("FromExamples: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := selector.SelectExamples(ctx, map[string]string{"input": "q"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("SelectExamples error = %v, want context.Canceled", err)
	}
	if err := selector.AddExample(ctx, exampleselectors.Example{"input": "x"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("AddExample error = %v, want context.Canceled", err)
	}
}
