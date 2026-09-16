package exampleselectors

import (
	"testing"
)

func TestLengthBasedSelector(t *testing.T) {
	selector, err := NewLengthBased([]Example{
		{"input": "happy", "output": "sad"},
		{"input": "tall", "output": "short"},
		{"input": "fast", "output": "slow"},
	}, func(e Example) (string, error) {
		return e["input"].(string) + " " + e["output"].(string), nil
	}, 5)
	if err != nil {
		t.Fatal(err)
	}
	got, err := selector.SelectExamples(t.Context(), map[string]string{"input": "large"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("selected %d examples, want 2", len(got))
	}
	got[0]["input"] = "mutated"
	if selector.Examples[0]["input"] == "mutated" {
		t.Fatal("selector returned internal example map")
	}
}
