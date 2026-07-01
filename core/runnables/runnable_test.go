package runnables

import (
	"context"
	"testing"

	"github.com/projanvil/langchain-golang/core/schema"
)

func TestFuncBatchPreservesOrder(t *testing.T) {
	r := NewFunc(
		func(_ context.Context, input int, _ ...Option) (int, error) {
			return input * 2, nil
		},
		schema.Integer("input"),
		schema.Integer("output"),
	)

	got, err := r.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("batch: %v", err)
	}

	want := []int{2, 4, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output[%d]: got %d want %d", i, got[i], want[i])
		}
	}
}

func TestFuncStream(t *testing.T) {
	r := NewFunc(
		func(_ context.Context, input string, _ ...Option) (string, error) {
			return input + "!", nil
		},
		schema.String("input"),
		schema.String("output"),
	)

	stream, err := r.Stream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer stream.Close()

	got, ok, err := stream.Next(context.Background())
	if err != nil {
		t.Fatalf("next: %v", err)
	}
	if !ok || got != "hello!" {
		t.Fatalf("first chunk: got %q ok=%v", got, ok)
	}

	_, ok, err = stream.Next(context.Background())
	if err != nil {
		t.Fatalf("next end: %v", err)
	}
	if ok {
		t.Fatal("expected stream end")
	}
}
