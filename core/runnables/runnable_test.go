package runnables

import (
	"context"
	"errors"
	"testing"

	"github.com/projanvil/langchain-golang/core/callbacks"
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

func TestConfigOptions(t *testing.T) {
	manager := callbacks.Manager{}
	cfg := NewConfig(
		WithName("named"),
		WithParentID("parent-1"),
		WithCallbacks(manager),
		WithMetadata("key", "value"),
		WithConfigurable("mode", "fast"),
	)
	if cfg.Name != "named" {
		t.Fatalf("name: %q", cfg.Name)
	}
	if cfg.ParentID != "parent-1" {
		t.Fatalf("parent id: %q", cfg.ParentID)
	}
	if cfg.Metadata["key"] != "value" {
		t.Fatalf("metadata: %#v", cfg.Metadata)
	}
	if cfg.Configurable["mode"] != "fast" {
		t.Fatalf("configurable: %#v", cfg.Configurable)
	}
}

func TestWithMetadataNilMap(t *testing.T) {
	// Applying metadata/configurable options to a zero-value Config must
	// initialize the maps instead of panicking.
	cfg := Config{}
	WithMetadata("k", "v")(&cfg)
	WithConfigurable("c", 1)(&cfg)
	if cfg.Metadata["k"] != "v" || cfg.Configurable["c"] != 1 {
		t.Fatalf("cfg: %#v", cfg)
	}
}

func TestFuncStreamPropagatesInvokeError(t *testing.T) {
	wantErr := errTestSentinel
	r := NewFunc(func(_ context.Context, input string, _ ...Option) (string, error) {
		return "", wantErr
	}, schema.String(""), schema.String(""))

	if _, err := r.Stream(context.Background(), "x"); err != wantErr {
		t.Fatalf("stream err: got %v want %v", err, wantErr)
	}
}

func TestFuncBatchJoinsErrors(t *testing.T) {
	r := NewFunc(func(_ context.Context, input int, _ ...Option) (int, error) {
		if input < 0 {
			return 0, errTestSentinel
		}
		return input, nil
	}, schema.Integer(""), schema.Integer(""))

	got, err := r.Batch(context.Background(), []int{1, -1})
	if err == nil {
		t.Fatal("expected joined error")
	}
	if got[0] != 1 {
		t.Fatalf("outputs: %#v", got)
	}
}

var errTestSentinel = errors.New("sentinel")

func TestConfigCloneNilMaps(t *testing.T) {
	cfg := Config{}.Clone()
	if cfg.Metadata == nil || cfg.Configurable == nil {
		t.Fatalf("clone must initialize nil maps: %#v", cfg)
	}
	if cfg.Tags != nil {
		t.Fatalf("clone of nil tags: %#v", cfg.Tags)
	}
}
